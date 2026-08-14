/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	gmsruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

// ──────────────────────────────────────────────────────────────────────────────
// Inter-pod GMS failover (Mode: interPod)
//
// A dedicated GMS weight server pod is created per rank. Engine pods share GPU
// memory via DRA ResourceClaims and a hostPath volume for UDS sockets.
// ──────────────────────────────────────────────────────────────────────────────

const (
	gmsSharedVolumeName = "gms-shared"
	gmsHostPathBase     = "/run/gms"
	gmsSharedMountPath  = "/run/gms/shared"
	gmsFailoverLockFile = "failover.lock"
	gmsPermFixInitName  = "fix-gms-perms"
)

// gmsWrapperScript generates a bash script that launches the GMS server
// (gpu_memory_service.cli.server), which auto-discovers DRA-allocated GPUs
// and exposes both "weights" and "kv_cache" UDS sockets per device. The
// wrapper cleans up stale sockets from a previous run, forwards SIGTERM/SIGINT
// to the process group, and propagates the GMS server's exit code so the
// container's exitCode in the Pod status reflects the actual failure mode
// (rather than always being 1).
func gmsWrapperScript() string {
	return fmt.Sprintf(
		`rm -f %s/gms_*.sock
rc=1
cleanup() { kill -- -$$ 2>/dev/null; exit "$rc"; }
trap cleanup SIGTERM SIGINT
python3 -m %s &
echo "Started GMS server pid=$!"
wait -n
rc=$?
echo "GMS server exited (code=$rc), shutting down"
cleanup`, gmsSharedMountPath, gmsruntime.ServerModule)
}

// gmsStartupProbeCommand returns the exec probe command that verifies the GMS
// server has opened both the weights and kv_cache UDS sockets for every
// allocated GPU (2 sockets per device).
func gmsStartupProbeCommand(gpuCount int) []string {
	return []string{
		"sh", "-c",
		fmt.Sprintf("test $(ls %s/gms_*.sock 2>/dev/null | wc -l) -ge %d", gmsSharedMountPath, 2*gpuCount),
	}
}

// applyGMSSharedResources attaches the resources common to both GMS weight
// server pods and engine pods: strips GPU limits (DRA handles allocation),
// adds the GPU toleration, mounts the rank-isolated hostPath shared volume,
// and prepends the permission-fix init container.
func applyGMSSharedResources(podSpec *corev1.PodSpec, c *corev1.Container, rank int32) {
	removeGPUFromLimits(c)
	addGPUToleration(podSpec)
	vol, mount := gmsSharedVolume(rank)
	podSpec.Volumes = append(podSpec.Volumes, vol)
	c.VolumeMounts = append(c.VolumeMounts, mount)
	podSpec.InitContainers = append(podSpec.InitContainers, gmsPermFixInitContainer(rank, c.Image))
}

// gmsWeightServerPodSpec builds a GMS weight server pod spec by cloning and
// modifying a base engine pod spec. The GMS pod runs a different command,
// has no liveness/readiness probes, and uses a startup probe that checks
// for the expected number of GMS UDS sockets.
//
// RestartPolicy is intentionally left unset here (i.e. inherits the base /
// Grove default, which is Always). A GMS server process holds only local
// state — GPU allocations (via DRA, which survive the container), hostPath
// UDS sockets (recreated by gmsWrapperScript on startup), and in-memory
// weight buffers (re-sharded on reconnection by the engine clients). So an
// in-place kubelet restart is a fast, correct recovery path.
//
// The paired engine pod mirrors this policy in the standalone inter-pod GMS
// layout (a restarted engine re-imports IPC handles from the still-running
// GMS server). In the inter-pod GMS failover layout, augmentEngineForGMS
// overrides the engine's RestartPolicy to Never so the cohort can only be
// recovered by the failover cascade controller; see
// failover_cascade_controller.go.
func gmsWeightServerPodSpec(basePodSpec *corev1.PodSpec, rank int32, gpuCount int) *corev1.PodSpec {
	podSpec := basePodSpec.DeepCopy()
	if len(podSpec.Containers) == 0 {
		return podSpec
	}

	c := &podSpec.Containers[0]
	c.Command = []string{"bash", "-c"}
	c.Args = []string{gmsWrapperScript()}

	c.StartupProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{Command: gmsStartupProbeCommand(gpuCount)},
		},
		PeriodSeconds:    2,
		TimeoutSeconds:   2,
		FailureThreshold: 150, // 2s * 150 = 5 min
	}
	c.LivenessProbe = nil
	c.ReadinessProbe = nil

	c.Env = append(c.Env, corev1.EnvVar{
		Name:  gmsruntime.EnvSocketDir,
		Value: gmsSharedMountPath,
	})

	applyGMSSharedResources(podSpec, c, rank)

	return podSpec
}

// gmsEngineEnvVars returns the backend-agnostic environment variables injected
// into engine pods when GMS failover is enabled. Backend-specific switches
// (e.g. the vLLM DYN_VLLM_GMS_SHADOW_MODE flag) are injected by the backend's
// UpdateContainer path so non-vLLM backends do not inherit stray env vars.
func gmsEngineEnvVars() []corev1.EnvVar {
	return []corev1.EnvVar{
		{
			Name: "ENGINE_ID",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.labels['grove.io/podclique-pod-index']",
				},
			},
		},
		{Name: gmsruntime.EnvSocketDir, Value: gmsSharedMountPath},
		{Name: "FAILOVER_LOCK_PATH", Value: gmsSharedMountPath + "/" + gmsFailoverLockFile},
		{Name: "DYN_SYSTEM_STARTING_HEALTH_STATUS", Value: "notready"},
	}
}

// augmentEngineForGMS modifies an engine pod spec in-place to work with the
// inter-pod GMS layout: injects env vars, shared volume, strips GPU limits,
// adds toleration, and prepends an init container to fix hostPath directory
// permissions.
//
// RestartPolicy behavior is layout-dependent and is the one asymmetry between
// standalone inter-pod GMS and inter-pod GMS failover:
//
//   - Standalone inter-pod GMS (isInterPodFailover=false): RestartPolicy is
//     left unset (inherits Always), matching the GMS weight-server pod. A
//     crashed engine is restarted in place by kubelet; the GMS server keeps
//     running and the new engine container reconnects to the existing UDS
//     sockets and re-imports CUDA IPC handles during --load-format gms
//     startup. There is no cohort state to protect because there is no
//     cohort — just one engine paired with one GMS server per rank.
//
//   - Inter-pod GMS failover (isInterPodFailover=true): RestartPolicy is
//     forced to Never. Engine pods in a failover cohort hold distributed
//     state that cannot survive an in-place container restart — active NCCL
//     collectives, torch.distributed TCPStore membership, and primary/shadow
//     coordination via the failover lock file and DYN_VLLM_GMS_SHADOW_MODE.
//     An in-place restart leaves the cohort in a half-torn-down state and
//     blocks recovery. The correct recovery path is for the pod to exit,
//     the failover cascade controller (failover_cascade_controller.go) to
//     force-delete the full engine group based on the
//     KubeLabelDynamoFailoverEngineGroupMember label, and Grove to recreate
//     the cohort from scratch. That label is applied in graph.go only when
//     isInterPodFailover is true, so forcing Never in the standalone case
//     would strand engine pods in Failed state with nothing listening to
//     force-delete them.
func augmentEngineForGMS(podSpec *corev1.PodSpec, rank int32, isInterPodFailover bool) {
	if len(podSpec.Containers) == 0 {
		return
	}
	c := &podSpec.Containers[0]

	c.Env = append(c.Env, gmsEngineEnvVars()...)
	removeEnvVar(c, "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS")

	applyGMSSharedResources(podSpec, c, rank)
	if isInterPodFailover {
		podSpec.RestartPolicy = corev1.RestartPolicyNever
	}
}

// gmsSharedVolume returns a hostPath volume and mount with a subPathExpr that
// isolates the shared directory per PCSG replica and per rank.
func gmsSharedVolume(rank int32) (corev1.Volume, corev1.VolumeMount) {
	hostPathType := corev1.HostPathDirectoryOrCreate
	vol := corev1.Volume{
		Name: gmsSharedVolumeName,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: gmsHostPathBase,
				Type: &hostPathType,
			},
		},
	}
	mount := corev1.VolumeMount{
		Name:        gmsSharedVolumeName,
		MountPath:   gmsSharedMountPath,
		SubPathExpr: fmt.Sprintf("$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)/rank-%d", rank),
	}
	return vol, mount
}

// gmsPermFixInitContainer returns an init container that runs as root and
// fixes the hostPath directory permissions so the non-root application user
// can write UDS sockets and lock files. It uses the same subPathExpr as the
// main container so kubelet creates the isolated subdirectory first.
func gmsPermFixInitContainer(rank int32, image string) corev1.Container {
	_, mount := gmsSharedVolume(rank)
	return corev1.Container{
		Name:    gmsPermFixInitName,
		Image:   image,
		Command: []string{"sh", "-c", fmt.Sprintf("chmod 1777 %s", gmsSharedMountPath)},
		SecurityContext: &corev1.SecurityContext{
			// Must run as uid 0 to chmod the hostPath mount for the non-root
			// engine/server processes. Explicitly set RunAsNonRoot=false so
			// cluster-wide baseline/restricted PodSecurity policies and some
			// pod-level SecurityContext defaults do not silently reject this
			// init container on admission.
			RunAsUser:    ptr.To[int64](0),
			RunAsNonRoot: ptr.To(false),
		},
		VolumeMounts: []corev1.VolumeMount{mount},
	}
}

// removeGPUFromLimits strips scalar GPU resources from the container's resource
// limits and requests because DRA handles GPU allocation for GMS pods.
func removeGPUFromLimits(c *corev1.Container) {
	dra.RemoveGPUResources(c.Resources.Limits)
	dra.RemoveGPUResources(c.Resources.Requests)
}

// addGPUToleration ensures pods without explicit GPU limits still get
// scheduled on GPU nodes.
func addGPUToleration(podSpec *corev1.PodSpec) {
	toleration := corev1.Toleration{
		Key:      "nvidia.com/gpu",
		Operator: corev1.TolerationOpExists,
		Effect:   corev1.TaintEffectNoSchedule,
	}
	for _, t := range podSpec.Tolerations {
		if t.Key == toleration.Key && t.Effect == toleration.Effect {
			return
		}
	}
	podSpec.Tolerations = append(podSpec.Tolerations, toleration)
}

// removeEnvVar removes all occurrences of the named env var from a container.
func removeEnvVar(c *corev1.Container, name string) {
	filtered := c.Env[:0]
	for _, e := range c.Env {
		if e.Name != name {
			filtered = append(filtered, e)
		}
	}
	c.Env = filtered
}

// getGPUCount extracts the GPU count from the component's Kubernetes resource requirements.
func getGPUCount(resources corev1.ResourceRequirements) (int32, error) {
	gpuCount, err := dra.ExtractGPUCountFromResourceRequirements(resources)
	if err != nil {
		return 0, err
	}
	return int32(gpuCount), nil
}

// getDeviceClassName returns the DRA device class name from the GMS config,
// falling back to the default device class shipped with the NVIDIA DRA
// driver. The literal "gpu.nvidia.com" is intentionally not duplicated
// here — it is the single source of truth in the dra package.
func getDeviceClassName(gmsSpec *v1beta1.GPUMemoryServiceSpec) string {
	if gmsSpec != nil && gmsSpec.DeviceClassName != "" {
		return gmsSpec.DeviceClassName
	}
	return dra.DefaultDeviceClassName
}

// gmsRCTName returns a deterministic ResourceClaimTemplate name for a given rank.
func gmsRCTName(serviceName string, rank int32) string {
	return fmt.Sprintf("%s-gpu-rank-%d", NormalizeKubeResourceName(serviceName), rank)
}

// gmsResourceClaimTemplateConfigs builds one PCS-level ResourceClaimTemplateConfig
// per rank. Each RCT has the same GPU spec but a distinct per-rank name so that
// each rank's GMS + engine pods get their own ResourceClaim.
func gmsResourceClaimTemplateConfigs(serviceName string, gmsSpec *v1beta1.GPUMemoryServiceSpec, resources corev1.ResourceRequirements, roles []ServiceRole) ([]grovev1alpha1.ResourceClaimTemplateConfig, error) {
	gpuCount, err := getGPUCount(resources)
	if err != nil {
		return nil, err
	}
	seen := map[int32]bool{}
	configs := make([]grovev1alpha1.ResourceClaimTemplateConfig, 0, len(roles))
	for _, r := range roles {
		if seen[r.Rank] {
			continue
		}
		seen[r.Rank] = true
		configs = append(configs, grovev1alpha1.ResourceClaimTemplateConfig{
			Name: gmsRCTName(serviceName, r.Rank),
			TemplateSpec: resourcev1.ResourceClaimTemplateSpec{
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "gpu",
								Exactly: &resourcev1.ExactDeviceRequest{
									DeviceClassName: getDeviceClassName(gmsSpec),
									AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
									Count:           int64(gpuCount),
								},
							},
						},
					},
				},
			},
		})
	}
	return configs, nil
}

// gmsResourceSharingEntries builds one PCSG-level ResourceSharingSpec per rank.
// Each entry uses PerReplica scope and a filter listing only the GMS clique
// and the engine clique for that rank, ensuring GPU isolation between ranks.
func gmsResourceSharingEntries(serviceName string, roles []ServiceRole) []grovev1alpha1.PCSGResourceSharingSpec {
	type rankGroup struct {
		cliqueNames []string
	}
	groups := map[int32]*rankGroup{}
	var rankOrder []int32

	for _, r := range roles {
		g, ok := groups[r.Rank]
		if !ok {
			g = &rankGroup{}
			groups[r.Rank] = g
			rankOrder = append(rankOrder, r.Rank)
		}
		g.cliqueNames = append(g.cliqueNames, strings.ToLower(r.Name))
	}

	refs := make([]grovev1alpha1.PCSGResourceSharingSpec, 0, len(groups))
	for _, rank := range rankOrder {
		g := groups[rank]
		refs = append(refs, grovev1alpha1.PCSGResourceSharingSpec{
			ResourceSharingSpec: grovev1alpha1.ResourceSharingSpec{
				Name:  gmsRCTName(serviceName, rank),
				Scope: grovev1alpha1.ResourceSharingScopePerReplica,
			},
			Filter: &grovev1alpha1.PCSGResourceSharingFilter{
				ChildCliqueNames: g.cliqueNames,
			},
		})
	}
	return refs
}

// ──────────────────────────────────────────────────────────────────────────────
// Intra-pod GMS failover (Mode: intraPod)
//
// The main container is cloned into active and standby engine containers within
// the same pod. GPU access is shared via DRA and a GMS sidecar injects weights
// via the shared emptyDir volume.
// ──────────────────────────────────────────────────────────────────────────────

// intraPodFailoverLockFile is the lock file path used by engine containers to
// coordinate active/standby election within the same pod.
var intraPodFailoverLockFile = filepath.Join(gmsruntime.SharedMountPath, "failover.lock")

const (
	failoverEngineCount                    = 2
	vllmModuleName                         = "dynamo.vllm"
	sglangModuleName                       = "dynamo.sglang"
	vllmLoadFormatFlag                     = "--load-format"
	vllmWorkerClassFlag                    = "--worker-cls"
	gmsV0VLLMWorkerClass                   = "gpu_memory_service.integrations.vllm.worker.GMSWorker"
	gmsV0VLLMWorkerClassAlt                = "gpu_memory_service.integrations.vllm.worker:GMSWorker"
	gmsV1VLLMWorkerClass                   = "gpu_memory_service.v1.integrations.vllm.worker.GMSV1Worker"
	checkpointFailoverCompatibilityMessage = "Snapshot with active/passive failover requires an operator-managed automatic single-node Worker checkpoint"
)

// IsDGDControlled reports whether a DCD has an exact DynamoGraphDeployment
// controller reference. Admission separately verifies the request principal.
func IsDGDControlled(dcd *v1beta1.DynamoComponentDeployment) bool {
	if dcd == nil {
		return false
	}
	owner := metav1.GetControllerOf(dcd)
	return owner != nil &&
		owner.APIVersion == v1beta1.GroupVersion.String() &&
		owner.Kind == "DynamoGraphDeployment" &&
		owner.Name != "" &&
		owner.UID != ""
}

// ValidateAutomaticFailoverCheckpointSource validates the DGD component that
// produces the one canonical checkpoint source.
func ValidateAutomaticFailoverCheckpointSource(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	backendFramework string,
) []error {
	if !hasCheckpointFailover(component) {
		return nil
	}
	violations := validateAutomaticFailoverCheckpointProfile(component, backendFramework)
	config := component.Experimental.Checkpoint
	if config.CheckpointRef != nil && *config.CheckpointRef != "" {
		violations = append(violations, errors.New("checkpointRef must be omitted so the DGD owns the automatic checkpoint"))
	}
	return wrapFailoverCompatibilityViolations(violations)
}

// ValidateAutomaticFailoverCheckpointTarget validates the DCD that restores
// the DGD-owned checkpoint into the configured failover engines.
func ValidateAutomaticFailoverCheckpointTarget(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	backendFramework string,
	operatorGenerated bool,
) []error {
	if !hasCheckpointFailover(component) {
		return nil
	}
	violations := validateAutomaticFailoverCheckpointProfile(component, backendFramework)
	if !operatorGenerated {
		violations = append(violations, errors.New("checkpoint failover is only supported for an operator-generated DCD"))
	}
	config := component.Experimental.Checkpoint
	if config.CheckpointRef == nil || *config.CheckpointRef == "" {
		violations = append(violations, errors.New("checkpointRef must name the DGD-owned automatic checkpoint"))
	}
	return wrapFailoverCompatibilityViolations(violations)
}

func hasCheckpointFailover(component *v1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return component != nil &&
		component.Experimental != nil &&
		component.Experimental.Checkpoint != nil &&
		component.Experimental.Checkpoint.Enabled &&
		(component.Experimental.GPUMemoryService == nil ||
			component.Experimental.GPUMemoryService.Mode != v1beta1.GMSModeInterPod) &&
		IsIntraPodFailoverEnabled(component)
}

func validateAutomaticFailoverCheckpointProfile(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	backendFramework string,
) []error {
	experimental := component.Experimental
	config := experimental.Checkpoint
	violations := make([]error, 0, 12)

	if config.Mode != "" && config.Mode != v1beta1.CheckpointModeAuto {
		violations = append(violations, errors.New("checkpoint mode must be automatic"))
	}
	if config.DeletionPolicy != "" && config.DeletionPolicy != v1beta1.CheckpointDeletionPolicyDelete {
		violations = append(violations, errors.New("deletionPolicy must be Delete"))
	}
	if config.TargetContainerName != "" && config.TargetContainerName != commonconsts.MainContainerName {
		violations = append(violations, errors.New("targetContainerName must be main"))
	}
	backend := BackendFramework(backendFramework)
	if backend != BackendFrameworkVLLM && backend != BackendFrameworkSGLang {
		violations = append(violations, errors.New("backendFramework must be vllm or sglang"))
	}
	if component.ComponentType != v1beta1.ComponentTypeWorker {
		violations = append(violations, errors.New("component type must be Worker"))
	}
	if experimental.GPUMemoryService == nil ||
		(experimental.GPUMemoryService.Mode != "" &&
			experimental.GPUMemoryService.Mode != v1beta1.GMSModeIntraPod) {
		violations = append(violations, errors.New("gpuMemoryService.mode must be IntraPod"))
	}
	if !IsIntraPodFailoverEnabled(component) {
		violations = append(violations, errors.New("failover.mode must be IntraPod"))
	}
	if experimental.Failover.NumShadows < 0 || experimental.Failover.NumShadows > 2 {
		violations = append(violations, errors.New("failover.numShadows must be 1 or 2"))
	}
	if component.GetNumberOfNodes() != 1 {
		violations = append(violations, errors.New("worker must use exactly one node"))
	}

	main := GetMainContainer(component)
	if main == nil {
		return append(violations, errors.New("podTemplate must contain the main container"))
	}
	gpuCount, err := getGPUCount(main.Resources)
	if err != nil {
		violations = append(violations, fmt.Errorf("main container GPU resources are invalid: %w", err))
	} else if gpuCount != 1 {
		violations = append(violations, errors.New("main container must request exactly one GPU"))
	}
	switch backend {
	case BackendFrameworkVLLM:
		if err := configureVLLMAutomaticSnapshotLoadProfile(main.DeepCopy()); err != nil {
			violations = append(violations, err)
			return violations
		}
		violations = append(violations, validateAutomaticSnapshotFlags(main.Args, []automaticSnapshotFlag{
			{flag: "--disaggregation-mode", defaultValue: "agg", want: "agg", description: "disaggregation mode must be aggregated"},
			{flag: "--request-plane", defaultValue: "tcp", want: "tcp", description: "request plane must be tcp"},
			{flag: tensorParallelSizeFlag, defaultValue: "1", want: "1", description: "tensor parallel size must be 1"},
			{flag: pipelineParallelSizeFlag, defaultValue: "1", want: "1", description: "pipeline parallel size must be 1"},
			{flag: dataParallelSizeFlag, defaultValue: "1", want: "1", description: "data parallel size must be 1"},
		})...)
	case BackendFrameworkSGLang:
		violations = append(violations, validateSGLangAutomaticSnapshotProfile(main)...)
	}
	return violations
}

type automaticSnapshotFlag struct {
	flag         string
	defaultValue string
	want         string
	description  string
}

func validateAutomaticSnapshotFlags(args []string, profiles []automaticSnapshotFlag) []error {
	var violations []error
	for _, profile := range profiles {
		value, _, _, found, err := tokenizedFlag(args, profile.flag)
		if err != nil {
			violations = append(violations, err)
			continue
		}
		if !found {
			value = profile.defaultValue
		}
		if value != profile.want {
			violations = append(violations, fmt.Errorf("%s (got %q)", profile.description, value))
		}
	}
	return violations
}

func validateSGLangAutomaticSnapshotProfile(container *corev1.Container) []error {
	if !isDirectDynamoModuleCommand(container, sglangModuleName) {
		return []error{fmt.Errorf(
			"requires a direct python -m %s command with tokenized arguments (command=%q args=%q)",
			sglangModuleName,
			container.Command,
			container.Args,
		)}
	}
	return validateAutomaticSnapshotFlags(container.Args, []automaticSnapshotFlag{
		{flag: "--disaggregation-mode", defaultValue: "agg", want: "agg", description: "disaggregation mode must be aggregated"},
		{flag: "--request-plane", defaultValue: "tcp", want: "tcp", description: "request plane must be tcp"},
		{flag: "--tp", defaultValue: "1", want: "1", description: "tensor parallel size must be 1"},
		{flag: "--dp-size", defaultValue: "1", want: "1", description: "data parallel size must be 1"},
		{flag: "--pp-size", defaultValue: "1", want: "1", description: "pipeline parallel size must be 1"},
	})
}

func wrapFailoverCompatibilityViolations(violations []error) []error {
	if len(violations) == 0 {
		return nil
	}
	return []error{fmt.Errorf("%s: %w", checkpointFailoverCompatibilityMessage, errors.Join(violations...))}
}

// PrepareVLLMAutomaticFailoverSnapshotSource applies only checkpoint-source
// changes to the canonical main container. Destination engine topology is
// added later when the DGD worker pod is rendered.
func PrepareVLLMAutomaticFailoverSnapshotSource(container *corev1.Container) error {
	if container == nil {
		return fmt.Errorf("automatic failover snapshot source container is nil")
	}
	updated := container.DeepCopy()
	removeEnvVar(updated, "DYN_FORWARDPASS_METRIC_PORT")
	updated.Env = MergeEnvs(updated.Env, []corev1.EnvVar{
		{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "false"},
		{Name: "DYN_VLLM_DISAGGREGATION_MODE", Value: "agg"},
		{Name: "DYN_REQUEST_PLANE", Value: "tcp"},
	})
	if err := configureVLLMAutomaticSnapshotLoadProfile(updated); err != nil {
		return fmt.Errorf("automatic failover snapshot source: %w", err)
	}
	*container = *updated
	return nil
}

// PrepareSGLangAutomaticFailoverSnapshotSource selects the GMS V1 snapshot
// plugin without adding destination-only election state to the source process.
func PrepareSGLangAutomaticFailoverSnapshotSource(container *corev1.Container) error {
	if container == nil {
		return fmt.Errorf("automatic failover snapshot source container is nil")
	}
	if violations := validateSGLangAutomaticSnapshotProfile(container); len(violations) > 0 {
		return fmt.Errorf("automatic failover snapshot source: %w", errors.Join(violations...))
	}
	removeEnvVar(container, "DYN_FORWARDPASS_METRIC_PORT")
	EnableSGLangGMSV1(container)
	container.Env = MergeEnvs(container.Env, []corev1.EnvVar{{
		Name:  "DYN_REQUEST_PLANE",
		Value: "tcp",
	}})
	return nil
}

func configureVLLMAutomaticSnapshotLoadProfile(container *corev1.Container) error {
	if !isDirectDynamoVLLMCommand(container) {
		return fmt.Errorf(
			"requires a direct python -m %s command with tokenized arguments (command=%q args=%q)",
			vllmModuleName,
			container.Command,
			container.Args,
		)
	}
	if slices.Contains(container.Args, "--") {
		return fmt.Errorf("arguments after -- are unsupported")
	}

	workerClass, _, _, _, err := tokenizedFlag(container.Args, vllmWorkerClassFlag)
	if err != nil {
		return err
	}
	switch workerClass {
	case gmsV1VLLMWorkerClass:
		loadFormat, _, _, found, err := tokenizedFlag(container.Args, vllmLoadFormatFlag)
		if err != nil {
			return err
		}
		if found && loadFormat != "auto" {
			return fmt.Errorf("%s %s requires %s auto", vllmWorkerClassFlag, gmsV1VLLMWorkerClass, vllmLoadFormatFlag)
		}
		return nil
	case "", "auto", gmsV0VLLMWorkerClass, gmsV0VLLMWorkerClassAlt:
		args, err := upsertTokenizedVLLMFlag(container.Args, vllmLoadFormatFlag, "gms")
		if err != nil {
			return err
		}
		container.Args = args
		return nil
	default:
		return fmt.Errorf("%s %q is unsupported for automatic snapshot failover", vllmWorkerClassFlag, workerClass)
	}
}

func isDirectDynamoVLLMCommand(container *corev1.Container) bool {
	return isDirectDynamoModuleCommand(container, vllmModuleName)
}

func isDirectDynamoModuleCommand(container *corev1.Container, module string) bool {
	args := container.Args
	switch {
	case len(container.Command) == 0 &&
		len(args) >= 3 &&
		isPythonCommand(args[0]) &&
		args[1] == "-m" &&
		args[2] == module:
		return true
	case len(container.Command) == 1 &&
		isPythonCommand(container.Command[0]) &&
		len(args) >= 2 &&
		args[0] == "-m" &&
		args[1] == module:
		return true
	case len(container.Command) == 3 &&
		isPythonCommand(container.Command[0]) &&
		container.Command[1] == "-m" &&
		container.Command[2] == module:
		return true
	default:
		return false
	}
}

func tokenizedFlag(args []string, flag string) (value string, index int, equalsForm, found bool, err error) {
	index = -1
	for i, arg := range args {
		switch {
		case arg == flag:
			if found {
				return "", -1, false, false, fmt.Errorf("%s must appear at most once", flag)
			}
			if i+1 >= len(args) || strings.HasPrefix(args[i+1], "--") {
				return "", -1, false, false, fmt.Errorf("%s requires a value", flag)
			}
			value, index, found = args[i+1], i, true
		case strings.HasPrefix(arg, flag+"="):
			if found {
				return "", -1, false, false, fmt.Errorf("%s must appear at most once", flag)
			}
			value = strings.TrimPrefix(arg, flag+"=")
			if value == "" {
				return "", -1, false, false, fmt.Errorf("%s requires a value", flag)
			}
			index, equalsForm, found = i, true, true
		}
	}
	return value, index, equalsForm, found, nil
}

func upsertTokenizedVLLMFlag(args []string, flag, value string) ([]string, error) {
	_, index, equalsForm, found, err := tokenizedFlag(args, flag)
	if err != nil {
		return nil, err
	}
	switch {
	case !found:
		args = append(args, flag, value)
	case equalsForm:
		args[index] = flag + "=" + value
	default:
		args[index+1] = value
	}
	return args, nil
}

func configureCheckpointFailoverEngines(
	podSpec *corev1.PodSpec,
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
	backendFramework BackendFramework,
) {
	engineNames := IntraPodFailoverEngineContainerNames(component)
	for i := range podSpec.Containers {
		if slices.Contains(engineNames, podSpec.Containers[i].Name) {
			env := []corev1.EnvVar{{Name: "DYN_REQUEST_PLANE", Value: "tcp"}}
			if backendFramework == BackendFrameworkVLLM {
				env = append(env, corev1.EnvVar{Name: "DYN_VLLM_DISAGGREGATION_MODE", Value: "agg"})
			}
			podSpec.Containers[i].Env = MergeEnvs(podSpec.Containers[i].Env, env)
		}
	}
}

// IsIntraPodFailoverEnabled is true only when failover clones engine
// containers inside one pod. Inter-pod failover keeps one main container per
// engine pod. v1beta1 FailoverSpec is presence-only: v1alpha1 conversion only
// creates it when Failover.Enabled was true, so non-nil means enabled. An empty
// mode means the API/defaulting path selected intra-pod.
func IsIntraPodFailoverEnabled(component *v1beta1.DynamoComponentDeploymentSharedSpec) bool {
	if component == nil || component.Experimental == nil || component.Experimental.Failover == nil {
		return false
	}
	mode := component.Experimental.Failover.Mode
	return mode == "" || mode == v1beta1.GMSModeIntraPod
}

func IntraPodFailoverEngineContainerNames(
	component *v1beta1.DynamoComponentDeploymentSharedSpec,
) []string {
	if !IsIntraPodFailoverEnabled(component) {
		return nil
	}
	numShadows := component.Experimental.Failover.NumShadows
	if numShadows == 0 {
		numShadows = 1
	}
	if numShadows < 1 || numShadows > 2 {
		return nil
	}
	engineCount := int(numShadows) + 1
	names := make([]string, 0, engineCount)
	for i := 0; i < engineCount; i++ {
		names = append(names, fmt.Sprintf("engine-%d", i))
	}
	return names
}

// buildFailoverPod clones the main container into active and standby engine containers.
// This runs AFTER applyGPUMemoryService, so the main container already has DRA claims,
// shared volume mount, and TMPDIR set. This function only handles engine duplication
// and failover-specific env vars.
//
// Non-main containers (e.g. frontend sidecar) are preserved in the final pod spec.
func buildFailoverPod(
	podSpec *corev1.PodSpec,
	numberOfNodes int32,
	backendFramework BackendFramework,
	engineCount int,
) error {
	if len(podSpec.Containers) == 0 {
		return fmt.Errorf("pod spec must have at least one container for failover transformation")
	}
	if engineCount < failoverEngineCount || engineCount > failoverEngineCount+1 {
		return fmt.Errorf("intra-pod failover supports one or two shadows")
	}

	mainContainer := podSpec.Containers[0]
	sidecars := podSpec.Containers[1:]

	engines := make([]corev1.Container, engineCount)
	for i := range engineCount {
		engines[i] = buildEngineContainer(mainContainer, i, commonconsts.DynamoSystemPort+i)
	}

	updated := podSpec.DeepCopy()
	updated.Containers = append(engines, sidecars...)

	// Backend-specific overrides
	switch backendFramework {
	case BackendFrameworkVLLM:
		if err := applyVLLMOverrides(updated, numberOfNodes); err != nil {
			return err
		}
	case BackendFrameworkSGLang:
		// SGLang only needs the backend-agnostic engine identity, health, and
		// lock wiring added above. GMS V1 is selected before cloning.
	default:
		return fmt.Errorf("failover is currently supported only for vLLM and SGLang (detected: %s)", backendFramework)
	}

	*podSpec = *updated
	return nil
}

// buildEngineContainer clones the main container with ENGINE_ID and failover env vars.
// Each engine gets a unique system port and named port for probe targeting.
func buildEngineContainer(base corev1.Container, engineID int, systemPort int) corev1.Container {
	engine := *base.DeepCopy()
	engine.Name = fmt.Sprintf("engine-%d", engineID)

	portName := fmt.Sprintf("system-%d", engineID)

	engine.Ports = []corev1.ContainerPort{
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          portName,
			ContainerPort: int32(systemPort),
		},
	}

	// Env vars to remove: replaced by failover-specific values or intentionally omitted.
	// DYN_FORWARDPASS_METRIC_PORT is removed here so we can override it per engine
	// below — both engines share the pod network namespace, so the base value
	// stamped by component_worker.go collides on bind.
	removeSet := map[string]bool{
		"DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS": true,
		"DYN_SYSTEM_PORT":                       true,
		"DYN_SYSTEM_ENABLED":                    true,
		"DYN_HEALTH_CHECK_ENABLED":              true,
		"CONTAINER_NAME":                        true,
		"DYN_FORWARDPASS_METRIC_PORT":           true,
	}

	var filtered []corev1.EnvVar
	for _, env := range engine.Env {
		if !removeSet[env.Name] {
			filtered = append(filtered, env)
		}
	}

	failoverEnvs := []corev1.EnvVar{
		{Name: "ENGINE_ID", Value: strconv.Itoa(engineID)},
		{Name: "CONTAINER_NAME", Value: engine.Name},
		{Name: "FAILOVER_LOCK_PATH", Value: intraPodFailoverLockFile},
		{Name: "DYN_SYSTEM_STARTING_HEALTH_STATUS", Value: "notready"},
		{Name: "DYN_SYSTEM_PORT", Value: strconv.Itoa(systemPort)},
		{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
		// Per-engine FPM port. data_parallel_index is 0 for both failover
		// engines (orthogonal axis), so without this override both bind to
		// the same base port and engine-1 fails with EADDRINUSE.
		{Name: "DYN_FORWARDPASS_METRIC_PORT", Value: strconv.Itoa(commonconsts.DynamoFPMBasePort + engineID)},
	}
	engine.Env = append(filtered, failoverEnvs...)

	// Retarget HTTP probes to this engine's named port. Each engine runs its
	// system server on a staggered port (e.g. 9090, 9091), and the probes
	// inherited from the base container still reference the original port name.
	portRef := intstr.FromString(portName)
	if engine.StartupProbe != nil && engine.StartupProbe.HTTPGet != nil {
		engine.StartupProbe.HTTPGet.Port = portRef
	}
	if engine.LivenessProbe != nil && engine.LivenessProbe.HTTPGet != nil {
		engine.LivenessProbe.HTTPGet.Port = portRef
	}
	if engine.ReadinessProbe != nil && engine.ReadinessProbe.HTTPGet != nil {
		engine.ReadinessProbe.HTTPGet.Port = portRef
	}

	return engine
}
