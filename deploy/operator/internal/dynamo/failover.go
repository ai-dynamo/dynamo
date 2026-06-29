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
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	gmsruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
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
	gmsSharedVolumeName                       = "gms-shared"
	gmsHostPathBase                           = "/run/gms"
	gmsSharedMountPath                        = "/run/gms/shared"
	gmsFailoverLockFile                       = "failover.lock"
	gmsPermFixInitName                        = "fix-gms-perms"
	gmsFastShutdownGracePeriodEnvVar          = "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS"
	gmsFastExitOnSigtermEnvVar                = "DYN_GMS_FAILOVER_FAST_EXIT_ON_SIGTERM"
	gmsKubeDiscoveryDebounceMsEnvVar          = "DYN_KUBE_DISCOVERY_DEBOUNCE_MS"
	gmsFastKubeDiscoveryDebounceMsDefault     = "25"
	gmsGatewayNotreadyGraceMsEnvVar           = "DYN_BULWARK_GATEWAY_NOTREADY_GRACE_MS"
	gmsGatewayNotreadyGraceMsDefault          = "1000"
	gmsMigrationRetryConcurrencyEnvVar        = "DYN_MIGRATION_RETRY_CONCURRENCY"
	gmsMigrationRetryConcurrencyDefault       = "2"
	gmsMigrationFirstChunkTimeoutMsEnvVar     = "DYN_MIGRATION_FIRST_CHUNK_TIMEOUT_MS"
	gmsMigrationFirstChunkTimeoutMsDefault    = "500"
	gmsMigrationRecentFailoverWindowMsEnvVar  = "DYN_MIGRATION_RECENT_FAILOVER_WINDOW_MS"
	gmsMigrationRecentFailoverWindowMsDefault = "1500"
)

var gmsServerTags = []string{"weights", "kv_cache"}

func gmsContainerName(tag string) string {
	return "gms-" + strings.ReplaceAll(tag, "_", "-")
}

// gmsWrapperScript generates a bash script that launches one logical GMS
// server group for a single tag. gpu_memory_service.cli.server still
// auto-discovers DRA-allocated GPUs and supervises one child server per device,
// but each Kubernetes container owns only one socket namespace (weights or
// kv_cache).
func gmsWrapperScript(tag string) string {
	return gmsWrapperScriptAt(tag, gmsSharedMountPath)
}

func gmsWrapperScriptAt(tag string, socketDir string) string {
	return fmt.Sprintf(
		`rm -f %s/gms_*_%s.sock
rc=1
cleanup() { kill -- -$$ 2>/dev/null; exit "$rc"; }
trap cleanup SIGTERM SIGINT
GMS_SERVER_TAGS=%s python3 -m %s &
echo "Started GMS %s server pid=$!"
wait -n
rc=$?
echo "GMS %s server exited (code=$rc), shutting down"
cleanup`, socketDir, tag, tag, gmsruntime.ServerModule, tag, tag)
}

// gmsProbeCommand returns the exec probe command that verifies the GMS server
// has opened its tag-specific UDS socket for every allocated GPU.
func gmsProbeCommand(tag string, gpuCount int) []string {
	return gmsProbeCommandAt(tag, gpuCount, gmsSharedMountPath)
}

func gmsProbeCommandAt(tag string, gpuCount int, socketDir string) []string {
	return []string{
		"sh", "-c",
		fmt.Sprintf("test $(ls %s/gms_*_%s.sock 2>/dev/null | wc -l) -ge %d", socketDir, tag, gpuCount),
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

// gmsWeightServerPodSpec builds a GMS server pod spec by cloning and
// modifying a base engine pod spec. The GMS pod runs one container per logical
// memory namespace so weights and kv_cache have independent process lifecycles
// and Kubernetes probes while still sharing the same DRA GPU allocation and
// rank-local socket directory.
//
// RestartPolicy is intentionally left unset here (i.e. inherits the base /
// Grove default, which is Always). A GMS server process holds only local
// state — GPU allocations (via DRA, which survive the container), hostPath
// UDS sockets (recreated by gmsWrapperScript on startup), and in-memory
// buffers (re-sharded/re-attached on reconnection by the engine clients). So
// an in-place kubelet restart is a fast, correct recovery path.
//
// The paired engine pod mirrors this policy in the standalone inter-pod GMS
// layout (a restarted engine re-imports IPC handles from the still-running
// GMS server). In the inter-pod GMS failover layout, augmentEngineForGMS
// overrides the engine's RestartPolicy to Never so the cohort can only be
// recovered via FailoverCascadeReconciler; see the comment there.
func gmsWeightServerPodSpec(basePodSpec *corev1.PodSpec, rank int32, gpuCount int) *corev1.PodSpec {
	podSpec := basePodSpec.DeepCopy()
	if len(podSpec.Containers) == 0 {
		return podSpec
	}

	baseContainer := podSpec.Containers[0].DeepCopy()
	vol, mount := gmsSharedVolume(rank)
	podSpec.Volumes = append(podSpec.Volumes, vol)
	podSpec.InitContainers = append(podSpec.InitContainers, gmsPermFixInitContainer(rank, baseContainer.Image))
	addGPUToleration(podSpec)

	containers := make([]corev1.Container, 0, len(gmsServerTags))
	for _, tag := range gmsServerTags {
		c := baseContainer.DeepCopy()
		c.Name = gmsContainerName(tag)
		c.Command = []string{"bash", "-c"}
		c.Args = []string{gmsWrapperScript(tag)}
		c.Ports = nil
		removeGPUFromLimits(c)

		probeCommand := gmsProbeCommand(tag, gpuCount)
		c.StartupProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{Command: probeCommand},
			},
			PeriodSeconds:    2,
			TimeoutSeconds:   2,
			FailureThreshold: 150, // 2s * 150 = 5 min
		}
		c.ReadinessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{Command: probeCommand},
			},
			PeriodSeconds:  5,
			TimeoutSeconds: 2,
		}
		c.LivenessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{Command: probeCommand},
			},
			PeriodSeconds:    10,
			TimeoutSeconds:   2,
			FailureThreshold: 3,
		}

		c.Env = append(c.Env,
			corev1.EnvVar{Name: gmsruntime.EnvSocketDir, Value: gmsSharedMountPath},
			corev1.EnvVar{Name: "GMS_SERVER_TAGS", Value: tag},
		)
		c.VolumeMounts = append(c.VolumeMounts, mount)
		containers = append(containers, *c)
	}
	podSpec.Containers = containers

	return podSpec
}

// appendIntraPodGMSServerContainers adds one regular container per logical GMS
// namespace inside an intra-pod failover worker pod. This mirrors the inter-pod
// split (weights + kv_cache) while using the intra-pod emptyDir socket volume
// instead of a hostPath shared across pods.
func appendIntraPodGMSServerContainers(podSpec *corev1.PodSpec, baseContainer corev1.Container, gpuCount int) {
	if podSpec == nil {
		return
	}
	if gpuCount <= 0 {
		gpuCount = 1
	}

	existing := map[string]bool{}
	for _, c := range podSpec.Containers {
		existing[c.Name] = true
	}

	mount := corev1.VolumeMount{
		Name:      gmsruntime.SharedVolumeName,
		MountPath: gmsruntime.SharedMountPath,
	}
	for _, tag := range gmsServerTags {
		name := gmsContainerName(tag)
		if existing[name] {
			continue
		}

		c := *baseContainer.DeepCopy()
		c.Name = name
		c.Command = []string{"bash", "-c"}
		c.Args = []string{gmsWrapperScriptAt(tag, gmsruntime.SharedMountPath)}
		c.Ports = nil
		removeGPUFromLimits(&c)

		probeCommand := gmsProbeCommandAt(tag, gpuCount, gmsruntime.SharedMountPath)
		c.StartupProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{Command: probeCommand},
			},
			PeriodSeconds:    2,
			TimeoutSeconds:   2,
			FailureThreshold: 150,
		}
		c.ReadinessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{Command: probeCommand},
			},
			PeriodSeconds:  5,
			TimeoutSeconds: 2,
		}
		c.LivenessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{Command: probeCommand},
			},
			PeriodSeconds:    10,
			TimeoutSeconds:   2,
			FailureThreshold: 3,
		}

		appendOrReplaceEnvVars(&c,
			corev1.EnvVar{Name: gmsruntime.EnvSocketDir, Value: gmsruntime.SharedMountPath},
			corev1.EnvVar{Name: "GMS_SERVER_TAGS", Value: tag},
		)
		c.VolumeMounts = appendMissingVolumeMounts(c.VolumeMounts, []corev1.VolumeMount{mount})
		podSpec.Containers = append(podSpec.Containers, c)
	}
}

// gmsEngineEnvVars returns the backend-agnostic environment variables injected
// into engine pods when GMS failover is enabled. Backend-specific switches
// (e.g. the vLLM DYN_VLLM_GMS_SHADOW_MODE flag) are injected by the backend's
// UpdateContainer path so non-vLLM backends do not inherit stray env vars.
func gmsLogicalInstanceKey(roleName string) string {
	// The key intentionally excludes grove.io/podclique-pod-index / ENGINE_ID so
	// primary and shadow pods in the same failover cohort publish one logical
	// worker identity to routers and KV indexers.
	return fmt.Sprintf("$(POD_NAMESPACE)/$(GROVE_PCSG_NAME)/$(GROVE_PCSG_INDEX)/%s", roleName)
}

func gmsEngineEnvVars(logicalInstanceKey string) []corev1.EnvVar {
	envs := []corev1.EnvVar{
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
		{Name: "GMS_KV_LEASES", Value: "1"},
		{Name: "GMS_KV_LEASE_SHM_DIR", Value: gmsSharedMountPath},
		{Name: "DYN_SYSTEM_STARTING_HEALTH_STATUS", Value: "notready"},
	}
	if logicalInstanceKey != "" {
		envs = append(envs, corev1.EnvVar{
			Name:  commonconsts.DynamoDiscoveryLogicalInstanceKeyEnvVar,
			Value: logicalInstanceKey,
		})
		envs = append(envs, corev1.EnvVar{
			Name:  "DYN_GMS_FAILOVER_SHADOW_MODE",
			Value: "true",
		})
	}
	return envs
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
//     FailoverCascadeReconciler (see failover_cascade_controller.go) to
//     force-delete the full engine group based on the
//     KubeLabelDynamoFailoverEngineGroupMember label, and Grove to recreate
//     the cohort from scratch. That label is applied in graph.go only when
//     isInterPodFailover is true, so forcing Never in the standalone case
//     would strand engine pods in Failed state with nothing listening to
//     force-delete them.
func augmentEngineForGMS(podSpec *corev1.PodSpec, rank int32, isInterPodFailover bool, roleName string) {
	if len(podSpec.Containers) == 0 {
		return
	}
	c := &podSpec.Containers[0]

	logicalInstanceKey := ""
	if isInterPodFailover {
		logicalInstanceKey = gmsLogicalInstanceKey(roleName)
	}
	appendOrReplaceEnvVars(c, gmsEngineEnvVars(logicalInstanceKey)...)
	removeEnvVar(c, "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS")

	applyGMSSharedResources(podSpec, c, rank)
	if isInterPodFailover {
		addDefaultFastFailoverShutdown(c)
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

func appendOrReplaceEnvVars(c *corev1.Container, envs ...corev1.EnvVar) {
	for _, env := range envs {
		removeEnvVar(c, env.Name)
		c.Env = append(c.Env, env)
	}
}

func applyTRTLLMFailoverOverrides(podSpec *corev1.PodSpec, numberOfNodes int32) {
	if numberOfNodes <= 1 {
		return
	}

	basePort := commonconsts.MpiRunSshPort
	for i := range podSpec.Containers {
		c := &podSpec.Containers[i]
		if !strings.HasPrefix(c.Name, "engine-") {
			continue
		}

		engineID, err := strconv.Atoi(strings.TrimPrefix(c.Name, "engine-"))
		if err != nil || engineID <= 0 {
			continue
		}

		retargetTRTLLMSshPort(c, basePort, basePort+engineID)
	}
}

func retargetTRTLLMSshPort(c *corev1.Container, from int, to int) {
	fromText := strconv.Itoa(from)
	toText := strconv.Itoa(to)

	for i, arg := range c.Args {
		c.Args[i] = strings.ReplaceAll(arg, "Port "+fromText, "Port "+toText)
		c.Args[i] = strings.ReplaceAll(c.Args[i], "-p "+fromText, "-p "+toText)
	}
	for i, cmd := range c.Command {
		c.Command[i] = strings.ReplaceAll(cmd, "Port "+fromText, "Port "+toText)
		c.Command[i] = strings.ReplaceAll(c.Command[i], "-p "+fromText, "-p "+toText)
	}

	if c.ReadinessProbe != nil && c.ReadinessProbe.TCPSocket != nil {
		c.ReadinessProbe.TCPSocket.Port = intstr.FromInt(to)
	}
}

func containerHasEnvVar(c *corev1.Container, name string) bool {
	for _, e := range c.Env {
		if e.Name == name {
			return true
		}
	}
	return false
}

func addDefaultFastFailoverShutdown(c *corev1.Container) {
	if !containerHasEnvVar(c, gmsFastShutdownGracePeriodEnvVar) {
		c.Env = append(c.Env, corev1.EnvVar{
			Name:  gmsFastShutdownGracePeriodEnvVar,
			Value: "0",
		})
	}
	if !containerHasEnvVar(c, gmsFastExitOnSigtermEnvVar) {
		c.Env = append(c.Env, corev1.EnvVar{
			Name:  gmsFastExitOnSigtermEnvVar,
			Value: "1",
		})
	}
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
// The main container is cloned into two engine containers (active + standby)
// within the same pod. GPU access is shared via DRA and a GMS sidecar
// injects weights via the shared emptyDir volume.
// ──────────────────────────────────────────────────────────────────────────────

// intraPodFailoverLockFile is the lock file path used by engine containers to
// coordinate active/standby election within the same pod.
var intraPodFailoverLockFile = filepath.Join(gmsruntime.SharedMountPath, "failover.lock")

const (
	failoverEngineCount = 2
)

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

func IntraPodFailoverEngineContainerNames() []string {
	names := make([]string, 0, failoverEngineCount)
	for i := 0; i < failoverEngineCount; i++ {
		names = append(names, fmt.Sprintf("engine-%d", i))
	}
	return names
}

// buildFailoverPod clones the main container into two engine containers (active + standby).
// This runs AFTER applyGPUMemoryService, so the main container already has DRA claims,
// shared volume mount, and TMPDIR set. This function only handles engine duplication
// and failover-specific env vars.
//
// Non-main containers (e.g. frontend sidecar) are preserved in the final pod spec.
func buildFailoverPod(
	podSpec *corev1.PodSpec,
	numberOfNodes int32,
	backendFramework BackendFramework,
) error {
	if len(podSpec.Containers) == 0 {
		return fmt.Errorf("pod spec must have at least one container for failover transformation")
	}

	mainContainer := podSpec.Containers[0]
	sidecars := podSpec.Containers[1:]

	engines := make([]corev1.Container, failoverEngineCount)
	for i := range failoverEngineCount {
		engines[i] = buildEngineContainer(mainContainer, i, commonconsts.DynamoSystemPort+i)
	}

	for i := range sidecars {
		if sidecars[i].Name == commonconsts.FrontendSidecarContainerName {
			appendOrReplaceEnvVars(&sidecars[i],
				corev1.EnvVar{
					Name:  gmsKubeDiscoveryDebounceMsEnvVar,
					Value: gmsFastKubeDiscoveryDebounceMsDefault,
				},
				corev1.EnvVar{
					Name:  gmsGatewayNotreadyGraceMsEnvVar,
					Value: gmsGatewayNotreadyGraceMsDefault,
				},
				corev1.EnvVar{
					Name:  gmsMigrationRetryConcurrencyEnvVar,
					Value: gmsMigrationRetryConcurrencyDefault,
				},
				corev1.EnvVar{
					Name:  gmsMigrationFirstChunkTimeoutMsEnvVar,
					Value: gmsMigrationFirstChunkTimeoutMsDefault,
				},
				corev1.EnvVar{
					Name:  gmsMigrationRecentFailoverWindowMsEnvVar,
					Value: gmsMigrationRecentFailoverWindowMsDefault,
				},
			)
		}
	}

	podSpec.Containers = append(engines, sidecars...)

	// Backend-specific overrides. vLLM needs extra port staggering for its
	// side-channel and torch.distributed sockets; SGLang and TRT-LLM use the
	// backend-neutral system/FPM port staggering above in aggregated mode.
	switch backendFramework {
	case BackendFrameworkVLLM:
		applyVLLMOverrides(podSpec, numberOfNodes)
	case BackendFrameworkTRTLLM:
		applyTRTLLMFailoverOverrides(podSpec, numberOfNodes)
	case BackendFrameworkSGLang:
		applySGLangFailoverOverrides(podSpec)
	default:
		return fmt.Errorf("failover is not supported for backend framework %s", backendFramework)
	}

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

	lockBeforeInit := "0"
	if engineID == 0 {
		lockBeforeInit = "1"
	}

	failoverEnvs := []corev1.EnvVar{
		{Name: "ENGINE_ID", Value: strconv.Itoa(engineID)},
		{Name: "CONTAINER_NAME", Value: engine.Name},
		{Name: "FAILOVER_LOCK_PATH", Value: intraPodFailoverLockFile},
		{Name: "GMS_KV_LEASES", Value: "1"},
		{Name: "GMS_KV_LEASE_SHM_DIR", Value: gmsruntime.SharedMountPath},
		{Name: "DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", Value: "0"},
		{Name: "DYN_GMS_FAILOVER_SHADOW_MODE", Value: "true"},
		{Name: "DYN_SYSTEM_STARTING_HEALTH_STATUS", Value: "notready"},
		{Name: "DYN_SYSTEM_PORT", Value: strconv.Itoa(systemPort)},
		{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
		{Name: gmsKubeDiscoveryDebounceMsEnvVar, Value: gmsFastKubeDiscoveryDebounceMsDefault},
		{Name: "DYN_VLLM_GMS_LOCK_BEFORE_INIT", Value: lockBeforeInit},
		{Name: "DYN_SGLANG_GMS_LOCK_BEFORE_INIT", Value: lockBeforeInit},
		{Name: "DYN_TRTLLM_GMS_LOCK_BEFORE_INIT", Value: lockBeforeInit},
		// Per-engine FPM port. data_parallel_index is 0 for both failover
		// engines (orthogonal axis), so without this override both bind to
		// the same base port and engine-1 fails with EADDRINUSE.
		{Name: "DYN_FORWARDPASS_METRIC_PORT", Value: strconv.Itoa(commonconsts.DynamoFPMBasePort + engineID)},
	}
	engine.Env = filtered
	appendOrReplaceEnvVars(&engine, failoverEnvs...)

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
