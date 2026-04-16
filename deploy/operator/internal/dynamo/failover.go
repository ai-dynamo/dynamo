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

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
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
	gmsSharedVolumeName = "gms-shared"
	gmsHostPathBase     = "/run/gms"
	gmsSharedMountPath  = "/run/gms/shared"
	gmsFailoverLockFile = "failover.lock"
	gmsPermFixInitName  = "fix-gms-perms"
)

// gmsWrapperScript generates a bash script that launches one GMS subprocess
// per GPU device, waits for any to exit, then tears down the process group.
func gmsWrapperScript(gpuCount int) string {
	devList := make([]string, gpuCount)
	for i := range gpuCount {
		devList[i] = strconv.Itoa(i)
	}
	return fmt.Sprintf(
		`rm -f %s/gms_*.sock
cleanup() { kill -- -$$ 2>/dev/null; exit 1; }
trap cleanup SIGTERM SIGINT
for dev in %s; do
  python3 -m gpu_memory_service --device "$dev" &
  echo "Started GMS device=$dev pid=$!"
done
wait -n
echo "A GMS subprocess exited, shutting down"
cleanup`, gmsSharedMountPath, strings.Join(devList, " "))
}

// gmsStartupProbeCommand returns the exec probe command that verifies the
// expected number of GMS UDS sockets exist on the shared volume.
func gmsStartupProbeCommand(gpuCount int) []string {
	return []string{
		"sh", "-c",
		fmt.Sprintf("test $(ls %s/gms_*.sock 2>/dev/null | wc -l) -ge %d", gmsSharedMountPath, gpuCount),
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
func gmsWeightServerPodSpec(basePodSpec *corev1.PodSpec, rank int32, gpuCount int) *corev1.PodSpec {
	podSpec := basePodSpec.DeepCopy()
	if len(podSpec.Containers) == 0 {
		return podSpec
	}

	c := &podSpec.Containers[0]
	c.Command = []string{"bash", "-c"}
	c.Args = []string{gmsWrapperScript(gpuCount)}

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
		Name:  "TMPDIR",
		Value: gmsSharedMountPath,
	})

	applyGMSSharedResources(podSpec, c, rank)

	return podSpec
}

// gmsEngineEnvVars returns the environment variables injected into engine pods
// when GMS failover is enabled.
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
		{Name: "TMPDIR", Value: gmsSharedMountPath},
		{Name: "FAILOVER_LOCK_PATH", Value: gmsSharedMountPath + "/" + gmsFailoverLockFile},
		{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "true"},
		{Name: "DYN_SYSTEM_STARTING_HEALTH_STATUS", Value: "notready"},
	}
}

// augmentEngineForGMS modifies an engine pod spec in-place to work with GMS failover:
// injects env vars, shared volume, strips GPU limits, adds toleration, and
// prepends an init container to fix hostPath directory permissions.
func augmentEngineForGMS(podSpec *corev1.PodSpec, rank int32) {
	if len(podSpec.Containers) == 0 {
		return
	}
	c := &podSpec.Containers[0]

	c.Env = append(c.Env, gmsEngineEnvVars()...)
	removeEnvVar(c, "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS")

	applyGMSSharedResources(podSpec, c, rank)
	podSpec.RestartPolicy = corev1.RestartPolicyNever
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
			RunAsUser: ptr.To[int64](0),
		},
		VolumeMounts: []corev1.VolumeMount{mount},
	}
}

// removeGPUFromLimits strips nvidia.com/gpu from the container's resource
// limits and requests because DRA handles GPU allocation for GMS pods.
func removeGPUFromLimits(c *corev1.Container) {
	delete(c.Resources.Limits, "nvidia.com/gpu")
	delete(c.Resources.Requests, "nvidia.com/gpu")
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

// getGPUCount extracts the GPU count from the component's resource limits.
func getGPUCount(resources *v1alpha1.Resources) int32 {
	if resources == nil || resources.Limits == nil || resources.Limits.GPU == "" {
		return 0
	}
	if n, err := strconv.ParseInt(resources.Limits.GPU, 10, 32); err == nil {
		return int32(n)
	}
	return 0
}

// getDeviceClassName returns the DRA device class name from gpuType,
// falling back to the standard nvidia.com/gpu resource name.
func getDeviceClassName(resources *v1alpha1.Resources) string {
	if resources != nil && resources.Limits != nil && resources.Limits.GPUType != "" {
		return resources.Limits.GPUType
	}
	return "gpu.nvidia.com"
}

// gmsRCTName returns a deterministic ResourceClaimTemplate name for a given rank.
func gmsRCTName(serviceName string, rank int32) string {
	return fmt.Sprintf("%s-gpu-rank-%d", serviceName, rank)
}

// gmsResourceClaimTemplateConfigs builds one PCS-level ResourceClaimTemplateConfig
// per rank. Each RCT has the same GPU spec but a distinct per-rank name so that
// each rank's GMS + engine pods get their own ResourceClaim.
func gmsResourceClaimTemplateConfigs(serviceName string, resources *v1alpha1.Resources, roles []ServiceRole) []grovev1alpha1.ResourceClaimTemplateConfig {
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
									DeviceClassName: getDeviceClassName(resources),
									AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
									Count:           int64(getGPUCount(resources)),
								},
							},
						},
					},
				},
			},
		})
	}
	return configs
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

// isFailoverEnabled returns true only for intra-pod failover mode, where the
// main container is cloned into active + standby containers within the same pod.
// Inter-pod failover (Mode=interPod) is handled separately via expandRolesForService
// and generatePodSpecForRole — it does not use container cloning.
func isFailoverEnabled(component *v1alpha1.DynamoComponentDeploymentSharedSpec) bool {
	return component.Failover != nil && component.Failover.Enabled &&
		component.Failover.Mode == v1alpha1.GMSModeIntraPod
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

	podSpec.Containers = append(engines, sidecars...)

	// Backend-specific overrides
	switch backendFramework {
	case BackendFrameworkVLLM:
		applyVLLMOverrides(podSpec, numberOfNodes)
	default:
		return fmt.Errorf("failover is currently supported only for vLLM (detected: %s)", backendFramework)
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
	removeSet := map[string]bool{
		"DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS": true,
		"DYN_SYSTEM_PORT":                       true,
		"DYN_SYSTEM_ENABLED":                    true,
		"DYN_HEALTH_CHECK_ENABLED":              true,
		"CONTAINER_NAME":                        true,
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
