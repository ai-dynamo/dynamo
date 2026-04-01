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
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
)

const (
	gmsSharedVolumeName = "gms-shared"
	gmsHostPathBase     = "/run/gms"
	gmsSharedMountPath  = "/run/gms/shared"
	gmsReadyFilePath    = "/tmp/gms_ready"
	gmsFailoverLockFile = "failover.lock"
)

// gmsWrapperScript returns the shell script that starts the GMS weight server
// and signals readiness by touching a sentinel file.
func gmsWrapperScript() string {
	return fmt.Sprintf(`#!/bin/sh
set -e
rm -f %s
python3 -m vllm.distributed.gms &
GMS_PID=$!

while [ ! -f %s ]; do
  if ! kill -0 $GMS_PID 2>/dev/null; then
    echo "GMS process exited unexpectedly"
    exit 1
  fi
  sleep 1
done

echo "GMS weight server ready"
wait $GMS_PID
`, gmsReadyFilePath, gmsReadyFilePath)
}

// gmsStartupProbeCommand returns the command for the GMS weight server startup probe.
func gmsStartupProbeCommand() []string {
	return []string{"test", "-f", gmsReadyFilePath}
}

// gmsWeightServerPodSpec builds a GMS weight server pod spec by cloning and
// modifying a base engine pod spec. The GMS pod runs a different command,
// has no liveness/readiness probes, and uses a startup probe that checks
// for the ready sentinel file.
func gmsWeightServerPodSpec(basePodSpec *corev1.PodSpec, rank int32) *corev1.PodSpec {
	podSpec := basePodSpec.DeepCopy()
	if len(podSpec.Containers) == 0 {
		return podSpec
	}

	c := &podSpec.Containers[0]
	c.Command = []string{"/bin/sh", "-c", gmsWrapperScript()}
	c.Args = nil

	c.StartupProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{Command: gmsStartupProbeCommand()},
		},
		PeriodSeconds:    5,
		TimeoutSeconds:   2,
		FailureThreshold: 1440, // 5s * 1440 = 7200s = 2h
	}
	c.LivenessProbe = nil
	c.ReadinessProbe = nil

	c.Env = append(c.Env, corev1.EnvVar{
		Name:  "TMPDIR",
		Value: gmsSharedMountPath,
	})

	removeGPUFromLimits(c)
	addGPUToleration(podSpec)

	vol, mount := gmsSharedVolume(rank)
	podSpec.Volumes = append(podSpec.Volumes, vol)
	c.VolumeMounts = append(c.VolumeMounts, mount)

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
		{Name: "DYN_SYSTEM_STARTING_HEALTH_STATUS", Value: "unhealthy"},
	}
}

// augmentEngineForGMS modifies an engine pod spec in-place to work with GMS failover:
// injects env vars, shared volume, strips GPU limits, and adds toleration.
func augmentEngineForGMS(podSpec *corev1.PodSpec, rank int32) {
	if len(podSpec.Containers) == 0 {
		return
	}
	c := &podSpec.Containers[0]

	c.Env = append(c.Env, gmsEngineEnvVars()...)
	removeEnvVar(c, "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS")

	removeGPUFromLimits(c)
	addGPUToleration(podSpec)

	vol, mount := gmsSharedVolume(rank)
	podSpec.Volumes = append(podSpec.Volumes, vol)
	c.VolumeMounts = append(c.VolumeMounts, mount)
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
	var configs []grovev1alpha1.ResourceClaimTemplateConfig
	for _, r := range roles {
		if seen[r.Rank] {
			continue
		}
		seen[r.Rank] = true
		configs = append(configs, grovev1alpha1.ResourceClaimTemplateConfig{
			Name: gmsRCTName(serviceName, r.Rank),
			Template: resourcev1.ResourceClaimTemplateSpec{
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

// gmsResourceSharingEntries builds one PCSG-level ResourceClaimTemplateRef per rank.
// Each entry uses PerReplica scope and an Include filter listing only the GMS clique
// and the engine clique for that rank, ensuring GPU isolation between ranks.
func gmsResourceSharingEntries(serviceName string, roles []ServiceRole) []grovev1alpha1.ResourceClaimTemplateRef {
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

	refs := make([]grovev1alpha1.ResourceClaimTemplateRef, 0, len(groups))
	for _, rank := range rankOrder {
		g := groups[rank]
		refs = append(refs, grovev1alpha1.ResourceClaimTemplateRef{
			Name:  gmsRCTName(serviceName, rank),
			Scope: grovev1alpha1.ResourceSharingScopePerReplica,
			Filter: &grovev1alpha1.ResourceSharingFilter{
				Mode:        grovev1alpha1.ResourceSharingFilterModeInclude,
				CliqueNames: g.cliqueNames,
			},
		})
	}
	return refs
}
