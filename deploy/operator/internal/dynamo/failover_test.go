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
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	k8sresource "k8s.io/apimachinery/pkg/api/resource"
)

func TestGmsWeightServerPodSpec(t *testing.T) {
	base := &corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:    "engine",
			Command: []string{"python3", "-m", "vllm.entrypoints.openai.api_server"},
			Args:    []string{"--model", "meta-llama/Llama-3-8B"},
			LivenessProbe: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					HTTPGet: &corev1.HTTPGetAction{Path: "/health"},
				},
			},
			ReadinessProbe: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					HTTPGet: &corev1.HTTPGetAction{Path: "/ready"},
				},
			},
			Resources: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{
					"nvidia.com/gpu": k8sresource.MustParse("8"),
					corev1.ResourceMemory: k8sresource.MustParse("64Gi"),
				},
			},
		}},
	}

	result := gmsWeightServerPodSpec(base, 0, 8)

	require.Len(t, result.Containers, 1)
	c := result.Containers[0]

	assert.Equal(t, []string{"bash", "-c"}, c.Command, "should use bash")
	require.Len(t, c.Args, 1)
	assert.Contains(t, c.Args[0], "gpu_memory_service", "should run gpu_memory_service")
	assert.Contains(t, c.Args[0], "--device", "should pass --device flag")
	assert.Nil(t, c.LivenessProbe, "liveness probe should be nil")
	assert.Nil(t, c.ReadinessProbe, "readiness probe should be nil")
	assert.NotNil(t, c.StartupProbe, "startup probe should be set")
	assert.Equal(t, gmsStartupProbeCommand(8), c.StartupProbe.Exec.Command)

	assert.NotContains(t, c.Resources.Limits, corev1.ResourceName("nvidia.com/gpu"), "GPU should be stripped")
	assert.Contains(t, c.Resources.Limits, corev1.ResourceMemory, "non-GPU limits should remain")

	assert.True(t, hasToleration(result, "nvidia.com/gpu"), "should have GPU toleration")
	assert.True(t, hasVolume(result, gmsSharedVolumeName), "should have shared volume")
	assert.True(t, hasVolumeMount(c, gmsSharedMountPath), "should have shared volume mount")
	assert.True(t, hasEnvVar(c, "TMPDIR", gmsSharedMountPath), "should set TMPDIR")

	require.Len(t, result.InitContainers, 1, "should have perm-fix init container")
	initC := result.InitContainers[0]
	assert.Equal(t, gmsPermFixInitName, initC.Name)
	assert.Equal(t, c.Image, initC.Image, "init container should reuse the service image")
	require.NotNil(t, initC.SecurityContext)
	assert.Equal(t, int64(0), *initC.SecurityContext.RunAsUser)

	// Verify original is not mutated
	assert.Len(t, base.Containers[0].Command, 3, "original command should be unchanged")
}

func TestGmsWeightServerPodSpec_EmptyContainers(t *testing.T) {
	base := &corev1.PodSpec{}
	result := gmsWeightServerPodSpec(base, 0, 1)
	assert.Empty(t, result.Containers)
}

func TestGmsWeightServerPodSpec_SubPathExpr(t *testing.T) {
	base := &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "engine"}},
	}

	t.Run("rank 0", func(t *testing.T) {
		result := gmsWeightServerPodSpec(base, 0, 4)
		mount := findVolumeMount(result.Containers[0], gmsSharedMountPath)
		require.NotNil(t, mount, "GMS container should mount shared volume")
		assert.Equal(t, "$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)/rank-0", mount.SubPathExpr)
	})

	t.Run("rank 3", func(t *testing.T) {
		result := gmsWeightServerPodSpec(base, 3, 4)
		mount := findVolumeMount(result.Containers[0], gmsSharedMountPath)
		require.NotNil(t, mount, "GMS container should mount shared volume")
		assert.Equal(t, "$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)/rank-3", mount.SubPathExpr)
	})
}

func TestAugmentEngineForGMS(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{{
			Name: "engine",
			Env: []corev1.EnvVar{
				{Name: "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", Value: "true"},
				{Name: "KEEP_ME", Value: "yes"},
			},
			Resources: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{
					"nvidia.com/gpu": k8sresource.MustParse("4"),
				},
			},
		}},
	}

	augmentEngineForGMS(podSpec, 1)
	c := podSpec.Containers[0]

	assert.True(t, hasEnvVar(c, "ENGINE_ID", ""), "ENGINE_ID should be set (via Downward API)")
	assert.True(t, hasEnvVar(c, "TMPDIR", gmsSharedMountPath))
	assert.True(t, hasEnvVar(c, "FAILOVER_LOCK_PATH", gmsSharedMountPath+"/"+gmsFailoverLockFile))
	assert.True(t, hasEnvVar(c, "DYN_VLLM_GMS_SHADOW_MODE", "true"))
	assert.True(t, hasEnvVar(c, "DYN_SYSTEM_STARTING_HEALTH_STATUS", "notready"))
	assert.True(t, hasEnvVar(c, "KEEP_ME", "yes"), "unrelated env vars should be preserved")

	for _, e := range c.Env {
		assert.NotEqual(t, "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", e.Name, "should be removed")
	}

	assert.NotContains(t, c.Resources.Limits, corev1.ResourceName("nvidia.com/gpu"))
	assert.True(t, hasToleration(podSpec, "nvidia.com/gpu"))
	assert.True(t, hasVolume(podSpec, gmsSharedVolumeName))

	require.Len(t, podSpec.InitContainers, 1, "should have perm-fix init container")
	initC := podSpec.InitContainers[0]
	assert.Equal(t, gmsPermFixInitName, initC.Name)
	assert.Equal(t, c.Image, initC.Image, "init container should reuse the service image")
	require.NotNil(t, initC.SecurityContext)
	assert.Equal(t, int64(0), *initC.SecurityContext.RunAsUser)
	initMount := findVolumeMount(initC, gmsSharedMountPath)
	require.NotNil(t, initMount, "init container should mount shared volume")
	assert.Equal(t, "$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)/rank-1", initMount.SubPathExpr)
}

func TestAugmentEngineForGMS_EmptyContainers(t *testing.T) {
	podSpec := &corev1.PodSpec{}
	augmentEngineForGMS(podSpec, 0)
	assert.Empty(t, podSpec.Containers)
}

func TestRemoveGPUFromLimits(t *testing.T) {
	c := &corev1.Container{
		Resources: corev1.ResourceRequirements{
			Limits: corev1.ResourceList{
				"nvidia.com/gpu":      k8sresource.MustParse("8"),
				corev1.ResourceMemory: k8sresource.MustParse("64Gi"),
			},
			Requests: corev1.ResourceList{
				"nvidia.com/gpu": k8sresource.MustParse("8"),
			},
		},
	}

	removeGPUFromLimits(c)
	assert.NotContains(t, c.Resources.Limits, corev1.ResourceName("nvidia.com/gpu"))
	assert.Contains(t, c.Resources.Limits, corev1.ResourceMemory)
	assert.NotContains(t, c.Resources.Requests, corev1.ResourceName("nvidia.com/gpu"))
}

func TestAddGPUToleration_Idempotent(t *testing.T) {
	podSpec := &corev1.PodSpec{}
	addGPUToleration(podSpec)
	addGPUToleration(podSpec)
	count := 0
	for _, tol := range podSpec.Tolerations {
		if tol.Key == "nvidia.com/gpu" {
			count++
		}
	}
	assert.Equal(t, 1, count, "toleration should be added only once")
}

func TestRemoveEnvVar(t *testing.T) {
	c := &corev1.Container{
		Env: []corev1.EnvVar{
			{Name: "A", Value: "1"},
			{Name: "REMOVE_ME", Value: "x"},
			{Name: "B", Value: "2"},
			{Name: "REMOVE_ME", Value: "y"},
		},
	}

	removeEnvVar(c, "REMOVE_ME")
	assert.Len(t, c.Env, 2)
	assert.Equal(t, "A", c.Env[0].Name)
	assert.Equal(t, "B", c.Env[1].Name)
}

func TestGetGPUCount(t *testing.T) {
	tests := []struct {
		name      string
		resources *v1alpha1.Resources
		want      int32
	}{
		{"nil resources", nil, 0},
		{"nil limits", &v1alpha1.Resources{}, 0},
		{"empty gpu string", &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPU: ""}}, 0},
		{"valid gpu count", &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPU: "8"}}, 8},
		{"invalid gpu string", &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPU: "abc"}}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, getGPUCount(tt.resources))
		})
	}
}

func TestGetDeviceClassName(t *testing.T) {
	tests := []struct {
		name      string
		resources *v1alpha1.Resources
		want      string
	}{
		{"nil resources", nil, "gpu.nvidia.com"},
		{"nil limits", &v1alpha1.Resources{}, "gpu.nvidia.com"},
		{"empty gpuType", &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{}}, "gpu.nvidia.com"},
		{"custom gpuType", &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPUType: "gpu.nvidia.com/h100"}}, "gpu.nvidia.com/h100"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, getDeviceClassName(tt.resources))
		})
	}
}

func TestGmsEngineEnvVars(t *testing.T) {
	envs := gmsEngineEnvVars()

	names := make(map[string]bool)
	for _, e := range envs {
		names[e.Name] = true
	}

	assert.True(t, names["ENGINE_ID"])
	assert.True(t, names["TMPDIR"])
	assert.True(t, names["FAILOVER_LOCK_PATH"])
	assert.True(t, names["DYN_VLLM_GMS_SHADOW_MODE"])
	assert.True(t, names["DYN_SYSTEM_STARTING_HEALTH_STATUS"])

	for _, e := range envs {
		if e.Name == "ENGINE_ID" {
			assert.NotNil(t, e.ValueFrom, "ENGINE_ID should use Downward API")
			assert.NotNil(t, e.ValueFrom.FieldRef)
			assert.Contains(t, e.ValueFrom.FieldRef.FieldPath, "grove.io/podclique-pod-index")
		}
	}
}

func TestGroveMultinodeDeployer_GMS(t *testing.T) {
	t.Run("GetNodeRank returns static rank for GMS", func(t *testing.T) {
		d := &GroveMultinodeDeployer{IsGMS: true, Rank: 2}
		rank, isShellExpr := d.GetNodeRank()
		assert.Equal(t, "2", rank)
		assert.False(t, isShellExpr, "GMS rank should be static, not a shell expression")
	})

	t.Run("GetNodeRank returns shell expr for non-GMS", func(t *testing.T) {
		d := &GroveMultinodeDeployer{IsGMS: false}
		rank, isShellExpr := d.GetNodeRank()
		assert.Contains(t, rank, "GROVE_PCLQ_POD_INDEX")
		assert.True(t, isShellExpr)
	})

	t.Run("GetHostNames for GMS multinode", func(t *testing.T) {
		d := &GroveMultinodeDeployer{IsGMS: true, Rank: 0}
		hostnames := d.GetHostNames("svc", 3)
		assert.Len(t, hostnames, 3)
		assert.Contains(t, hostnames[0], "ldr-$(GROVE_PCLQ_POD_INDEX)")
		assert.Contains(t, hostnames[1], "wkr-1-$(GROVE_PCLQ_POD_INDEX)")
		assert.Contains(t, hostnames[2], "wkr-2-$(GROVE_PCLQ_POD_INDEX)")
	})

	t.Run("GetHostNames for non-GMS multinode", func(t *testing.T) {
		d := &GroveMultinodeDeployer{IsGMS: false}
		hostnames := d.GetHostNames("svc", 3)
		assert.Len(t, hostnames, 3)
		assert.Contains(t, hostnames[0], "ldr")
		assert.Contains(t, hostnames[1], "wkr-0")
		assert.Contains(t, hostnames[2], "wkr-1")
	})
}

func TestGmsRCTName(t *testing.T) {
	assert.Equal(t, "my-svc-gpu-rank-0", gmsRCTName("my-svc", 0))
	assert.Equal(t, "llama-gpu-rank-2", gmsRCTName("llama", 2))
}

func TestGmsResourceClaimTemplateConfigs_SingleNode(t *testing.T) {
	resources := &v1alpha1.Resources{
		Limits: &v1alpha1.ResourceItem{GPU: "8", GPUType: "gpu.nvidia.com/h100"},
	}
	roles := []ServiceRole{
		{Name: "svc-gms-0", Role: RoleGMS, Rank: 0, Replicas: 1},
		{Name: "svc", Role: RoleMain, Rank: 0, Replicas: 2},
	}

	configs := gmsResourceClaimTemplateConfigs("svc", resources, roles)

	require.Len(t, configs, 1)
	assert.Equal(t, "svc-gpu-rank-0", configs[0].Name)

	req := configs[0].TemplateSpec.Spec.Devices.Requests[0]
	require.NotNil(t, req.Exactly)
	assert.Equal(t, "gpu.nvidia.com/h100", req.Exactly.DeviceClassName)
	assert.Equal(t, int64(8), req.Exactly.Count)
}

func TestGmsResourceClaimTemplateConfigs_Multinode(t *testing.T) {
	resources := &v1alpha1.Resources{
		Limits: &v1alpha1.ResourceItem{GPU: "4"},
	}
	roles := []ServiceRole{
		{Name: "svc-gms-0", Role: RoleGMS, Rank: 0, Replicas: 1},
		{Name: "svc-ldr", Role: RoleLeader, Rank: 0, Replicas: 3},
		{Name: "svc-gms-1", Role: RoleGMS, Rank: 1, Replicas: 1},
		{Name: "svc-wkr-1", Role: RoleWorker, Rank: 1, Replicas: 3},
	}

	configs := gmsResourceClaimTemplateConfigs("svc", resources, roles)

	require.Len(t, configs, 2)
	assert.Equal(t, "svc-gpu-rank-0", configs[0].Name)
	assert.Equal(t, "svc-gpu-rank-1", configs[1].Name)

	req := configs[1].TemplateSpec.Spec.Devices.Requests[0]
	require.NotNil(t, req.Exactly)
	assert.Equal(t, "gpu.nvidia.com", req.Exactly.DeviceClassName)
	assert.Equal(t, int64(4), req.Exactly.Count)
}

func TestGmsResourceSharingEntries_SingleNode(t *testing.T) {
	roles := []ServiceRole{
		{Name: "svc-gms-0", Role: RoleGMS, Rank: 0, Replicas: 1},
		{Name: "svc", Role: RoleMain, Rank: 0, Replicas: 2},
	}

	refs := gmsResourceSharingEntries("svc", roles)

	require.Len(t, refs, 1)
	assert.Equal(t, "svc-gpu-rank-0", refs[0].Name)
	assert.Equal(t, grovev1alpha1.ResourceSharingScopePerReplica, refs[0].Scope)
	require.NotNil(t, refs[0].Filter)
	assert.Equal(t, []string{"svc-gms-0", "svc"}, refs[0].Filter.ChildCliqueNames)
}

func TestGmsResourceSharingEntries_Multinode(t *testing.T) {
	roles := []ServiceRole{
		{Name: "svc-gms-0", Role: RoleGMS, Rank: 0, Replicas: 1},
		{Name: "svc-ldr", Role: RoleLeader, Rank: 0, Replicas: 3},
		{Name: "svc-gms-1", Role: RoleGMS, Rank: 1, Replicas: 1},
		{Name: "svc-wkr-1", Role: RoleWorker, Rank: 1, Replicas: 3},
	}

	refs := gmsResourceSharingEntries("svc", roles)

	require.Len(t, refs, 2)

	assert.Equal(t, "svc-gpu-rank-0", refs[0].Name)
	assert.Equal(t, grovev1alpha1.ResourceSharingScopePerReplica, refs[0].Scope)
	require.NotNil(t, refs[0].Filter)
	assert.Equal(t, []string{"svc-gms-0", "svc-ldr"}, refs[0].Filter.ChildCliqueNames)

	assert.Equal(t, "svc-gpu-rank-1", refs[1].Name)
	assert.Equal(t, grovev1alpha1.ResourceSharingScopePerReplica, refs[1].Scope)
	require.NotNil(t, refs[1].Filter)
	assert.Equal(t, []string{"svc-gms-1", "svc-wkr-1"}, refs[1].Filter.ChildCliqueNames)
}

// --- helpers ---

func hasToleration(podSpec *corev1.PodSpec, key string) bool {
	for _, t := range podSpec.Tolerations {
		if t.Key == key {
			return true
		}
	}
	return false
}

func hasVolume(podSpec *corev1.PodSpec, name string) bool {
	for _, v := range podSpec.Volumes {
		if v.Name == name {
			return true
		}
	}
	return false
}

func hasVolumeMount(c corev1.Container, mountPath string) bool {
	for _, m := range c.VolumeMounts {
		if m.MountPath == mountPath {
			return true
		}
	}
	return false
}

func findVolumeMount(c corev1.Container, mountPath string) *corev1.VolumeMount {
	for i := range c.VolumeMounts {
		if c.VolumeMounts[i].MountPath == mountPath {
			return &c.VolumeMounts[i]
		}
	}
	return nil
}

func hasEnvVar(c corev1.Container, name, value string) bool {
	for _, e := range c.Env {
		if e.Name == name {
			if value == "" || e.Value == value {
				return true
			}
		}
	}
	return false
}
