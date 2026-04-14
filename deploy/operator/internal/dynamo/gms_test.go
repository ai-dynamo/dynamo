/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"context"
	"strconv"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func gmsComponent(gpuCount int) *v1alpha1.DynamoComponentDeploymentSharedSpec {
	return &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType:    commonconsts.ComponentTypeWorker,
		GPUMemoryService: &v1alpha1.GPUMemoryServiceSpec{Enabled: true},
		Resources: &v1alpha1.Resources{
			Limits: &v1alpha1.ResourceItem{GPU: strconv.Itoa(gpuCount)},
		},
	}
}

func gmsBasePodSpec() corev1.PodSpec {
	httpPort := intstr.FromString("system")
	return corev1.PodSpec{
		Containers: []corev1.Container{
			{
				Name:    "main",
				Image:   "test-image:latest",
				Command: []string{"python3", "-m", "dynamo.vllm"},
				Env: []corev1.EnvVar{
					{Name: "DYN_SYSTEM_PORT", Value: "9090"},
					{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"},
				},
				Ports: []corev1.ContainerPort{
					{Name: "system", ContainerPort: 9090, Protocol: corev1.ProtocolTCP},
				},
				StartupProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: httpPort},
					},
				},
				Resources: corev1.ResourceRequirements{
					Limits: corev1.ResourceList{
						corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("2"),
					},
				},
			},
		},
	}
}

// --- applyGPUMemoryService ---

func TestApplyGPUMemoryService_EmptyContainers(t *testing.T) {
	ps := corev1.PodSpec{}
	err := applyGPUMemoryService(&ps, gmsComponent(2), "myapp-worker-gpu")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least one container")
}

func TestApplyGPUMemoryService_MainContainerTransformed(t *testing.T) {
	ps := gmsBasePodSpec()
	err := applyGPUMemoryService(&ps, gmsComponent(2), "myapp-worker-gpu")
	require.NoError(t, err)

	main := ps.Containers[0]

	// GPU resources should be removed
	gpuResource := corev1.ResourceName(commonconsts.KubeResourceGPUNvidia)
	_, hasGPU := main.Resources.Limits[gpuResource]
	assert.False(t, hasGPU, "main container should not have GPU limits")

	// Should have DRA claim
	require.Len(t, main.Resources.Claims, 1)
	assert.Equal(t, gmsDRAClaimName, main.Resources.Claims[0].Name)

	// Should have shared volume mount
	var hasSharedMount bool
	for _, vm := range main.VolumeMounts {
		if vm.Name == gmsSharedVolumeName && vm.MountPath == gmsSharedMountPath {
			hasSharedMount = true
		}
	}
	assert.True(t, hasSharedMount, "main container should have gms-shared volume mount")

	// Should have TMPDIR
	envMap := envToMap(main.Env)
	assert.Equal(t, gmsSharedMountPath, envMap["TMPDIR"])
}

func TestApplyGPUMemoryService_GMSSidecarInjected(t *testing.T) {
	ps := gmsBasePodSpec()
	err := applyGPUMemoryService(&ps, gmsComponent(2), "myapp-worker-gpu")
	require.NoError(t, err)

	require.Len(t, ps.InitContainers, 1)
	gms := ps.InitContainers[0]
	assert.Equal(t, "gms-weights", gms.Name)
	assert.Equal(t, "test-image:latest", gms.Image)
	assert.Equal(t, []string{"bash", "-c"}, gms.Command)
	assert.Contains(t, gms.Args[0], "gpu_memory_service --device")
	assert.NotNil(t, gms.RestartPolicy)
	assert.Equal(t, corev1.ContainerRestartPolicyAlways, *gms.RestartPolicy)

	// GMS sidecar should have DRA claim
	require.Len(t, gms.Resources.Claims, 1)
	assert.Equal(t, gmsDRAClaimName, gms.Resources.Claims[0].Name)

	// GMS sidecar should have TMPDIR
	gmsEnv := envToMap(gms.Env)
	assert.Equal(t, gmsSharedMountPath, gmsEnv["TMPDIR"])
}

func TestApplyGPUMemoryService_SharedVolume(t *testing.T) {
	ps := gmsBasePodSpec()
	err := applyGPUMemoryService(&ps, gmsComponent(2), "myapp-worker-gpu")
	require.NoError(t, err)

	var found bool
	for _, v := range ps.Volumes {
		if v.Name == gmsSharedVolumeName {
			assert.NotNil(t, v.EmptyDir)
			found = true
		}
	}
	assert.True(t, found, "should have gms-shared volume")
}

func TestApplyGPUMemoryService_GPUToleration(t *testing.T) {
	ps := gmsBasePodSpec()
	err := applyGPUMemoryService(&ps, gmsComponent(2), "myapp-worker-gpu")
	require.NoError(t, err)

	var found bool
	for _, tol := range ps.Tolerations {
		if tol.Key == commonconsts.KubeResourceGPUNvidia && tol.Effect == corev1.TaintEffectNoSchedule {
			assert.Equal(t, corev1.TolerationOpExists, tol.Operator)
			found = true
		}
	}
	assert.True(t, found, "should have nvidia.com/gpu NoSchedule toleration")
}

func TestApplyGPUMemoryService_DRAResourceClaim(t *testing.T) {
	ps := gmsBasePodSpec()
	err := applyGPUMemoryService(&ps, gmsComponent(2), "myapp-worker-gpu")
	require.NoError(t, err)

	require.Len(t, ps.ResourceClaims, 1)
	assert.Equal(t, gmsDRAClaimName, ps.ResourceClaims[0].Name)
	assert.Equal(t, "myapp-worker-gpu", *ps.ResourceClaims[0].ResourceClaimTemplateName)
}

func TestApplyGPUMemoryService_PreservesExistingEnv(t *testing.T) {
	ps := gmsBasePodSpec()
	err := applyGPUMemoryService(&ps, gmsComponent(2), "myapp-worker-gpu")
	require.NoError(t, err)

	main := ps.Containers[0]
	envMap := envToMap(main.Env)
	assert.Equal(t, "kubernetes", envMap[commonconsts.DynamoDiscoveryBackendEnvVar])
	assert.Equal(t, "9090", envMap["DYN_SYSTEM_PORT"])
}

func TestApplyGPUMemoryService_SingleContainer(t *testing.T) {
	ps := gmsBasePodSpec()
	err := applyGPUMemoryService(&ps, gmsComponent(2), "myapp-worker-gpu")
	require.NoError(t, err)

	// Should still have exactly 1 regular container (no duplication)
	assert.Len(t, ps.Containers, 1)
	assert.Equal(t, "main", ps.Containers[0].Name)
}

// --- GMS sidecar helpers ---

func TestGmsWrapperScript_TwoTagsPerDevice(t *testing.T) {
	script := gmsWrapperScript(3)
	assert.Contains(t, script, "for dev in 0 1 2")
	assert.Contains(t, script, "--tag weights")
	assert.Contains(t, script, "--tag kv_cache")
	assert.Contains(t, script, "trap 'kill 0")
	assert.Contains(t, script, "wait -n")
}

func TestGmsReadyCheckCommand_TwoSocketsPerGPU(t *testing.T) {
	cmd := gmsReadyCheckCommand(2)
	assert.Equal(t, "sh", cmd[0])
	assert.Equal(t, "-c", cmd[1])
	assert.Contains(t, cmd[2], "gms_*.sock")
	// 2 GPUs * 2 tags = 4 sockets
	assert.Contains(t, cmd[2], "-ge 4")
}

func TestGmsReadyCheckCommand_SingleGPU(t *testing.T) {
	cmd := gmsReadyCheckCommand(1)
	// 1 GPU * 2 tags = 2 sockets
	assert.Contains(t, cmd[2], "-ge 2")
}

// --- GenerateGMSResourceClaimTemplate ---

func TestGenerateGMSResourceClaimTemplate_Enabled(t *testing.T) {
	component := gmsComponent(4)
	tmpl, toDelete, err := GenerateGMSResourceClaimTemplate(context.Background(), nil, "myapp-worker-gpu", "default", component)

	require.NoError(t, err)
	assert.False(t, toDelete)
	assert.Equal(t, "myapp-worker-gpu", tmpl.Name)
	assert.Equal(t, "default", tmpl.Namespace)

	require.Len(t, tmpl.Spec.Spec.Devices.Requests, 1)
	req := tmpl.Spec.Spec.Devices.Requests[0]
	assert.Equal(t, "gpus", req.Name)
	require.NotNil(t, req.Exactly)
	assert.Equal(t, defaultDeviceClassName, req.Exactly.DeviceClassName)
	assert.Equal(t, int64(4), req.Exactly.Count)
}

func TestGenerateGMSResourceClaimTemplate_CustomGPUType(t *testing.T) {
	component := gmsComponent(2)
	component.Resources.Limits.GPUType = "gpu.intel.com/xe"
	tmpl, toDelete, err := GenerateGMSResourceClaimTemplate(context.Background(), nil, "myapp-worker-gpu", "default", component)

	require.NoError(t, err)
	assert.False(t, toDelete)
	assert.Equal(t, "gpu.intel.com/xe", tmpl.Spec.Spec.Devices.Requests[0].Exactly.DeviceClassName)
}

func TestGenerateGMSResourceClaimTemplate_DisabledReturnsDelete(t *testing.T) {
	component := &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType: commonconsts.ComponentTypeWorker,
	}
	tmpl, toDelete, err := GenerateGMSResourceClaimTemplate(context.Background(), nil, "myapp-worker-gpu", "default", component)

	require.NoError(t, err)
	assert.True(t, toDelete)
	assert.Equal(t, "myapp-worker-gpu", tmpl.Name)
}

func TestGenerateGMSResourceClaimTemplate_NoGPUCountError(t *testing.T) {
	component := &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType:    commonconsts.ComponentTypeWorker,
		GPUMemoryService: &v1alpha1.GPUMemoryServiceSpec{Enabled: true},
	}
	_, _, err := GenerateGMSResourceClaimTemplate(context.Background(), nil, "myapp-worker-gpu", "default", component)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "resources must be specified")
}

// --- GMSResourceClaimTemplateName ---

func TestGMSResourceClaimTemplateName(t *testing.T) {
	assert.Equal(t, "myapp-worker-gpu", GMSResourceClaimTemplateName("myapp", "Worker"))
	assert.Equal(t, "app-vllmdecodeworker-gpu", GMSResourceClaimTemplateName("app", "VllmDecodeWorker"))
}

// --- isGMSEnabled ---

func TestIsGMSEnabled(t *testing.T) {
	assert.True(t, isGMSEnabled(&v1alpha1.DynamoComponentDeploymentSharedSpec{
		GPUMemoryService: &v1alpha1.GPUMemoryServiceSpec{Enabled: true},
	}))
	assert.False(t, isGMSEnabled(&v1alpha1.DynamoComponentDeploymentSharedSpec{
		GPUMemoryService: &v1alpha1.GPUMemoryServiceSpec{Enabled: false},
	}))
	assert.False(t, isGMSEnabled(&v1alpha1.DynamoComponentDeploymentSharedSpec{}))
}

// --- getGPUCount ---

func TestGetGPUCount(t *testing.T) {
	tests := []struct {
		name      string
		component *v1alpha1.DynamoComponentDeploymentSharedSpec
		want      int
		wantErr   bool
	}{
		{
			name:      "from limits",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{Resources: &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPU: "4"}}},
			want:      4,
		},
		{
			name:      "from requests",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{Resources: &v1alpha1.Resources{Requests: &v1alpha1.ResourceItem{GPU: "2"}}},
			want:      2,
		},
		{
			name:      "no resources",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			wantErr:   true,
		},
		{
			name:      "invalid GPU string",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{Resources: &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPU: "abc"}}},
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := getGPUCount(tt.component)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

// --- getDeviceClassName ---

func TestGetDeviceClassName(t *testing.T) {
	assert.Equal(t, defaultDeviceClassName, getDeviceClassName(&v1alpha1.DynamoComponentDeploymentSharedSpec{}))
	assert.Equal(t, defaultDeviceClassName, getDeviceClassName(&v1alpha1.DynamoComponentDeploymentSharedSpec{
		Resources: &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPU: "1"}},
	}))
	assert.Equal(t, "gpu.intel.com/xe", getDeviceClassName(&v1alpha1.DynamoComponentDeploymentSharedSpec{
		Resources: &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPUType: "gpu.intel.com/xe"}},
	}))
}

// helpers

func envToMap(envs []corev1.EnvVar) map[string]string {
	m := make(map[string]string, len(envs))
	for _, e := range envs {
		m[e.Name] = e.Value
	}
	return m
}
