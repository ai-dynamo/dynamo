/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dra

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

func basePodSpec() corev1.PodSpec {
	httpPort := intstr.FromString("system")
	return corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:    "main",
			Image:   "test-image:latest",
			Command: []string{"python3", "-m", "dynamo.vllm"},
			Env: []corev1.EnvVar{
				{Name: "DYN_SYSTEM_PORT", Value: "9090"},
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
		}},
	}
}

func TestApplyClaim_EmptyContainers(t *testing.T) {
	ps := corev1.PodSpec{}
	err := ApplyClaim(&ps, "myapp-worker-gpu")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least one container")
}

func TestApplyClaim_ReplacesGPUWithDRAClaim(t *testing.T) {
	ps := basePodSpec()
	err := ApplyClaim(&ps, "myapp-worker-gpu")
	require.NoError(t, err)

	main := ps.Containers[0]

	gpuResource := corev1.ResourceName(commonconsts.KubeResourceGPUNvidia)
	_, hasGPU := main.Resources.Limits[gpuResource]
	assert.False(t, hasGPU)

	require.Len(t, main.Resources.Claims, 1)
	assert.Equal(t, ClaimName, main.Resources.Claims[0].Name)

	require.Len(t, ps.ResourceClaims, 1)
	assert.Equal(t, ClaimName, ps.ResourceClaims[0].Name)
	assert.Equal(t, "myapp-worker-gpu", *ps.ResourceClaims[0].ResourceClaimTemplateName)

	var hasToleration bool
	for _, tol := range ps.Tolerations {
		if tol.Key == commonconsts.KubeResourceGPUNvidia && tol.Effect == corev1.TaintEffectNoSchedule {
			hasToleration = true
		}
	}
	assert.True(t, hasToleration)
	assert.Empty(t, ps.InitContainers)
}

func TestApplyClaim_AlwaysTargetsFirstContainer(t *testing.T) {
	ps := basePodSpec()
	ps.Containers = append(ps.Containers, corev1.Container{Name: "sidecar", Image: "sidecar:latest"})

	err := ApplyClaim(&ps, "myapp-worker-gpu")
	require.NoError(t, err)

	require.Len(t, ps.Containers[0].Resources.Claims, 1)
	assert.Equal(t, ClaimName, ps.Containers[0].Resources.Claims[0].Name)
	assert.Empty(t, ps.Containers[1].Resources.Claims)
}

func TestGenerateResourceClaimTemplate_Enabled(t *testing.T) {
	tmpl, toDelete, err := GenerateResourceClaimTemplate(context.Background(), nil, "myapp-worker-gpu", "default", 4, "")
	require.NoError(t, err)
	assert.False(t, toDelete)
	assert.Equal(t, "myapp-worker-gpu", tmpl.Name)
	require.Len(t, tmpl.Spec.Spec.Devices.Requests, 1)
	req := tmpl.Spec.Spec.Devices.Requests[0]
	assert.Equal(t, DefaultDeviceClassName, req.Exactly.DeviceClassName)
	assert.Equal(t, int64(4), req.Exactly.Count)
}

func TestGenerateResourceClaimTemplate_CustomDeviceClass(t *testing.T) {
	tmpl, _, err := GenerateResourceClaimTemplate(context.Background(), nil, "myapp-worker-gpu", "default", 2, "gpu.intel.com/xe")
	require.NoError(t, err)
	assert.Equal(t, "gpu.intel.com/xe", tmpl.Spec.Spec.Devices.Requests[0].Exactly.DeviceClassName)
}

func TestGenerateResourceClaimTemplate_DisabledReturnsDelete(t *testing.T) {
	tmpl, toDelete, err := GenerateResourceClaimTemplate(context.Background(), nil, "myapp-worker-gpu", "default", 0, "")
	require.NoError(t, err)
	assert.True(t, toDelete)
	assert.Equal(t, "myapp-worker-gpu", tmpl.Name)
}

func TestResourceClaimTemplateName(t *testing.T) {
	assert.Equal(t, "myapp-worker-gpu", ResourceClaimTemplateName("myapp", "Worker"))
	assert.Equal(t, "app-vllmdecodeworker-gpu", ResourceClaimTemplateName("app", "VllmDecodeWorker"))
}

func TestExtractGPUParams(t *testing.T) {
	count, dc := ExtractGPUParams(nil, nil)
	assert.Equal(t, 0, count)
	assert.Equal(t, "", dc)

	count, dc = ExtractGPUParams(
		&v1alpha1.GPUMemoryServiceSpec{Enabled: true},
		&v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPU: strconv.Itoa(4)}},
	)
	assert.Equal(t, 4, count)
	assert.Equal(t, "", dc)

	count, dc = ExtractGPUParams(
		&v1alpha1.GPUMemoryServiceSpec{Enabled: true, DeviceClassName: "gpu.intel.com/xe"},
		&v1alpha1.Resources{Requests: &v1alpha1.ResourceItem{GPU: "2"}},
	)
	assert.Equal(t, 2, count)
	assert.Equal(t, "gpu.intel.com/xe", dc)
}
