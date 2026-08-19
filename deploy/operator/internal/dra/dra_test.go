/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dra

import (
	"context"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
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

func TestApplyClaim_ReplacesMIGResourceWithDRAClaim(t *testing.T) {
	migResource := corev1.ResourceName("nvidia.com/mig-3g.20gb")
	ps := basePodSpec()
	ps.Containers[0].Resources.Limits = corev1.ResourceList{
		migResource: resource.MustParse("1"),
	}
	ps.Containers[0].Resources.Requests = corev1.ResourceList{
		migResource: resource.MustParse("1"),
	}

	err := ApplyClaim(&ps, "myapp-worker-gpu")
	require.NoError(t, err)

	main := ps.Containers[0]
	assert.NotContains(t, main.Resources.Limits, migResource)
	assert.NotContains(t, main.Resources.Requests, migResource)
	require.Len(t, main.Resources.Claims, 1)
	assert.Equal(t, ClaimName, main.Resources.Claims[0].Name)
}

func TestApplyClaimOverridesOperatorOwnedClaim(t *testing.T) {
	oldTemplate := "old-template"
	ps := basePodSpec()
	ps.ResourceClaims = []corev1.PodResourceClaim{{
		Name:                      ClaimName,
		ResourceClaimTemplateName: &oldTemplate,
	}}

	require.NoError(t, ApplyClaim(&ps, "new-template"))

	require.Len(t, ps.ResourceClaims, 1)
	assert.Equal(t, "new-template", *ps.ResourceClaims[0].ResourceClaimTemplateName)
}

func TestApplyClaimPreservesNamedResourceClaim(t *testing.T) {
	claimName := "dest-gpus"
	ps := basePodSpec()
	ps.ResourceClaims = []corev1.PodResourceClaim{{
		Name:              ClaimName,
		ResourceClaimName: &claimName,
	}}

	require.NoError(t, ApplyClaim(&ps, "should-not-replace-named-claim"))

	require.Len(t, ps.ResourceClaims, 1)
	require.NotNil(t, ps.ResourceClaims[0].ResourceClaimName)
	assert.Equal(t, claimName, *ps.ResourceClaims[0].ResourceClaimName)
	assert.Nil(t, ps.ResourceClaims[0].ResourceClaimTemplateName)
	require.Len(t, ps.Containers[0].Resources.Claims, 1)
	assert.Equal(t, ClaimName, ps.Containers[0].Resources.Claims[0].Name)
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

func TestExtractGPUCountFromResourceRequirements_DeterministicResourceSelection(t *testing.T) {
	resources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceName("nvidia.com/mig-3g.20gb"):           resource.MustParse("1"),
			corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("4"),
		},
	}

	gpuCount, err := ExtractGPUCountFromResourceRequirements(resources)
	require.NoError(t, err)
	assert.Equal(t, 4, gpuCount)
}

func TestExtractGPUCountFromResourceRequirements_MIGResource(t *testing.T) {
	resources := corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceName("nvidia.com/mig-3g.20gb"): resource.MustParse("2"),
		},
	}

	gpuCount, err := ExtractGPUCountFromResourceRequirements(resources)
	require.NoError(t, err)
	assert.Equal(t, 2, gpuCount)
}

func TestExtractGPUCountFromResourceRequirements_RejectsFractionalGPU(t *testing.T) {
	resources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("500m"),
		},
	}

	_, err := ExtractGPUCountFromResourceRequirements(resources)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "must be a whole number")
	assert.Contains(t, err.Error(), "500m")
}

func TestResolveGPUCountFromResourceClaims(t *testing.T) {
	driverSelectors := func(driver string) []resourcev1.DeviceSelector {
		return []resourcev1.DeviceSelector{{
			CEL: &resourcev1.CELDeviceSelector{Expression: "device.driver == '" + driver + "'"},
		}}
	}
	exactRequest := func(name, deviceClass string, count int64, mode resourcev1.DeviceAllocationMode) resourcev1.DeviceRequest {
		return resourcev1.DeviceRequest{
			Name: name,
			Exactly: &resourcev1.ExactDeviceRequest{
				DeviceClassName: deviceClass,
				AllocationMode:  mode,
				Count:           count,
			},
		}
	}
	exactRequestWithDriver := func(name, deviceClass, driver string, count int64) resourcev1.DeviceRequest {
		request := exactRequest(name, deviceClass, count, resourcev1.DeviceAllocationModeExactCount)
		request.Exactly.Selectors = driverSelectors(driver)
		return request
	}
	claimTemplate := func(name string, requests ...resourcev1.DeviceRequest) client.Object {
		return &resourcev1.ResourceClaimTemplate{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default"},
			Spec: resourcev1.ResourceClaimTemplateSpec{
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{Requests: requests},
				},
			},
		}
	}
	claim := func(name string, requests ...resourcev1.DeviceRequest) client.Object {
		return &resourcev1.ResourceClaim{
			ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default"},
			Spec: resourcev1.ResourceClaimSpec{
				Devices: resourcev1.DeviceClaim{Requests: requests},
			},
		}
	}
	templatePodClaim := func(claimName, templateName string) corev1.PodResourceClaim {
		return corev1.PodResourceClaim{
			Name:                      claimName,
			ResourceClaimTemplateName: ptr.To(templateName),
		}
	}
	deviceClass := func(name, driver string) client.Object {
		return &resourcev1.DeviceClass{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Spec: resourcev1.DeviceClassSpec{
				Selectors: driverSelectors(driver),
			},
		}
	}
	deviceClasses := []client.Object{
		deviceClass("gpu.nvidia.com", "gpu.nvidia.com"),
		deviceClass("gpu.intel.com", "gpu.intel.com"),
		deviceClass("rdma-dranet", "dra.net.example.com"),
	}

	tests := []struct {
		name        string
		objects     []client.Object
		podClaim    corev1.PodResourceClaim
		requestName string
		want        int
		wantErr     string
	}{
		{
			name: "ResourceClaimTemplate exact NVIDIA GPU count",
			objects: []client.Object{claimTemplate(
				"gpu-template",
				exactRequest("gpus", "gpu.nvidia.com", 4, resourcev1.DeviceAllocationModeExactCount),
			)},
			podClaim: templatePodClaim("accelerators", "gpu-template"),
			want:     4,
		},
		{
			name: "ResourceClaim request selector excludes non-GPU devices",
			objects: []client.Object{claim(
				"devices",
				exactRequest("gpus", "gpu.intel.com", 2, ""),
				exactRequest("network", "rdma-dranet", 1, ""),
			)},
			podClaim: corev1.PodResourceClaim{
				Name:              "accelerators",
				ResourceClaimName: ptr.To("devices"),
			},
			requestName: "gpus",
			want:        2,
		},
		{
			name: "default exact count is one",
			objects: []client.Object{claimTemplate(
				"gpu-template",
				exactRequest("gpus", "gpu.nvidia.com", 0, ""),
			)},
			podClaim: templatePodClaim("accelerators", "gpu-template"),
			want:     1,
		},
		{
			name: "custom DeviceClass name uses its GPU driver selector",
			objects: []client.Object{
				deviceClass("shared-h100.example.com", "gpu.nvidia.com"),
				claimTemplate(
					"gpu-template",
					exactRequest("gpus", "shared-h100.example.com", 8, resourcev1.DeviceAllocationModeExactCount),
				),
			},
			podClaim: templatePodClaim("accelerators", "gpu-template"),
			want:     8,
		},
		{
			name: "custom DeviceClass name uses its extended GPU resource",
			objects: []client.Object{
				&resourcev1.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: "shared-gpus.example.com"},
					Spec: resourcev1.DeviceClassSpec{
						ExtendedResourceName: ptr.To(commonconsts.KubeResourceGPUNvidia),
					},
				},
				claimTemplate(
					"gpu-template",
					exactRequest("gpus", "shared-gpus.example.com", 4, resourcev1.DeviceAllocationModeExactCount),
				),
			},
			podClaim: templatePodClaim("accelerators", "gpu-template"),
			want:     4,
		},
		{
			name: "request selector overrides a GPU-like empty DeviceClass",
			objects: []client.Object{
				&resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "gpu.example.com"}},
				claimTemplate(
					"network-template",
					exactRequestWithDriver("network", "gpu.example.com", "dra.net.example.com", 1),
				),
			},
			podClaim: templatePodClaim("network", "network-template"),
			want:     0,
		},
		{
			name: "request selector classifies an otherwise unknown DeviceClass as GPU",
			objects: []client.Object{
				&resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "accelerators.example.com"}},
				claimTemplate(
					"gpu-template",
					exactRequestWithDriver("gpus", "accelerators.example.com", "gpu.nvidia.com", 2),
				),
			},
			podClaim: templatePodClaim("accelerators", "gpu-template"),
			want:     2,
		},
		{
			name: "same DeviceClass is classified independently for each request",
			objects: []client.Object{
				&resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "devices.example.com"}},
				claimTemplate(
					"devices-template",
					exactRequestWithDriver("gpus", "devices.example.com", "gpu.nvidia.com", 2),
					exactRequestWithDriver("network", "devices.example.com", "dra.net.example.com", 1),
				),
			},
			podClaim: templatePodClaim("devices", "devices-template"),
			want:     2,
		},
		{
			name: "conflicting class and request driver selectors are rejected",
			objects: []client.Object{
				deviceClass("shared-gpus.example.com", "gpu.nvidia.com"),
				claimTemplate(
					"conflicting-template",
					exactRequestWithDriver("devices", "shared-gpus.example.com", "dra.net.example.com", 1),
				),
			},
			podClaim: templatePodClaim("devices", "conflicting-template"),
			wantErr:  "constrain device.driver to both GPU and non-GPU drivers",
		},
		{
			name: "non-GPU claim is ignored",
			objects: []client.Object{claimTemplate(
				"network-template",
				exactRequest("network", "rdma-dranet", 1, ""),
			)},
			podClaim: templatePodClaim("network", "network-template"),
			want:     0,
		},
		{
			name: "allocation mode All is rejected",
			objects: []client.Object{claimTemplate(
				"gpu-template",
				exactRequest("gpus", "gpu.nvidia.com", 0, resourcev1.DeviceAllocationModeAll),
			)},
			podClaim: templatePodClaim("accelerators", "gpu-template"),
			wantErr:  "has no deterministic per-node device count",
		},
		{
			name: "mixed firstAvailable alternatives are rejected",
			objects: []client.Object{claimTemplate(
				"mixed-template",
				resourcev1.DeviceRequest{
					Name: "devices",
					FirstAvailable: []resourcev1.DeviceSubRequest{
						{Name: "gpu", DeviceClassName: "gpu.nvidia.com", Count: 1},
						{Name: "network", DeviceClassName: "rdma-dranet", Count: 1},
					},
				},
			)},
			podClaim: templatePodClaim("devices", "mixed-template"),
			wantErr:  "mixes GPU and non-GPU",
		},
		{
			name: "equal-count GPU firstAvailable alternatives are supported",
			objects: []client.Object{
				&resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "nvidia.example.com"}},
				&resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "intel.example.com"}},
				claimTemplate(
					"gpu-template",
					resourcev1.DeviceRequest{
						Name: "gpus",
						FirstAvailable: []resourcev1.DeviceSubRequest{
							{Name: "nvidia", DeviceClassName: "nvidia.example.com", Selectors: driverSelectors("gpu.nvidia.com"), Count: 2},
							{Name: "intel", DeviceClassName: "intel.example.com", Selectors: driverSelectors("gpu.intel.com"), Count: 2},
						},
					},
				),
			},
			podClaim: templatePodClaim("accelerators", "gpu-template"),
			want:     2,
		},
		{
			name: "unclassifiable DeviceClass is actionable",
			objects: []client.Object{
				&resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "accelerators.example.com"}},
				claimTemplate(
					"accelerator-template",
					exactRequest("accelerators", "accelerators.example.com", 1, ""),
				),
			},
			podClaim: templatePodClaim("accelerators", "accelerator-template"),
			wantErr:  "cannot determine whether DeviceClass \"accelerators.example.com\" provides GPUs",
		},
		{
			name: "unsupported selector does not fall back to GPU-like class name",
			objects: []client.Object{
				&resourcev1.DeviceClass{
					ObjectMeta: metav1.ObjectMeta{Name: "gpu.example.com"},
					Spec: resourcev1.DeviceClassSpec{
						Selectors: []resourcev1.DeviceSelector{{CEL: &resourcev1.CELDeviceSelector{
							Expression: `device.driver != "cpu.example.com"`,
						}}},
					},
				},
				claimTemplate(
					"gpu-template",
					exactRequest("gpus", "gpu.example.com", 1, ""),
				),
			},
			podClaim: templatePodClaim("accelerators", "gpu-template"),
			wantErr:  "cannot determine whether DeviceClass \"gpu.example.com\" provides GPUs",
		},
		{
			name: "missing DeviceClass is actionable",
			objects: []client.Object{claimTemplate(
				"missing-class-template",
				exactRequest("accelerators", "missing.example.com", 1, ""),
			)},
			podClaim: templatePodClaim("accelerators", "missing-class-template"),
			wantErr:  "failed to get DeviceClass \"missing.example.com\"",
		},
		{
			name: "request without a supported shape is rejected",
			objects: []client.Object{claimTemplate(
				"invalid-template",
				resourcev1.DeviceRequest{Name: "devices"},
			)},
			podClaim: templatePodClaim("devices", "invalid-template"),
			wantErr:  "must set exactly or firstAvailable",
		},
		{
			name:     "missing ResourceClaimTemplate is actionable",
			podClaim: templatePodClaim("accelerators", "missing"),
			wantErr:  "failed to get ResourceClaimTemplate default/missing",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Create a client containing the claim source")
			scheme := runtime.NewScheme()
			require.NoError(t, resourcev1.AddToScheme(scheme))
			objects := append([]client.Object{}, deviceClasses...)
			objects = append(objects, tt.objects...)
			kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build()

			t.Log("Reference the pod claim from the main container")
			podSpec := &corev1.PodSpec{ResourceClaims: []corev1.PodResourceClaim{tt.podClaim}}
			resources := corev1.ResourceRequirements{
				Claims: []corev1.ResourceClaim{{
					Name:    tt.podClaim.Name,
					Request: tt.requestName,
				}},
			}

			t.Log("Resolve the exact GPU count")
			got, err := ResolveGPUCount(context.Background(), kubeClient, "default", podSpec, resources)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestResolveGPUCountPrefersScalarResources(t *testing.T) {
	t.Log("Combine a scalar GPU request with an unresolved ResourceClaim")
	resources := corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("8"),
		},
		Claims: []corev1.ResourceClaim{{Name: "missing"}},
	}

	t.Log("Resolve without a Kubernetes client")
	got, err := ResolveGPUCount(context.Background(), nil, "default", nil, resources)
	require.NoError(t, err)
	assert.Equal(t, 8, got)
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
