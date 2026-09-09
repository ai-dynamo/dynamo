// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestResolveLPXGPUShape(t *testing.T) {
	for _, test := range []struct {
		name string
		want GPUShape
	}{
		{"scalar-init-peak", GPUShape{GPUsPerEngine: 8, GPUsPerReplica: 21}},
		{"claim-template", GPUShape{GPUsPerEngine: 4, GPUsPerReplica: 7}},
		{"shared-claim", GPUShape{GPUsPerEngine: 2, GPUsPerReplica: 5}},
		{"lpu-only", GPUShape{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Count two GPU workers and three auxiliary Pods in each complete engine replica")
			nativeSidecar := gpuContainer("native-sidecar", "1")
			nativeSidecar.RestartPolicy = ptr.To(corev1.ContainerRestartPolicyAlways)
			gpu := &grovev1alpha1.PodCliqueTemplateSpec{
				Name: "cyborg", Labels: map[string]string{lpx.ExecutionRoleLabel: lpxGPUExecutionRole},
				Spec: grovev1alpha1.PodCliqueSpec{Replicas: 2, PodSpec: corev1.PodSpec{
					Containers:     []corev1.Container{gpuContainer(consts.MainContainerName, "4"), gpuContainer("sidecar", "1")},
					InitContainers: []corev1.Container{nativeSidecar, gpuContainer("prepare", "8")},
				}},
			}
			auxiliary := &grovev1alpha1.PodCliqueTemplateSpec{
				Name: "agents", Labels: map[string]string{lpx.ExecutionRoleLabel: "lpu"},
				Spec: grovev1alpha1.PodCliqueSpec{Replicas: 3, PodSpec: corev1.PodSpec{
					Containers: []corev1.Container{gpuContainer("agent", "0"), gpuContainer("metrics", "1")},
				}},
			}
			pcs := &grovev1alpha1.PodCliqueSet{ObjectMeta: metav1.ObjectMeta{Namespace: "test"},
				Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{
					Cliques:                      []*grovev1alpha1.PodCliqueTemplateSpec{gpu, auxiliary},
					PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{{Replicas: ptr.To(int32(3))}},
				}},
			}
			claim := &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Namespace: pcs.Namespace, Name: "gpu"},
				Spec: resourcev1.ResourceClaimSpec{Devices: resourcev1.DeviceClaim{Requests: []resourcev1.DeviceRequest{{
					Name: "gpu", Exactly: &resourcev1.ExactDeviceRequest{DeviceClassName: "gpu.nvidia.com", Count: 2, AllocationMode: resourcev1.DeviceAllocationModeExactCount},
				}}}},
			}
			claimTemplate := &resourcev1.ResourceClaimTemplate{ObjectMeta: claim.ObjectMeta,
				Spec: resourcev1.ResourceClaimTemplateSpec{Spec: claim.Spec},
			}
			scheme := runtime.NewScheme()
			require.NoError(t, resourcev1.AddToScheme(scheme))
			reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(claim, claimTemplate,
				&resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "gpu.nvidia.com"}}).Build()
			switch test.name {
			case "claim-template", "shared-claim":
				pod := &gpu.Spec.PodSpec
				pod.InitContainers = nil
				for i := range pod.Containers {
					pod.Containers[i].Resources = corev1.ResourceRequirements{Claims: []corev1.ResourceClaim{{Name: "gpu"}}}
				}
				pod.ResourceClaims = []corev1.PodResourceClaim{{Name: "gpu", ResourceClaimTemplateName: ptr.To(claimTemplate.Name)}}
				if test.name == "shared-claim" {
					pod.ResourceClaims[0].ResourceClaimTemplateName = nil
					pod.ResourceClaims[0].ResourceClaimName = ptr.To(claim.Name)
				}
			case "lpu-only":
				pcs.Spec.Template.Cliques = []*grovev1alpha1.PodCliqueTemplateSpec{auxiliary}
				auxiliary.Spec.PodSpec.Containers = auxiliary.Spec.PodSpec.Containers[:1]
			}
			before := pcs.DeepCopy()
			shape, err := ResolveLPXGPUShape(t.Context(), reader, pcs)
			require.NoError(t, err)
			require.Equal(t, test.want, shape)
			require.Equal(t, before, pcs, "GPU accounting must not mutate shared Pod specs")

			if test.name == "claim-template" {
				t.Log("Missing dependencies must not produce a misleading zero-GPU result")
				require.NoError(t, reader.Delete(t.Context(), claimTemplate))
				_, err := ResolveLPXGPUShape(t.Context(), reader, pcs)
				require.ErrorContains(t, err, "resolve LPX engine GPUs")
			}
		})
	}
}
