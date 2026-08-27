/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controllercommon "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestResolveGPUShapeSeparatesEngineWidthFromReplicaCost(t *testing.T) {
	tests := []struct {
		name               string
		sidecarGPU         string
		wantGPUsPerEngine  int64
		wantGPUsPerReplica int64
	}{
		{
			name:               "zero-GPU sidecars preserve the original multinode shape",
			sidecarGPU:         "0",
			wantGPUsPerEngine:  8,
			wantGPUsPerReplica: 8,
		},
		{
			name:               "independent sidecar GPUs increase only replica cost",
			sidecarGPU:         "1",
			wantGPUsPerEngine:  8,
			wantGPUsPerReplica: 10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a two-node engine with four GPUs and one sidecar per node")
			main := corev1.Container{
				Name: commonconsts.MainContainerName,
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
					corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("4"),
				}},
			}
			sidecar := corev1.Container{
				Name: "sidecar",
				Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
					corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse(tt.sidecarGPU),
				}},
			}
			component := &v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				ComponentType: v1beta1.ComponentTypeDecode,
				Multinode:     &v1beta1.MultinodeSpec{NodeCount: 2},
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{main, sidecar},
				}},
			}
			leader := &corev1.PodSpec{Containers: []corev1.Container{main, sidecar}}
			worker := leader.DeepCopy()

			t.Log("Resolve per-engine width and the unique cost of one component replica")
			shape, err := ResolveGPUShape(
				t.Context(),
				nil,
				"default",
				component,
				[]PodSpecMultiplicity{
					{PodSpec: leader, Count: 1},
					{PodSpec: worker, Count: 1},
				},
			)
			require.NoError(t, err)
			assert.Equal(t, tt.wantGPUsPerEngine, shape.GPUsPerEngine)
			assert.Equal(t, tt.wantGPUsPerReplica, shape.GPUsPerReplica)
		})
	}
}

func TestResolveGroveGPUShapesIncludesUntypedAndMultinodeComponents(t *testing.T) {
	for _, sidecarGPU := range []string{"0", "1"} {
		t.Run("sidecar GPU "+sidecarGPU, func(t *testing.T) {
			untypedMain := gpuContainer(commonconsts.MainContainerName, "2")
			multinodeMain := gpuContainer(commonconsts.MainContainerName, "4")
			sidecar := gpuContainer("sidecar", sidecarGPU)
			dgd := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Namespace: "default"},
				Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
					{
						ComponentName: "custom",
						PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
							Containers: []corev1.Container{untypedMain},
						}},
					},
					{
						ComponentName: "decode",
						ComponentType: v1beta1.ComponentTypeDecode,
						Multinode:     &v1beta1.MultinodeSpec{NodeCount: 2},
						PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
							Containers: []corev1.Container{multinodeMain, sidecar},
						}},
					},
				}},
			}
			pcs := &grovev1alpha1.PodCliqueSet{Spec: grovev1alpha1.PodCliqueSetSpec{
				Template: grovev1alpha1.PodCliqueSetTemplateSpec{Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					groveClique("custom", "custom", []corev1.Container{untypedMain}),
					groveClique("decode-ldr", "decode", []corev1.Container{multinodeMain, sidecar}),
					groveClique("decode-wkr", "decode", []corev1.Container{multinodeMain, sidecar}),
				}},
			}}

			shapes, err := ResolveGroveGPUShapes(t.Context(), nil, dgd, pcs)
			require.NoError(t, err)
			assert.Equal(t, GPUShape{GPUsPerEngine: 2, GPUsPerReplica: 2}, shapes["custom"])
			wantReplica := int64(8)
			if sidecarGPU == "1" {
				wantReplica = 10
			}
			assert.Equal(t, GPUShape{GPUsPerEngine: 8, GPUsPerReplica: wantReplica}, shapes["decode"])
		})
	}
}

func TestResolveGPUShapeDeduplicatesConcreteClaimAcrossNodes(t *testing.T) {
	main := corev1.Container{
		Name: commonconsts.MainContainerName,
		Resources: corev1.ResourceRequirements{
			Claims: []corev1.ResourceClaim{{Name: "gpu"}},
		},
	}
	podSpec := corev1.PodSpec{
		ResourceClaims: []corev1.PodResourceClaim{{
			Name:              "gpu",
			ResourceClaimName: ptr.To("shared-gpu"),
		}},
		Containers: []corev1.Container{main},
	}
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "decode",
		ComponentType: v1beta1.ComponentTypeDecode,
		Multinode:     &v1beta1.MultinodeSpec{NodeCount: 2},
		PodTemplate:   &corev1.PodTemplateSpec{Spec: podSpec},
	}
	claim := &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{Name: "shared-gpu", Namespace: "default"},
		Spec: resourcev1.ResourceClaimSpec{Devices: resourcev1.DeviceClaim{
			Requests: []resourcev1.DeviceRequest{{
				Name: "gpu",
				Exactly: &resourcev1.ExactDeviceRequest{
					DeviceClassName: "gpu.nvidia.com",
					AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
					Count:           2,
				},
			}},
		}},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, resourcev1.AddToScheme(scheme))
	reader := fake.NewClientBuilder().WithScheme(scheme).WithObjects(
		claim,
		&resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "gpu.nvidia.com"}},
	).Build()

	shape, err := ResolveGPUShape(t.Context(), reader, "default", component, []PodSpecMultiplicity{
		{PodSpec: podSpec.DeepCopy(), Count: 1},
		{PodSpec: podSpec.DeepCopy(), Count: 1},
	})
	require.NoError(t, err)
	assert.Equal(t, GPUShape{GPUsPerEngine: 2, GPUsPerReplica: 2}, shape)
}

func TestResolveGroveGPUShapesCountsInterPodSharedGPUsOnce(t *testing.T) {
	for _, tt := range []struct {
		name        string
		sidecarGPU  string
		wantReplica int64
	}{
		{name: "zero-GPU sidecar preserves engine allocation", sidecarGPU: "0", wantReplica: 8},
		{name: "provider-cloned GPU sidecar increases replica cost", sidecarGPU: "1", wantReplica: 12},
	} {
		t.Run(tt.name, func(t *testing.T) {
			main := gpuContainer(commonconsts.MainContainerName, "4")
			sidecar := gpuContainer("sidecar", tt.sidecarGPU)
			dgd := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
						ComponentName: "decode",
						ComponentType: v1beta1.ComponentTypeDecode,
						Multinode:     &v1beta1.MultinodeSpec{NodeCount: 2},
						PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
							Containers: []corev1.Container{main, sidecar},
						}},
						Experimental: &v1beta1.ExperimentalSpec{GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{
							Mode: v1beta1.GMSModeInterPod,
						}},
					}},
				},
			}
			pcs, err := GenerateGrovePodCliqueSet(
				t.Context(),
				dgd,
				&configv1alpha1.OperatorConfiguration{
					Discovery: configv1alpha1.DiscoveryConfiguration{Backend: "kubernetes"},
					Infrastructure: configv1alpha1.InfrastructureConfiguration{
						ETCDAddress: "etcd-address",
						NATSAddress: "nats-address",
					},
				},
				&controllercommon.RuntimeConfig{Gate: features.Gates{DRA: true}},
				nil,
				nil,
				nil,
				nil,
				nil,
			)
			require.NoError(t, err)

			shapes, err := ResolveGroveGPUShapes(t.Context(), nil, dgd, pcs)
			require.NoError(t, err)
			assert.Equal(t, GPUShape{GPUsPerEngine: 8, GPUsPerReplica: tt.wantReplica}, shapes["decode"])
		})
	}
}

func gpuContainer(name, gpuCount string) corev1.Container {
	return corev1.Container{
		Name: name,
		Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{
			corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse(gpuCount),
		}},
	}
}

func groveClique(name, component string, containers []corev1.Container) *grovev1alpha1.PodCliqueTemplateSpec {
	return &grovev1alpha1.PodCliqueTemplateSpec{
		Name: name,
		Labels: map[string]string{
			commonconsts.KubeLabelDynamoComponent: component,
		},
		Spec: grovev1alpha1.PodCliqueSpec{
			PodSpec: corev1.PodSpec{Containers: containers},
		},
	}
}
