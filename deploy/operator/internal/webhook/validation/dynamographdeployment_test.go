/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package validation

import (
	"context"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	authenticationv1 "k8s.io/api/authentication/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sptr "k8s.io/utils/ptr"
)

func TestDynamoGraphDeploymentValidator_Validate(t *testing.T) {
	tests := []struct {
		name         string
		deployment   *nvidiacomv1beta1.DynamoGraphDeployment
		groveEnabled bool
		wantErr      string
	}{
		{
			name:       "valid deployment with components",
			deployment: newBetaDGDForValidation(),
		},
		{
			name: "no components",
			deployment: &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test-graph", Namespace: "default"},
			},
			wantErr: "spec.components must have at least one component",
		},
		{
			name: "component name is required",
			deployment: &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test-graph", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{{}},
				},
			},
			wantErr: "spec.components[0].name is required",
		},
		{
			name: "component names are unique case-insensitively",
			deployment: &nvidiacomv1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test-graph", Namespace: "default"},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
					Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						{ComponentName: "worker"},
						{ComponentName: "WORKER"},
					},
				},
			},
			wantErr: `spec.components[1].name "WORKER" duplicates component "worker" case-insensitively`,
		},
		{
			name: "component replicas must be non-negative",
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Replicas = k8sptr.To(int32(-1))
			}),
			wantErr: "spec.components[worker].replicas must be non-negative",
		},
		{
			name: "component minAvailable requires Grove",
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.MinAvailable = k8sptr.To(int32(1))
			}),
			wantErr: "spec.components[worker].minAvailable is currently supported only for Grove-backed DynamoGraphDeployment components",
		},
		{
			name: "restart parallel strategy cannot specify order",
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = &nvidiacomv1beta1.Restart{
					ID: "roll",
					Strategy: &nvidiacomv1beta1.RestartStrategy{
						Type:  nvidiacomv1beta1.RestartStrategyTypeParallel,
						Order: []string{"frontend", "worker"},
					},
				}
			}),
			wantErr: "spec.restart.strategy.order cannot be specified when strategy is parallel",
		},
		{
			name: "component topology constraint requires deployment topology",
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.TopologyConstraint = &nvidiacomv1beta1.TopologyConstraint{PackDomain: "rack"}
			}),
			wantErr: "spec.topologyConstraint with clusterTopologyName is required when any topology constraint is set",
		},
		{
			name:         "inter-pod GMS requires Grove",
			groveEnabled: false,
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaInterPodGMS(worker)
			}),
			wantErr: `spec.components[worker]: experimental.gpuMemoryService.mode="InterPod" requires the Grove pathway`,
		},
		{
			name:         "inter-pod GMS requires vLLM backend",
			groveEnabled: true,
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.BackendFramework = "sglang"
				enableBetaInterPodGMS(&spec.Components[1])
			}),
			wantErr: `spec.components[worker]: the inter-pod GMS layout (experimental.gpuMemoryService.mode="InterPod") is currently supported only for vLLM`,
		},
		{
			name: "kv transfer policy requires exactly one topology selector",
			deployment: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Experimental = &nvidiacomv1beta1.DynamoGraphDeploymentExperimentalSpec{
					KvTransferPolicy: &nvidiacomv1beta1.KvTransferPolicy{
						Domain: "rack",
					},
				}
			}),
			wantErr: "spec.experimental.kvTransferPolicy: exactly one of labelKey or clusterTopologyName is required",
		},
		{
			name: "intra-pod failover requires container discovery",
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				enableBetaIntraPodGMS(worker)
				worker.Experimental.Failover = &nvidiacomv1beta1.FailoverSpec{
					Mode: nvidiacomv1beta1.GMSModeIntraPod,
				}
			}),
			wantErr: `failover requires per-container K8s discovery; set annotation "nvidia.com/dynamo-kube-discovery-mode" to "container"`,
		},
		{
			name: "checkpoint job cannot be combined with checkpointRef",
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{
						Enabled:       true,
						CheckpointRef: k8sptr.To("existing-checkpoint"),
						Job:           &nvidiacomv1beta1.ComponentCheckpointJobConfig{},
					},
				}
			}),
			wantErr: "spec.components[worker].experimental.checkpoint.job cannot be set when checkpointRef is specified",
		},
		{
			name: "GMS requires GPU resources on the main container",
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
						Mode: nvidiacomv1beta1.GMSModeIntraPod,
					},
				}
			}),
			wantErr: "spec.components[worker].experimental.gpuMemoryService: GPU memory service requires podTemplate.spec.containers[main].resources.limits.nvidia.com/gpu >= 1",
		},
		{
			name: "sidecars must provide an image",
			deployment: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.PodTemplate = &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: consts.MainContainerName},
							{Name: "metrics"},
						},
					},
				}
			}),
			wantErr: `spec.components[worker].podTemplate.spec.containers[1].image is required for sidecar container "metrics"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentValidator(nil, tt.groveEnabled)
			_, err := validator.Validate(context.Background(), tt.deployment)
			assertBetaValidationError(t, err, tt.wantErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_ValidateAlphaCompatibility(t *testing.T) {
	tests := []struct {
		name    string
		mutate  func(*nvidiacomv1alpha1.DynamoGraphDeployment)
		wantErr string
	}{
		{
			name: "alpha PVC create requires storage fields",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.PVCs = []nvidiacomv1alpha1.PVC{
					{
						Name:   k8sptr.To("cache"),
						Create: k8sptr.To(true),
					},
				}
			},
			wantErr: "spec.pvcs[0].storageClass is required when create is true",
		},
		{
			name: "alpha ingress requires host",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				className := "nginx"
				dgd.Spec.Services["frontend"] = &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeFrontend,
					Ingress: &nvidiacomv1alpha1.IngressSpec{
						Enabled:                    true,
						IngressControllerClassName: &className,
					},
				}
			},
			wantErr: "spec.services[frontend].ingress.host is required when ingress is enabled",
		},
		{
			name: "alpha service annotations are validated",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].Annotations = map[string]string{
					consts.KubeAnnotationVLLMDistributedExecutorBackend: "typo",
				}
			},
			wantErr: `spec.services[worker].annotations[nvidia.com/vllm-distributed-executor-backend] has invalid value "typo"`,
		},
		{
			name: "alpha sharedMemory requires size when enabled",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].SharedMemory = &nvidiacomv1alpha1.SharedMemorySpec{
					Disabled: false,
				}
			},
			wantErr: "spec.services[worker].sharedMemory.size is required when disabled is false",
		},
		{
			name: "alpha frontend sidecar rejects generated container name conflict",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["frontend"] = &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: consts.ComponentTypeFrontend,
					FrontendSidecar: &nvidiacomv1alpha1.FrontendSidecarSpec{
						Image: "custom/frontend:latest",
					},
					ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
						PodSpec: &corev1.PodSpec{
							Containers: []corev1.Container{
								{
									Name:  consts.FrontendSidecarContainerName,
									Image: "custom/frontend:latest",
								},
							},
						},
					},
				}
			},
			wantErr: `spec.services[frontend]: cannot inject frontend sidecar: a container named "sidecar-frontend" already exists in extraPodSpec.containers`,
		},
		{
			name: "disabled alpha GMS still validates extra client container names",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["worker"].GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled:               false,
					ExtraClientContainers: []string{"Bad_Name"},
				}
			},
			wantErr: `spec.services[worker].gpuMemoryService.extraClientContainers[0] "Bad_Name" is not a valid Kubernetes container name`,
		},
		{
			name: "nil alpha service entry is rejected",
			mutate: func(dgd *nvidiacomv1alpha1.DynamoGraphDeployment) {
				dgd.Spec.Services["ghost"] = nil
			},
			wantErr: "spec.services[ghost] must not be null",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			deployment := betaDGDFromAlpha(t, tt.mutate)
			validator := NewDynamoGraphDeploymentValidator(nil, true)
			_, err := validator.Validate(context.Background(), deployment)
			assertBetaValidationError(t, err, tt.wantErr)
		})
	}
}

func TestDynamoGraphDeploymentValidator_ValidateUpdate(t *testing.T) {
	const operatorPrincipal = "system:serviceaccount:dynamo-system:dynamo-operator"

	tests := []struct {
		name      string
		oldDGD    *nvidiacomv1beta1.DynamoGraphDeployment
		newDGD    *nvidiacomv1beta1.DynamoGraphDeployment
		userInfo  *authenticationv1.UserInfo
		principal string
		wantErr   string
		wantWarns bool
	}{
		{
			name:   "component topology is immutable",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Components = append(spec.Components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
					ComponentName: "extra",
					Replicas:      k8sptr.To(int32(1)),
				})
			}),
			wantErr: "component topology is immutable and cannot be modified after creation: components added: [extra]",
		},
		{
			name: "scaling adapter blocks direct replica changes",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(2))
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(3))
			}),
			userInfo: &authenticationv1.UserInfo{
				Username: "system:serviceaccount:default:regular-user",
			},
			principal: operatorPrincipal,
			wantErr:   "spec.components[worker].replicas cannot be modified directly when scaling adapter is enabled",
		},
		{
			name: "operator can change scaling-adapter-owned replicas",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(2))
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.ScalingAdapter = &nvidiacomv1beta1.ScalingAdapter{}
				worker.Replicas = k8sptr.To(int32(3))
			}),
			userInfo: &authenticationv1.UserInfo{
				Username: operatorPrincipal,
			},
			principal: operatorPrincipal,
		},
		{
			name: "minAvailable is immutable once set",
			oldDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.MinAvailable = k8sptr.To(int32(1))
			}),
			newDGD: betaDGDWithWorker(func(worker *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
				worker.MinAvailable = k8sptr.To(int32(2))
			}),
			wantErr: "spec.components[worker].minAvailable is immutable after creation",
		},
		{
			name:   "backend framework changes warn and fail",
			oldDGD: newBetaDGDForValidation(),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.BackendFramework = "sglang"
			}),
			wantErr:   "spec.backendFramework is immutable and cannot be changed after creation",
			wantWarns: true,
		},
		{
			name: "restart id cannot change during active rolling update",
			oldDGD: betaDGDWithStatus(
				func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
					spec.Restart = &nvidiacomv1beta1.Restart{ID: "old"}
				},
				func(status *nvidiacomv1beta1.DynamoGraphDeploymentStatus) {
					status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
						Phase: nvidiacomv1beta1.RollingUpdatePhaseInProgress,
					}
				},
			),
			newDGD: betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
				spec.Restart = &nvidiacomv1beta1.Restart{ID: "new"}
			}),
			wantErr: "spec.restart.id cannot be changed while a rolling update is InProgress",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentValidator(nil, true)
			warnings, err := validator.ValidateUpdate(tt.oldDGD, tt.newDGD, tt.userInfo, tt.principal)
			assertBetaValidationError(t, err, tt.wantErr)
			if tt.wantWarns && len(warnings) == 0 {
				t.Fatal("ValidateUpdate() expected warnings but got none")
			}
			if !tt.wantWarns && len(warnings) != 0 {
				t.Fatalf("ValidateUpdate() unexpected warnings: %v", warnings)
			}
		})
	}
}

func newBetaDGDForValidation() *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-graph",
			Namespace: "default",
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "frontend",
					ComponentType: nvidiacomv1beta1.ComponentTypeFrontend,
					Replicas:      k8sptr.To(int32(1)),
				},
				{
					ComponentName: "worker",
					ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
					Replicas:      k8sptr.To(int32(2)),
				},
			},
		},
	}
}

func betaDGDFromAlpha(
	t *testing.T,
	mutate func(*nvidiacomv1alpha1.DynamoGraphDeployment),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	t.Helper()

	alpha := newAlphaDGDForCompatibilityValidation()
	mutate(alpha)

	beta := &nvidiacomv1beta1.DynamoGraphDeployment{}
	if err := alpha.ConvertTo(beta); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}
	return beta
}

func newAlphaDGDForCompatibilityValidation() *nvidiacomv1alpha1.DynamoGraphDeployment {
	return &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-graph",
			Namespace: "default",
		},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      k8sptr.To(int32(1)),
				},
			},
		},
	}
}

func betaDGDWithSpec(
	mutate func(*nvidiacomv1beta1.DynamoGraphDeploymentSpec),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	dgd := newBetaDGDForValidation()
	mutate(&dgd.Spec)
	return dgd
}

func betaDGDWithWorker(
	mutate func(*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	return betaDGDWithSpec(func(spec *nvidiacomv1beta1.DynamoGraphDeploymentSpec) {
		for i := range spec.Components {
			if spec.Components[i].ComponentName == "worker" {
				mutate(&spec.Components[i])
				return
			}
		}
	})
}

func betaDGDWithStatus(
	mutateSpec func(*nvidiacomv1beta1.DynamoGraphDeploymentSpec),
	mutateStatus func(*nvidiacomv1beta1.DynamoGraphDeploymentStatus),
) *nvidiacomv1beta1.DynamoGraphDeployment {
	dgd := betaDGDWithSpec(mutateSpec)
	mutateStatus(&dgd.Status)
	return dgd
}

func enableBetaInterPodGMS(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
	component.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
		GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
			Mode: nvidiacomv1beta1.GMSModeInterPod,
		},
	}
	component.PodTemplate = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name: consts.MainContainerName,
					Resources: corev1.ResourceRequirements{
						Limits: corev1.ResourceList{
							corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
						},
					},
				},
			},
		},
	}
}

func enableBetaIntraPodGMS(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) {
	component.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
		GPUMemoryService: &nvidiacomv1beta1.GPUMemoryServiceSpec{
			Mode: nvidiacomv1beta1.GMSModeIntraPod,
		},
	}
	component.PodTemplate = &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name: consts.MainContainerName,
					Resources: corev1.ResourceRequirements{
						Limits: corev1.ResourceList{
							corev1.ResourceName(consts.KubeResourceGPUNvidia): resource.MustParse("1"),
						},
					},
				},
			},
		},
	}
}

func assertBetaValidationError(t *testing.T, err error, wantErr string) {
	t.Helper()
	if wantErr == "" {
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("expected error containing %q but got nil", wantErr)
	}
	if !strings.Contains(err.Error(), wantErr) {
		t.Fatalf("error = %q, want to contain %q", err.Error(), wantErr)
	}
}
