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

package validation

import (
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

const dcdAdmissionSGLangBackend = "sglang"

func TestDynamoComponentDeploymentValidator_Validate(t *testing.T) {
	requestValidators := requestValidatorsFromCRD(t, "nvidia.com_dynamocomponentdeployments.yaml")

	var (
		oneReplica       = int32(1)
		validReplicas    = int32(3)
		negativeReplicas = int32(-1)
		validMinAvail    = int32(2)
	)

	tests := []struct {
		name           string
		deployment     runtime.Object
		oldDeployment  runtime.Object
		wantSchemaErr  string
		wantCELErr     string
		wantWebhookErr string
		wantWarning    string
	}{
		{
			name: "valid deployment",
			deployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						Replicas: &validReplicas,
					},
					BackendFramework: dcdAdmissionSGLangBackend,
				},
			},
		},
		{
			name: "v1alpha1 checkpoint with active-passive failover is rejected",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.Resources = &nvidiacomv1alpha1.Resources{
					Limits: &nvidiacomv1alpha1.ResourceItem{GPU: "1"},
				}
				dcd.Spec.GPUMemoryService = &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
				dcd.Spec.Failover = &nvidiacomv1alpha1.FailoverSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				}
				dcd.Spec.Checkpoint = &nvidiacomv1alpha1.ServiceCheckpointConfig{Enabled: true}
			}),
			wantWebhookErr: "spec.checkpoint cannot be enabled with failover.enabled=true: checkpoint/snapshot is not supported with active/passive failover",
		},
		{
			name: "invalid replicas",
			deployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						Replicas: &negativeReplicas,
					},
				},
			},
			wantSchemaErr: "spec.replicas: Invalid value: -1: spec.replicas in body should be greater than or equal to 0",
		},
		{
			name: "minAvailable is unsupported for standalone DCD",
			deployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						Replicas:     &validReplicas,
						MinAvailable: &validMinAvail,
					},
				},
			},
			wantWebhookErr: "spec.minAvailable is currently supported only for Grove-backed DynamoGraphDeployment components",
		},
		{
			name: "invalid ingress",
			deployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						Ingress: &nvidiacomv1alpha1.IngressSpec{
							Enabled: true,
							Host:    "",
						},
					},
				},
			},
			wantWebhookErr: "spec.ingress.host is required when ingress is enabled",
		},
		{
			name: "invalid volume mount",
			deployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						VolumeMounts: []nvidiacomv1alpha1.VolumeMount{
							{
								Name:                  "data",
								UseAsCompilationCache: false,
							},
						},
					},
				},
			},
			wantWebhookErr: "spec.volumeMounts[0].mountPoint is required when useAsCompilationCache is false",
		},
		{
			name: "invalid shared memory",
			deployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{
							Disabled: false,
							Size:     resource.Quantity{},
						},
					},
				},
			},
			wantCELErr: "spec.sharedMemory: Invalid value: size is required when disabled is false",
		},

		// Pair shared pod-template validation across both served source versions.
		{
			name: "v1beta1 sidecar without image is rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers: []corev1.Container{{Name: consts.MainContainerName}, {Name: "metrics"}},
				}}
			}),
			wantCELErr: "spec.podTemplate.spec.containers[1]: Invalid value: sidecar containers must specify a non-empty image",
		},
		{
			name: "v1alpha1 sidecar without image reaches the webhook",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
					Containers: []corev1.Container{{Name: "metrics"}},
				}}
			}),
		},
		{
			name: "v1beta1 init container without image is rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{Spec: corev1.PodSpec{
					Containers:     []corev1.Container{{Name: consts.MainContainerName}},
					InitContainers: []corev1.Container{{Name: "prepare"}},
				}}
			}),
			wantCELErr: "spec.podTemplate.spec.initContainers[0]: Invalid value: init containers must specify a non-empty image",
		},
		{
			name: "v1alpha1 init container without image reaches the webhook",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.ExtraPodSpec = &nvidiacomv1alpha1.ExtraPodSpec{PodSpec: &corev1.PodSpec{
					InitContainers: []corev1.Container{{Name: "prepare"}},
				}}
			}),
		},
		{
			name: "v1beta1 invalid pod template backend annotation is rejected by CEL",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
					}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName}}},
				}
			}),
			wantCELErr: "spec.podTemplate.metadata.annotations: Invalid value: podTemplate backend annotation must be mp or ray, case-insensitively",
		},
		{
			name: "v1beta1 valid pod template backend annotation reaches the webhook",
			deployment: betaDCDForAdmission(func(dcd *nvidiacomv1beta1.DynamoComponentDeployment) {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "RaY",
					}},
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName}}},
				}
			}),
		},
		{
			name: "v1alpha1 invalid extra pod metadata annotation reaches the webhook",
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.ExtraPodMetadata = &nvidiacomv1alpha1.ExtraPodMetadata{
					Annotations: map[string]string{consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid"},
				}
			}),
		},
		{
			name:       "valid v1beta1 deployment reaches the converted webhook",
			deployment: betaDCDForAdmission(nil),
		},
		{
			name: "v1alpha1 update without changes reaches the webhook",
			oldDeployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
			}),
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
			}),
		},
		{
			name: "v1alpha1 backend framework update reaches and is rejected by the webhook",
			oldDeployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
			}),
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = "vllm"
			}),
			wantWebhookErr: "spec.backendFramework is immutable and cannot be changed after creation",
			wantWarning:    "Changing spec.backendFramework may cause unexpected behavior",
		},
		{
			name: "v1alpha1 replicas update reaches the webhook",
			oldDeployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
				dcd.Spec.Replicas = &oneReplica
			}),
			deployment: alphaDCDForAdmission(func(dcd *nvidiacomv1alpha1.DynamoComponentDeployment) {
				dcd.Spec.BackendFramework = dcdAdmissionSGLangBackend
				dcd.Spec.Replicas = &validReplicas
			}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := admissionUnstructured(t, tt.deployment)
			var oldRequest map[string]any
			if tt.oldDeployment != nil {
				oldRequest = admissionUnstructured(t, tt.oldDeployment)
			}
			version := admissionSourceVersion(t, tt.deployment)
			requestValidator, ok := requestValidators[version]
			if !ok {
				t.Fatalf("no request validator for source version %q", version)
			}

			schemaErrs := requestValidator.validateSchema(request, oldRequest)
			if tt.wantSchemaErr != "" {
				assertRequestValidationError(t, schemaErrs, tt.wantSchemaErr)
				return
			}
			if len(schemaErrs) != 0 {
				t.Fatalf("schema errors = %v, want none", schemaErrs)
			}

			celErrs := requestValidator.celValidator(request, oldRequest)
			if tt.wantCELErr != "" {
				assertRequestValidationError(t, celErrs, tt.wantCELErr)
				return
			}
			if len(celErrs) != 0 {
				t.Fatalf("CEL errors = %v, want none", celErrs)
			}

			handler := NewDynamoComponentDeploymentHandler()
			ctx := dgdAdmissionContext(dgdAdmissionOperation(tt.oldDeployment), nvidiacomv1alpha1.DynamoComponentDeploymentGVK)
			var warnings []string
			var err error
			if tt.oldDeployment == nil {
				warnings, err = handler.ValidateCreate(ctx, dcdAdmissionAlpha(t, tt.deployment))
			} else {
				warnings, err = handler.ValidateUpdate(
					ctx,
					dcdAdmissionAlpha(t, tt.oldDeployment),
					dcdAdmissionAlpha(t, tt.deployment),
				)
			}
			assertWebhookError(t, err, tt.wantWebhookErr)
			if tt.wantWarning == "" {
				if len(warnings) != 0 {
					t.Fatalf("webhook warnings = %v, want none", warnings)
				}
			} else if len(warnings) != 1 || warnings[0] != tt.wantWarning {
				t.Fatalf("webhook warnings = %v, want exactly %q", warnings, tt.wantWarning)
			}
		})
	}
}

func alphaDCDForAdmission(
	mutate func(*nvidiacomv1alpha1.DynamoComponentDeployment),
) *nvidiacomv1alpha1.DynamoComponentDeployment {
	dcd := &nvidiacomv1alpha1.DynamoComponentDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
			Kind:       "DynamoComponentDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ServiceName:   "worker",
				ComponentType: consts.ComponentTypeWorker,
			},
		},
	}
	if mutate != nil {
		mutate(dcd)
	}
	return dcd
}

func betaDCDForAdmission(
	mutate func(*nvidiacomv1beta1.DynamoComponentDeployment),
) *nvidiacomv1beta1.DynamoComponentDeployment {
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
		TypeMeta: metav1.TypeMeta{
			APIVersion: nvidiacomv1beta1.GroupVersion.String(),
			Kind:       "DynamoComponentDeployment",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: "default"},
		Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: "vllm",
			DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "worker",
				ComponentType: nvidiacomv1beta1.ComponentTypeWorker,
			},
		},
	}
	if mutate != nil {
		mutate(dcd)
	}
	return dcd
}

func dcdAdmissionAlpha(t *testing.T, deployment runtime.Object) *nvidiacomv1alpha1.DynamoComponentDeployment {
	t.Helper()
	switch deployment := deployment.(type) {
	case *nvidiacomv1alpha1.DynamoComponentDeployment:
		return deployment.DeepCopy()
	case *nvidiacomv1beta1.DynamoComponentDeployment:
		alpha := &nvidiacomv1alpha1.DynamoComponentDeployment{}
		if err := alpha.ConvertFrom(deployment); err != nil {
			t.Fatalf("convert v1beta1 DCD to v1alpha1: %v", err)
		}
		return alpha
	default:
		t.Fatalf("unsupported DCD type %T", deployment)
		return nil
	}
}
