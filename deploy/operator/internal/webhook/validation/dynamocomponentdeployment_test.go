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
	"context"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestDynamoComponentDeploymentValidator_Validate(t *testing.T) {
	var (
		validReplicas    = int32(3)
		negativeReplicas = int32(-1)
		validMinAvail    = int32(2)
	)

	tests := []struct {
		name       string
		deployment *nvidiacomv1alpha1.DynamoComponentDeployment
		wantErr    bool
		errMsg     string
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
					BackendFramework: "sglang",
				},
			},
			wantErr: false,
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
			wantErr: true,
			errMsg:  "spec.replicas must be non-negative",
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
			wantErr: true,
			errMsg:  "spec.minAvailable is currently supported only for Grove-backed DynamoGraphDeployment components",
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
			wantErr: true,
			errMsg:  "spec.ingress.host is required when ingress is enabled",
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
			wantErr: true,
			errMsg:  "spec.volumeMounts[0].mountPoint is required when useAsCompilationCache is false",
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
			wantErr: true,
			errMsg:  "spec.sharedMemory.size is required when disabled is false",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoComponentDeploymentValidator(tt.deployment)
			_, err := validator.Validate(context.Background())

			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoComponentDeploymentValidator.Validate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && err.Error() != tt.errMsg {
				t.Errorf("DynamoComponentDeploymentValidator.Validate() error message = %v, want %v", err.Error(), tt.errMsg)
			}
		})
	}
}

func TestDynamoComponentDeploymentValidator_Admission(t *testing.T) {
	requestValidators := requestValidatorsFromCRD(t, "nvidia.com_dynamocomponentdeployments.yaml")

	tests := []struct {
		name       string
		deployment runtime.Object
		wantCELErr string
	}{
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
			wantCELErr: "spec.podTemplate: Invalid value: podTemplate backend annotation must be mp or ray, case-insensitively",
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := admissionUnstructured(t, tt.deployment)
			version := tt.deployment.GetObjectKind().GroupVersionKind().Version
			requestValidator, ok := requestValidators[version]
			if !ok {
				t.Fatalf("no request validator for source version %q", version)
			}
			if schemaErrs := requestValidator.validateSchema(request, nil); len(schemaErrs) != 0 {
				t.Fatalf("schema errors = %v, want none", schemaErrs)
			}

			celErrs := requestValidator.celValidator(request, nil)
			if tt.wantCELErr != "" {
				assertRequestValidationError(t, celErrs, tt.wantCELErr)
				return
			}
			if len(celErrs) != 0 {
				t.Fatalf("CEL errors = %v, want none", celErrs)
			}

			handler := NewDynamoComponentDeploymentHandler()
			_, err := handler.ValidateCreate(
				dgdAdmissionContext(admissionv1.Create, nvidiacomv1alpha1.DynamoComponentDeploymentGVK),
				dcdAdmissionAlpha(t, tt.deployment),
			)
			if err != nil {
				t.Fatalf("webhook error = %v, want none", err)
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

func TestDynamoComponentDeploymentValidator_ValidateUpdate(t *testing.T) {
	tests := []struct {
		name            string
		oldDeployment   *nvidiacomv1alpha1.DynamoComponentDeployment
		newDeployment   *nvidiacomv1alpha1.DynamoComponentDeployment
		wantErr         bool
		wantWarnings    bool
		errMsg          string
		expectedWarnMsg string
	}{
		{
			name: "no changes",
			oldDeployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					BackendFramework: "sglang",
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					BackendFramework: "sglang",
				},
			},
			wantErr: false,
		},
		{
			name: "changing backend framework",
			oldDeployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					BackendFramework: "sglang",
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					BackendFramework: "vllm",
				},
			},
			wantErr:         true,
			wantWarnings:    true,
			errMsg:          "spec.backendFramework is immutable and cannot be changed after creation",
			expectedWarnMsg: "Changing spec.backendFramework may cause unexpected behavior",
		},
		{
			name: "changing replicas is allowed",
			oldDeployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						Replicas: func() *int32 { r := int32(1); return &r }(),
					},
					BackendFramework: "sglang",
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						Replicas: func() *int32 { r := int32(3); return &r }(),
					},
					BackendFramework: "sglang",
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoComponentDeploymentValidator(tt.newDeployment)
			warnings, err := validator.ValidateUpdate(tt.oldDeployment)

			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoComponentDeploymentValidator.ValidateUpdate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && err.Error() != tt.errMsg {
				t.Errorf("DynamoComponentDeploymentValidator.ValidateUpdate() error message = %v, want %v", err.Error(), tt.errMsg)
			}

			if tt.wantWarnings && len(warnings) == 0 {
				t.Errorf("DynamoComponentDeploymentValidator.ValidateUpdate() expected warnings but got none")
			}

			if tt.wantWarnings && len(warnings) > 0 && warnings[0] != tt.expectedWarnMsg {
				t.Errorf("DynamoComponentDeploymentValidator.ValidateUpdate() warning = %v, want %v", warnings[0], tt.expectedWarnMsg)
			}
		})
	}
}
