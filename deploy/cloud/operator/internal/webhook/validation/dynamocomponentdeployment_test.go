/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDynamoComponentDeploymentValidator_Validate(t *testing.T) {
	var (
		validReplicas    = int32(3)
		negativeReplicas = int32(-1)
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
						Autoscaling: &nvidiacomv1alpha1.Autoscaling{
							Enabled:     true,
							MinReplicas: 1,
							MaxReplicas: 10,
						},
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
			name: "invalid autoscaling",
			deployment: &nvidiacomv1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						Autoscaling: &nvidiacomv1alpha1.Autoscaling{
							Enabled:     true,
							MinReplicas: 5,
							MaxReplicas: 3,
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.autoscaling.maxReplicas must be > minReplicas",
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
			_, err := validator.Validate()

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
