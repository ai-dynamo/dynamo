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
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDynamoGraphDeploymentValidator_Validate(t *testing.T) {
	var (
		validReplicas    = int32(3)
		negativeReplicas = int32(-1)
		pvcName          = "test-pvc"
		trueVal          = true
		falseVal         = false
	)

	tests := []struct {
		name        string
		deployment  *nvidiacomv1alpha1.DynamoGraphDeployment
		wantErr     bool
		errMsg      string
		errContains bool
	}{
		{
			name: "valid deployment with services",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Replicas: &validReplicas,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "no services",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{},
				},
			},
			wantErr: true,
			errMsg:  "spec.services must have at least one service",
		},
		{
			name: "service with invalid replicas",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Replicas: &negativeReplicas,
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.services[main].replicas must be non-negative",
		},
		{
			name: "service with invalid autoscaling",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"prefill": {
							Autoscaling: &nvidiacomv1alpha1.Autoscaling{
								Enabled:     true,
								MinReplicas: 10,
								MaxReplicas: 5,
							},
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.services[prefill].autoscaling.maxReplicas must be > minReplicas",
		},
		{
			name: "service with invalid ingress",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"gateway": {
							Ingress: &nvidiacomv1alpha1.IngressSpec{
								Enabled: true,
								Host:    "",
							},
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.services[gateway].ingress.host is required when ingress is enabled",
		},
		{
			name: "pvc with create=true and missing storageClass",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					PVCs: []nvidiacomv1alpha1.PVC{
						{
							Create:           &trueVal,
							Name:             &pvcName,
							StorageClass:     "",
							Size:             resource.MustParse("10Gi"),
							VolumeAccessMode: corev1.ReadWriteOnce,
						},
					},
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.pvcs[0].storageClass is required when create is true",
		},
		{
			name: "pvc with create=true and missing size",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					PVCs: []nvidiacomv1alpha1.PVC{
						{
							Create:           &trueVal,
							Name:             &pvcName,
							StorageClass:     "standard",
							Size:             resource.Quantity{},
							VolumeAccessMode: corev1.ReadWriteOnce,
						},
					},
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.pvcs[0].size is required when create is true",
		},
		{
			name: "pvc with create=true and missing volumeAccessMode",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					PVCs: []nvidiacomv1alpha1.PVC{
						{
							Create:           &trueVal,
							Name:             &pvcName,
							StorageClass:     "standard",
							Size:             resource.MustParse("10Gi"),
							VolumeAccessMode: "",
						},
					},
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.pvcs[0].volumeAccessMode is required when create is true",
		},
		{
			name: "pvc with create=false and missing fields",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					PVCs: []nvidiacomv1alpha1.PVC{
						{
							Create: &falseVal,
							Name:   &pvcName,
						},
					},
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "pvc with missing name",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					PVCs: []nvidiacomv1alpha1.PVC{
						{
							Create: &falseVal,
						},
					},
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.pvcs[0].name is required",
		},
		{
			name: "pvc with multiple errors (name, storageClass, size, volumeAccessMode all missing)",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					PVCs: []nvidiacomv1alpha1.PVC{
						{
							Create: &trueVal,
						},
					},
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr:     true,
			errMsg:      "spec.pvcs[0].name is required\nspec.pvcs[0].storageClass is required when create is true\nspec.pvcs[0].size is required when create is true\nspec.pvcs[0].volumeAccessMode is required when create is true",
			errContains: true,
		},
		{
			name: "valid pvc with create=true",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					PVCs: []nvidiacomv1alpha1.PVC{
						{
							Create:           &trueVal,
							Name:             &pvcName,
							StorageClass:     "standard",
							Size:             resource.MustParse("10Gi"),
							VolumeAccessMode: corev1.ReadWriteOnce,
						},
					},
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "service with invalid volume mount",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							VolumeMounts: []nvidiacomv1alpha1.VolumeMount{
								{
									Name:                  "data",
									UseAsCompilationCache: false,
								},
							},
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.services[main].volumeMounts[0].mountPoint is required when useAsCompilationCache is false",
		},
		{
			name: "service with invalid shared memory",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{
								Disabled: false,
								Size:     resource.Quantity{},
							},
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.services[main].sharedMemory.size is required when disabled is false",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentValidator(tt.deployment)
			_, err := validator.Validate()

			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoGraphDeploymentValidator.Validate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				if tt.errContains {
					// For multiple errors, check that all expected error messages are present
					errStr := err.Error()
					for _, expectedMsg := range strings.Split(tt.errMsg, "\n") {
						if !strings.Contains(errStr, expectedMsg) {
							t.Errorf("DynamoGraphDeploymentValidator.Validate() error message = %v, want to contain %v", errStr, expectedMsg)
						}
					}
				} else {
					if err.Error() != tt.errMsg {
						t.Errorf("DynamoGraphDeploymentValidator.Validate() error message = %v, want %v", err.Error(), tt.errMsg)
					}
				}
			}
		})
	}
}

func TestDynamoGraphDeploymentValidator_ValidateUpdate(t *testing.T) {
	tests := []struct {
		name            string
		oldDeployment   *nvidiacomv1alpha1.DynamoGraphDeployment
		newDeployment   *nvidiacomv1alpha1.DynamoGraphDeployment
		wantErr         bool
		wantWarnings    bool
		errMsg          string
		expectedWarnMsg string
	}{
		{
			name: "no changes",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
				},
			},
			wantErr: false,
		},
		{
			name: "changing backend framework",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
				},
			},
			wantErr:         true,
			wantWarnings:    true,
			errMsg:          "spec.backendFramework is immutable and cannot be changed after creation",
			expectedWarnMsg: "Changing spec.backendFramework may cause unexpected behavior",
		},
		{
			name: "adding new service is allowed",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main":    {},
						"prefill": {},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentValidator(tt.newDeployment)
			warnings, err := validator.ValidateUpdate(tt.oldDeployment)

			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoGraphDeploymentValidator.ValidateUpdate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && err.Error() != tt.errMsg {
				t.Errorf("DynamoGraphDeploymentValidator.ValidateUpdate() error message = %v, want %v", err.Error(), tt.errMsg)
			}

			if tt.wantWarnings && len(warnings) == 0 {
				t.Errorf("DynamoGraphDeploymentValidator.ValidateUpdate() expected warnings but got none")
			}

			if tt.wantWarnings && len(warnings) > 0 && warnings[0] != tt.expectedWarnMsg {
				t.Errorf("DynamoGraphDeploymentValidator.ValidateUpdate() warning = %v, want %v", warnings[0], tt.expectedWarnMsg)
			}
		})
	}
}
