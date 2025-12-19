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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestSharedSpecValidator_Validate(t *testing.T) {
	var (
		negativeReplicas = int32(-1)
		validReplicas    = int32(3)
	)

	tests := []struct {
		name      string
		spec      *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		fieldPath string
		wantErr   bool
		errMsg    string
	}{
		{
			name: "valid spec with all fields",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas: &validReplicas,
				Ingress: &nvidiacomv1alpha1.IngressSpec{
					Enabled: true,
					Host:    "example.com",
				},
				VolumeMounts: []nvidiacomv1alpha1.VolumeMount{
					{
						Name:       "cache",
						MountPoint: "/cache",
					},
					{
						Name:                  "compilation",
						UseAsCompilationCache: true,
					},
				},
				SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{
					Disabled: false,
					Size:     resource.MustParse("1Gi"),
				},
			},
			fieldPath: "spec",
			wantErr:   false,
		},
		{
			name: "negative replicas",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas: &negativeReplicas,
			},
			fieldPath: "spec",
			wantErr:   true,
			errMsg:    "spec.replicas must be non-negative",
		},
		{
			name: "ingress enabled without host",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Ingress: &nvidiacomv1alpha1.IngressSpec{
					Enabled: true,
					Host:    "",
				},
			},
			fieldPath: "spec",
			wantErr:   true,
			errMsg:    "spec.ingress.host is required when ingress is enabled",
		},
		{
			name: "ingress disabled - no validation",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Ingress: &nvidiacomv1alpha1.IngressSpec{
					Enabled: false,
					Host:    "",
				},
			},
			fieldPath: "spec",
			wantErr:   false,
		},
		{
			name: "volume mount without mountPoint and not compilation cache",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []nvidiacomv1alpha1.VolumeMount{
					{
						Name:                  "data",
						MountPoint:            "",
						UseAsCompilationCache: false,
					},
				},
			},
			fieldPath: "spec",
			wantErr:   true,
			errMsg:    "spec.volumeMounts[0].mountPoint is required when useAsCompilationCache is false",
		},
		{
			name: "volume mount with mountPoint",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []nvidiacomv1alpha1.VolumeMount{
					{
						Name:       "data",
						MountPoint: "/data",
					},
				},
			},
			fieldPath: "spec",
			wantErr:   false,
		},
		{
			name: "volume mount as compilation cache without mountPoint",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []nvidiacomv1alpha1.VolumeMount{
					{
						Name:                  "cache",
						UseAsCompilationCache: true,
					},
				},
			},
			fieldPath: "spec",
			wantErr:   false,
		},
		{
			name: "shared memory enabled without size",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{
					Disabled: false,
					Size:     resource.Quantity{},
				},
			},
			fieldPath: "spec",
			wantErr:   true,
			errMsg:    "spec.sharedMemory.size is required when disabled is false",
		},
		{
			name: "shared memory enabled with size",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{
					Disabled: false,
					Size:     resource.MustParse("2Gi"),
				},
			},
			fieldPath: "spec",
			wantErr:   false,
		},
		{
			name: "shared memory disabled without size",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{
					Disabled: true,
					Size:     resource.Quantity{},
				},
			},
			fieldPath: "spec",
			wantErr:   false,
		},
		{
			name: "custom field path for service validation",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas: &negativeReplicas,
			},
			fieldPath: "spec.services[main]",
			wantErr:   true,
			errMsg:    "spec.services[main].replicas must be non-negative",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewSharedSpecValidator(tt.spec, tt.fieldPath)
			_, err := validator.Validate()

			if (err != nil) != tt.wantErr {
				t.Errorf("SharedSpecValidator.Validate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr && err.Error() != tt.errMsg {
				t.Errorf("SharedSpecValidator.Validate() error message = %v, want %v", err.Error(), tt.errMsg)
			}
		})
	}
}

func TestSharedSpecValidator_Validate_Warnings(t *testing.T) {
	validReplicas := int32(3)

	tests := []struct {
		name         string
		spec         *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		fieldPath    string
		wantWarnings int
	}{
		{
			name: "no warnings for spec without autoscaling",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas: &validReplicas,
			},
			fieldPath:    "spec",
			wantWarnings: 0,
		},
		{
			name: "warning for deprecated autoscaling field enabled",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas: &validReplicas,
				//nolint:staticcheck // SA1019: Intentionally testing deprecated field
				Autoscaling: &nvidiacomv1alpha1.Autoscaling{
					Enabled:     true,
					MinReplicas: 1,
					MaxReplicas: 10,
				},
			},
			fieldPath:    "spec",
			wantWarnings: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewSharedSpecValidator(tt.spec, tt.fieldPath)
			warnings, err := validator.Validate()

			if err != nil {
				t.Errorf("SharedSpecValidator.Validate() unexpected error = %v", err)
				return
			}

			if len(warnings) != tt.wantWarnings {
				t.Errorf("SharedSpecValidator.Validate() warnings count = %d, want %d", len(warnings), tt.wantWarnings)
			}
		})
	}
}
