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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

// ptr is a helper function to create a pointer to a string
func ptr(s string) *string {
	return &s
}

func TestSharedSpecValidator_Validate(t *testing.T) {
	var (
		negativeReplicas = int32(-1)
		validReplicas    = int32(3)
		workerGPU        = &nvidiacomv1alpha1.Resources{
			Limits: &nvidiacomv1alpha1.ResourceItem{GPU: "1"},
		}
	)

	tests := []struct {
		name                string
		spec                *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		fieldPath           string
		calculatedNamespace string
		wantErr             bool
		errMsg              string
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
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "negative replicas",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas: &negativeReplicas,
			},
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             true,
			errMsg:              "spec.replicas must be non-negative",
		},
		{
			name: "nil dynamoNamespace is allowed",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				DynamoNamespace: nil,
			},
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "empty string dynamoNamespace is allowed",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				DynamoNamespace: ptr(""),
			},
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "ingress enabled without host",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Ingress: &nvidiacomv1alpha1.IngressSpec{
					Enabled: true,
					Host:    "",
				},
			},
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             true,
			errMsg:              "spec.ingress.host is required when ingress is enabled",
		},
		{
			name: "ingress disabled - no validation",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Ingress: &nvidiacomv1alpha1.IngressSpec{
					Enabled: false,
					Host:    "",
				},
			},
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
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
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             true,
			errMsg:              "spec.volumeMounts[0].mountPoint is required when useAsCompilationCache is false",
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
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
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
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "shared memory enabled without size",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{
					Disabled: false,
					Size:     resource.Quantity{},
				},
			},
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             true,
			errMsg:              "spec.sharedMemory.size is required when disabled is false",
		},
		{
			name: "shared memory enabled with size",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{
					Disabled: false,
					Size:     resource.MustParse("2Gi"),
				},
			},
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "shared memory disabled without size",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				SharedMemory: &nvidiacomv1alpha1.SharedMemorySpec{
					Disabled: true,
					Size:     resource.Quantity{},
				},
			},
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "custom field path for service validation",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas: &negativeReplicas,
			},
			fieldPath:           "spec.services[main]",
			calculatedNamespace: "default-my-dgd",
			wantErr:             true,
			errMsg:              "spec.services[main].replicas must be non-negative",
		},
		{
			name: "valid service annotation vllm-distributed-executor-backend=ray",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Annotations: map[string]string{
					"nvidia.com/vllm-distributed-executor-backend": "ray",
				},
			},
			fieldPath:           "spec.services[decode]",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "valid service annotation vllm-distributed-executor-backend=mp",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Annotations: map[string]string{
					"nvidia.com/vllm-distributed-executor-backend": "mp",
				},
			},
			fieldPath:           "spec.services[decode]",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "invalid service annotation vllm-distributed-executor-backend",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Annotations: map[string]string{
					"nvidia.com/vllm-distributed-executor-backend": "invalid",
				},
			},
			fieldPath:           "spec.services[decode]",
			calculatedNamespace: "default-my-dgd",
			wantErr:             true,
			errMsg:              `spec.services[decode].annotations[nvidia.com/vllm-distributed-executor-backend] has invalid value "invalid": must be "mp" or "ray"`,
		},
		{
			name: "checkpoint with gpuMemoryService is temporarily rejected",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Enabled: true,
					Identity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
						Model:            "model",
						BackendFramework: "vllm",
					},
				},
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
			},
			fieldPath:           "spec.services[worker]",
			calculatedNamespace: "default-my-dgd",
			wantErr:             true,
			errMsg:              "spec.services[worker].checkpoint: checkpointing with gpuMemoryService is temporarily disabled due to known GPU driver issues; disable either checkpointing or gpuMemoryService for this service",
		},
		{
			name: "checkpoint without gpuMemoryService is accepted",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Enabled: true,
					Identity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
						Model:            "model",
						BackendFramework: "vllm",
					},
				},
			},
			fieldPath:           "spec.services[worker]",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "disabled checkpoint with gpuMemoryService is accepted",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				Checkpoint: &nvidiacomv1alpha1.ServiceCheckpointConfig{
					Enabled: false,
				},
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
			},
			fieldPath:           "spec.services[worker]",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "frontendSidecar with no extraPodSpec containers is valid",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				FrontendSidecar: &nvidiacomv1alpha1.FrontendSidecarSpec{
					Image: "my-frontend:latest",
				},
			},
			fieldPath:           "spec.services[worker]",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
		{
			name: "frontendSidecar rejects duplicate container name in extraPodSpec",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				FrontendSidecar: &nvidiacomv1alpha1.FrontendSidecarSpec{
					Image: "my-frontend:latest",
				},
				ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
					PodSpec: &corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: consts.FrontendSidecarContainerName, Image: "conflict:latest"},
						},
					},
				},
			},
			fieldPath:           "spec.services[worker]",
			calculatedNamespace: "default-my-dgd",
			wantErr:             true,
			errMsg:              `spec.services[worker]: cannot inject frontend sidecar: a container named "sidecar-frontend" already exists in extraPodSpec.containers`,
		},
		{
			name: "frontendSidecar with non-conflicting extraPodSpec containers is valid",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				FrontendSidecar: &nvidiacomv1alpha1.FrontendSidecarSpec{
					Image: "my-frontend:latest",
				},
				ExtraPodSpec: &nvidiacomv1alpha1.ExtraPodSpec{
					PodSpec: &corev1.PodSpec{
						Containers: []corev1.Container{
							{Name: "other-sidecar", Image: "other:latest"},
						},
					},
				},
			},
			fieldPath:           "spec.services[worker]",
			calculatedNamespace: "default-my-dgd",
			wantErr:             false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewSharedSpecValidator(tt.spec, tt.fieldPath, tt.calculatedNamespace)
			_, err := validator.Validate(context.Background())

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
		name                string
		spec                *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		fieldPath           string
		calculatedNamespace string
		wantWarnings        int
		wantWarningContains string // optional substring to check in warning
	}{
		{
			name: "no warnings for spec without autoscaling",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas: &validReplicas,
			},
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantWarnings:        0,
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
			fieldPath:           "spec",
			calculatedNamespace: "default-my-dgd",
			wantWarnings:        1,
		},
		{
			name: "warning for deprecated dynamoNamespace field shows calculated namespace",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				Replicas:        &validReplicas,
				DynamoNamespace: ptr("my-custom-namespace"),
			},
			fieldPath:           "spec.services[Frontend]",
			calculatedNamespace: "hannahz-trtllm-disagg",
			wantWarnings:        1,
			wantWarningContains: "Value 'my-custom-namespace' will be replaced with 'hannahz-trtllm-disagg'",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewSharedSpecValidator(tt.spec, tt.fieldPath, tt.calculatedNamespace)
			warnings, err := validator.Validate(context.Background())

			if err != nil {
				t.Errorf("SharedSpecValidator.Validate() unexpected error = %v", err)
				return
			}

			if len(warnings) != tt.wantWarnings {
				t.Errorf("SharedSpecValidator.Validate() warnings count = %d, want %d", len(warnings), tt.wantWarnings)
			}

			if tt.wantWarningContains != "" && len(warnings) > 0 {
				found := false
				for _, w := range warnings {
					if contains(w, tt.wantWarningContains) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("SharedSpecValidator.Validate() warnings = %v, want warning containing %q", warnings, tt.wantWarningContains)
				}
			}
		})
	}
}

// TestSharedSpecValidator_Failover_ModeConstraints covers the layout/failover
// symmetry invariants enforced by validateFailover / validateGPUMemoryService:
//
//  1. gpuMemoryService declares the layout (intra-pod sidecar vs. inter-pod
//     weight-server pod). Both modes are valid on their own (standalone GMS
//     with no failover), and both may be paired with failover of a matching
//     mode.
//  2. failover.mode=intraPod requires gpuMemoryService.enabled=true and a
//     matching (or unset) gpuMemoryService.mode.
//  3. failover.mode=interPod requires gpuMemoryService.enabled=true AND
//     gpuMemoryService.mode=interPod — the symmetric counterpart of (2).
//  4. intraPod failover with numShadows != 1 is rejected (intraPod is a
//     fixed 1 primary + 1 shadow layout).
//  5. When failover.enabled=false, sub-fields (mode, numShadows) are dormant
//     configuration and are intentionally NOT validated — the render path
//     ignores them and users may stage a config before enabling failover.
func TestSharedSpecValidator_Failover_ModeConstraints(t *testing.T) {
	workerGPU := &nvidiacomv1alpha1.Resources{
		Limits: &nvidiacomv1alpha1.ResourceItem{GPU: "1"},
	}

	tests := []struct {
		name      string
		spec      *nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		wantErr   bool
		errSubstr string
	}{
		{
			name: "standalone inter-pod GMS (no failover) is accepted",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeInterPod,
				},
			},
			wantErr: false,
		},
		{
			name: "sidecar gpuMemoryService mode=intraPod is accepted",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
			},
			wantErr: false,
		},
		{
			name: "sidecar gpuMemoryService mode unset is accepted",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
				},
			},
			wantErr: false,
		},
		{
			name: "inter-pod failover requires gpuMemoryService.enabled",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeInterPod,
					NumShadows: 1,
				},
			},
			wantErr:   true,
			errSubstr: "gpuMemoryService.enabled=true",
		},
		{
			name: "inter-pod failover requires gpuMemoryService.mode=interPod",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeInterPod,
					NumShadows: 1,
				},
			},
			wantErr:   true,
			errSubstr: "requires gpuMemoryService.mode",
		},
		{
			name: "inter-pod failover with matching gpuMemoryService.mode=interPod is accepted",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeInterPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeInterPod,
					NumShadows: 1,
				},
			},
			wantErr: false,
		},
		{
			// numShadows is dormant configuration when failover.enabled=false
			// and GetNumShadows returns 0; validateFailover deliberately does
			// not constrain sub-fields on a disabled feature so users can
			// stage a config before flipping enabled=true.
			name: "numShadows with failover.enabled=false is accepted (dormant config)",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeInterPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    false,
					NumShadows: 2,
				},
			},
			wantErr: false,
		},
		{
			name: "intraPod failover with numShadows=2 is rejected",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeIntraPod,
					NumShadows: 2,
				},
			},
			wantErr:   true,
			errSubstr: "numShadows",
		},
		{
			name: "intraPod failover with numShadows=1 is accepted",
			spec: &nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: consts.ComponentTypeWorker,
				Resources:     workerGPU,
				GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
					Enabled: true,
					Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
				},
				Failover: &nvidiacomv1alpha1.FailoverSpec{
					Enabled:    true,
					Mode:       nvidiacomv1alpha1.GMSModeIntraPod,
					NumShadows: 1,
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := NewSharedSpecValidator(tt.spec, "spec", "default-my-dgd")
			_, err := v.Validate(context.Background())

			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				if tt.errSubstr != "" && !contains(err.Error(), tt.errSubstr) {
					t.Errorf("error %q does not contain %q", err.Error(), tt.errSubstr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// contains checks if s contains substr
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
