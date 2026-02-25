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
	"sort"
	"strings"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
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
		// Restart validation test cases
		{
			name: "restart with nil at",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "",
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.restart.id is required",
		},
		{
			name: "restart with valid id and no strategy",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "restart-id",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "restart with parallel strategy and order specified",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main":    {},
						"prefill": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "restart-id",
						Strategy: &nvidiacomv1alpha1.RestartStrategy{
							Type:  nvidiacomv1alpha1.RestartStrategyTypeParallel,
							Order: []string{"main", "prefill"},
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.restart.strategy.order cannot be specified when strategy is parallel",
		},
		{
			name: "restart with sequential strategy and duplicate services in order",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main":    {},
						"prefill": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "restart-id",
						Strategy: &nvidiacomv1alpha1.RestartStrategy{
							Type:  nvidiacomv1alpha1.RestartStrategyTypeSequential,
							Order: []string{"main", "main", "prefill"},
						},
					},
				},
			},
			wantErr:     true,
			errMsg:      "spec.restart.strategy.order must be unique",
			errContains: true,
		},
		{
			name: "restart with sequential strategy and unknown service in order",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main":    {},
						"prefill": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "restart-id",
						Strategy: &nvidiacomv1alpha1.RestartStrategy{
							Type:  nvidiacomv1alpha1.RestartStrategyTypeSequential,
							Order: []string{"main", "unknown"},
						},
					},
				},
			},
			wantErr:     true,
			errMsg:      "spec.restart.strategy.order contains unknown service: unknown",
			errContains: true,
		},
		{
			name: "restart with sequential strategy and missing service in order",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main":    {},
						"prefill": {},
						"decode":  {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "restart-id",
						Strategy: &nvidiacomv1alpha1.RestartStrategy{
							Type:  nvidiacomv1alpha1.RestartStrategyTypeSequential,
							Order: []string{"main", "prefill"},
						},
					},
				},
			},
			wantErr:     true,
			errMsg:      "spec.restart.strategy.order must have the same number of unique services as the deployment",
			errContains: true,
		},
		{
			name: "restart with valid sequential strategy and order",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main":    {},
						"prefill": {},
						"decode":  {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "restart-id",
						Strategy: &nvidiacomv1alpha1.RestartStrategy{
							Type:  nvidiacomv1alpha1.RestartStrategyTypeSequential,
							Order: []string{"prefill", "decode", "main"},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "restart with sequential strategy and empty order is valid",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "restart-id",
						Strategy: &nvidiacomv1alpha1.RestartStrategy{
							Type:  nvidiacomv1alpha1.RestartStrategyTypeSequential,
							Order: []string{},
						},
					},
				},
			},
			wantErr: false,
		},
		// Service name length validation tests
		{
			name: "service name too long for single-node deployment",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "verylongdynamographdeploymentname",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"VeryLongServiceNameThatExceedsLimit": {},
					},
				},
			},
			wantErr:     true,
			errContains: true,
			errMsg:      "combined resource name length",
		},
		{
			name: "service name too long for multinode deployment",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "vllm-agg",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"VllmPrefillWorker": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
						},
					},
				},
			},
			wantErr:     true,
			errContains: true,
			errMsg:      "combined resource name length",
		},
		{
			name: "valid service name length for single-node",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "dgd",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"Frontend": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "valid service name length for multinode",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "dgd",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"Worker": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "boundary case - exactly at 45 char limit for single-node",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					// DGD name (3 chars) + service name (42 chars) = 45 chars (exactly at limit)
					Name:      "dgd",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						// 42 character service name
						"abcdefghijklmnopqrstuvwxyz0123456789ABCDEF": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "boundary case - one char over limit for single-node",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					// DGD name (3 chars) + service name (43 chars) = 46 chars (over limit)
					Name:      "dgd",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						// 43 character service name
						"abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG": {},
					},
				},
			},
			wantErr:     true,
			errContains: true,
			errMsg:      "combined resource name length 46 exceeds 45-character limit",
		},
		// Grove disabled tests - service name length validation should be skipped
		{
			name: "long service name allowed when Grove disabled via annotation",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "verylongdynamographdeploymentname",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationEnableGrove: "false",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"VeryLongServiceNameThatExceedsLimit": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "long multinode service name allowed when Grove disabled via annotation",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "vllm-agg",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationEnableGrove: "false",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"VllmPrefillWorker": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Grove annotation case insensitive - FALSE",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "verylongdynamographdeploymentname",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationEnableGrove: "FALSE",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"VeryLongServiceNameThatExceedsLimit": {},
					},
				},
			},
			wantErr: false,
		},
		// Annotation validation test cases
		{
			name: "valid annotation vllm-distributed-executor-backend=mp",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "mp",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "valid annotation vllm-distributed-executor-backend=ray",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "ray",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "valid annotation vllm-distributed-executor-backend case insensitive MP",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "MP",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "invalid annotation vllm-distributed-executor-backend",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: true,
			errMsg:  `annotation nvidia.com/vllm-distributed-executor-backend has invalid value "invalid": must be "mp" or "ray"`,
		},
		{
			name: "no annotations is valid",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "valid annotation dynamo-operator-origin-version with semver",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: "1.0.0",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "valid annotation dynamo-operator-origin-version with pre-release",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: "1.0.0-dev",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "valid annotation dynamo-operator-origin-version fallback version",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: "0.0.0-unknown",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "invalid annotation dynamo-operator-origin-version not semver",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion: "not-a-version",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: true,
			errMsg:  `annotation nvidia.com/dynamo-operator-origin-version has invalid value "not-a-version": must be valid semver`,
		},
		{
			name: "both annotations valid",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion:    "1.0.0",
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "mp",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "both annotations invalid",
			deployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-graph",
					Namespace: "default",
					Annotations: map[string]string{
						consts.KubeAnnotationDynamoOperatorOriginVersion:    "bad",
						consts.KubeAnnotationVLLMDistributedExecutorBackend: "invalid",
					},
				},
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {},
					},
				},
			},
			wantErr:     true,
			errMsg:      "annotation nvidia.com/dynamo-operator-origin-version has invalid value \"bad\": must be valid semver\nannotation nvidia.com/vllm-distributed-executor-backend has invalid value \"invalid\": must be \"mp\" or \"ray\"",
			errContains: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentValidator(tt.deployment)
			_, err := validator.Validate(context.Background())

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
			name: "adding single service is prohibited",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend": {},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend":  {},
						"frontend": {},
					},
				},
			},
			wantErr: true,
			errMsg:  "service topology is immutable and cannot be modified after creation: services added: [frontend]",
		},
		{
			name: "adding multiple services is prohibited",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend": {},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend":  {},
						"cache":    {},
						"frontend": {},
					},
				},
			},
			wantErr: true,
			errMsg:  "service topology is immutable and cannot be modified after creation: services added: [cache frontend]",
		},
		{
			name: "removing single service is prohibited",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend":  {},
						"frontend": {},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend": {},
					},
				},
			},
			wantErr: true,
			errMsg:  "service topology is immutable and cannot be modified after creation: services removed: [frontend]",
		},
		{
			name: "removing multiple services is prohibited",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend":  {},
						"cache":    {},
						"frontend": {},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend": {},
					},
				},
			},
			wantErr: true,
			errMsg:  "service topology is immutable and cannot be modified after creation: services removed: [cache frontend]",
		},
		{
			name: "adding and removing services simultaneously is prohibited",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend": {},
						"cache":   {},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend":  {},
						"frontend": {},
					},
				},
			},
			wantErr: true,
			errMsg:  "service topology is immutable and cannot be modified after creation: services added: [frontend], services removed: [cache]",
		},
		{
			name: "modifying service specifications is allowed",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend": {
							Replicas: func() *int32 { r := int32(1); return &r }(),
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend": {
							Replicas: func() *int32 { r := int32(3); return &r }(),
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "service topology unchanged with same services",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend":  {},
						"frontend": {},
						"cache":    {},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"backend":  {},
						"frontend": {},
						"cache":    {},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "changing service from single-node to multi-node",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							// Single-node (nil Multinode)
							Multinode: nil,
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							// Multi-node (NodeCount > 1)
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.services[main] cannot change node topology (between single-node and multi-node) after creation",
		},
		{
			name: "changing service from multi-node to single-node",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							// Multi-node
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 3,
							},
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							// Single-node (nil Multinode)
							Multinode: nil,
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.services[main] cannot change node topology (between single-node and multi-node) after creation",
		},
		{
			name: "changing multinode NodeCount within multi-node range is allowed",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 4,
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "keeping service as single-node is allowed",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: nil,
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: nil,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "keeping service as multi-node is allowed",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 3,
							},
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 3,
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "changing from single-node (NodeCount=1) to multi-node (NodeCount=2)",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 1,
							},
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.services[main] cannot change node topology (between single-node and multi-node) after creation",
		},
		{
			name: "changing from multi-node (NodeCount=2) to single-node (NodeCount=1)",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 1,
							},
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.services[main] cannot change node topology (between single-node and multi-node) after creation",
		},
		{
			name: "multiple services with one changing topology",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: nil,
						},
						"prefill": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							// Changing from single-node to multi-node
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 3,
							},
						},
						"prefill": {
							// Keeping as multi-node (OK)
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 4,
							},
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.services[main] cannot change node topology (between single-node and multi-node) after creation",
		},
		{
			name: "adding new service with multinode is not allowed", // service topology is immutable
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: nil,
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: nil,
						},
						"decode": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 4,
							},
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "service topology is immutable and cannot be modified after creation: services added: [decode]",
		},
		{
			name: "adding new service without multinode is not allowed", // service topology is immutable
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
						},
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &nvidiacomv1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
						},
						"gateway": {
							// New service without multinode - should be allowed
							Multinode: nil,
						},
					},
				},
			},
			wantErr: true,
			errMsg:  "service topology is immutable and cannot be modified after creation: services added: [gateway]",
		},
		{
			name: "restart.id change while rolling update Pending - rejected",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "old-restart-id",
					},
				},
				Status: nvidiacomv1alpha1.DynamoGraphDeploymentStatus{
					RollingUpdate: &nvidiacomv1alpha1.RollingUpdateStatus{
						Phase: nvidiacomv1alpha1.RollingUpdatePhasePending,
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "new-restart-id",
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.restart.id cannot be changed while a rolling update is Pending",
		},
		{
			name: "restart.id change while rolling update InProgress - rejected",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "old-restart-id",
					},
				},
				Status: nvidiacomv1alpha1.DynamoGraphDeploymentStatus{
					RollingUpdate: &nvidiacomv1alpha1.RollingUpdateStatus{
						Phase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress,
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "new-restart-id",
					},
				},
			},
			wantErr: true,
			errMsg:  "spec.restart.id cannot be changed while a rolling update is InProgress",
		},
		{
			name: "restart.id change while rolling update Completed - allowed",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "old-restart-id",
					},
				},
				Status: nvidiacomv1alpha1.DynamoGraphDeploymentStatus{
					RollingUpdate: &nvidiacomv1alpha1.RollingUpdateStatus{
						Phase: nvidiacomv1alpha1.RollingUpdatePhaseCompleted,
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "new-restart-id",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "restart.id change with no rolling update - allowed",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "old-restart-id",
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "new-restart-id",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "spec change without restart.id change during rolling update - allowed",
			oldDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {
							Replicas: func() *int32 { r := int32(1); return &r }(),
						},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "same-restart-id",
					},
				},
				Status: nvidiacomv1alpha1.DynamoGraphDeploymentStatus{
					RollingUpdate: &nvidiacomv1alpha1.RollingUpdateStatus{
						Phase: nvidiacomv1alpha1.RollingUpdatePhaseInProgress,
					},
				},
			},
			newDeployment: &nvidiacomv1alpha1.DynamoGraphDeployment{
				Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
					Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						"worker": {
							Replicas: func() *int32 { r := int32(3); return &r }(),
						},
					},
					Restart: &nvidiacomv1alpha1.Restart{
						ID: "same-restart-id",
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			validator := NewDynamoGraphDeploymentValidator(tt.newDeployment)
			// Pass nil userInfo - these tests don't modify replicas, so it's safe
			warnings, err := validator.ValidateUpdate(tt.oldDeployment, nil)

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

func TestGetServiceNames(t *testing.T) {
	tests := []struct {
		name     string
		services map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec
		want     map[string]struct{}
	}{
		{
			name:     "empty services",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{},
			want:     map[string]struct{}{},
		},
		{
			name: "single service",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"backend": {},
			},
			want: map[string]struct{}{
				"backend": {},
			},
		},
		{
			name: "multiple services",
			services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"backend":  {},
				"frontend": {},
				"cache":    {},
			},
			want: map[string]struct{}{
				"backend":  {},
				"frontend": {},
				"cache":    {},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := getServiceNames(tt.services)
			if len(got) != len(tt.want) {
				t.Errorf("getServiceNames() length = %v, want %v", len(got), len(tt.want))
				return
			}
			for name := range tt.want {
				if _, exists := got[name]; !exists {
					t.Errorf("getServiceNames() missing service %v", name)
				}
			}
		})
	}
}

func TestDifference(t *testing.T) {
	tests := []struct {
		name string
		a    map[string]struct{}
		b    map[string]struct{}
		want []string
	}{
		{
			name: "empty sets",
			a:    map[string]struct{}{},
			b:    map[string]struct{}{},
			want: nil,
		},
		{
			name: "a is empty",
			a:    map[string]struct{}{},
			b: map[string]struct{}{
				"backend": {},
			},
			want: nil,
		},
		{
			name: "b is empty",
			a: map[string]struct{}{
				"backend": {},
			},
			b:    map[string]struct{}{},
			want: []string{"backend"},
		},
		{
			name: "no difference - identical sets",
			a: map[string]struct{}{
				"backend":  {},
				"frontend": {},
			},
			b: map[string]struct{}{
				"backend":  {},
				"frontend": {},
			},
			want: nil,
		},
		{
			name: "single element difference",
			a: map[string]struct{}{
				"backend":  {},
				"frontend": {},
			},
			b: map[string]struct{}{
				"backend": {},
			},
			want: []string{"frontend"},
		},
		{
			name: "multiple element difference",
			a: map[string]struct{}{
				"backend":  {},
				"frontend": {},
				"cache":    {},
			},
			b: map[string]struct{}{
				"backend": {},
			},
			want: []string{"cache", "frontend"},
		},
		{
			name: "completely different sets",
			a: map[string]struct{}{
				"frontend": {},
				"cache":    {},
			},
			b: map[string]struct{}{
				"backend": {},
			},
			want: []string{"cache", "frontend"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := difference(tt.a, tt.b)

			// Sort both slices for comparison (since map iteration order is undefined)
			sort.Strings(got)
			want := make([]string, len(tt.want))
			copy(want, tt.want)
			sort.Strings(want)

			if len(got) != len(want) {
				t.Errorf("difference() length = %v, want %v", len(got), len(want))
				return
			}

			for i := range got {
				if got[i] != want[i] {
					t.Errorf("difference() = %v, want %v", got, want)
					return
				}
			}
		})
	}
}
