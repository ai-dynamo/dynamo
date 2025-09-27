/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
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
 * Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
 */

package v1alpha1

import (
	"reflect"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestDynamoComponentDeployment_IsFrontendComponent(t *testing.T) {
	type fields struct {
		TypeMeta   metav1.TypeMeta
		ObjectMeta metav1.ObjectMeta
		Spec       DynamoComponentDeploymentSpec
		Status     DynamoComponentDeploymentStatus
	}
	tests := []struct {
		name   string
		fields fields
		want   bool
	}{
		{
			name: "main component",
			fields: fields{
				Spec: DynamoComponentDeploymentSpec{
					DynamoTag: "dynamo-component:main",
					DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
						ServiceName: "main",
					},
				},
			},
			want: true,
		},
		{
			name: "not main component",
			fields: fields{
				Spec: DynamoComponentDeploymentSpec{
					DynamoTag: "dynamo-component:main",
					DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
						ServiceName: "not-main",
					},
				},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &DynamoComponentDeployment{
				TypeMeta:   tt.fields.TypeMeta,
				ObjectMeta: tt.fields.ObjectMeta,
				Spec:       tt.fields.Spec,
				Status:     tt.fields.Status,
			}
			if got := s.IsFrontendComponent(); got != tt.want {
				t.Errorf("DynamoComponentDeployment.IsFrontendComponent() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDynamoComponentDeployment_GetDynamoDeploymentConfig(t *testing.T) {
	type fields struct {
		TypeMeta   metav1.TypeMeta
		ObjectMeta metav1.ObjectMeta
		Spec       DynamoComponentDeploymentSpec
		Status     DynamoComponentDeploymentStatus
	}
	tests := []struct {
		name   string
		fields fields
		want   []byte
	}{
		{
			name: "no config",
			fields: fields{
				Spec: DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
						Envs: []corev1.EnvVar{},
					},
				},
			},
			want: nil,
		},
		{
			name: "with config",
			fields: fields{
				Spec: DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
						Envs: []corev1.EnvVar{
							{
								Name:  commonconsts.DynamoDeploymentConfigEnvVar,
								Value: `{"Frontend":{"port":8080},"Planner":{"environment":"kubernetes"}}`,
							},
						},
					},
				},
			},
			want: []byte(`{"Frontend":{"port":8080},"Planner":{"environment":"kubernetes"}}`),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &DynamoComponentDeployment{
				TypeMeta:   tt.fields.TypeMeta,
				ObjectMeta: tt.fields.ObjectMeta,
				Spec:       tt.fields.Spec,
				Status:     tt.fields.Status,
			}
			if got := s.GetDynamoDeploymentConfig(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("DynamoComponentDeployment.GetDynamoDeploymentConfig() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDynamoComponentDeployment_SetDynamoDeploymentConfig(t *testing.T) {
	type fields struct {
		TypeMeta   metav1.TypeMeta
		ObjectMeta metav1.ObjectMeta
		Spec       DynamoComponentDeploymentSpec
		Status     DynamoComponentDeploymentStatus
	}
	type args struct {
		config []byte
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   []corev1.EnvVar
	}{
		{
			name: "no config",
			fields: fields{
				Spec: DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
						Envs: nil,
					},
				},
			},
			args: args{
				config: []byte(`{"Frontend":{"port":8080},"Planner":{"environment":"kubernetes"}}`),
			},
			want: []corev1.EnvVar{
				{
					Name:  commonconsts.DynamoDeploymentConfigEnvVar,
					Value: `{"Frontend":{"port":8080},"Planner":{"environment":"kubernetes"}}`,
				},
			},
		},
		{
			name: "with config",
			fields: fields{
				Spec: DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
						Envs: []corev1.EnvVar{
							{
								Name:  commonconsts.DynamoDeploymentConfigEnvVar,
								Value: `{"Frontend":{"port":8080},"Planner":{"environment":"kubernetes"}}`,
							},
						},
					},
				},
			},
			args: args{
				config: []byte(`{"Frontend":{"port":9000},"Planner":{"environment":"kubernetes"}}`),
			},
			want: []corev1.EnvVar{
				{
					Name:  commonconsts.DynamoDeploymentConfigEnvVar,
					Value: `{"Frontend":{"port":9000},"Planner":{"environment":"kubernetes"}}`,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &DynamoComponentDeployment{
				TypeMeta:   tt.fields.TypeMeta,
				ObjectMeta: tt.fields.ObjectMeta,
				Spec:       tt.fields.Spec,
				Status:     tt.fields.Status,
			}
			s.SetDynamoDeploymentConfig(tt.args.config)
			if !reflect.DeepEqual(s.Spec.DynamoComponentDeploymentSharedSpec.Envs, tt.want) {
				t.Errorf("DynamoComponentDeployment.SetDynamoDeploymentConfig() = %v, want %v", s.Spec.DynamoComponentDeploymentSharedSpec.Envs, tt.want)
			}
		})
	}
}

func TestDynamoComponentDeployment_GetParentGraphDeploymentName(t *testing.T) {
	type fields struct {
		TypeMeta   metav1.TypeMeta
		ObjectMeta metav1.ObjectMeta
		Spec       DynamoComponentDeploymentSpec
		Status     DynamoComponentDeploymentStatus
	}
	tests := []struct {
		name   string
		fields fields
		want   string
	}{
		{
			name: "test",
			fields: fields{
				ObjectMeta: metav1.ObjectMeta{
					OwnerReferences: []metav1.OwnerReference{
						{
							Kind: "DynamoGraphDeployment",
							Name: "name",
						},
					},
				},
			},
			want: "name",
		},
		{
			name: "no owner reference",
			fields: fields{
				ObjectMeta: metav1.ObjectMeta{
					OwnerReferences: []metav1.OwnerReference{},
				},
			},
			want: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := &DynamoComponentDeployment{
				TypeMeta:   tt.fields.TypeMeta,
				ObjectMeta: tt.fields.ObjectMeta,
				Spec:       tt.fields.Spec,
				Status:     tt.fields.Status,
			}
			if got := s.GetParentGraphDeploymentName(); got != tt.want {
				t.Errorf("DynamoComponentDeployment.GetParentGraphDeploymentName() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCompilationCachePVC(t *testing.T) {
	tests := []struct {
		name     string
		pvc      CompilationCachePVC
		expected CompilationCachePVC
	}{
		{
			name: "CompilationCachePVC with all fields set",
			pvc: CompilationCachePVC{
				PVC: PVC{
					Create:           ptr.To(true),
					Name:             ptr.To("test-compilation-cache"),
					StorageClass:     "fast-ssd",
					Size:             resource.MustParse("50Gi"),
					VolumeAccessMode: corev1.ReadWriteMany,
					MountPoint:       ptr.To("/cache/compilation"),
				},
			},
			expected: CompilationCachePVC{
				PVC: PVC{
					Create:           ptr.To(true),
					Name:             ptr.To("test-compilation-cache"),
					StorageClass:     "fast-ssd",
					Size:             resource.MustParse("50Gi"),
					VolumeAccessMode: corev1.ReadWriteMany,
					MountPoint:       ptr.To("/cache/compilation"),
				},
			},
		},
		{
			name: "CompilationCachePVC with minimal fields",
			pvc: CompilationCachePVC{
				PVC: PVC{
					Create:     ptr.To(false),
					Name:       ptr.To("existing-cache"),
					MountPoint: ptr.To("/root/.cache"),
				},
			},
			expected: CompilationCachePVC{
				PVC: PVC{
					Create:     ptr.To(false),
					Name:       ptr.To("existing-cache"),
					MountPoint: ptr.To("/root/.cache"),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !reflect.DeepEqual(tt.expected, tt.pvc) {
				t.Errorf("CompilationCachePVC = %v, want %v", tt.pvc, tt.expected)
			}

			// Test that CompilationCachePVC embeds PVC correctly
			if !reflect.DeepEqual(tt.expected.PVC.Create, tt.pvc.Create) {
				t.Errorf("CompilationCachePVC.Create = %v, want %v", tt.pvc.Create, tt.expected.PVC.Create)
			}
			if !reflect.DeepEqual(tt.expected.PVC.Name, tt.pvc.Name) {
				t.Errorf("CompilationCachePVC.Name = %v, want %v", tt.pvc.Name, tt.expected.PVC.Name)
			}
			if tt.expected.PVC.StorageClass != tt.pvc.StorageClass {
				t.Errorf("CompilationCachePVC.StorageClass = %v, want %v", tt.pvc.StorageClass, tt.expected.PVC.StorageClass)
			}
			if !tt.expected.PVC.Size.Equal(tt.pvc.Size) {
				t.Errorf("CompilationCachePVC.Size = %v, want %v", tt.pvc.Size, tt.expected.PVC.Size)
			}
			if tt.expected.PVC.VolumeAccessMode != tt.pvc.VolumeAccessMode {
				t.Errorf("CompilationCachePVC.VolumeAccessMode = %v, want %v", tt.pvc.VolumeAccessMode, tt.expected.PVC.VolumeAccessMode)
			}
			if !reflect.DeepEqual(tt.expected.PVC.MountPoint, tt.pvc.MountPoint) {
				t.Errorf("CompilationCachePVC.MountPoint = %v, want %v", tt.pvc.MountPoint, tt.expected.PVC.MountPoint)
			}
		})
	}
}

func TestDynamoComponentDeploymentSharedSpec_CompilationCache(t *testing.T) {
	tests := []struct {
		name              string
		spec              DynamoComponentDeploymentSharedSpec
		expectCompilation bool
		expectMountPoint  string
	}{
		{
			name: "Spec with compilation cache enabled",
			spec: DynamoComponentDeploymentSharedSpec{
				CompilationCache: &CompilationCachePVC{
					PVC: PVC{
						Create:     ptr.To(true),
						Name:       ptr.To("vllm-compilation-cache"),
						MountPoint: ptr.To("/root/.cache/vllm"),
					},
				},
			},
			expectCompilation: true,
			expectMountPoint:  "/root/.cache/vllm",
		},
		{
			name: "Spec without compilation cache",
			spec: DynamoComponentDeploymentSharedSpec{
				CompilationCache: nil,
			},
			expectCompilation: false,
			expectMountPoint:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.expectCompilation {
				if tt.spec.CompilationCache == nil {
					t.Errorf("Expected compilation cache to be set, but it was nil")
				}
				if tt.spec.CompilationCache.MountPoint == nil {
					t.Errorf("Expected mount point to be set, but it was nil")
				} else if *tt.spec.CompilationCache.MountPoint != tt.expectMountPoint {
					t.Errorf("Mount point = %v, want %v", *tt.spec.CompilationCache.MountPoint, tt.expectMountPoint)
				}
			} else {
				if tt.spec.CompilationCache != nil {
					t.Errorf("Expected compilation cache to be nil, but it was %v", tt.spec.CompilationCache)
				}
			}
		})
	}
}
