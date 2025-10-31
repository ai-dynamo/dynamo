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
					DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
						ComponentType: commonconsts.ComponentTypeFrontend,
					},
				},
			},
			want: true,
		},
		{
			name: "not main component",
			fields: fields{
				Spec: DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
						ComponentType: commonconsts.ComponentTypeWorker,
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

func TestDynamoComponentDeploymentSharedSpec_VolumeMounts(t *testing.T) {
	tests := []struct {
		name               string
		spec               DynamoComponentDeploymentSharedSpec
		expectedMountCount int
		expectedMounts     []VolumeMount
	}{
		{
			name: "Spec with multiple volume mounts",
			spec: DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []VolumeMount{
					{Name: "data-pvc", MountPoint: "/data"},
					{Name: "logs-pvc", MountPoint: "/logs"},
				},
			},
			expectedMountCount: 2,
			expectedMounts: []VolumeMount{
				{Name: "data-pvc", MountPoint: "/data"},
				{Name: "logs-pvc", MountPoint: "/logs"},
			},
		},
		{
			name: "Spec with single volume mount",
			spec: DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []VolumeMount{
					{Name: "shared-storage", MountPoint: "/shared"},
				},
			},
			expectedMountCount: 1,
			expectedMounts: []VolumeMount{
				{Name: "shared-storage", MountPoint: "/shared"},
			},
		},
		{
			name: "Spec without volume mounts",
			spec: DynamoComponentDeploymentSharedSpec{
				VolumeMounts: nil,
			},
			expectedMountCount: 0,
			expectedMounts:     nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.spec.VolumeMounts) != tt.expectedMountCount {
				t.Errorf("Volume mount count = %v, want %v", len(tt.spec.VolumeMounts), tt.expectedMountCount)
			}
			if !reflect.DeepEqual(tt.spec.VolumeMounts, tt.expectedMounts) {
				t.Errorf("VolumeMounts = %v, want %v", tt.spec.VolumeMounts, tt.expectedMounts)
			}
		})
	}
}

func TestPVC_Validation(t *testing.T) {
	tests := []struct {
		name        string
		pvc         PVC
		expectValid bool
		description string
	}{
		{
			name: "valid PVC with create false",
			pvc: PVC{
				Create: ptr.To(false),
				Name:   ptr.To("test-pvc"),
			},
			expectValid: true,
			description: "When create is false, size/storageClass/volumeAccessMode are not required",
		},
		{
			name: "valid PVC with create nil (omitted)",
			pvc: PVC{
				Name: ptr.To("test-pvc"),
			},
			expectValid: true,
			description: "When create is omitted, size/storageClass/volumeAccessMode are not required",
		},
		{
			name: "valid PVC with create true and all required fields",
			pvc: PVC{
				Create:           ptr.To(true),
				Name:             ptr.To("test-pvc"),
				Size:             resource.MustParse("10Gi"),
				StorageClass:     "fast-ssd",
				VolumeAccessMode: corev1.ReadWriteOnce,
			},
			expectValid: true,
			description: "When create is true and all required fields are provided",
		},
		{
			name: "invalid PVC with create true but missing size",
			pvc: PVC{
				Create:           ptr.To(true),
				Name:             ptr.To("test-pvc"),
				StorageClass:     "fast-ssd",
				VolumeAccessMode: corev1.ReadWriteOnce,
			},
			expectValid: false,
			description: "When create is true but size is missing",
		},
		{
			name: "invalid PVC with create true but missing storageClass",
			pvc: PVC{
				Create:           ptr.To(true),
				Name:             ptr.To("test-pvc"),
				Size:             resource.MustParse("10Gi"),
				VolumeAccessMode: corev1.ReadWriteOnce,
			},
			expectValid: false,
			description: "When create is true but storageClass is missing",
		},
		{
			name: "invalid PVC with create true but missing volumeAccessMode",
			pvc: PVC{
				Create:       ptr.To(true),
				Name:         ptr.To("test-pvc"),
				Size:         resource.MustParse("10Gi"),
				StorageClass: "fast-ssd",
			},
			expectValid: false,
			description: "When create is true but volumeAccessMode is missing",
		},
		{
			name: "invalid PVC with create true but missing all required fields",
			pvc: PVC{
				Create: ptr.To(true),
				Name:   ptr.To("test-pvc"),
			},
			expectValid: false,
			description: "When create is true but all required fields are missing",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.pvc.Create != nil && *tt.pvc.Create {
				hasSize := !tt.pvc.Size.IsZero()
				hasStorageClass := tt.pvc.StorageClass != ""
				hasVolumeAccessMode := tt.pvc.VolumeAccessMode != ""

				isValid := hasSize && hasStorageClass && hasVolumeAccessMode

				if isValid != tt.expectValid {
					t.Errorf("PVC validation = %v, expected %v. %s", isValid, tt.expectValid, tt.description)
					t.Errorf("  hasSize: %v, hasStorageClass: %v, hasVolumeAccessMode: %v", hasSize, hasStorageClass, hasVolumeAccessMode)
				}
			} else {
				if !tt.expectValid {
					t.Errorf("PVC validation should be valid when create is false/nil. %s", tt.description)
				}
			}
		})
	}
}
