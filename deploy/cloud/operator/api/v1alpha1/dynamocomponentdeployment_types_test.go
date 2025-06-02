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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDynamoComponentDeployment_IsMainComponent(t *testing.T) {
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
			if got := s.IsMainComponent(); got != tt.want {
				t.Errorf("DynamoComponentDeployment.IsMainComponent() = %v, want %v", got, tt.want)
			}
		})
	}
}
