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

package defaulting

import (
	"context"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestDCDDefaulter_DefaultsComponentNameOnCreate(t *testing.T) {
	tests := []struct {
		name string
		ctx  context.Context
		dcd  *nvidiacomv1beta1.DynamoComponentDeployment
		want string
	}{
		{
			name: "CREATE defaults empty spec name from metadata name",
			ctx:  admissionCtx(admissionv1.Create),
			dcd: &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker"},
			},
			want: "worker",
		},
		{
			name: "CREATE preserves explicit spec name",
			ctx:  admissionCtx(admissionv1.Create),
			dcd: &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker"},
				Spec: nvidiacomv1beta1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentName: "custom",
					},
				},
			},
			want: "custom",
		},
		{
			name: "UPDATE does not default empty spec name",
			ctx:  admissionCtx(admissionv1.Update),
			dcd: &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker"},
			},
			want: "",
		},
		{
			name: "missing admission request skips defaulting gracefully",
			ctx:  context.Background(),
			dcd: &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker"},
			},
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defaulter := NewDCDDefaulter()

			if err := defaulter.Default(tt.ctx, tt.dcd); err != nil {
				t.Fatalf("Default() unexpected error: %v", err)
			}

			if got := tt.dcd.Spec.ComponentName; got != tt.want {
				t.Fatalf("spec.name = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDCDDefaulter_DefaultRejectsWrongType(t *testing.T) {
	defaulter := NewDCDDefaulter()

	if err := defaulter.Default(admissionCtx(admissionv1.Create), &corev1.Pod{}); err == nil {
		t.Fatal("Default() error = nil, want type error")
	}
}
