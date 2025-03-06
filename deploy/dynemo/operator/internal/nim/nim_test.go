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

package nim

import (
	"testing"

	compounaiCommon "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/common"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/v1alpha1"
	"github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGenerateCompoundAINIMDeployments(t *testing.T) {
	type args struct {
		parentCompoundAIDeployment *v1alpha1.CompoundAIDeployment
		config                     *CompoundAINIMConfig
	}
	tests := []struct {
		name    string
		args    args
		want    map[string]*v1alpha1.CompoundAINimDeployment
		wantErr bool
	}{
		{
			name: "Test GenerateCompoundAINIMDeployments http dependency",
			args: args{
				parentCompoundAIDeployment: &v1alpha1.CompoundAIDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-compoundaideployment",
						Namespace: "default",
					},
					Spec: v1alpha1.CompoundAIDeploymentSpec{
						CompoundAINim: "compoundainim:ac4e234",
					},
				},
				config: &CompoundAINIMConfig{
					Services: []ServiceConfig{
						{
							Name:         "service1",
							Dependencies: []map[string]string{{"service": "service2"}},
							Config: Config{
								Nova: &NovaConfig{
									Enabled:   true,
									Namespace: "default",
									Name:      "service1",
								},
								Resources: &Resources{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
								Autoscaling: &Autoscaling{
									MinReplicas: 1,
									MaxReplicas: 5,
								},
							},
						},
						{
							Name:         "service2",
							Dependencies: []map[string]string{},
							Config: Config{
								Nova: &NovaConfig{
									Enabled: false,
								},
							},
						},
					},
				},
			},
			want: map[string]*v1alpha1.CompoundAINimDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-compoundaideployment-service1",
						Namespace: "default",
					},
					Spec: v1alpha1.CompoundAINimDeploymentSpec{
						CompoundAINim: "compoundainim",
						ServiceName:   "service1",
						Resources: &compounaiCommon.Resources{
							Requests: &compounaiCommon.ResourceItem{
								CPU:    "1",
								Memory: "1Gi",
								GPU:    "0",
								Custom: map[string]string{},
							},
							Limits: &compounaiCommon.ResourceItem{
								CPU:    "1",
								Memory: "1Gi",
								GPU:    "0",
								Custom: map[string]string{},
							},
						},
						Autoscaling: &v1alpha1.Autoscaling{
							MinReplicas: 1,
							MaxReplicas: 5,
						},
						ExternalServices: map[string]v1alpha1.ExternalService{
							"service2": {
								DeploymentSelectorKey:   "name",
								DeploymentSelectorValue: "service2",
							},
						},
					},
				},
				"service2": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-compoundaideployment-service2",
						Namespace: "default",
					},
					Spec: v1alpha1.CompoundAINimDeploymentSpec{
						ServiceName:   "service2",
						CompoundAINim: "compoundainim",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateCompoundAINIMDeployments dynemo dependency",
			args: args{
				parentCompoundAIDeployment: &v1alpha1.CompoundAIDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-compoundaideployment",
						Namespace: "default",
					},
					Spec: v1alpha1.CompoundAIDeploymentSpec{
						CompoundAINim: "compoundainim:ac4e234",
					},
				},
				config: &CompoundAINIMConfig{
					Services: []ServiceConfig{
						{
							Name:         "service1",
							Dependencies: []map[string]string{{"service": "service2"}},
							Config: Config{
								Nova: &NovaConfig{
									Enabled:   true,
									Namespace: "default",
									Name:      "service1",
								},
								Resources: &Resources{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
								Autoscaling: &Autoscaling{
									MinReplicas: 1,
									MaxReplicas: 5,
								},
							},
						},
						{
							Name:         "service2",
							Dependencies: []map[string]string{},
							Config: Config{
								Nova: &NovaConfig{
									Enabled:   true,
									Namespace: "default",
									Name:      "service2",
								},
							},
						},
					},
				},
			},
			want: map[string]*v1alpha1.CompoundAINimDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-compoundaideployment-service1",
						Namespace: "default",
					},
					Spec: v1alpha1.CompoundAINimDeploymentSpec{
						CompoundAINim: "compoundainim",
						ServiceName:   "service1",
						Resources: &compounaiCommon.Resources{
							Requests: &compounaiCommon.ResourceItem{
								CPU:    "1",
								Memory: "1Gi",
								GPU:    "0",
								Custom: map[string]string{},
							},
							Limits: &compounaiCommon.ResourceItem{
								CPU:    "1",
								Memory: "1Gi",
								GPU:    "0",
								Custom: map[string]string{},
							},
						},
						Autoscaling: &v1alpha1.Autoscaling{
							MinReplicas: 1,
							MaxReplicas: 5,
						},
						ExternalServices: map[string]v1alpha1.ExternalService{
							"service2": {
								DeploymentSelectorKey:   "nova",
								DeploymentSelectorValue: "service2/default",
							},
						},
					},
				},
				"service2": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-compoundaideployment-service2",
						Namespace: "default",
					},
					Spec: v1alpha1.CompoundAINimDeploymentSpec{
						CompoundAINim: "compoundainim",
						ServiceName:   "service2",
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateCompoundAINIMDeployments dependency not found",
			args: args{
				parentCompoundAIDeployment: &v1alpha1.CompoundAIDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-compoundaideployment",
						Namespace: "default",
					},
					Spec: v1alpha1.CompoundAIDeploymentSpec{
						CompoundAINim: "compoundainim:ac4e234",
					},
				},
				config: &CompoundAINIMConfig{
					Services: []ServiceConfig{
						{
							Name:         "service1",
							Dependencies: []map[string]string{{"service": "service2"}},
							Config: Config{
								Nova: &NovaConfig{
									Enabled:   true,
									Namespace: "default",
									Name:      "service1",
								},
								Resources: &Resources{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
								Autoscaling: &Autoscaling{
									MinReplicas: 1,
									MaxReplicas: 5,
								},
							},
						},
						{
							Name:         "service3",
							Dependencies: []map[string]string{},
							Config: Config{
								Nova: &NovaConfig{
									Enabled:   true,
									Namespace: "default",
									Name:      "service3",
								},
							},
						},
					},
				},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			got, err := GenerateCompoundAINIMDeployments(tt.args.parentCompoundAIDeployment, tt.args.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("GenerateCompoundAINIMDeployments() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			g.Expect(got).To(gomega.Equal(tt.want))
		})
	}
}
