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

	compounaiCommon "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/v1alpha1"
	"github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGenerateDynamoNIMDeployments(t *testing.T) {
	type args struct {
		parentDynamoDeployment *v1alpha1.DynamoDeployment
		config                 *DynamoNIMConfig
	}
	tests := []struct {
		name    string
		args    args
		want    map[string]*v1alpha1.DynamoNimDeployment
		wantErr bool
	}{
		{
			name: "Test GenerateDynamoNIMDeployments http dependency",
			args: args{
				parentDynamoDeployment: &v1alpha1.DynamoDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamodeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoDeploymentSpec{
						DynamoNim: "dynamonim:ac4e234",
					},
				},
				config: &DynamoNIMConfig{
					DynamoTag: "dynamonim:MyService1",
					Services: []ServiceConfig{
						{
							Name:         "service1",
							Dependencies: []map[string]string{{"service": "service2"}},
							Config: Config{
								Dynamo: &DynamoConfig{
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
								Dynamo: &DynamoConfig{
									Enabled: false,
								},
							},
						},
					},
				},
			},
			want: map[string]*v1alpha1.DynamoNimDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamodeployment-service1",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoNimDeploymentSpec{
						DynamoNim:   "dynamonim--ac4e234",
						DynamoTag:   "dynamonim:MyService1",
						ServiceName: "service1",
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
							Enabled:     true,
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
						Name:      "test-dynamodeployment-service2",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoNimDeploymentSpec{
						DynamoNim:   "dynamonim--ac4e234",
						DynamoTag:   "dynamonim:MyService1",
						ServiceName: "service2",
						Autoscaling: &v1alpha1.Autoscaling{
							Enabled: false,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoNIMDeployments dynamo dependency",
			args: args{
				parentDynamoDeployment: &v1alpha1.DynamoDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamodeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoDeploymentSpec{
						DynamoNim: "dynamonim:ac4e234",
					},
				},
				config: &DynamoNIMConfig{
					DynamoTag:    "dynamonim:MyService2",
					EntryService: "service1",
					Services: []ServiceConfig{
						{
							Name:         "service1",
							Dependencies: []map[string]string{{"service": "service2"}},
							Config: Config{
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
								Dynamo: &DynamoConfig{
									Enabled:   true,
									Namespace: "default",
									Name:      "service2",
								},
							},
						},
					},
				},
			},
			want: map[string]*v1alpha1.DynamoNimDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamodeployment-service1",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoNimDeploymentSpec{
						DynamoNim:   "dynamonim--ac4e234",
						DynamoTag:   "dynamonim:MyService2",
						ServiceName: "service1",
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
							Enabled:     true,
							MinReplicas: 1,
							MaxReplicas: 5,
						},
						ExternalServices: map[string]v1alpha1.ExternalService{
							"service2": {
								DeploymentSelectorKey:   "dynamo",
								DeploymentSelectorValue: "service2/default",
							},
						},
						Ingress: v1alpha1.IngressSpec{
							Enabled:           true,
							UseVirtualService: &[]bool{true}[0],
						},
					},
				},
				"service2": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamodeployment-service2",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoNimDeploymentSpec{
						DynamoNim:   "dynamonim--ac4e234",
						DynamoTag:   "dynamonim:MyService2",
						ServiceName: "service2",
						Autoscaling: &v1alpha1.Autoscaling{
							Enabled: false,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoNIMDeployments dependency not found",
			args: args{
				parentDynamoDeployment: &v1alpha1.DynamoDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamodeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoDeploymentSpec{
						DynamoNim: "dynamonim:ac4e234",
					},
				},
				config: &DynamoNIMConfig{
					DynamoTag: "dynamonim:MyService3",
					Services: []ServiceConfig{
						{
							Name:         "service1",
							Dependencies: []map[string]string{{"service": "service2"}},
							Config: Config{
								Dynamo: &DynamoConfig{
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
								Dynamo: &DynamoConfig{
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
			got, err := GenerateDynamoNIMDeployments(tt.args.parentDynamoDeployment, tt.args.config)
			if (err != nil) != tt.wantErr {
				t.Errorf("GenerateDynamoNIMDeployments() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			g.Expect(got).To(gomega.Equal(tt.want))
		})
	}
}
