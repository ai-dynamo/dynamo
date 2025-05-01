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

package dynamo

import (
	"context"
	"testing"

	compounaiCommon "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/dynamo/operator/internal/consts"
	"github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestGenerateDynamoComponentsDeployments(t *testing.T) {
	type args struct {
		parentDynamoGraphDeployment *v1alpha1.DynamoGraphDeployment
		config                      *DynamoGraphConfig
		ingressSpec                 *v1alpha1.IngressSpec
	}
	tests := []struct {
		name    string
		args    args
		want    map[string]*v1alpha1.DynamoComponentDeployment
		wantErr bool
	}{
		{
			name: "Test GenerateDynamoComponentsDeployments http dependency",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						DynamoGraph: "dynamocomponent:ac4e234",
					},
				},
				config: &DynamoGraphConfig{
					DynamoTag: "dynamocomponent:MyService1",
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
				ingressSpec: &v1alpha1.IngressSpec{},
			},
			want: map[string]*v1alpha1.DynamoComponentDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service1",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "service1",
							commonconsts.KubeLabelDynamoNamespace: "default",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponent: "dynamocomponent:ac4e234",
						DynamoTag:       "dynamocomponent:MyService1",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default"}[0],
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
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
								commonconsts.KubeLabelDynamoNamespace: "default",
							},
						},
					},
				},
				"service2": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service2",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "service2",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponent: "dynamocomponent:ac4e234",
						DynamoTag:       "dynamocomponent:MyService1",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "service2",
							Autoscaling: &v1alpha1.Autoscaling{
								Enabled: false,
							},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments dynamo dependency",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						DynamoGraph: "dynamocomponent:ac4e234",
					},
				},
				config: &DynamoGraphConfig{
					DynamoTag:    "dynamocomponent:MyService2",
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
				ingressSpec: &v1alpha1.IngressSpec{
					Enabled: true,
					Host:    "test-dynamographdeployment",
				},
			},
			want: map[string]*v1alpha1.DynamoComponentDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service1",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "service1",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponent: "dynamocomponent:ac4e234",
						DynamoTag:       "dynamocomponent:MyService2",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
								Enabled: true,
								Host:    "test-dynamographdeployment",
							},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
							},
						},
					},
				},
				"service2": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service2",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "service2",
							commonconsts.KubeLabelDynamoNamespace: "default",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponent: "dynamocomponent:ac4e234",
						DynamoTag:       "dynamocomponent:MyService2",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default"}[0],
							Autoscaling: &v1alpha1.Autoscaling{
								Enabled: false,
							},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
								commonconsts.KubeLabelDynamoNamespace: "default",
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments dynamo dependency, default namespace",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						DynamoGraph: "dynamocomponent:ac4e234",
					},
				},
				config: &DynamoGraphConfig{
					DynamoTag:    "dynamocomponent:MyService2",
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
									Enabled: true,
									Name:    "service2",
								},
							},
						},
					},
				},
				ingressSpec: &v1alpha1.IngressSpec{},
			},
			want: map[string]*v1alpha1.DynamoComponentDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service1",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "service1",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponent: "dynamocomponent:ac4e234",
						DynamoTag:       "dynamocomponent:MyService2",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
									DeploymentSelectorValue: "service2/dynamo-test-dynamographdeployment",
								},
							},
							Ingress: v1alpha1.IngressSpec{},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
							},
						},
					},
				},
				"service2": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service2",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "service2",
							commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponent: "dynamocomponent:ac4e234",
						DynamoTag:       "dynamocomponent:MyService2",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "service2",
							Autoscaling: &v1alpha1.Autoscaling{
								Enabled: false,
							},
							DynamoNamespace: &[]string{"dynamo-test-dynamographdeployment"}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
								commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments dependency not found",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						DynamoGraph: "dynamocomponent:ac4e234",
					},
				},
				config: &DynamoGraphConfig{
					DynamoTag: "dynamocomponent:MyService3",
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
				ingressSpec: &v1alpha1.IngressSpec{},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			got, err := GenerateDynamoComponentsDeployments(context.Background(), tt.args.parentDynamoGraphDeployment, tt.args.config, tt.args.ingressSpec)
			if (err != nil) != tt.wantErr {
				t.Errorf("GenerateDynamoComponentsDeployments() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			g.Expect(got).To(gomega.Equal(tt.want))
		})
	}
}
