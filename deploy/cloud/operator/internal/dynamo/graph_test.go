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
	"reflect"
	"sort"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	compounaiCommon "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/google/go-cmp/cmp"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
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
									CPU:    &[]string{"1"}[0],
									Memory: &[]string{"1Gi"}[0],
									GPU:    &[]string{"0"}[0],
									Custom: map[string]string{},
								},
								Autoscaling: &Autoscaling{
									MinReplicas: 1,
									MaxReplicas: 5,
								},
								Workers: &[]int32{3}[0],
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
							Replicas:        &[]int32{3}[0],
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
								HttpExposed: true,
								Resources: &Resources{
									CPU:    &[]string{"1"}[0],
									Memory: &[]string{"1Gi"}[0],
									GPU:    &[]string{"0"}[0],
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
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions:  nil,
						PodSelector: nil,
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
							Ingress: v1alpha1.IngressSpec{
								Enabled:                    false,
								Host:                       "",
								UseVirtualService:          false,
								VirtualServiceGateway:      nil,
								HostPrefix:                 nil,
								Annotations:                nil,
								Labels:                     nil,
								TLS:                        nil,
								HostSuffix:                 nil,
								IngressControllerClassName: nil,
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments with config override from parent deployment",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						DynamoGraph: "dynamocomponent:ac4e234",
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"service1": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "10",
											Memory: "10Gi",
										},
									},
								},
							},
						},
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
								HttpExposed: true,
								Resources: &Resources{
									CPU:    &[]string{"1"}[0],
									Memory: &[]string{"1Gi"}[0],
									GPU:    &[]string{"0"}[0],
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
									CPU:    "10",
									Memory: "10Gi",
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
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions:  nil,
						PodSelector: nil,
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
							Ingress: v1alpha1.IngressSpec{
								Enabled:                    false,
								Host:                       "",
								UseVirtualService:          false,
								VirtualServiceGateway:      nil,
								HostPrefix:                 nil,
								Annotations:                nil,
								Labels:                     nil,
								TLS:                        nil,
								HostSuffix:                 nil,
								IngressControllerClassName: nil,
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments generate config from DYN_DEPLOYMENT_CONFIG env var",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						DynamoGraph: "dynamocomponent:ac4e234",
						Envs: []corev1.EnvVar{
							{
								Name:  "DYN_DEPLOYMENT_CONFIG",
								Value: `{"service1":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`,
							},
						},
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"service1": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "10",
											Memory: "10Gi",
										},
									},
								},
							},
						},
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
								HttpExposed: true,
								Resources: &Resources{
									CPU:    &[]string{"1"}[0],
									Memory: &[]string{"1Gi"}[0],
									GPU:    &[]string{"0"}[0],
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
							Envs: []corev1.EnvVar{
								{
									Name:  "DYN_DEPLOYMENT_CONFIG",
									Value: `{"service1":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`,
								},
							},
							ServiceName: "service1",
							Replicas:    &[]int32{3}[0],
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
									CPU:    "2",
									Memory: "2Gi",
									GPU:    "2",
									Custom: map[string]string{},
								},
								Limits: &compounaiCommon.ResourceItem{
									CPU:    "2",
									Memory: "2Gi",
									GPU:    "2",
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
						Config: &apiextensionsv1.JSON{
							Raw: []byte(`{"service1":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions:  nil,
						PodSelector: nil,
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
							Envs: []corev1.EnvVar{
								{
									Name:  "DYN_DEPLOYMENT_CONFIG",
									Value: `{"service1":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`,
								},
							},
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default"}[0],
							Autoscaling: &v1alpha1.Autoscaling{
								Enabled: false,
							},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
								commonconsts.KubeLabelDynamoNamespace: "default",
							},
							Ingress: v1alpha1.IngressSpec{
								Enabled:                    false,
								Host:                       "",
								UseVirtualService:          false,
								VirtualServiceGateway:      nil,
								HostPrefix:                 nil,
								Annotations:                nil,
								Labels:                     nil,
								TLS:                        nil,
								HostSuffix:                 nil,
								IngressControllerClassName: nil,
							},
						},
						Config: &apiextensionsv1.JSON{
							Raw: []byte(`{"service1":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`),
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments, number of replicas always set by the parent CR",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						DynamoGraph: "dynamocomponent:ac4e234",
						Envs: []corev1.EnvVar{
							{
								Name:  "DYN_DEPLOYMENT_CONFIG",
								Value: `{"service1":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`,
							},
						},
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"service1": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "10",
											Memory: "10Gi",
										},
									},
									Replicas: &[]int32{10}[0],
								},
							},
						},
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
								HttpExposed: true,
								Resources: &Resources{
									CPU:    &[]string{"1"}[0],
									Memory: &[]string{"1Gi"}[0],
									GPU:    &[]string{"0"}[0],
									Custom: map[string]string{},
								},
								Autoscaling: &Autoscaling{
									MinReplicas: 1,
									MaxReplicas: 5,
								},
								Workers: &[]int32{2}[0],
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
							Envs: []corev1.EnvVar{
								{
									Name:  "DYN_DEPLOYMENT_CONFIG",
									Value: `{"service1":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`,
								},
							},
							ServiceName: "service1",
							Replicas:    &[]int32{10}[0],
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
									CPU:    "2",
									Memory: "2Gi",
									GPU:    "2",
									Custom: map[string]string{},
								},
								Limits: &compounaiCommon.ResourceItem{
									CPU:    "2",
									Memory: "2Gi",
									GPU:    "2",
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
						Config: &apiextensionsv1.JSON{
							Raw: []byte(`{"service1":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`),
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions:  nil,
						PodSelector: nil,
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
							Envs: []corev1.EnvVar{
								{
									Name:  "DYN_DEPLOYMENT_CONFIG",
									Value: `{"service1":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`,
								},
							},
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default"}[0],
							Autoscaling: &v1alpha1.Autoscaling{
								Enabled: false,
							},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
								commonconsts.KubeLabelDynamoNamespace: "default",
							},
							Ingress: v1alpha1.IngressSpec{
								Enabled:                    false,
								Host:                       "",
								UseVirtualService:          false,
								VirtualServiceGateway:      nil,
								HostPrefix:                 nil,
								Annotations:                nil,
								Labels:                     nil,
								TLS:                        nil,
								HostSuffix:                 nil,
								IngressControllerClassName: nil,
							},
						},
						Config: &apiextensionsv1.JSON{
							Raw: []byte(`{"service1":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`),
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
								HttpExposed: true,
								Resources: &Resources{
									CPU:    &[]string{"1"}[0],
									Memory: &[]string{"1Gi"}[0],
									GPU:    &[]string{"0"}[0],
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
							Ingress: v1alpha1.IngressSpec{
								Enabled:                    false,
								Host:                       "",
								UseVirtualService:          false,
								VirtualServiceGateway:      nil,
								HostPrefix:                 nil,
								Annotations:                nil,
								Labels:                     nil,
								TLS:                        nil,
								HostSuffix:                 nil,
								IngressControllerClassName: nil,
							},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
							},
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions:  nil,
						PodSelector: nil,
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
							Ingress: v1alpha1.IngressSpec{
								Enabled:                    false,
								Host:                       "",
								UseVirtualService:          false,
								VirtualServiceGateway:      nil,
								HostPrefix:                 nil,
								Annotations:                nil,
								Labels:                     nil,
								TLS:                        nil,
								HostSuffix:                 nil,
								IngressControllerClassName: nil,
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
									CPU:    &[]string{"1"}[0],
									Memory: &[]string{"1Gi"}[0],
									GPU:    &[]string{"0"}[0],
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
		{
			name: "Test GenerateDynamoComponentsDeployments planner",
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
							Name: "service1",
							Config: Config{
								Dynamo: &DynamoConfig{
									Enabled:       true,
									Namespace:     "default",
									Name:          "service1",
									ComponentType: ComponentTypePlanner,
								},
								Resources: &Resources{
									CPU:    &[]string{"1"}[0],
									Memory: &[]string{"1Gi"}[0],
									GPU:    &[]string{"0"}[0],
									Custom: map[string]string{},
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
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
								commonconsts.KubeLabelDynamoNamespace: "default",
							},
							ExtraPodSpec: &compounaiCommon.ExtraPodSpec{
								ServiceAccountName: PlannerServiceAccountName,
							},
							Autoscaling: &v1alpha1.Autoscaling{},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments dynamo dependency, different namespace",
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
								Dynamo: &DynamoConfig{
									Enabled:   true,
									Namespace: "namespace1",
									Name:      "service1",
								},
								Resources: &Resources{
									CPU:    &[]string{"1"}[0],
									Memory: &[]string{"1Gi"}[0],
									GPU:    &[]string{"0"}[0],
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
									Namespace: "namespace2",
									Name:      "service2",
								},
							},
						},
					},
				},
				ingressSpec: &v1alpha1.IngressSpec{},
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments ingress enabled by default",
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
					DynamoTag:    "dynamocomponent:MyServiceIngressEnabled",
					EntryService: "service1",
					Services: []ServiceConfig{
						{
							Name: "service1",
							Config: Config{
								HttpExposed: true,
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
						DynamoTag:       "dynamocomponent:MyServiceIngressEnabled",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							Annotations: nil,
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
							},
							ServiceName:     "service1",
							DynamoNamespace: nil,
							Resources:       nil,
							Autoscaling: &v1alpha1.Autoscaling{
								Enabled:     false,
								MinReplicas: 0,
								MaxReplicas: 0,
								Behavior:    nil,
								Metrics:     nil,
							},
							Envs:             nil,
							EnvFromSecret:    nil,
							PVC:              nil,
							RunMode:          nil,
							ExternalServices: nil,
							Ingress: v1alpha1.IngressSpec{
								Enabled: true,
								Host:    "test-dynamographdeployment",
							},
							ExtraPodMetadata: nil,
							ExtraPodSpec:     nil,
							LivenessProbe:    nil,
							ReadinessProbe:   nil,
							Replicas:         nil,
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions:  nil,
						PodSelector: nil,
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments ingress explicitly disabled",
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
					DynamoTag: "dynamocomponent:MyServiceIngressDisabled",
					Services: []ServiceConfig{
						{
							Name:   "service1",
							Config: Config{},
						},
					},
				},
				ingressSpec: &v1alpha1.IngressSpec{
					Enabled: false,
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
						DynamoTag:       "dynamocomponent:MyServiceIngressDisabled",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							Annotations: nil,
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
							},
							ServiceName:     "service1",
							DynamoNamespace: nil,
							Resources:       nil,
							Autoscaling: &v1alpha1.Autoscaling{
								Enabled:     false,
								MinReplicas: 0,
								MaxReplicas: 0,
								Behavior:    nil,
								Metrics:     nil,
							},
							Envs:             nil,
							EnvFromSecret:    nil,
							PVC:              nil,
							RunMode:          nil,
							ExternalServices: nil,
							Ingress: v1alpha1.IngressSpec{
								Enabled:                    false,
								Host:                       "",
								UseVirtualService:          false,
								VirtualServiceGateway:      nil,
								HostPrefix:                 nil,
								Annotations:                nil,
								Labels:                     nil,
								TLS:                        nil,
								HostSuffix:                 nil,
								IngressControllerClassName: nil,
							},
							ExtraPodMetadata: nil,
							ExtraPodSpec:     nil,
							LivenessProbe:    nil,
							ReadinessProbe:   nil,
							Replicas:         nil,
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions:  nil,
						PodSelector: nil,
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments ingress custom host",
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
					DynamoTag:    "dynamocomponent:MyServiceIngressCustomHost",
					EntryService: "service1",
					Services: []ServiceConfig{
						{
							Name: "service1",
							Config: Config{
								HttpExposed: true,
							},
						},
					},
				},
				ingressSpec: &v1alpha1.IngressSpec{
					Enabled: true,
					Host:    "custom-host",
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
						DynamoTag:       "dynamocomponent:MyServiceIngressCustomHost",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							Annotations: nil,
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
							},
							ServiceName:     "service1",
							DynamoNamespace: nil,
							Resources:       nil,
							Autoscaling: &v1alpha1.Autoscaling{
								Enabled:     false,
								MinReplicas: 0,
								MaxReplicas: 0,
								Behavior:    nil,
								Metrics:     nil,
							},
							Envs:             nil,
							EnvFromSecret:    nil,
							PVC:              nil,
							RunMode:          nil,
							ExternalServices: nil,
							Ingress: v1alpha1.IngressSpec{
								Enabled: true,
								Host:    "custom-host",
							},
							ExtraPodMetadata: nil,
							ExtraPodSpec:     nil,
							LivenessProbe:    nil,
							ReadinessProbe:   nil,
							Replicas:         nil,
						},
					},
					Status: v1alpha1.DynamoComponentDeploymentStatus{
						Conditions:  nil,
						PodSelector: nil,
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GenerateDynamoComponentsDeployments(context.Background(), tt.args.parentDynamoGraphDeployment, tt.args.config, tt.args.ingressSpec)
			if (err != nil) != tt.wantErr {
				t.Errorf("GenerateDynamoComponentsDeployments() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("GenerateDynamoComponentsDeployments() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestSetLwsAnnotations(t *testing.T) {
	type args struct {
		serviceArgs *ServiceArgs
		deployment  *v1alpha1.DynamoComponentDeployment
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
		want    *v1alpha1.DynamoComponentDeployment
	}{
		{
			name: "Test SetLwsAnnotations for 16 GPUs",
			args: args{
				serviceArgs: &ServiceArgs{
					Resources: &Resources{
						GPU: &[]string{"8"}[0],
					},
					TotalGpus: &[]int32{16}[0],
				},
				deployment: &v1alpha1.DynamoComponentDeployment{},
			},
			wantErr: false,
			want: &v1alpha1.DynamoComponentDeployment{
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						Annotations: map[string]string{
							"nvidia.com/deployment-type": "leader-worker",
							"nvidia.com/lws-size":        "2",
						},
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := SetLwsAnnotations(tt.args.serviceArgs, tt.args.deployment); (err != nil) != tt.wantErr {
				t.Errorf("SetLwsAnnotations() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func Test_updateDynDeploymentConfig(t *testing.T) {
	type args struct {
		dynamoDeploymentComponent *nvidiacomv1alpha1.DynamoComponentDeployment
		newPort                   int
	}
	tests := []struct {
		name    string
		args    args
		want    *apiextensionsv1.JSON
		wantErr bool
	}{
		{
			name: "main component",
			args: args{
				dynamoDeploymentComponent: &nvidiacomv1alpha1.DynamoComponentDeployment{
					Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
						DynamoTag: "graphs.agg:Frontend",
						DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "Frontend",
							Envs: []corev1.EnvVar{
								{
									Name:  "OTHER",
									Value: `value`,
								},
							},
						},
						Config: &apiextensionsv1.JSON{
							Raw: []byte(`{"Frontend":{"port":8080},"Planner":{"environment":"kubernetes"}}`),
						},
					},
				},
				newPort: 3000,
			},
			want: &apiextensionsv1.JSON{
				Raw: []byte(`{"Frontend":{"port":3000},"Planner":{"environment":"kubernetes"}}`),
			},
			wantErr: false,
		},
		{
			name: "not main component",
			args: args{
				dynamoDeploymentComponent: &nvidiacomv1alpha1.DynamoComponentDeployment{
					Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
						DynamoTag: "graphs.agg:Frontend",
						DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "Other",
						},
						Config: &apiextensionsv1.JSON{
							Raw: []byte(`{"Frontend":{"port":8000},"Planner":{"environment":"kubernetes"}}`),
						},
					},
				},
				newPort: 3000,
			},
			want: &apiextensionsv1.JSON{
				Raw: []byte(`{"Frontend":{"port":8000},"Planner":{"environment":"kubernetes"}}`),
			},
			wantErr: false,
		},
		{
			name: "no config variable",
			args: args{
				dynamoDeploymentComponent: &nvidiacomv1alpha1.DynamoComponentDeployment{
					Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
						DynamoTag: "graphs.agg:Frontend",
						DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "Frontend",
							Envs: []corev1.EnvVar{
								{
									Name:  "OTHER",
									Value: `value`,
								},
							},
						},
						Config: nil,
					},
				},
				newPort: 8080,
			},
			want:    nil,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := updateDynDeploymentConfig(tt.args.dynamoDeploymentComponent, tt.args.newPort)
			if (err != nil) != tt.wantErr {
				t.Errorf("updateDynDeploymentConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			g := gomega.NewGomegaWithT(t)
			g.Expect(tt.args.dynamoDeploymentComponent.Spec.Config).To(gomega.Equal(tt.want))
		})
	}
}

func Test_overrideWithDynDeploymentConfig(t *testing.T) {
	type args struct {
		ctx                       context.Context
		dynamoDeploymentComponent *nvidiacomv1alpha1.DynamoComponentDeployment
	}
	tests := []struct {
		name     string
		args     args
		wantErr  bool
		expected *nvidiacomv1alpha1.DynamoComponentDeployment
	}{
		{
			name: "no env var",
			args: args{
				ctx: context.Background(),
				dynamoDeploymentComponent: &nvidiacomv1alpha1.DynamoComponentDeployment{
					Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "Frontend",
							Replicas:    &[]int32{1}[0],
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "1",
								},
							},
						},
					},
				},
			},
			wantErr: false,
			expected: &nvidiacomv1alpha1.DynamoComponentDeployment{
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						ServiceName: "Frontend",
						Replicas:    &[]int32{1}[0],
						Resources: &common.Resources{
							Requests: &common.ResourceItem{
								CPU:    "1",
								Memory: "1Gi",
								GPU:    "1",
							},
						},
					},
				},
			},
		},
		{
			name: "override workers and resources",
			args: args{
				ctx: context.Background(),
				dynamoDeploymentComponent: &nvidiacomv1alpha1.DynamoComponentDeployment{
					Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "Frontend",
							Replicas:    &[]int32{1}[0],
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "1",
								},
							},
						},
						Config: &apiextensionsv1.JSON{
							Raw: []byte(`{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}},"Planner":{"environment":"kubernetes"}}`),
						},
					},
				},
			},
			wantErr: false,
			expected: &nvidiacomv1alpha1.DynamoComponentDeployment{
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						ServiceName: "Frontend",
						Replicas:    &[]int32{3}[0],
						Resources: &common.Resources{
							Requests: &common.ResourceItem{
								CPU:    "2",
								Memory: "2Gi",
								GPU:    "2",
							},
							Limits: &common.ResourceItem{
								CPU:    "2",
								Memory: "2Gi",
								GPU:    "2",
							},
						},
					},
					Config: &apiextensionsv1.JSON{
						Raw: []byte(`{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}},"Planner":{"environment":"kubernetes"}}`),
					},
				},
			},
		},
		{
			name: "override workers and resources with gpusPerNode",
			args: args{
				ctx: context.Background(),
				dynamoDeploymentComponent: &nvidiacomv1alpha1.DynamoComponentDeployment{
					Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "Frontend",
							Replicas:    nil,
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "1",
								},
							},
						},
						Config: &apiextensionsv1.JSON{
							Raw: []byte(`{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"8"}, "total_gpus":16}},"Planner":{"environment":"kubernetes"}}`),
						},
					},
				},
			},
			wantErr: false,
			expected: &nvidiacomv1alpha1.DynamoComponentDeployment{
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						ServiceName: "Frontend",
						Replicas:    &[]int32{3}[0],
						Resources: &common.Resources{
							Requests: &common.ResourceItem{
								CPU:    "2",
								Memory: "2Gi",
								GPU:    "8",
							},
							Limits: &common.ResourceItem{
								CPU:    "2",
								Memory: "2Gi",
								GPU:    "8",
							},
						},
						Annotations: map[string]string{
							"nvidia.com/deployment-type": "leader-worker",
							"nvidia.com/lws-size":        "2",
						},
					},
					Config: &apiextensionsv1.JSON{
						Raw: []byte(`{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"8"}, "total_gpus":16}},"Planner":{"environment":"kubernetes"}}`),
					},
				},
			},
		},
		{
			name: "override subset of resources",
			args: args{
				ctx: context.Background(),
				dynamoDeploymentComponent: &nvidiacomv1alpha1.DynamoComponentDeployment{
					Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName: "Frontend",
							Replicas:    nil,
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "1",
								},
							},
						},
						Config: &apiextensionsv1.JSON{
							Raw: []byte(`{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"GPU":"2"}}},"Planner":{"environment":"kubernetes"}}`),
						},
					},
				},
			},
			wantErr: false,
			expected: &nvidiacomv1alpha1.DynamoComponentDeployment{
				Spec: nvidiacomv1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
						ServiceName: "Frontend",
						Replicas:    &[]int32{3}[0],
						Resources: &common.Resources{
							Requests: &common.ResourceItem{
								CPU:    "1",
								Memory: "1Gi",
								GPU:    "2",
							},
							Limits: &common.ResourceItem{
								CPU:    "",
								Memory: "",
								GPU:    "2",
							},
						},
					},
					Config: &apiextensionsv1.JSON{
						Raw: []byte(`{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"GPU":"2"}}},"Planner":{"environment":"kubernetes"}}`),
					},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := overrideWithDynDeploymentConfig(tt.args.ctx, tt.args.dynamoDeploymentComponent); (err != nil) != tt.wantErr {
				t.Errorf("overrideWithDynDeploymentConfig() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.args.dynamoDeploymentComponent, tt.expected); diff != "" {
				t.Errorf("overrideWithDynDeploymentConfig() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func Test_mergeEnvs(t *testing.T) {
	type args struct {
		common   []corev1.EnvVar
		specific []corev1.EnvVar
	}
	tests := []struct {
		name string
		args args
		want []corev1.EnvVar
	}{
		{
			name: "no_common_envs",
			args: args{
				common:   []corev1.EnvVar{},
				specific: []corev1.EnvVar{{Name: "FOO", Value: "BAR"}},
			},
			want: []corev1.EnvVar{{Name: "FOO", Value: "BAR"}},
		},
		{
			name: "no_specific_envs",
			args: args{
				common:   []corev1.EnvVar{{Name: "FOO", Value: "BAR"}},
				specific: []corev1.EnvVar{},
			},
			want: []corev1.EnvVar{{Name: "FOO", Value: "BAR"}},
		},
		{
			name: "common_and_specific_envs",
			args: args{
				specific: []corev1.EnvVar{{Name: "BAZ", Value: "QUX"}},
				common:   []corev1.EnvVar{{Name: "FOO", Value: "BAR"}},
			},
			want: []corev1.EnvVar{{Name: "BAZ", Value: "QUX"}, {Name: "FOO", Value: "BAR"}},
		},
		{
			name: "common_and_specific_envs_with_same_name",
			args: args{
				common:   []corev1.EnvVar{{Name: "FOO", Value: "BAR"}},
				specific: []corev1.EnvVar{{Name: "FOO", Value: "QUX"}},
			},
			want: []corev1.EnvVar{{Name: "FOO", Value: "QUX"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := mergeEnvs(tt.args.common, tt.args.specific)
			sort.Slice(got, func(i, j int) bool {
				return got[i].Name < got[j].Name
			})
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("mergeEnvs() = %v, want %v", got, tt.want)
			}
		})
	}
}
