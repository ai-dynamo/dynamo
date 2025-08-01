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
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"

	grovev1alpha1 "github.com/NVIDIA/grove/operator/api/core/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	compounaiCommon "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestGenerateDynamoComponentsDeployments(t *testing.T) {
	type args struct {
		parentDynamoGraphDeployment *v1alpha1.DynamoGraphDeployment
		ingressSpec                 *v1alpha1.IngressSpec
	}
	tests := []struct {
		name    string
		args    args
		want    map[string]*v1alpha1.DynamoComponentDeployment
		wantErr bool
	}{
		{
			name: "Test GenerateDynamoComponentsDeployments",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"service1": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: &[]string{"default"}[0],
									ComponentType:   "main",
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
								},
							},
							"service2": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: &[]string{"default"}[0],
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
								},
							},
						},
					},
				},
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
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default"}[0],
							ComponentType:   "main",
							Replicas:        &[]int32{3}[0],
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
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
							Autoscaling: nil,
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
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
								commonconsts.KubeLabelDynamoNamespace: "default",
							},
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
							},
							Autoscaling: nil,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments with default dynamo namespace",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"service1": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: nil,
									ComponentType:   "main",
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
								},
							},
							"service2": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: nil,
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
								},
							},
						},
					},
				},
			},
			want: map[string]*v1alpha1.DynamoComponentDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service1",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "service1",
							commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"dynamo-test-dynamographdeployment"}[0],
							ComponentType:   "main",
							Replicas:        &[]int32{3}[0],
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
							},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
								commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
							},
							Autoscaling: nil,
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
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"dynamo-test-dynamographdeployment"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
								commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
							},
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
							},
							Autoscaling: nil,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments with different namespaces",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"service1": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: &[]string{"default"}[0],
									ComponentType:   "main",
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
								},
							},
							"service2": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: &[]string{"another"}[0],
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
								},
							},
						},
					},
				},
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments with ingress enabled",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"service1": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: nil,
									ComponentType:   "main",
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
								},
							},
							"service2": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: nil,
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
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
							commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"dynamo-test-dynamographdeployment"}[0],
							ComponentType:   "main",
							Replicas:        &[]int32{3}[0],
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
							},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
								commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
							},
							Autoscaling: nil,
							Ingress: &v1alpha1.IngressSpec{
								Enabled: true,
								Host:    "test-dynamographdeployment",
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
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"dynamo-test-dynamographdeployment"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
								commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
							},
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
							},
							Autoscaling: nil,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments with config from DYN_DEPLOYMENT_CONFIG env var",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						Envs: []corev1.EnvVar{
							{
								Name:  "DYN_DEPLOYMENT_CONFIG",
								Value: `{"service1":{"port":8080,"ServiceArgs":{"Workers":2, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`,
							},
						},
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"service1": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: nil,
									ComponentType:   "main",
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
								},
							},
							"service2": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: nil,
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
								},
							},
						},
					},
				},
			},
			want: map[string]*v1alpha1.DynamoComponentDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service1",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "service1",
							commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"dynamo-test-dynamographdeployment"}[0],
							ComponentType:   "main",
							Replicas:        &[]int32{3}[0],
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
									Custom: nil,
								},
							},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service1",
								commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
							},
							Autoscaling: nil,
							Envs: []corev1.EnvVar{
								{
									Name:  "DYN_DEPLOYMENT_CONFIG",
									Value: fmt.Sprintf(`{"service1":{"ServiceArgs":{"Resources":{"CPU":"2","GPU":"2","Memory":"2Gi"},"Workers":2},"port":%d}}`, commonconsts.DynamoServicePort),
								},
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
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"dynamo-test-dynamographdeployment"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
								commonconsts.KubeLabelDynamoNamespace: "dynamo-test-dynamographdeployment",
							},
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
							},
							Autoscaling: nil,
							Envs: []corev1.EnvVar{
								{
									Name:  "DYN_DEPLOYMENT_CONFIG",
									Value: `{"service1":{"port":8080,"ServiceArgs":{"Workers":2, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}}}`,
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments with ExtraPodSpec.MainContainer Command and Args",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"service1": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: &[]string{"default"}[0],
									ComponentType:   "main",
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
									ExtraPodSpec: &compounaiCommon.ExtraPodSpec{
										MainContainer: &corev1.Container{
											Command: []string{"sh", "-c"},
											Args:    []string{"echo hello world", "sleep 99999"},
										},
									},
								},
							},
							"service2": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: &[]string{"default"}[0],
									Replicas:        &[]int32{3}[0],
									Resources: &compounaiCommon.Resources{
										Requests: &compounaiCommon.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
								},
							},
						},
					},
				},
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
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default"}[0],
							ComponentType:   "main",
							Replicas:        &[]int32{3}[0],
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
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
							Autoscaling: nil,
							ExtraPodSpec: &compounaiCommon.ExtraPodSpec{
								MainContainer: &corev1.Container{
									Command: []string{"sh", "-c"},
									Args:    []string{"echo hello world", "sleep 99999"},
								},
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
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
								commonconsts.KubeLabelDynamoNamespace: "default",
							},
							Resources: &compounaiCommon.Resources{
								Requests: &compounaiCommon.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
							},
							Autoscaling: nil,
						},
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GenerateDynamoComponentsDeployments(context.Background(), tt.args.parentDynamoGraphDeployment, tt.args.ingressSpec)
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
		want    []byte
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
								{
									Name:  commonconsts.DynamoDeploymentConfigEnvVar,
									Value: `{"Frontend":{"port":8080},"Planner":{"environment":"kubernetes"}}`,
								},
							},
						},
					},
				},
				newPort: commonconsts.DynamoServicePort,
			},
			want:    []byte(fmt.Sprintf(`{"Frontend":{"port":%d},"Planner":{"environment":"kubernetes"}}`, commonconsts.DynamoServicePort)),
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
							Envs: []corev1.EnvVar{
								{
									Name:  commonconsts.DynamoDeploymentConfigEnvVar,
									Value: `{"Frontend":{"port":8000},"Planner":{"environment":"kubernetes"}}`,
								},
							},
						},
					},
				},
				newPort: commonconsts.DynamoServicePort,
			},
			want:    []byte(fmt.Sprintf(`{"Frontend":{"port":%d},"Planner":{"environment":"kubernetes"}}`, commonconsts.DynamoServicePort)),
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
			if diff := cmp.Diff(tt.args.dynamoDeploymentComponent.GetDynamoDeploymentConfig(), tt.want); diff != "" {
				t.Errorf("updateDynDeploymentConfig() mismatch (-want +got):\n%s", diff)
			}
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
							Envs: []corev1.EnvVar{
								{
									Name:  commonconsts.DynamoDeploymentConfigEnvVar,
									Value: `{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}},"Planner":{"environment":"kubernetes"}}`,
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
						Envs: []corev1.EnvVar{
							{
								Name:  commonconsts.DynamoDeploymentConfigEnvVar,
								Value: `{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"2"}}},"Planner":{"environment":"kubernetes"}}`,
							},
						},
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
							Envs: []corev1.EnvVar{
								{
									Name:  commonconsts.DynamoDeploymentConfigEnvVar,
									Value: `{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"8"}, "total_gpus":16}},"Planner":{"environment":"kubernetes"}}`,
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
						Envs: []corev1.EnvVar{
							{
								Name:  commonconsts.DynamoDeploymentConfigEnvVar,
								Value: `{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"CPU":"2", "Memory":"2Gi", "GPU":"8"}, "total_gpus":16}},"Planner":{"environment":"kubernetes"}}`,
							},
						},
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
							Envs: []corev1.EnvVar{
								{
									Name:  commonconsts.DynamoDeploymentConfigEnvVar,
									Value: `{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"GPU":"2"}}},"Planner":{"environment":"kubernetes"}}`,
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
						Envs: []corev1.EnvVar{
							{
								Name:  commonconsts.DynamoDeploymentConfigEnvVar,
								Value: `{"Frontend":{"port":8080,"ServiceArgs":{"Workers":3, "Resources":{"GPU":"2"}}},"Planner":{"environment":"kubernetes"}}`,
							},
						},
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
			got := MergeEnvs(tt.args.common, tt.args.specific)
			sort.Slice(got, func(i, j int) bool {
				return got[i].Name < got[j].Name
			})
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("mergeEnvs() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGenerateGrovePodGangSet(t *testing.T) {
	type args struct {
		ctx              context.Context
		dynamoDeployment *v1alpha1.DynamoGraphDeployment
		controllerConfig controller_common.Config
	}
	tests := []struct {
		name    string
		args    args
		want    *grovev1alpha1.PodGangSet
		wantErr bool
	}{
		{
			name: "test_generate_grove_pod_gang_set",
			args: args{
				ctx: context.Background(),
				controllerConfig: controller_common.Config{
					EtcdAddress: "etcd-address",
					NatsAddress: "nats-address",
					Grove: controller_common.GroveConfig{
						TerminationDelay: 15 * time.Minute,
					},
				},
				dynamoDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamo-graph-deployment",
						Namespace: "test-namespace",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						Envs: []corev1.EnvVar{
							{
								Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
								Value: "1",
							},
						},
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"Frontend": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									Replicas: &[]int32{1}[0],
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
										},
										Limits: &common.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "1",
										},
									},
									Envs: []corev1.EnvVar{
										{
											Name:  "FRONTEND_ENV_1",
											Value: "1",
										},
									},
									EnvFromSecret: &[]string{"frontend-secret"}[0],
									LivenessProbe: &corev1.Probe{
										ProbeHandler: corev1.ProbeHandler{
											HTTPGet: &corev1.HTTPGetAction{
												Path: "/health",
												Port: intstr.FromInt(8080),
											},
										},
									},
									ReadinessProbe: &corev1.Probe{
										ProbeHandler: corev1.ProbeHandler{
											HTTPGet: &corev1.HTTPGetAction{
												Path: "/ready",
												Port: intstr.FromInt(8080),
											},
										},
									},
									ExtraPodSpec: &common.ExtraPodSpec{
										MainContainer: &corev1.Container{
											Command: []string{
												"/bin/sh",
												"-c",
												"echo $FRONTEND_ENV_1",
											},
											Args: []string{
												"--frontend-env-1",
												"1",
											},
											Image: "frontend-image",
										},
									},
								},
							},
							"Planner": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									Replicas: &[]int32{2}[0],
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
											CPU:    "2",
											Memory: "2Gi",
										},
										Limits: &common.ResourceItem{
											CPU:    "2",
											Memory: "2Gi",
											GPU:    "2",
										},
									},
									Envs: []corev1.EnvVar{
										{
											Name:  "PLANNER_ENV_1",
											Value: "2",
										},
									},
									PVC: &v1alpha1.PVC{
										Name:       &[]string{"planner-pvc"}[0],
										MountPoint: &[]string{"/planner"}[0],
									},
									EnvFromSecret: &[]string{"planner-secret"}[0],
									LivenessProbe: &corev1.Probe{
										ProbeHandler: corev1.ProbeHandler{
											HTTPGet: &corev1.HTTPGetAction{
												Path: "/health",
												Port: intstr.FromInt(8080),
											},
										},
									},
									ReadinessProbe: &corev1.Probe{
										ProbeHandler: corev1.ProbeHandler{
											HTTPGet: &corev1.HTTPGetAction{
												Path: "/ready",
												Port: intstr.FromInt(8080),
											},
										},
									},
									ExtraPodSpec: &common.ExtraPodSpec{
										MainContainer: &corev1.Container{
											Command: []string{
												"/bin/sh",
												"-c",
												"echo $PLANNER_ENV_1",
											},
											Args: []string{
												"--planner-env-1",
												"1",
											},
											Image: "planner-image",
										},
									},
								},
							},
						},
					},
				},
			},
			want: &grovev1alpha1.PodGangSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dynamo-graph-deployment",
					Namespace: "test-namespace",
				},
				Spec: grovev1alpha1.PodGangSetSpec{
					Replicas: 1,
					Template: grovev1alpha1.PodGangSetTemplateSpec{
						TerminationDelay: &metav1.Duration{Duration: 15 * time.Minute},
						Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
							{
								Name: "frontend",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-frontend",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "frontend",
									Replicas: 1,
									PodSpec: corev1.PodSpec{
										ImagePullSecrets: []corev1.LocalObjectReference{},
										Containers: []corev1.Container{
											{
												Name:  "main",
												Image: "frontend-image",
												Command: []string{
													"/bin/sh",
													"-c",
													"echo $FRONTEND_ENV_1",
												},
												Args: []string{
													"--frontend-env-1",
													"1",
												},
												EnvFrom: []corev1.EnvFromSource{
													{
														SecretRef: &corev1.SecretEnvSource{
															LocalObjectReference: corev1.LocalObjectReference{
																Name: "frontend-secret",
															},
														},
													},
												},
												LivenessProbe: &corev1.Probe{
													ProbeHandler: corev1.ProbeHandler{
														HTTPGet: &corev1.HTTPGetAction{
															Path: "/health",
															Port: intstr.FromInt(8080),
														},
													},
												},
												ReadinessProbe: &corev1.Probe{
													ProbeHandler: corev1.ProbeHandler{
														HTTPGet: &corev1.HTTPGetAction{
															Path: "/ready",
															Port: intstr.FromInt(8080),
														},
													},
												},
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "FRONTEND_ENV_1",
														Value: "1",
													},
													{
														Name:  "DYNAMO_PORT",
														Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort),
													},
													{
														Name:  "NATS_SERVER",
														Value: "nats-address",
													},
													{
														Name:  "ETCD_ENDPOINTS",
														Value: "etcd-address",
													},
												},
												Resources: corev1.ResourceRequirements{
													Requests: corev1.ResourceList{
														corev1.ResourceCPU:    resource.MustParse("1"),
														corev1.ResourceMemory: resource.MustParse("1Gi"),
													},
													Limits: corev1.ResourceList{
														corev1.ResourceCPU:                    resource.MustParse("1"),
														corev1.ResourceMemory:                 resource.MustParse("1Gi"),
														corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
													},
												},
												Ports: []corev1.ContainerPort{
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoContainerPortName,
														ContainerPort: int32(commonconsts.DynamoServicePort),
													},
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoHealthPortName,
														ContainerPort: int32(commonconsts.DynamoHealthPort),
													},
												},
											},
										},
									},
								},
							},
							{
								Name: "planner",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-planner",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "planner",
									Replicas: 2,
									PodSpec: corev1.PodSpec{
										ImagePullSecrets: []corev1.LocalObjectReference{},
										Volumes: []corev1.Volume{
											{
												Name: "planner-pvc",
												VolumeSource: corev1.VolumeSource{
													PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
														ClaimName: "planner-pvc",
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  "main",
												Image: "planner-image",
												Command: []string{
													"/bin/sh",
													"-c",
													"echo $PLANNER_ENV_1",
												},
												Args: []string{
													"--planner-env-1",
													"1",
												},
												EnvFrom: []corev1.EnvFromSource{
													{
														SecretRef: &corev1.SecretEnvSource{
															LocalObjectReference: corev1.LocalObjectReference{
																Name: "planner-secret",
															},
														},
													},
												},
												LivenessProbe: &corev1.Probe{
													ProbeHandler: corev1.ProbeHandler{
														HTTPGet: &corev1.HTTPGetAction{
															Path: "/health",
															Port: intstr.FromInt(8080),
														},
													},
												},
												ReadinessProbe: &corev1.Probe{
													ProbeHandler: corev1.ProbeHandler{
														HTTPGet: &corev1.HTTPGetAction{
															Path: "/ready",
															Port: intstr.FromInt(8080),
														},
													},
												},
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "PLANNER_ENV_1",
														Value: "2",
													},
													{
														Name:  "DYNAMO_PORT",
														Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort),
													},
													{
														Name:  "NATS_SERVER",
														Value: "nats-address",
													},
													{
														Name:  "ETCD_ENDPOINTS",
														Value: "etcd-address",
													},
												},
												Resources: corev1.ResourceRequirements{
													Requests: corev1.ResourceList{
														corev1.ResourceCPU:    resource.MustParse("2"),
														corev1.ResourceMemory: resource.MustParse("2Gi"),
													},
													Limits: corev1.ResourceList{
														corev1.ResourceCPU:                    resource.MustParse("2"),
														corev1.ResourceMemory:                 resource.MustParse("2Gi"),
														corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("2"),
													},
												},
												VolumeMounts: []corev1.VolumeMount{
													{
														Name:      "planner-pvc",
														MountPath: "/planner",
													},
												},
												Ports: []corev1.ContainerPort{
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoContainerPortName,
														ContainerPort: int32(commonconsts.DynamoServicePort),
													},
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoHealthPortName,
														ContainerPort: int32(commonconsts.DynamoHealthPort),
													},
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GenerateGrovePodGangSet(tt.args.ctx, tt.args.dynamoDeployment, tt.args.controllerConfig, nil)
			if (err != nil) != tt.wantErr {
				t.Errorf("GenerateGrovePodGangSet() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			sort.Slice(got.Spec.Template.Cliques, func(i, j int) bool {
				return got.Spec.Template.Cliques[i].Name < got.Spec.Template.Cliques[j].Name
			})
			sort.Slice(tt.want.Spec.Template.Cliques, func(i, j int) bool {
				return tt.want.Spec.Template.Cliques[i].Name < tt.want.Spec.Template.Cliques[j].Name
			})
			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("GenerateGrovePodGangSet() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
