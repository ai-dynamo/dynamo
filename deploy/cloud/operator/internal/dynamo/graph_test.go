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
	"strings"
	"testing"
	"time"

	grovev1alpha1 "github.com/NVIDIA/grove/operator/api/core/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	ptr "k8s.io/utils/ptr"
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
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
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
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
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
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
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
									CPU:    "2",
									Memory: "2Gi",
									GPU:    "2",
									Custom: map[string]string{},
								},
								Limits: &common.ResourceItem{
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
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
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
						BackendFramework: string(BackendFrameworkSGLang),
						Services: map[string]*v1alpha1.DynamoComponentDeploymentOverridesSpec{
							"service1": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									DynamoNamespace: &[]string{"default"}[0],
									ComponentType:   "main",
									Replicas:        &[]int32{3}[0],
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
											CPU:    "1",
											Memory: "1Gi",
											GPU:    "0",
											Custom: map[string]string{},
										},
									},
									ExtraPodSpec: &common.ExtraPodSpec{
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
									Resources: &common.Resources{
										Requests: &common.ResourceItem{
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
						BackendFramework: string(BackendFrameworkSGLang),
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default"}[0],
							ComponentType:   "main",
							Replicas:        &[]int32{3}[0],
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
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
							ExtraPodSpec: &common.ExtraPodSpec{
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
						BackendFramework: string(BackendFrameworkSGLang),
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent: "service2",
								commonconsts.KubeLabelDynamoNamespace: "default",
							},
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
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
		dynamoDeploymentComponent *v1alpha1.DynamoComponentDeployment
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
				dynamoDeploymentComponent: &v1alpha1.DynamoComponentDeployment{
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoTag: "graphs.agg:Frontend",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
				dynamoDeploymentComponent: &v1alpha1.DynamoComponentDeployment{
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoTag: "graphs.agg:Frontend",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
				dynamoDeploymentComponent: &v1alpha1.DynamoComponentDeployment{
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoTag: "graphs.agg:Frontend",
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
		dynamoDeploymentComponent *v1alpha1.DynamoComponentDeployment
	}
	tests := []struct {
		name     string
		args     args
		wantErr  bool
		expected *v1alpha1.DynamoComponentDeployment
	}{
		{
			name: "no env var",
			args: args{
				ctx: context.Background(),
				dynamoDeploymentComponent: &v1alpha1.DynamoComponentDeployment{
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
			expected: &v1alpha1.DynamoComponentDeployment{
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
				dynamoDeploymentComponent: &v1alpha1.DynamoComponentDeployment{
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
			expected: &v1alpha1.DynamoComponentDeployment{
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
				dynamoDeploymentComponent: &v1alpha1.DynamoComponentDeployment{
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
			expected: &v1alpha1.DynamoComponentDeployment{
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
				dynamoDeploymentComponent: &v1alpha1.DynamoComponentDeployment{
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
			expected: &v1alpha1.DynamoComponentDeployment{
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
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
			name: "test_generate_grove_pod_gang_set_single_node",
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
									ExtraPodMetadata: &common.ExtraPodMetadata{
										Annotations: map[string]string{
											"nvidia.com/annotation1": "annotation1",
											"nvidia.com/annotation2": "annotation2",
										},
										Labels: map[string]string{
											"nvidia.com/label1": "label1",
											"nvidia.com/label2": "label2",
										},
									},
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
										PodSpec: &corev1.PodSpec{
											TerminationGracePeriodSeconds: ptr.To(int64(10)),
											ImagePullSecrets: []corev1.LocalObjectReference{
												{
													Name: "frontend-secret",
												},
											},
										},
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
									"nvidia.com/label1":                  "label1",
									"nvidia.com/label2":                  "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "frontend",
									Replicas: 1,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(536870912, resource.BinarySI),
													},
												},
											},
										},
										TerminationGracePeriodSeconds: ptr.To(int64(10)),
										ImagePullSecrets: []corev1.LocalObjectReference{
											{
												Name: "frontend-secret",
											},
										},
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
												VolumeMounts: []corev1.VolumeMount{
													{
														Name:      "shared-memory",
														MountPath: "/dev/shm",
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
										Volumes: []corev1.Volume{
											{
												Name: "planner-pvc",
												VolumeSource: corev1.VolumeSource{
													PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
														ClaimName: "planner-pvc",
													},
												},
											},
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(536870912, resource.BinarySI),
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
													{
														Name:      "shared-memory",
														MountPath: "/dev/shm",
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
		{
			name: "test_generate_grove_pod_gang_set_multinode sglang",
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
						BackendFramework: string(BackendFrameworkSGLang),
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
										PodSpec: &corev1.PodSpec{
											ImagePullSecrets: []corev1.LocalObjectReference{
												{
													Name: "frontend-secret",
												},
											},
											TerminationGracePeriodSeconds: ptr.To(int64(10)),
										},
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
							"worker": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									ExtraPodMetadata: &common.ExtraPodMetadata{
										Annotations: map[string]string{
											"nvidia.com/annotation1": "annotation1",
											"nvidia.com/annotation2": "annotation2",
										},
										Labels: map[string]string{
											"nvidia.com/label1": "label1",
											"nvidia.com/label2": "label2",
										},
									},
									Replicas:      &[]int32{5}[0],
									ComponentType: commonconsts.ComponentTypeWorker,
									DynamoConfig: &v1alpha1.DynamoConfig{
										NumberOfNodes:      ptr.To(int32(3)),
										TensorParallelSize: ptr.To(int32(2)),
										DataParallelSize:   ptr.To(int32(3)),
										FlagOverrides: map[string]*string{
											"custom-flag": ptr.To("custom-value"),
										},
										ExtraArgs: []string{"--extra", "arg"},
									},
									ExtraPodSpec: &common.ExtraPodSpec{
										MainContainer: &corev1.Container{
											Image: "worker-image",
										},
									},
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
											Name:  "WORKER_ENV_1",
											Value: "1",
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
						PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
							{
								Name: "worker-sg",
								CliqueNames: []string{
									"worker-ldr",
									"worker-wkr",
								},
								Replicas: ptr.To(int32(5)),
							},
						},
						Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
							{
								Name: "worker-ldr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-worker-ldr",
									"nvidia.com/label1":                  "label1",
									"nvidia.com/label2":                  "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "worker-ldr",
									Replicas: 1,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(1*1024*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  "main",
												Image: "worker-image",
												Command: []string{
													"/bin/sh",
													"-c",
												},
												Args: []string{
													"python3 -m dynamo.sglang.worker --custom-flag custom-value --dist-init-addr ${GROVE_HEADLESS_SERVICE}:29500 --dp-size 3 --nnodes 3 --node-rank 0 --tp-size 2 --extra arg",
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
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "WORKER_ENV_1",
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
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
													},
												},
											},
										},
									},
								},
							},
							{
								Name: "worker-wkr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-worker-wkr",
									"nvidia.com/label1":                  "label1",
									"nvidia.com/label2":                  "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "worker-wkr",
									Replicas: 2,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(1*1024*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  "main",
												Image: "worker-image",
												Command: []string{
													"/bin/sh",
													"-c",
												},
												Args: []string{
													"python3 -m dynamo.sglang.worker --custom-flag custom-value --dist-init-addr ${GROVE_HEADLESS_SERVICE}:29500 --dp-size 3 --nnodes 3 --node-rank $((GROVE_PCLQ_POD_INDEX + 1)) --tp-size 2 --extra arg",
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
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "WORKER_ENV_1",
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
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
													},
												},
											},
										},
									},
								},
							},
							{
								Name: "frontend",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-frontend",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "frontend",
									Replicas: 1,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(512*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										ImagePullSecrets: []corev1.LocalObjectReference{
											{
												Name: "frontend-secret",
											},
										},
										TerminationGracePeriodSeconds: ptr.To(int64(10)),
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
												VolumeMounts: []corev1.VolumeMount{
													{
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
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
										Volumes: []corev1.Volume{
											{
												Name: "planner-pvc",
												VolumeSource: corev1.VolumeSource{
													PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
														ClaimName: "planner-pvc",
													},
												},
											},
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(512*1024*1024, resource.BinarySI),
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
													{
														Name:      "shared-memory",
														MountPath: "/dev/shm",
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
		{
			name: "test_generate_grove_pod_gang_set_multinode sglang - custom commands",
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
						BackendFramework: string(BackendFrameworkSGLang),
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
										PodSpec: &corev1.PodSpec{
											ImagePullSecrets: []corev1.LocalObjectReference{
												{
													Name: "frontend-secret",
												},
											},
											TerminationGracePeriodSeconds: ptr.To(int64(10)),
										},
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
							"worker": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									ExtraPodMetadata: &common.ExtraPodMetadata{
										Annotations: map[string]string{
											"nvidia.com/annotation1": "annotation1",
											"nvidia.com/annotation2": "annotation2",
										},
										Labels: map[string]string{
											"nvidia.com/label1": "label1",
											"nvidia.com/label2": "label2",
										},
									},
									Replicas:      &[]int32{5}[0],
									ComponentType: commonconsts.ComponentTypeWorker,
									DynamoConfig: &v1alpha1.DynamoConfig{
										NumberOfNodes:      ptr.To(int32(3)),
										TensorParallelSize: ptr.To(int32(2)),
										DataParallelSize:   ptr.To(int32(3)),
										FlagOverrides: map[string]*string{
											"custom-flag": ptr.To("custom-value"),
										},
										ExtraArgs: []string{"--extra", "arg"},
									},
									ExtraPodSpec: &common.ExtraPodSpec{
										MainContainer: &corev1.Container{
											Image: "worker-image",
											Command: []string{
												"/bin/bash",
												"-c",
											},
											Args: []string{
												"python3 -m dynamo.sglang.worker --my-option my-value",
											},
										},
									},
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
											Name:  "WORKER_ENV_1",
											Value: "1",
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
						PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
							{
								Name: "worker-sg",
								CliqueNames: []string{
									"worker-ldr",
									"worker-wkr",
								},
								Replicas: ptr.To(int32(5)),
							},
						},
						Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
							{
								Name: "worker-ldr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-worker-ldr",
									"nvidia.com/label1":                  "label1",
									"nvidia.com/label2":                  "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "worker-ldr",
									Replicas: 1,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(1*1024*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  "main",
												Image: "worker-image",
												Command: []string{
													"/bin/bash",
													"-c",
												},
												Args: []string{
													"python3 -m dynamo.sglang.worker --my-option my-value --custom-flag custom-value --dist-init-addr ${GROVE_HEADLESS_SERVICE}:29500 --dp-size 3 --nnodes 3 --node-rank 0 --tp-size 2 --extra arg",
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
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "WORKER_ENV_1",
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
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
													},
												},
											},
										},
									},
								},
							},
							{
								Name: "worker-wkr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-worker-wkr",
									"nvidia.com/label1":                  "label1",
									"nvidia.com/label2":                  "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "worker-wkr",
									Replicas: 2,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(1*1024*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  "main",
												Image: "worker-image",
												Command: []string{
													"/bin/bash",
													"-c",
												},
												Args: []string{
													"python3 -m dynamo.sglang.worker --my-option my-value --custom-flag custom-value --dist-init-addr ${GROVE_HEADLESS_SERVICE}:29500 --dp-size 3 --nnodes 3 --node-rank $((GROVE_PCLQ_POD_INDEX + 1)) --tp-size 2 --extra arg",
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
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "WORKER_ENV_1",
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
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
													},
												},
											},
										},
									},
								},
							},
							{
								Name: "frontend",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-frontend",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "frontend",
									Replicas: 1,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(512*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										ImagePullSecrets: []corev1.LocalObjectReference{
											{
												Name: "frontend-secret",
											},
										},
										TerminationGracePeriodSeconds: ptr.To(int64(10)),
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
												VolumeMounts: []corev1.VolumeMount{
													{
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
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
										Volumes: []corev1.Volume{
											{
												Name: "planner-pvc",
												VolumeSource: corev1.VolumeSource{
													PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
														ClaimName: "planner-pvc",
													},
												},
											},
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(512*1024*1024, resource.BinarySI),
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
													{
														Name:      "shared-memory",
														MountPath: "/dev/shm",
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
		{
			name: "test_generate_grove_pod_gang_set_multinode vllm",
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
						BackendFramework: string(BackendFrameworkVLLM),
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
										PodSpec: &corev1.PodSpec{
											ImagePullSecrets: []corev1.LocalObjectReference{
												{
													Name: "frontend-secret",
												},
											},
											TerminationGracePeriodSeconds: ptr.To(int64(10)),
										},
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
							"worker": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									ExtraPodMetadata: &common.ExtraPodMetadata{
										Annotations: map[string]string{
											"nvidia.com/annotation1": "annotation1",
											"nvidia.com/annotation2": "annotation2",
										},
										Labels: map[string]string{
											"nvidia.com/label1": "label1",
											"nvidia.com/label2": "label2",
										},
									},
									Replicas:      &[]int32{5}[0],
									ComponentType: commonconsts.ComponentTypeWorker,
									DynamoConfig: &v1alpha1.DynamoConfig{
										NumberOfNodes:      ptr.To(int32(3)),
										TensorParallelSize: ptr.To(int32(2)),
										DataParallelSize:   ptr.To(int32(3)),
										FlagOverrides: map[string]*string{
											"custom-flag": ptr.To("custom-value"),
										},
										ExtraArgs: []string{"--extra", "arg"},
									},
									ExtraPodSpec: &common.ExtraPodSpec{
										MainContainer: &corev1.Container{
											Image: "worker-image",
										},
									},
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
											Name:  "WORKER_ENV_1",
											Value: "1",
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
						PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
							{
								Name: "worker-sg",
								CliqueNames: []string{
									"worker-ldr",
									"worker-wkr",
								},
								Replicas: ptr.To(int32(5)),
							},
						},
						Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
							{
								Name: "worker-ldr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-worker-ldr",
									"nvidia.com/label1":                  "label1",
									"nvidia.com/label2":                  "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "worker-ldr",
									Replicas: 1,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(1*1024*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  "main",
												Image: "worker-image",
												Command: []string{
													"/bin/sh",
													"-c",
												},
												Args: []string{
													"ray start --head --port=6379 && python3 -m dynamo.vllm --custom-flag custom-value --data-parallel-size 3 --tensor-parallel-size 2 --extra arg",
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
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "WORKER_ENV_1",
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
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
													},
												},
											},
										},
									},
								},
							},
							{
								Name: "worker-wkr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-worker-wkr",
									"nvidia.com/label1":                  "label1",
									"nvidia.com/label2":                  "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "worker-wkr",
									Replicas: 2,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(1*1024*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  "main",
												Image: "worker-image",
												Command: []string{
													"/bin/sh",
													"-c",
												},
												Args: []string{
													"ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block",
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
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "WORKER_ENV_1",
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
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
													},
												},
											},
										},
									},
								},
							},
							{
								Name: "frontend",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-frontend",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "frontend",
									Replicas: 1,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(512*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										ImagePullSecrets: []corev1.LocalObjectReference{
											{
												Name: "frontend-secret",
											},
										},
										TerminationGracePeriodSeconds: ptr.To(int64(10)),
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
												VolumeMounts: []corev1.VolumeMount{
													{
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
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
										Volumes: []corev1.Volume{
											{
												Name: "planner-pvc",
												VolumeSource: corev1.VolumeSource{
													PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
														ClaimName: "planner-pvc",
													},
												},
											},
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(512*1024*1024, resource.BinarySI),
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
													{
														Name:      "shared-memory",
														MountPath: "/dev/shm",
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
		{
			name: "test_generate_grove_pod_gang_set_multinode vllm with - custom command",
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
						BackendFramework: string(BackendFrameworkVLLM),
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
										PodSpec: &corev1.PodSpec{
											ImagePullSecrets: []corev1.LocalObjectReference{
												{
													Name: "frontend-secret",
												},
											},
											TerminationGracePeriodSeconds: ptr.To(int64(10)),
										},
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
							"Worker": {
								DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
									ExtraPodMetadata: &common.ExtraPodMetadata{
										Annotations: map[string]string{
											"nvidia.com/annotation1": "annotation1",
											"nvidia.com/annotation2": "annotation2",
										},
										Labels: map[string]string{
											"nvidia.com/label1": "label1",
											"nvidia.com/label2": "label2",
										},
									},
									Replicas:      &[]int32{5}[0],
									ComponentType: commonconsts.ComponentTypeWorker,
									DynamoConfig: &v1alpha1.DynamoConfig{
										NumberOfNodes:      ptr.To(int32(3)),
										TensorParallelSize: ptr.To(int32(2)),
										DataParallelSize:   ptr.To(int32(3)),
										FlagOverrides: map[string]*string{
											"custom-flag": ptr.To("custom-value"),
										},
										ExtraArgs: []string{"--extra", "arg"},
									},
									ExtraPodSpec: &common.ExtraPodSpec{
										MainContainer: &corev1.Container{
											Image: "worker-image",
											Command: []string{
												"/bin/bash",
												"-c",
											},
											Args: []string{
												"python3 -m dynamo.vllm --my-flag my-value",
											},
										},
									},
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
											Name:  "WORKER_ENV_1",
											Value: "1",
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
						PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
							{
								Name: "worker-sg",
								CliqueNames: []string{
									"worker-ldr",
									"worker-wkr",
								},
								Replicas: ptr.To(int32(5)),
							},
						},
						Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
							{
								Name: "worker-ldr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-worker-ldr",
									"nvidia.com/label1":                  "label1",
									"nvidia.com/label2":                  "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "worker-ldr",
									Replicas: 1,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(1*1024*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  "main",
												Image: "worker-image",
												Command: []string{
													"/bin/bash",
													"-c",
												},
												Args: []string{
													"ray start --head --port=6379 && python3 -m dynamo.vllm --my-flag my-value --custom-flag custom-value --data-parallel-size 3 --tensor-parallel-size 2 --extra arg",
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
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "WORKER_ENV_1",
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
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
													},
												},
											},
										},
									},
								},
							},
							{
								Name: "worker-wkr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-worker-wkr",
									"nvidia.com/label1":                  "label1",
									"nvidia.com/label2":                  "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "worker-wkr",
									Replicas: 2,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(1*1024*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  "main",
												Image: "worker-image",
												Command: []string{
													"/bin/bash",
													"-c",
												},
												Args: []string{
													"ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block",
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
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "WORKER_ENV_1",
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
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
													},
												},
											},
										},
									},
								},
							},
							{
								Name: "frontend",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector: "test-dynamo-graph-deployment-frontend",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName: "frontend",
									Replicas: 1,
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(512*1024*1024, resource.BinarySI),
													},
												},
											},
										},
										ImagePullSecrets: []corev1.LocalObjectReference{
											{
												Name: "frontend-secret",
											},
										},
										TerminationGracePeriodSeconds: ptr.To(int64(10)),
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
												VolumeMounts: []corev1.VolumeMount{
													{
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: "/dev/shm",
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
										Volumes: []corev1.Volume{
											{
												Name: "planner-pvc",
												VolumeSource: corev1.VolumeSource{
													PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
														ClaimName: "planner-pvc",
													},
												},
											},
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: resource.NewQuantity(512*1024*1024, resource.BinarySI),
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
													{
														Name:      "shared-memory",
														MountPath: "/dev/shm",
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

// Mock SecretsRetriever for testing
type mockSecretsRetriever struct{}

func (m *mockSecretsRetriever) RetrieveImagePullSecrets(ctx context.Context, deployment *v1alpha1.DynamoGraphDeployment) ([]corev1.LocalObjectReference, error) {
	return []corev1.LocalObjectReference{}, nil
}

func (m *mockSecretsRetriever) GetSecrets(namespace, registry string) ([]string, error) {
	return []string{}, nil
}

func TestGeneratePodSpecForComponent_SGLang(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	dynamoDeployment := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-deployment",
			Namespace: "default",
		},
	}
	controllerConfig := controller_common.Config{}

	tests := []struct {
		name              string
		component         *v1alpha1.DynamoComponentDeploymentOverridesSpec
		backendFramework  BackendFramework
		role              Role
		numberOfNodes     int32
		expectError       bool
		expectContains    []string
		expectNotContains []string
	}{
		{
			name: "SGLang single node worker",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: commonconsts.ComponentTypeWorker,
					DynamoConfig: &v1alpha1.DynamoConfig{
						TensorParallelSize: ptr.To(int32(2)),
					},
				},
			},
			backendFramework:  BackendFrameworkSGLang,
			role:              RoleMain,
			numberOfNodes:     1,
			expectError:       false,
			expectContains:    []string{"python3 -m dynamo.sglang.worker", "tp-size", "2"},
			expectNotContains: []string{"dist-init-addr", "nnodes"},
		},
		{
			name: "SGLang multinode leader",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: commonconsts.ComponentTypeWorker,
					DynamoConfig:  &v1alpha1.DynamoConfig{},
				},
			},
			backendFramework: BackendFrameworkSGLang,
			role:             RoleLeader,
			numberOfNodes:    3,
			expectError:      false,
			expectContains:   []string{"python3 -m dynamo.sglang.worker", "dist-init-addr", "nnodes", "node-rank"},
		},
		{
			name: "SGLang multinode worker",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: commonconsts.ComponentTypeWorker,
					DynamoConfig:  &v1alpha1.DynamoConfig{},
				},
			},
			backendFramework: BackendFrameworkSGLang,
			role:             RoleWorker,
			numberOfNodes:    3,
			expectError:      false,
			expectContains:   []string{"python3 -m dynamo.sglang.worker", "dist-init-addr", "nnodes", "node-rank"},
		},
		{
			name: "SGLang with user command override",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: commonconsts.ComponentTypeWorker,
					DynamoConfig:  &v1alpha1.DynamoConfig{},
					ExtraPodSpec: &common.ExtraPodSpec{
						MainContainer: &corev1.Container{
							Command: []string{"custom", "command"},
						},
					},
				},
			},
			backendFramework: BackendFrameworkSGLang,
			role:             RoleMain,
			numberOfNodes:    1,
			expectError:      false,
			expectContains:   []string{},
		},
		{
			name: "SGLang with resources",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: commonconsts.ComponentTypeWorker,
					DynamoConfig:  &v1alpha1.DynamoConfig{},
					ExtraPodSpec: &common.ExtraPodSpec{
						MainContainer: &corev1.Container{
							Resources: corev1.ResourceRequirements{
								Requests: corev1.ResourceList{
									corev1.ResourceCPU:    resource.MustParse("1"),
									corev1.ResourceMemory: resource.MustParse("2Gi"),
								},
							},
						},
					},
				},
			},
			backendFramework: BackendFrameworkSGLang,
			role:             RoleMain,
			numberOfNodes:    1,
			expectError:      false,
			expectContains:   []string{"python3 -m dynamo.sglang.worker"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec, err := GeneratePodSpecForComponent(
				tt.component,
				tt.backendFramework,
				secretsRetriever,
				dynamoDeployment,
				tt.role,
				tt.numberOfNodes,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
			)

			if tt.expectError {
				if err == nil {
					t.Errorf("GeneratePodSpecForComponent() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("GeneratePodSpecForComponent() unexpected error: %v", err)
				return
			}

			// Check container exists
			if len(podSpec.Containers) == 0 {
				t.Errorf("GeneratePodSpecForComponent() no containers in podSpec")
				return
			}

			container := podSpec.Containers[0]

			// Check command and args contain expected strings
			allArgs := append(container.Command, container.Args...)
			allArgsStr := strings.Join(allArgs, " ")

			for _, expected := range tt.expectContains {
				if !strings.Contains(allArgsStr, expected) {
					t.Errorf("GeneratePodSpecForComponent() args = %v, should contain %s", allArgs, expected)
				}
			}

			for _, notExpected := range tt.expectNotContains {
				if strings.Contains(allArgsStr, notExpected) {
					t.Errorf("GeneratePodSpecForComponent() args = %v, should NOT contain %s", allArgs, notExpected)
				}
			}

			// Check that container name is set
			if container.Name != "main" {
				t.Errorf("GeneratePodSpecForComponent() container name = %s, want main", container.Name)
			}
		})
	}
}

func TestGeneratePodSpecForComponent_VLLM(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	dynamoDeployment := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-deployment",
			Namespace: "default",
		},
	}
	controllerConfig := controller_common.Config{}

	tests := []struct {
		name              string
		component         *v1alpha1.DynamoComponentDeploymentOverridesSpec
		backendFramework  BackendFramework
		role              Role
		numberOfNodes     int32
		expectError       bool
		expectContains    []string
		expectNotContains []string
	}{
		{
			name: "VLLM single node worker",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: commonconsts.ComponentTypeWorker,
					DynamoConfig:  &v1alpha1.DynamoConfig{},
				},
			},
			backendFramework:  BackendFrameworkVLLM,
			role:              RoleMain,
			numberOfNodes:     1,
			expectError:       false,
			expectContains:    []string{"python3 -m dynamo.vllm"},
			expectNotContains: []string{"ray start"},
		},
		{
			name: "VLLM multinode leader",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: commonconsts.ComponentTypeWorker,
					DynamoConfig:  &v1alpha1.DynamoConfig{},
				},
			},
			backendFramework: BackendFrameworkVLLM,
			role:             RoleLeader,
			numberOfNodes:    3,
			expectError:      false,
			expectContains:   []string{"ray start --head --port=6379", "python3 -m dynamo.vllm"},
		},
		{
			name: "VLLM multinode worker",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: commonconsts.ComponentTypeWorker,
					DynamoConfig:  &v1alpha1.DynamoConfig{},
				},
			},
			backendFramework:  BackendFrameworkVLLM,
			role:              RoleWorker,
			numberOfNodes:     3,
			expectError:       false,
			expectContains:    []string{"ray start --address=${GROVE_HEADLESS_SERVICE}:6379 --block"},
			expectNotContains: []string{"python3 -m dynamo.vllm"},
		},
		{
			name: "VLLM prefill worker single node",
			component: &v1alpha1.DynamoComponentDeploymentOverridesSpec{
				DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
					ComponentType: commonconsts.ComponentTypePrefillWorker,
					DynamoConfig:  &v1alpha1.DynamoConfig{},
				},
			},
			backendFramework:  BackendFrameworkVLLM,
			role:              RoleMain,
			numberOfNodes:     1,
			expectError:       false,
			expectContains:    []string{"python3 -m dynamo.vllm", "--is-prefill-worker"},
			expectNotContains: []string{"ray start"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec, err := GeneratePodSpecForComponent(
				tt.component,
				tt.backendFramework,
				secretsRetriever,
				dynamoDeployment,
				tt.role,
				tt.numberOfNodes,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
			)

			if tt.expectError {
				if err == nil {
					t.Errorf("GeneratePodSpecForComponent() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("GeneratePodSpecForComponent() unexpected error: %v", err)
				return
			}

			// Check container exists
			if len(podSpec.Containers) == 0 {
				t.Errorf("GeneratePodSpecForComponent() no containers in podSpec")
				return
			}

			container := podSpec.Containers[0]

			// Check command and args contain expected strings
			allArgs := append(container.Command, container.Args...)
			allArgsStr := strings.Join(allArgs, " ")

			for _, expected := range tt.expectContains {
				if !strings.Contains(allArgsStr, expected) {
					t.Errorf("GeneratePodSpecForComponent() args = %v, should contain %s", allArgs, expected)
				}
			}

			for _, notExpected := range tt.expectNotContains {
				if strings.Contains(allArgsStr, notExpected) {
					t.Errorf("GeneratePodSpecForComponent() args = %v, should NOT contain %s", allArgs, notExpected)
				}
			}
		})
	}
}

func TestGeneratePodSpecForComponent_UnsupportedBackend(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	dynamoDeployment := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-deployment",
			Namespace: "default",
		},
	}
	controllerConfig := controller_common.Config{}

	component := &v1alpha1.DynamoComponentDeploymentOverridesSpec{
		DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
			ComponentType: commonconsts.ComponentTypeWorker,
			DynamoConfig:  &v1alpha1.DynamoConfig{},
		},
	}

	tests := []struct {
		name             string
		backendFramework BackendFramework
		expectError      bool
		errorContains    string
	}{
		{
			name:             "TRTLLM backend not implemented",
			backendFramework: BackendFrameworkTRTLLM,
			expectError:      true,
			errorContains:    "unsupported backend framework",
		},
		{
			name:             "unknown backend",
			backendFramework: BackendFramework("unknown"),
			expectError:      true,
			errorContains:    "unsupported backend framework",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := GeneratePodSpecForComponent(
				component,
				tt.backendFramework,
				secretsRetriever,
				dynamoDeployment,
				RoleMain,
				1,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
			)

			if tt.expectError {
				if err == nil {
					t.Errorf("GeneratePodSpecForComponent() expected error, got nil")
					return
				}
				if !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("GeneratePodSpecForComponent() error = %v, should contain %s", err, tt.errorContains)
				}
			} else {
				if err != nil {
					t.Errorf("GeneratePodSpecForComponent() unexpected error: %v", err)
				}
			}
		})
	}
}

func TestMergeContainerCommand(t *testing.T) {
	tests := []struct {
		name       string
		defaultCmd []string
		userCmd    []string
		expected   []string
	}{
		{
			name:       "user command overrides default",
			defaultCmd: []string{"python", "default.py"},
			userCmd:    []string{"python", "custom.py"},
			expected:   []string{"python", "custom.py"},
		},
		{
			name:       "empty user command returns default",
			defaultCmd: []string{"python", "default.py"},
			userCmd:    []string{},
			expected:   []string{"python", "default.py"},
		},
		{
			name:       "nil user command returns default",
			defaultCmd: []string{"python", "default.py"},
			userCmd:    nil,
			expected:   []string{"python", "default.py"},
		},
		{
			name:       "both empty returns empty",
			defaultCmd: []string{},
			userCmd:    []string{},
			expected:   []string{},
		},
		{
			name:       "default empty user provided",
			defaultCmd: []string{},
			userCmd:    []string{"python", "user.py"},
			expected:   []string{"python", "user.py"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := mergeContainerCommand(tt.defaultCmd, tt.userCmd)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("mergeContainerCommand() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestExpandRolesForService(t *testing.T) {
	tests := []struct {
		name            string
		serviceName     string
		numberOfNodes   int32
		serviceReplicas int32
		expected        []ServiceRole
	}{
		{
			name:            "single node",
			serviceName:     "test-service",
			numberOfNodes:   1,
			serviceReplicas: 2,
			expected: []ServiceRole{
				{Name: "test-service", Role: RoleMain, Replicas: 2},
			},
		},
		{
			name:          "multinode 2 nodes",
			serviceName:   "test-service",
			numberOfNodes: 2,
			expected: []ServiceRole{
				{Name: "test-service-ldr", Role: RoleLeader, Replicas: 1},
				{Name: "test-service-wkr", Role: RoleWorker, Replicas: 1},
			},
		},
		{
			name:          "multinode 5 nodes",
			serviceName:   "test-service",
			numberOfNodes: 5,
			expected: []ServiceRole{
				{Name: "test-service-ldr", Role: RoleLeader, Replicas: 1},
				{Name: "test-service-wkr", Role: RoleWorker, Replicas: 4},
			},
		},
		{
			name:            "zero nodes should return main",
			serviceName:     "test-service",
			numberOfNodes:   0,
			serviceReplicas: 1,
			expected: []ServiceRole{
				{Name: "test-service", Role: RoleMain, Replicas: 1},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := expandRolesForService(tt.serviceName, &tt.serviceReplicas, tt.numberOfNodes)
			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("expandRolesForService() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestRoleEnum(t *testing.T) {
	// Test that role constants are defined correctly
	if RoleLeader != "leader" {
		t.Errorf("RoleLeader = %v, want \"leader\"", RoleLeader)
	}
	if RoleWorker != "worker" {
		t.Errorf("RoleWorker = %v, want \"worker\"", RoleWorker)
	}
	if RoleMain != "main" {
		t.Errorf("RoleMain = %v, want \"main\"", RoleMain)
	}

	// Test that roles can be compared
	roles := []Role{RoleLeader, RoleWorker, RoleMain}
	for _, role := range roles {
		switch role {
		case RoleLeader, RoleWorker, RoleMain:
			// Expected
		default:
			t.Errorf("Unexpected role value: %v", role)
		}
	}
}

func TestBackendFrameworkEnum(t *testing.T) {
	// Test that backend framework constants are defined correctly
	if BackendFrameworkSGLang != "sglang" {
		t.Errorf("BackendFrameworkSGLang = %v, want \"sglang\"", BackendFrameworkSGLang)
	}
	if BackendFrameworkVLLM != "vllm" {
		t.Errorf("BackendFrameworkVLLM = %v, want \"vllm\"", BackendFrameworkVLLM)
	}
	if BackendFrameworkTRTLLM != "trtllm" {
		t.Errorf("BackendFrameworkTRTLLM = %v, want \"trtllm\"", BackendFrameworkTRTLLM)
	}

	// Test that frameworks can be compared
	frameworks := []BackendFramework{BackendFrameworkSGLang, BackendFrameworkVLLM, BackendFrameworkTRTLLM}
	for _, framework := range frameworks {
		switch framework {
		case BackendFrameworkSGLang, BackendFrameworkVLLM, BackendFrameworkTRTLLM:
			// Expected
		default:
			t.Errorf("Unexpected framework value: %v", framework)
		}
	}
}

func TestServiceRoleStruct(t *testing.T) {
	// Test ServiceRole struct creation and field access
	sr := ServiceRole{
		Name:     "test-service",
		Role:     RoleLeader,
		Replicas: 3,
	}

	if sr.Name != "test-service" {
		t.Errorf("ServiceRole.Name = %v, want \"test-service\"", sr.Name)
	}
	if sr.Role != RoleLeader {
		t.Errorf("ServiceRole.Role = %v, want %v", sr.Role, RoleLeader)
	}
	if sr.Replicas != 3 {
		t.Errorf("ServiceRole.Replicas = %v, want 3", sr.Replicas)
	}
}
