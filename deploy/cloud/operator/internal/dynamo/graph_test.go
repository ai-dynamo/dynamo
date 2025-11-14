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
	"github.com/stretchr/testify/assert"
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
						Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
							"service1": {
								DynamoNamespace:  &[]string{"default"}[0],
								ComponentType:    "frontend",
								SubComponentType: "test-sub-component",
								Replicas:         &[]int32{3}[0],
								Resources: &common.Resources{
									Requests: &common.ResourceItem{
										CPU:    "1",
										Memory: "1Gi",
										GPU:    "0",
										Custom: map[string]string{},
									},
								},
							},
							"service2": {
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
			want: map[string]*v1alpha1.DynamoComponentDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service1",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:           "service1",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:      "service1",
							DynamoNamespace:  &[]string{"default-test-dynamographdeployment"}[0],
							ComponentType:    "frontend",
							SubComponentType: "test-sub-component",
							Replicas:         &[]int32{3}[0],
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
									CPU:    "1",
									Memory: "1Gi",
									GPU:    "0",
									Custom: map[string]string{},
								},
							},
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent:           "service1",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
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
							commonconsts.KubeLabelDynamoComponent:           "service2",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent:           "service2",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
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
						Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
							"service1": {
								DynamoNamespace: nil,
								ComponentType:   "frontend",
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
							"service2": {
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
			want: map[string]*v1alpha1.DynamoComponentDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service1",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:           "service1",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
							ComponentType:   "frontend",
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
								commonconsts.KubeLabelDynamoComponent:           "service1",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
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
							commonconsts.KubeLabelDynamoComponent:           "service2",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent:           "service2",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
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
			name: "Test GenerateDynamoComponentsDeployments with ingress enabled",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
							"service1": {
								DynamoNamespace: nil,
								ComponentType:   "frontend",
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
							"service2": {
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
							commonconsts.KubeLabelDynamoComponent:           "service1",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
							ComponentType:   "frontend",
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
								commonconsts.KubeLabelDynamoComponent:           "service1",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
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
							commonconsts.KubeLabelDynamoComponent:           "service2",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent:           "service2",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
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
						Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
							"service1": {
								DynamoNamespace: nil,
								ComponentType:   "frontend",
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
							"service2": {
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
			want: map[string]*v1alpha1.DynamoComponentDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service1",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:           "service1",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
							ComponentType:   "frontend",
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
								commonconsts.KubeLabelDynamoComponent:           "service1",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
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
							commonconsts.KubeLabelDynamoComponent:           "service2",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent:           "service2",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
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
						Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
							"service1": {
								DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
								ComponentType:   "frontend",
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
							"service2": {
								DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
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
						Envs: []corev1.EnvVar{
							{
								Name:  "TEST_ENV",
								Value: "test-value",
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
							commonconsts.KubeLabelDynamoComponent:           "service1",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: string(BackendFrameworkSGLang),
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
							ComponentType:   "frontend",
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
								commonconsts.KubeLabelDynamoComponent:           "service1",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
							},
							Autoscaling: nil,
							ExtraPodSpec: &common.ExtraPodSpec{
								MainContainer: &corev1.Container{
									Command: []string{"sh", "-c"},
									Args:    []string{"echo hello world", "sleep 99999"},
								},
							},
							Envs: []corev1.EnvVar{
								{
									Name:  "TEST_ENV",
									Value: "test-value",
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
							commonconsts.KubeLabelDynamoComponent:           "service2",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: string(BackendFrameworkSGLang),
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service2",
							DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
							Replicas:        &[]int32{3}[0],
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponent:           "service2",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
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
									Name:  "TEST_ENV",
									Value: "test-value",
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "Test GenerateDynamoComponentsDeployments with Discover Backend and Metrics Annotatitions",
			args: args{
				parentDynamoGraphDeployment: &v1alpha1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment",
						Namespace: "default",
						Annotations: map[string]string{
							commonconsts.KubeAnnotationEnableMetrics:          "false",
							commonconsts.KubeAnnotationDynamoDiscoveryBackend: "test",
						},
					},
					Spec: v1alpha1.DynamoGraphDeploymentSpec{
						BackendFramework: string(BackendFrameworkSGLang),
						Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
							"service1": {
								DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
								ComponentType:   "frontend",
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
			want: map[string]*v1alpha1.DynamoComponentDeployment{
				"service1": {
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-dynamographdeployment-service1",
						Namespace: "default",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:           "service1",
							commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
							commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
						},
					},
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						BackendFramework: string(BackendFrameworkSGLang),
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							Annotations: map[string]string{
								commonconsts.KubeAnnotationEnableMetrics:          "false",
								commonconsts.KubeAnnotationDynamoDiscoveryBackend: "test",
							},
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default-test-dynamographdeployment"}[0],
							ComponentType:   "frontend",
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
								commonconsts.KubeLabelDynamoComponent:           "service1",
								commonconsts.KubeLabelDynamoNamespace:           "default-test-dynamographdeployment",
								commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamographdeployment",
							},
						},
					},
				},
			},
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

func Test_GetDynamoComponentDeploymentsGlobalNamespace(t *testing.T) {
	dgd := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dynamographdeployment",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: string(BackendFrameworkSGLang),
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"service1": {
					ComponentType:         "frontend",
					GlobalDynamoNamespace: true,
					Replicas:              &[]int32{3}[0],
				},
				"service2": {
					ComponentType: "worker",
					Replicas:      &[]int32{3}[0],
				},
			},
		},
	}

	got, err := GenerateDynamoComponentsDeployments(context.Background(), dgd, nil)
	if !assert.NoError(t, err) {
		return
	}

	if !assert.Len(t, got, 2) {
		return
	}

	for _, d := range got {
		switch d.Spec.ComponentType {
		case commonconsts.ComponentTypeFrontend:
			assert.Equal(t, commonconsts.GlobalDynamoNamespace, *d.Spec.DynamoNamespace)
			assert.Equal(t, commonconsts.GlobalDynamoNamespace, d.Labels[commonconsts.KubeLabelDynamoNamespace])
		case commonconsts.ComponentTypeWorker:
			expectedNamespace := fmt.Sprintf("%s-%s", dgd.Namespace, dgd.Name)
			assert.Equal(t, expectedNamespace, *d.Spec.DynamoNamespace)
			assert.Equal(t, expectedNamespace, d.Labels[commonconsts.KubeLabelDynamoNamespace])
		default:
			t.Errorf("unexpected component type: %s", d.Spec.ComponentType)
		}
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
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:   "Frontend",
							ComponentType: commonconsts.ComponentTypeFrontend,
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

func sortEnvVars(envs []corev1.EnvVar) []corev1.EnvVar {
	sorted := make([]corev1.EnvVar, len(envs))
	copy(sorted, envs)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Name < sorted[j].Name
	})
	return sorted
}

func TestGenerateGrovePodCliqueSet(t *testing.T) {
	type args struct {
		ctx              context.Context
		dynamoDeployment *v1alpha1.DynamoGraphDeployment
		controllerConfig controller_common.Config
	}
	tests := []struct {
		name    string
		args    args
		want    *grovev1alpha1.PodCliqueSet
		wantErr bool
	}{
		{
			name: "test_generate_grove_pod_clique_set_single_node",
			args: args{
				ctx: context.Background(),
				controllerConfig: controller_common.Config{
					EtcdAddress:     "etcd-address",
					NatsAddress:     "nats-address",
					ModelExpressURL: "model-express-url",
					Grove: controller_common.GroveConfig{
						TerminationDelay: 15 * time.Minute,
					},
					PrometheusEndpoint: "http://localhost:9090",
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
						Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
							"Frontend": {
								ComponentType:    "frontend", // Frontend component
								SubComponentType: "test-sub-component",
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
							"Planner": {
								Replicas:      &[]int32{2}[0],
								ComponentType: commonconsts.ComponentTypePlanner,
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
								VolumeMounts: []v1alpha1.VolumeMount{
									{
										Name:       "dynamo-pvc",
										MountPoint: "/planner",
									},
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
			want: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dynamo-graph-deployment",
					Namespace: "test-namespace",
				},
				Spec: grovev1alpha1.PodCliqueSetSpec{
					Replicas: 1,
					Template: grovev1alpha1.PodCliqueSetTemplateSpec{
						StartupType: ptr.To(grovev1alpha1.CliqueStartupTypeAnyOrder),
						HeadlessServiceConfig: &grovev1alpha1.HeadlessServiceConfig{
							PublishNotReadyAddresses: true,
						},
						TerminationDelay: &metav1.Duration{Duration: 15 * time.Minute},
						Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
							{
								Name: "frontend",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector:            "test-dynamo-graph-deployment-frontend",
									commonconsts.KubeLabelMetricsEnabled:            commonconsts.KubeLabelValueTrue,
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeFrontend,
									commonconsts.KubeLabelDynamoSubComponentType:    "test-sub-component",
									commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamo-graph-deployment",
									commonconsts.KubeLabelDynamoNamespace:           "test-namespace-test-dynamo-graph-deployment",
									"nvidia.com/label1":                             "label1",
									"nvidia.com/label2":                             "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName:     "frontend",
									Replicas:     1,
									MinAvailable: ptr.To(int32(1)),
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
													},
												},
											},
										},
										TerminationGracePeriodSeconds: ptr.To(int64(10)),
										SecurityContext: &corev1.PodSecurityContext{
											FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
										},
										ImagePullSecrets: []corev1.LocalObjectReference{
											{
												Name: "frontend-secret",
											},
										},
										RestartPolicy: corev1.RestartPolicyAlways,
										Containers: []corev1.Container{
											{
												Name:  commonconsts.MainContainerName,
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
														Name:  "DYN_HTTP_PORT",
														Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort),
													},
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
														Name: "POD_NAME",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.name",
															},
														},
													},
													{
														Name: "POD_NAMESPACE",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.namespace",
															},
														},
													},
													{
														Name:  "ETCD_ENDPOINTS",
														Value: "etcd-address",
													},
													{
														Name:  commonconsts.DynamoNamespaceEnvVar,
														Value: "test-namespace-test-dynamo-graph-deployment",
													},
													{
														Name:  commonconsts.DynamoComponentEnvVar,
														Value: commonconsts.ComponentTypeFrontend,
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAME",
														Value: "test-dynamo-graph-deployment",
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
														Value: "test-namespace",
													},
													{
														Name:  "MODEL_EXPRESS_URL",
														Value: "model-express-url",
													},
													{
														Name:  "PROMETHEUS_ENDPOINT",
														Value: "http://localhost:9090",
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
														MountPath: commonconsts.DefaultSharedMemoryMountPath,
													},
												},
												Ports: []corev1.ContainerPort{
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoContainerPortName,
														ContainerPort: int32(commonconsts.DynamoServicePort),
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
									commonconsts.KubeLabelMetricsEnabled:            commonconsts.KubeLabelValueTrue,
									commonconsts.KubeLabelDynamoSelector:            "test-dynamo-graph-deployment-planner",
									commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamo-graph-deployment",
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypePlanner,
									commonconsts.KubeLabelDynamoNamespace:           "test-namespace-test-dynamo-graph-deployment",
								},
								Annotations: map[string]string{},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName:     "planner",
									Replicas:     2,
									MinAvailable: ptr.To(int32(1)),
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "dynamo-pvc",
												VolumeSource: corev1.VolumeSource{
													PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
														ClaimName: "dynamo-pvc",
													},
												},
											},
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
													},
												},
											},
										},
										ServiceAccountName:            commonconsts.PlannerServiceAccountName,
										TerminationGracePeriodSeconds: ptr.To(int64(60)),
										SecurityContext: &corev1.PodSecurityContext{
											FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
										},
										RestartPolicy: corev1.RestartPolicyAlways,

										Containers: []corev1.Container{
											{
												Name:  commonconsts.MainContainerName,
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
														Name:  "NATS_SERVER",
														Value: "nats-address",
													},
													{
														Name:  "ETCD_ENDPOINTS",
														Value: "etcd-address",
													},
													{
														Name:  commonconsts.DynamoNamespaceEnvVar,
														Value: "test-namespace-test-dynamo-graph-deployment",
													},
													{
														Name:  commonconsts.DynamoComponentEnvVar,
														Value: commonconsts.ComponentTypePlanner,
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAME",
														Value: "test-dynamo-graph-deployment",
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
														Value: "test-namespace",
													},
													{
														Name:  "MODEL_EXPRESS_URL",
														Value: "model-express-url",
													},
													{
														Name:  "PROMETHEUS_ENDPOINT",
														Value: "http://localhost:9090",
													},
													{
														Name:  "PLANNER_PROMETHEUS_PORT",
														Value: fmt.Sprintf("%d", commonconsts.DynamoPlannerMetricsPort),
													},
													{
														Name: "POD_NAME",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.name",
															},
														},
													},
													{
														Name: "POD_NAMESPACE",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.namespace",
															},
														},
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
														Name:      "dynamo-pvc",
														MountPath: "/planner",
													},
													{
														Name:      "shared-memory",
														MountPath: commonconsts.DefaultSharedMemoryMountPath,
													},
												},
												Ports: []corev1.ContainerPort{
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoMetricsPortName,
														ContainerPort: int32(commonconsts.DynamoPlannerMetricsPort),
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
						Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
							"Frontend": {
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
								ComponentType: commonconsts.ComponentTypeFrontend,
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
							"worker": {
								Multinode: &v1alpha1.MultinodeSpec{
									NodeCount: 3,
								},
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
								Replicas:         &[]int32{5}[0],
								ComponentType:    commonconsts.ComponentTypeWorker,
								SubComponentType: "test-sub-component",
								ExtraPodSpec: &common.ExtraPodSpec{
									MainContainer: &corev1.Container{
										Image: "worker-image",
										Command: []string{
											"/bin/sh",
											"-c",
										},
										Args: []string{
											"python3 -m dynamo.sglang --custom-flag custom-value",
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
							"Planner": {
								ComponentType: commonconsts.ComponentTypePlanner,
								Replicas:      &[]int32{2}[0],
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
								VolumeMounts: []v1alpha1.VolumeMount{
									{
										Name:       "dynamo-pvc",
										MountPoint: "/planner",
									},
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
			want: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dynamo-graph-deployment",
					Namespace: "test-namespace",
				},
				Spec: grovev1alpha1.PodCliqueSetSpec{
					Replicas: 1,
					Template: grovev1alpha1.PodCliqueSetTemplateSpec{
						HeadlessServiceConfig: &grovev1alpha1.HeadlessServiceConfig{
							PublishNotReadyAddresses: true,
						},
						TerminationDelay: &metav1.Duration{Duration: 15 * time.Minute},
						PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
							{
								Name: "worker",
								CliqueNames: []string{
									"worker-ldr",
									"worker-wkr",
								},
								Replicas:     ptr.To(int32(5)),
								MinAvailable: ptr.To(int32(1)),
							},
						},
						// StartupType: ptr.To(grovev1alpha1.CliqueStartupTypeExplicit),
						StartupType: ptr.To(grovev1alpha1.CliqueStartupTypeAnyOrder),
						Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
							{
								Name: "worker-ldr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
									commonconsts.KubeLabelDynamoSubComponentType:    "test-sub-component",
									commonconsts.KubeLabelMetricsEnabled:            commonconsts.KubeLabelValueTrue,
									commonconsts.KubeLabelDynamoSelector:            "test-dynamo-graph-deployment-worker-ldr",
									commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamo-graph-deployment",
									commonconsts.KubeLabelDynamoNamespace:           "test-namespace-test-dynamo-graph-deployment",
									"nvidia.com/label1":                             "label1",
									"nvidia.com/label2":                             "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName:     "worker-ldr",
									Replicas:     1,
									MinAvailable: ptr.To(int32(1)),
									PodSpec: corev1.PodSpec{
										RestartPolicy:                 corev1.RestartPolicyAlways,
										TerminationGracePeriodSeconds: ptr.To(int64(60)),
										SecurityContext: &corev1.PodSecurityContext{
											FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
										},
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  commonconsts.MainContainerName,
												Image: "worker-image",
												Command: []string{
													"/bin/sh",
													"-c",
												},
												Args: []string{
													"python3 -m dynamo.sglang --dist-init-addr $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-worker-ldr-0.$(GROVE_HEADLESS_SERVICE):29500 --nnodes 3 --node-rank 0 --custom-flag custom-value",
												},
												Ports: []corev1.ContainerPort{
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoSystemPortName,
														ContainerPort: int32(commonconsts.DynamoSystemPort),
													},
												},
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "DYN_SYSTEM_ENABLED",
														Value: "true",
													},
													{
														Name:  "DYN_SYSTEM_PORT",
														Value: "9090",
													},
													{
														Name:  "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
														Value: "[\"generate\"]",
													},
													{
														Name:  "WORKER_ENV_1",
														Value: "1",
													},
													{
														Name:  "NATS_SERVER",
														Value: "nats-address",
													},
													{
														Name:  "ETCD_ENDPOINTS",
														Value: "etcd-address",
													},
													{
														Name:  commonconsts.DynamoNamespaceEnvVar,
														Value: "test-namespace-test-dynamo-graph-deployment",
													},
													{
														Name:  commonconsts.DynamoComponentEnvVar,
														Value: commonconsts.ComponentTypeWorker,
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAME",
														Value: "test-dynamo-graph-deployment",
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
														Value: "test-namespace",
													},
													{
														Name: "POD_NAME",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.name",
															},
														},
													},
													{
														Name: "POD_NAMESPACE",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.namespace",
															},
														},
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
														MountPath: commonconsts.DefaultSharedMemoryMountPath,
													},
												},
												LivenessProbe: &corev1.Probe{
													ProbeHandler: corev1.ProbeHandler{
														HTTPGet: &corev1.HTTPGetAction{
															Path: "/live",
															Port: intstr.FromString(commonconsts.DynamoSystemPortName),
														},
													},
													TimeoutSeconds:   4,
													PeriodSeconds:    5,
													SuccessThreshold: 0,
													FailureThreshold: 1,
												},
												ReadinessProbe: &corev1.Probe{
													ProbeHandler: corev1.ProbeHandler{
														HTTPGet: &corev1.HTTPGetAction{
															Path: "/health",
															Port: intstr.FromString(commonconsts.DynamoSystemPortName),
														},
													},
													TimeoutSeconds:   4,
													PeriodSeconds:    10,
													SuccessThreshold: 0,
													FailureThreshold: 3,
												},
												StartupProbe: &corev1.Probe{
													ProbeHandler: corev1.ProbeHandler{
														HTTPGet: &corev1.HTTPGetAction{
															Path: "/live",
															Port: intstr.FromString(commonconsts.DynamoSystemPortName),
														},
													},
													TimeoutSeconds:   5,
													PeriodSeconds:    10,
													SuccessThreshold: 0,
													FailureThreshold: 720,
												},
											},
										},
									},
								},
							},
							{
								Name: "worker-wkr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
									commonconsts.KubeLabelDynamoSubComponentType:    "test-sub-component",
									commonconsts.KubeLabelMetricsEnabled:            commonconsts.KubeLabelValueTrue,
									commonconsts.KubeLabelDynamoSelector:            "test-dynamo-graph-deployment-worker-wkr",
									commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamo-graph-deployment",
									commonconsts.KubeLabelDynamoNamespace:           "test-namespace-test-dynamo-graph-deployment",
									"nvidia.com/label1":                             "label1",
									"nvidia.com/label2":                             "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName:     "worker-wkr",
									Replicas:     2,
									MinAvailable: ptr.To(int32(1)),
									// StartsAfter: []string{"worker-ldr"},
									PodSpec: corev1.PodSpec{
										RestartPolicy:                 corev1.RestartPolicyAlways,
										TerminationGracePeriodSeconds: ptr.To(int64(60)),
										SecurityContext: &corev1.PodSecurityContext{
											FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
										},
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  commonconsts.MainContainerName,
												Image: "worker-image",
												Command: []string{
													"/bin/sh",
													"-c",
												},
												Args: []string{
													"python3 -m dynamo.sglang --dist-init-addr $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-worker-ldr-0.$(GROVE_HEADLESS_SERVICE):29500 --nnodes 3 --node-rank $((GROVE_PCLQ_POD_INDEX + 1)) --custom-flag custom-value",
												},
												Ports: []corev1.ContainerPort{
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoSystemPortName,
														ContainerPort: int32(commonconsts.DynamoSystemPort),
													},
												},
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "DYN_SYSTEM_ENABLED",
														Value: "true",
													},
													{
														Name:  "DYN_SYSTEM_PORT",
														Value: "9090",
													},
													{
														Name:  "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
														Value: "[\"generate\"]",
													},
													{
														Name:  "WORKER_ENV_1",
														Value: "1",
													},
													{
														Name:  "NATS_SERVER",
														Value: "nats-address",
													},
													{
														Name:  "ETCD_ENDPOINTS",
														Value: "etcd-address",
													},
													{
														Name:  commonconsts.DynamoNamespaceEnvVar,
														Value: "test-namespace-test-dynamo-graph-deployment",
													},
													{
														Name:  commonconsts.DynamoComponentEnvVar,
														Value: commonconsts.ComponentTypeWorker,
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAME",
														Value: "test-dynamo-graph-deployment",
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
														Value: "test-namespace",
													},
													{
														Name: "POD_NAME",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.name",
															},
														},
													},
													{
														Name: "POD_NAMESPACE",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.namespace",
															},
														},
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
														MountPath: commonconsts.DefaultSharedMemoryMountPath,
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
									commonconsts.KubeLabelMetricsEnabled:            commonconsts.KubeLabelValueTrue,
									commonconsts.KubeLabelDynamoSelector:            "test-dynamo-graph-deployment-frontend",
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeFrontend,
									commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamo-graph-deployment",
									commonconsts.KubeLabelDynamoNamespace:           "test-namespace-test-dynamo-graph-deployment",
								},
								Annotations: map[string]string{},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName:     "frontend",
									Replicas:     1,
									MinAvailable: ptr.To(int32(1)),
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
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
										SecurityContext: &corev1.PodSecurityContext{
											FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
										},
										RestartPolicy: corev1.RestartPolicyAlways,
										Containers: []corev1.Container{
											{
												Name:  commonconsts.MainContainerName,
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
														Name:  "DYN_HTTP_PORT",
														Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort),
													},
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
													{
														Name:  commonconsts.DynamoNamespaceEnvVar,
														Value: "test-namespace-test-dynamo-graph-deployment",
													},
													{
														Name:  commonconsts.DynamoComponentEnvVar,
														Value: commonconsts.ComponentTypeFrontend,
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAME",
														Value: "test-dynamo-graph-deployment",
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
														Value: "test-namespace",
													},
													{
														Name: "POD_NAME",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.name",
															},
														},
													},
													{
														Name: "POD_NAMESPACE",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.namespace",
															},
														},
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
												},
												VolumeMounts: []corev1.VolumeMount{
													{
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: commonconsts.DefaultSharedMemoryMountPath,
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
									commonconsts.KubeLabelDynamoSelector:            "test-dynamo-graph-deployment-planner",
									commonconsts.KubeLabelMetricsEnabled:            commonconsts.KubeLabelValueTrue,
									commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamo-graph-deployment",
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypePlanner,
									commonconsts.KubeLabelDynamoNamespace:           "test-namespace-test-dynamo-graph-deployment",
								},
								Annotations: map[string]string{},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName:     "planner",
									Replicas:     2,
									MinAvailable: ptr.To(int32(1)),
									PodSpec: corev1.PodSpec{
										TerminationGracePeriodSeconds: ptr.To(int64(60)),
										ServiceAccountName:            commonconsts.PlannerServiceAccountName,
										SecurityContext: &corev1.PodSecurityContext{
											FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
										},
										RestartPolicy: corev1.RestartPolicyAlways,
										Volumes: []corev1.Volume{
											{
												Name: "dynamo-pvc",
												VolumeSource: corev1.VolumeSource{
													PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
														ClaimName: "dynamo-pvc",
													},
												},
											},
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
													},
												},
											},
										},
										Containers: []corev1.Container{
											{
												Name:  commonconsts.MainContainerName,
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
														Name:  "NATS_SERVER",
														Value: "nats-address",
													},
													{
														Name:  "ETCD_ENDPOINTS",
														Value: "etcd-address",
													},
													{
														Name:  commonconsts.DynamoNamespaceEnvVar,
														Value: "test-namespace-test-dynamo-graph-deployment",
													},
													{
														Name:  commonconsts.DynamoComponentEnvVar,
														Value: commonconsts.ComponentTypePlanner,
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAME",
														Value: "test-dynamo-graph-deployment",
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
														Value: "test-namespace",
													},
													{
														Name:  "PLANNER_PROMETHEUS_PORT",
														Value: fmt.Sprintf("%d", commonconsts.DynamoPlannerMetricsPort),
													},
													{
														Name: "POD_NAME",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.name",
															},
														},
													},
													{
														Name: "POD_NAMESPACE",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.namespace",
															},
														},
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
														Name:      "dynamo-pvc",
														MountPath: "/planner",
													},
													{
														Name:      "shared-memory",
														MountPath: commonconsts.DefaultSharedMemoryMountPath,
													},
												},
												Ports: []corev1.ContainerPort{
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoMetricsPortName,
														ContainerPort: int32(commonconsts.DynamoPlannerMetricsPort),
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
						Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
							"Frontend": {
								Replicas:      &[]int32{1}[0],
								ComponentType: commonconsts.ComponentTypeFrontend,
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
							"worker": {

								Multinode: &v1alpha1.MultinodeSpec{
									NodeCount: 3,
								},
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
								ExtraPodSpec: &common.ExtraPodSpec{
									MainContainer: &corev1.Container{
										Image: "worker-image",
										Command: []string{
											"python3",
											"-m",
											"dynamo.vllm",
										},
										Args: []string{
											"--custom-flag",
											"custom-value",
											"--tensor-parallel-size",
											"4",
											"--pipeline-parallel-size",
											"1",
										},
										StartupProbe: &corev1.Probe{
											ProbeHandler: corev1.ProbeHandler{
												HTTPGet: &corev1.HTTPGetAction{
													Path: "/startup",
													Port: intstr.FromInt(8080),
												},
											},
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
								LivenessProbe: &corev1.Probe{
									ProbeHandler: corev1.ProbeHandler{
										HTTPGet: &corev1.HTTPGetAction{
											Path: "/health",
											Port: intstr.FromInt(8080),
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
							"Planner": {

								ComponentType: commonconsts.ComponentTypePlanner,
								Replicas:      &[]int32{2}[0],
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
								VolumeMounts: []v1alpha1.VolumeMount{
									{
										Name:       "dynamo-pvc",
										MountPoint: "/planner",
									},
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
			want: &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-dynamo-graph-deployment",
					Namespace: "test-namespace",
				},
				Spec: grovev1alpha1.PodCliqueSetSpec{
					Replicas: 1,
					Template: grovev1alpha1.PodCliqueSetTemplateSpec{
						StartupType: ptr.To(grovev1alpha1.CliqueStartupTypeAnyOrder),
						HeadlessServiceConfig: &grovev1alpha1.HeadlessServiceConfig{
							PublishNotReadyAddresses: true,
						},
						TerminationDelay: &metav1.Duration{Duration: 15 * time.Minute},
						PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{
							{
								Name: "worker",
								CliqueNames: []string{
									"worker-ldr",
									"worker-wkr",
								},
								Replicas:     ptr.To(int32(5)),
								MinAvailable: ptr.To(int32(1)),
							},
						},
						// StartupType: ptr.To(grovev1alpha1.CliqueStartupTypeExplicit),
						Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
							{
								Name: "worker-ldr",
								Labels: map[string]string{
									commonconsts.KubeLabelDynamoSelector:            "test-dynamo-graph-deployment-worker-ldr",
									commonconsts.KubeLabelMetricsEnabled:            commonconsts.KubeLabelValueTrue,
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
									commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamo-graph-deployment",
									commonconsts.KubeLabelDynamoNamespace:           "test-namespace-test-dynamo-graph-deployment",
									"nvidia.com/label1":                             "label1",
									"nvidia.com/label2":                             "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName:     "worker-ldr",
									Replicas:     1,
									MinAvailable: ptr.To(int32(1)),
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
													},
												},
											},
										},
										TerminationGracePeriodSeconds: ptr.To(int64(60)),
										SecurityContext: &corev1.PodSecurityContext{
											FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
										},
										RestartPolicy: corev1.RestartPolicyAlways,
										Containers: []corev1.Container{
											{
												Name:  commonconsts.MainContainerName,
												Image: "worker-image",
												Command: []string{
													"/bin/sh",
													"-c",
												},
												Args: []string{
													"ray start --head --port=6379 && python3 -m dynamo.vllm --custom-flag custom-value --tensor-parallel-size 4 --pipeline-parallel-size 1",
												},
												Ports: []corev1.ContainerPort{
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoSystemPortName,
														ContainerPort: int32(commonconsts.DynamoSystemPort),
													},
												},
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "DYN_SYSTEM_PORT",
														Value: "9090",
													},
													{
														Name:  "DYN_SYSTEM_ENABLED",
														Value: "true",
													},
													{
														Name:  "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
														Value: "[\"generate\"]",
													},
													{
														Name:  "WORKER_ENV_1",
														Value: "1",
													},
													{
														Name:  "NATS_SERVER",
														Value: "nats-address",
													},
													{
														Name:  "ETCD_ENDPOINTS",
														Value: "etcd-address",
													},
													{
														Name:  commonconsts.DynamoNamespaceEnvVar,
														Value: "test-namespace-test-dynamo-graph-deployment",
													},
													{
														Name:  commonconsts.DynamoComponentEnvVar,
														Value: commonconsts.ComponentTypeWorker,
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAME",
														Value: "test-dynamo-graph-deployment",
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
														Value: "test-namespace",
													},
													{
														Name: "POD_NAME",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.name",
															},
														},
													},
													{
														Name: "POD_NAMESPACE",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.namespace",
															},
														},
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
														MountPath: commonconsts.DefaultSharedMemoryMountPath,
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
												LivenessProbe: &corev1.Probe{
													ProbeHandler: corev1.ProbeHandler{
														HTTPGet: &corev1.HTTPGetAction{
															Path: "/health",
															Port: intstr.FromInt(8080),
														},
													},
												},
												StartupProbe: &corev1.Probe{
													ProbeHandler: corev1.ProbeHandler{
														HTTPGet: &corev1.HTTPGetAction{
															Path: "/startup",
															Port: intstr.FromInt(8080),
														},
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
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
									commonconsts.KubeLabelMetricsEnabled:            commonconsts.KubeLabelValueTrue,
									commonconsts.KubeLabelDynamoSelector:            "test-dynamo-graph-deployment-worker-wkr",
									commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamo-graph-deployment",
									commonconsts.KubeLabelDynamoNamespace:           "test-namespace-test-dynamo-graph-deployment",
									"nvidia.com/label1":                             "label1",
									"nvidia.com/label2":                             "label2",
								},
								Annotations: map[string]string{
									"nvidia.com/annotation1": "annotation1",
									"nvidia.com/annotation2": "annotation2",
								},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName:     "worker-wkr",
									Replicas:     2,
									MinAvailable: ptr.To(int32(1)),
									// StartsAfter: []string{"worker-ldr"},
									PodSpec: corev1.PodSpec{
										TerminationGracePeriodSeconds: ptr.To(int64(60)),
										SecurityContext: &corev1.PodSecurityContext{
											FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
										},
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
													},
												},
											},
										},
										RestartPolicy: corev1.RestartPolicyAlways,
										Containers: []corev1.Container{
											{
												Name:  commonconsts.MainContainerName,
												Image: "worker-image",
												Command: []string{
													"/bin/sh",
													"-c",
												},
												Args: []string{
													"ray start --address=$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-worker-ldr-0.$(GROVE_HEADLESS_SERVICE):6379 --block",
												},
												Ports: []corev1.ContainerPort{
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoSystemPortName,
														ContainerPort: int32(commonconsts.DynamoSystemPort),
													},
												},
												Env: []corev1.EnvVar{
													{
														Name:  "DYNAMO_POD_GANG_SET_REPLICAS",
														Value: "1",
													},
													{
														Name:  "DYN_SYSTEM_PORT",
														Value: "9090",
													},
													{
														Name:  "DYN_SYSTEM_ENABLED",
														Value: "true",
													},
													{
														Name:  "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
														Value: "[\"generate\"]",
													},
													{
														Name:  "WORKER_ENV_1",
														Value: "1",
													},
													{
														Name:  "NATS_SERVER",
														Value: "nats-address",
													},
													{
														Name:  "ETCD_ENDPOINTS",
														Value: "etcd-address",
													},
													{
														Name:  commonconsts.DynamoNamespaceEnvVar,
														Value: "test-namespace-test-dynamo-graph-deployment",
													},
													{
														Name:  commonconsts.DynamoComponentEnvVar,
														Value: commonconsts.ComponentTypeWorker,
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAME",
														Value: "test-dynamo-graph-deployment",
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
														Value: "test-namespace",
													},
													{
														Name: "POD_NAME",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.name",
															},
														},
													},
													{
														Name: "POD_NAMESPACE",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.namespace",
															},
														},
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
														MountPath: commonconsts.DefaultSharedMemoryMountPath,
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
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeFrontend,
									commonconsts.KubeLabelMetricsEnabled:            commonconsts.KubeLabelValueTrue,
									commonconsts.KubeLabelDynamoSelector:            "test-dynamo-graph-deployment-frontend",
									commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamo-graph-deployment",
									commonconsts.KubeLabelDynamoNamespace:           "test-namespace-test-dynamo-graph-deployment",
								},
								Annotations: map[string]string{},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName:     "frontend",
									Replicas:     1,
									MinAvailable: ptr.To(int32(1)),
									PodSpec: corev1.PodSpec{
										Volumes: []corev1.Volume{
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
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
										SecurityContext: &corev1.PodSecurityContext{
											FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
										},
										RestartPolicy: corev1.RestartPolicyAlways,
										Containers: []corev1.Container{
											{
												Name:  commonconsts.MainContainerName,
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
														Name:  "DYN_HTTP_PORT",
														Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort),
													},
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
													{
														Name:  commonconsts.DynamoNamespaceEnvVar,
														Value: "test-namespace-test-dynamo-graph-deployment",
													},
													{
														Name:  commonconsts.DynamoComponentEnvVar,
														Value: commonconsts.ComponentTypeFrontend,
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAME",
														Value: "test-dynamo-graph-deployment",
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
														Value: "test-namespace",
													},
													{
														Name: "POD_NAME",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.name",
															},
														},
													},
													{
														Name: "POD_NAMESPACE",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.namespace",
															},
														},
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
												},
												VolumeMounts: []corev1.VolumeMount{
													{
														Name:      commonconsts.KubeValueNameSharedMemory,
														MountPath: commonconsts.DefaultSharedMemoryMountPath,
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
									commonconsts.KubeLabelMetricsEnabled:            commonconsts.KubeLabelValueTrue,
									commonconsts.KubeLabelDynamoSelector:            "test-dynamo-graph-deployment-planner",
									commonconsts.KubeLabelDynamoGraphDeploymentName: "test-dynamo-graph-deployment",
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypePlanner,
									commonconsts.KubeLabelDynamoNamespace:           "test-namespace-test-dynamo-graph-deployment",
								},
								Annotations: map[string]string{},
								Spec: grovev1alpha1.PodCliqueSpec{
									RoleName:     "planner",
									Replicas:     2,
									MinAvailable: ptr.To(int32(1)),
									PodSpec: corev1.PodSpec{
										TerminationGracePeriodSeconds: ptr.To(int64(60)),
										ServiceAccountName:            commonconsts.PlannerServiceAccountName,
										SecurityContext: &corev1.PodSecurityContext{
											FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
										},
										Volumes: []corev1.Volume{
											{
												Name: "dynamo-pvc",
												VolumeSource: corev1.VolumeSource{
													PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
														ClaimName: "dynamo-pvc",
													},
												},
											},
											{
												Name: "shared-memory",
												VolumeSource: corev1.VolumeSource{
													EmptyDir: &corev1.EmptyDirVolumeSource{
														Medium:    corev1.StorageMediumMemory,
														SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
													},
												},
											},
										},
										RestartPolicy: corev1.RestartPolicyAlways,
										Containers: []corev1.Container{
											{
												Name:  commonconsts.MainContainerName,
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
												Ports: []corev1.ContainerPort{
													{
														Protocol:      corev1.ProtocolTCP,
														Name:          commonconsts.DynamoMetricsPortName,
														ContainerPort: int32(commonconsts.DynamoPlannerMetricsPort),
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
														Name:  "NATS_SERVER",
														Value: "nats-address",
													},
													{
														Name:  "ETCD_ENDPOINTS",
														Value: "etcd-address",
													},
													{
														Name:  commonconsts.DynamoNamespaceEnvVar,
														Value: "test-namespace-test-dynamo-graph-deployment",
													},
													{
														Name:  commonconsts.DynamoComponentEnvVar,
														Value: commonconsts.ComponentTypePlanner,
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAME",
														Value: "test-dynamo-graph-deployment",
													},
													{
														Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
														Value: "test-namespace",
													},
													{
														Name:  "PLANNER_PROMETHEUS_PORT",
														Value: fmt.Sprintf("%d", commonconsts.DynamoPlannerMetricsPort),
													},
													{
														Name: "POD_NAME",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.name",
															},
														},
													},
													{
														Name: "POD_NAMESPACE",
														ValueFrom: &corev1.EnvVarSource{
															FieldRef: &corev1.ObjectFieldSelector{
																FieldPath: "metadata.namespace",
															},
														},
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
														Name:      "dynamo-pvc",
														MountPath: "/planner",
													},
													{
														Name:      "shared-memory",
														MountPath: commonconsts.DefaultSharedMemoryMountPath,
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
			got, err := GenerateGrovePodCliqueSet(tt.args.ctx, tt.args.dynamoDeployment, tt.args.controllerConfig, nil)
			if (err != nil) != tt.wantErr {
				t.Errorf("GenerateGrovePodCliqueSet() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			sort.Slice(got.Spec.Template.Cliques, func(i, j int) bool {
				return got.Spec.Template.Cliques[i].Name < got.Spec.Template.Cliques[j].Name
			})
			sort.Slice(tt.want.Spec.Template.Cliques, func(i, j int) bool {
				return tt.want.Spec.Template.Cliques[i].Name < tt.want.Spec.Template.Cliques[j].Name
			})

			// Sort environment variables for all containers in all cliques
			for _, clique := range got.Spec.Template.Cliques {
				for i := range clique.Spec.PodSpec.Containers {
					clique.Spec.PodSpec.Containers[i].Env = sortEnvVars(clique.Spec.PodSpec.Containers[i].Env)
				}
			}
			for _, clique := range tt.want.Spec.Template.Cliques {
				for i := range clique.Spec.PodSpec.Containers {
					clique.Spec.PodSpec.Containers[i].Env = sortEnvVars(clique.Spec.PodSpec.Containers[i].Env)
				}
			}

			if diff := cmp.Diff(got, tt.want); diff != "" {
				t.Errorf("GenerateGrovePodCliqueSet() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func Test_GeneratePodCliqueSetGlobalDynamoNamespace(t *testing.T) {
	dynamoDeployment := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-dynamo-graph",
			Namespace: "k8s-namespace",
		},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
				"Frontend": {
					ComponentType:         commonconsts.ComponentTypeFrontend,
					GlobalDynamoNamespace: true,
					Replicas:              ptr.To(int32(1)),
				},
				"Planner": {
					ComponentType: commonconsts.ComponentTypePlanner,
					Replicas:      ptr.To(int32(1)),
				},
			},
		},
	}

	got, err := GenerateGrovePodCliqueSet(context.Background(), dynamoDeployment, controller_common.Config{}, nil)
	if !assert.NoError(t, err) {
		return
	}

	if !assert.Len(t, got.Spec.Template.Cliques, 2) {
		return
	}

	for _, clique := range got.Spec.Template.Cliques {
		switch clique.Name {
		case "frontend":
			assert.Equal(t, commonconsts.GlobalDynamoNamespace, clique.Labels[commonconsts.KubeLabelDynamoNamespace])
			assertDYNNamespace(t, clique.Spec.PodSpec, commonconsts.GlobalDynamoNamespace)
		case "planner":
			expectedNamespace := fmt.Sprintf("%s-%s", dynamoDeployment.Namespace, dynamoDeployment.Name)
			assert.Equal(t, expectedNamespace, clique.Labels[commonconsts.KubeLabelDynamoNamespace])
			assertDYNNamespace(t, clique.Spec.PodSpec, expectedNamespace)
		default:
			t.Errorf("GenerateGrovePodCliqueSet() clique = %v, want %v", clique.Name, "frontend or planner")
		}
	}
}

func assertDYNNamespace(t *testing.T, podSpec corev1.PodSpec, expectedNamespace string) {
	if assert.Len(t, podSpec.Containers, 1) {
		foundDYNNamespace := false
		for _, env := range podSpec.Containers[0].Env {
			if env.Name == commonconsts.DynamoNamespaceEnvVar {
				assert.Equal(t, expectedNamespace, env.Value)
				foundDYNNamespace = true
				break
			}
		}
		assert.True(t, foundDYNNamespace, fmt.Sprintf("%s not found in container environment variables", commonconsts.DynamoNamespaceEnvVar))
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

// Mock SecretsRetriever that returns secrets for testing docker secrets functionality
type mockSecretsRetrieverWithSecrets struct{}

func (m *mockSecretsRetrieverWithSecrets) RetrieveImagePullSecrets(ctx context.Context, deployment *v1alpha1.DynamoGraphDeployment) ([]corev1.LocalObjectReference, error) {
	return []corev1.LocalObjectReference{
		{Name: "test-docker-secret"},
	}, nil
}

func (m *mockSecretsRetrieverWithSecrets) GetSecrets(namespace, registry string) ([]string, error) {
	// Return some mock secrets when called
	return []string{"test-docker-secret"}, nil
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
		component         *v1alpha1.DynamoComponentDeploymentSharedSpec
		backendFramework  BackendFramework
		role              Role
		numberOfNodes     int32
		expectError       bool
		expectContains    []string
		expectNotContains []string
	}{
		{
			name: "SGLang single node worker",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"python3 -m dynamo.sglang"},
					},
				},
			},
			backendFramework:  BackendFrameworkSGLang,
			role:              RoleMain,
			numberOfNodes:     1,
			expectError:       false,
			expectContains:    []string{"python3", "-m", "dynamo.sglang"},
			expectNotContains: []string{"dist-init-addr", "nnodes", "tp-size"},
		},
		{
			name: "SGLang multinode leader",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"python3 -m dynamo.sglang"},
					},
				},
			},
			backendFramework: BackendFrameworkSGLang,
			role:             RoleLeader,
			numberOfNodes:    3,
			expectError:      false,
			expectContains:   []string{"python3", "-m", "dynamo.sglang", "dist-init-addr", "nnodes", "node-rank"},
		},
		{
			name: "SGLang multinode worker",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"python3 -m dynamo.sglang"},
					},
				},
			},
			backendFramework: BackendFrameworkSGLang,
			role:             RoleWorker,
			numberOfNodes:    3,
			expectError:      false,
			expectContains:   []string{"python3", "-m", "dynamo.sglang", "dist-init-addr", "nnodes", "node-rank"},
		},
		{
			name: "SGLang with user command override",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,

				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Command: []string{"custom", "command"},
					},
				},
			},
			backendFramework: BackendFrameworkSGLang,
			role:             RoleMain,
			numberOfNodes:    1,
			expectError:      false,
			expectContains:   []string{},
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
				"worker",
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
			if container.Name != commonconsts.MainContainerName {
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
		component         *v1alpha1.DynamoComponentDeploymentSharedSpec
		backendFramework  BackendFramework
		role              Role
		numberOfNodes     int32
		expectError       bool
		expectContains    []string
		expectNotContains []string
	}{
		{
			name: "VLLM single node worker",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"python3", "-m", "dynamo.vllm"},
					},
				},
			},
			backendFramework:  BackendFrameworkVLLM,
			role:              RoleMain,
			numberOfNodes:     1,
			expectError:       false,
			expectContains:    []string{"python3", "-m", "dynamo.vllm"},
			expectNotContains: []string{"ray start"},
		},
		{
			name: "VLLM multinode leader",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"python3", "-m", "dynamo.vllm", "--tensor-parallel-size", "4", "--pipeline-parallel-size", "1"},
					},
				},
			},
			backendFramework: BackendFrameworkVLLM,
			role:             RoleLeader,
			numberOfNodes:    3,
			expectError:      false,
			expectContains:   []string{"ray start --head --port=6379", "python3", "-m", "dynamo.vllm"},
		},
		{
			name: "VLLM multinode worker",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"python3", "-m", "dynamo.vllm", "--tensor-parallel-size", "4", "--pipeline-parallel-size", "1"},
					},
				},
			},
			backendFramework:  BackendFrameworkVLLM,
			role:              RoleWorker,
			numberOfNodes:     3,
			expectError:       false,
			expectContains:    []string{"ray start --address=$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-worker-ldr-0.$(GROVE_HEADLESS_SERVICE):6379 --block"},
			expectNotContains: []string{"python3 -m dynamo.vllm"},
		},
		{
			name: "VLLM worker single node",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"python3", "-m", "dynamo.vllm", "--is-prefill-worker"},
					},
				},
			},
			backendFramework:  BackendFrameworkVLLM,
			role:              RoleMain,
			numberOfNodes:     1,
			expectError:       false,
			expectContains:    []string{"python3", "-m", "dynamo.vllm", "--is-prefill-worker"},
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
				"worker",
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

	component := &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType: commonconsts.ComponentTypeWorker,
	}

	tests := []struct {
		name             string
		backendFramework BackendFramework
		expectError      bool
		errorContains    string
	}{
		{
			name:             "TRTLLM backend implemented",
			backendFramework: BackendFrameworkTRTLLM,
			expectError:      false,
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
				"worker",
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

func TestDetectBackendFrameworkFromArgs(t *testing.T) {
	tests := []struct {
		name        string
		command     []string
		args        []string
		expected    BackendFramework
		expectError bool
	}{
		{
			name:     "detect VLLM from args",
			command:  []string{"/bin/sh", "-c"},
			args:     []string{"python -m dynamo.vllm.worker --model test"},
			expected: BackendFrameworkVLLM,
		},
		{
			name:     "detect SGLang from args",
			command:  []string{"/bin/sh", "-c"},
			args:     []string{"python -m dynamo.sglang --model test"},
			expected: BackendFrameworkSGLang,
		},
		{
			name:     "detect TRTLLM from args",
			command:  []string{"/bin/sh", "-c"},
			args:     []string{"python -m dynamo.trtllm.worker --model test"},
			expected: BackendFrameworkTRTLLM,
		},
		{
			name:     "detect from complex command with pipes",
			command:  []string{},
			args:     []string{"echo start && python -m dynamo.vllm.worker --model test | tee /tmp/log"},
			expected: BackendFrameworkVLLM,
		},
		{
			name:     "detect from python3.11",
			command:  []string{},
			args:     []string{"python3.11 -m dynamo.sglang"},
			expected: BackendFrameworkSGLang,
		},
		{
			name:     "no backend detected",
			command:  []string{"/bin/sh", "-c"},
			args:     []string{"echo hello world"},
			expected: BackendFrameworkNoop,
		},
		{
			name:        "multiple backends detected",
			command:     []string{},
			args:        []string{"python -m dynamo.vllm.worker && python -m dynamo.sglang"},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := detectBackendFrameworkFromArgs(tt.command, tt.args)

			if tt.expectError {
				if err == nil {
					t.Errorf("detectBackendFrameworkFromArgs() expected error, got none")
				}
				return
			}

			if err != nil {
				t.Errorf("detectBackendFrameworkFromArgs() unexpected error: %v", err)
				return
			}

			if result != tt.expected {
				t.Errorf("detectBackendFrameworkFromArgs() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestDetermineBackendFramework(t *testing.T) {
	tests := []struct {
		name                     string
		componentType            string
		command                  []string
		args                     []string
		explicitBackendFramework string
		expected                 BackendFramework
		expectError              bool
		errorContains            string
	}{
		{
			name:          "non-worker component returns noop",
			componentType: "frontend",
			command:       []string{"/bin/sh", "-c"},
			args:          []string{"echo hello world"},
			expected:      BackendFrameworkNoop,
		},
		{
			name:          "worker with VLLM detection",
			componentType: "worker",
			command:       []string{},
			args:          []string{"python -m dynamo.vllm.worker --model test"},
			expected:      BackendFrameworkVLLM,
		},
		{
			name:                     "worker with explicit framework only",
			componentType:            "worker",
			explicitBackendFramework: "sglang",
			expected:                 BackendFrameworkSGLang,
		},
		{
			name:                     "worker with detected matching explicit",
			componentType:            "worker",
			args:                     []string{"python -m dynamo.sglang"},
			explicitBackendFramework: "sglang",
			expected:                 BackendFrameworkSGLang,
		},
		{
			name:                     "worker with detected conflicting explicit",
			componentType:            "worker",
			args:                     []string{"python -m dynamo.vllm.worker"},
			explicitBackendFramework: "sglang",
			expectError:              true,
			errorContains:            "backend framework mismatch",
		},
		{
			name:          "worker with no detection, no explicit - returns noop",
			componentType: "worker",
			expected:      BackendFrameworkNoop,
			expectError:   false,
		},
		{
			name:          "worker with detection failure, no explicit - returns noop",
			componentType: "worker",
			args:          []string{"echo hello world"},
			expected:      BackendFrameworkNoop,
			expectError:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := determineBackendFramework(
				tt.componentType,
				tt.command,
				tt.args,
				tt.explicitBackendFramework,
			)

			if tt.expectError {
				if err == nil {
					t.Errorf("determineBackendFramework() expected error, got none")
					return
				}
				if tt.errorContains != "" && !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("determineBackendFramework() error = %v, should contain %q", err, tt.errorContains)
				}
				return
			}

			if err != nil {
				t.Errorf("determineBackendFramework() unexpected error: %v", err)
				return
			}

			if result != tt.expected {
				t.Errorf("determineBackendFramework() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGetBackendFrameworkFromComponent(t *testing.T) {
	tests := []struct {
		name          string
		component     *v1alpha1.DynamoComponentDeploymentSharedSpec
		deployment    *v1alpha1.DynamoGraphDeployment
		expected      BackendFramework
		expectError   bool
		errorContains string
	}{
		{
			name: "detect from args - VLLM",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: "worker", // Worker component
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"python -m dynamo.vllm.worker --model test"},
					},
				},
			},
			deployment: &v1alpha1.DynamoGraphDeployment{},
			expected:   BackendFrameworkVLLM,
		},
		{
			name: "explicit framework only",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: "worker", // Worker component
			},
			deployment: &v1alpha1.DynamoGraphDeployment{
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
				},
			},
			expected: BackendFrameworkSGLang,
		},
		{
			name: "detected matches explicit",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: "worker", // Worker component
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"python -m dynamo.sglang"},
					},
				},
			},
			deployment: &v1alpha1.DynamoGraphDeployment{
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
				},
			},
			expected: BackendFrameworkSGLang,
		},
		{
			name: "detected conflicts with explicit",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: "worker", // Worker component
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"python -m dynamo.vllm.worker"},
					},
				},
			},
			deployment: &v1alpha1.DynamoGraphDeployment{
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: "sglang",
				},
			},
			expectError:   true,
			errorContains: "backend framework mismatch",
		},
		{
			name: "non-worker component returns noop",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: "frontend", // Frontend component
			},
			deployment: &v1alpha1.DynamoGraphDeployment{},
			expected:   BackendFrameworkNoop,
		},
		{
			name: "worker with no detection, no explicit - returns noop",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: "worker", // Worker component
			},
			deployment:  &v1alpha1.DynamoGraphDeployment{},
			expected:    BackendFrameworkNoop,
			expectError: false,
		},
		{
			name: "worker with detection failure, no explicit - returns noop",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: "worker", // Worker component
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Args: []string{"echo hello world"},
					},
				},
			},
			deployment:  &v1alpha1.DynamoGraphDeployment{},
			expected:    BackendFrameworkNoop,
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := getBackendFrameworkFromComponent(tt.component, tt.deployment)

			if tt.expectError {
				if err == nil {
					t.Errorf("getBackendFrameworkFromComponent() expected error, got none")
					return
				}
				if tt.errorContains != "" && !strings.Contains(err.Error(), tt.errorContains) {
					t.Errorf("getBackendFrameworkFromComponent() error = %v, should contain %q", err, tt.errorContains)
				}
				return
			}

			if err != nil {
				t.Errorf("getBackendFrameworkFromComponent() unexpected error: %v", err)
				return
			}

			if result != tt.expected {
				t.Errorf("getBackendFrameworkFromComponent() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestApplyCliqueStartupDependencies(t *testing.T) {
	tests := []struct {
		name              string
		roles             []ServiceRole
		backendFramework  BackendFramework
		numberOfNodes     int32
		expectedDeps      map[string][]string // clique name -> expected StartsAfter dependencies
		expectStartupType bool
	}{
		{
			name: "vllm_multinode_applies_dependencies",
			roles: []ServiceRole{
				{Name: "service-ldr", Role: RoleLeader, Replicas: 1},
				{Name: "service-wkr", Role: RoleWorker, Replicas: 2},
			},
			backendFramework: BackendFrameworkVLLM,
			numberOfNodes:    3,
			expectedDeps: map[string][]string{
				"service-ldr": nil,
				"service-wkr": nil,
			},
			expectStartupType: false,
		},
		{
			name: "sglang_multinode_applies_dependencies",
			roles: []ServiceRole{
				{Name: "service-ldr", Role: RoleLeader, Replicas: 1},
				{Name: "service-wkr", Role: RoleWorker, Replicas: 2},
			},
			backendFramework: BackendFrameworkSGLang,
			numberOfNodes:    3,
			expectedDeps: map[string][]string{
				"service-ldr": nil,
				"service-wkr": nil,
			},
			expectStartupType: false,
		},
		{
			name: "trtllm_multinode_applies_dependencies",
			roles: []ServiceRole{
				{Name: "service-ldr", Role: RoleLeader, Replicas: 1},
				{Name: "service-wkr", Role: RoleWorker, Replicas: 2},
			},
			backendFramework: BackendFrameworkTRTLLM,
			numberOfNodes:    3,
			expectedDeps: map[string][]string{
				"service-ldr": {"service-wkr"},
				"service-wkr": nil,
			},
			expectStartupType: true,
		},
		{
			name: "single_node_no_dependencies",
			roles: []ServiceRole{
				{Name: "service", Role: RoleMain, Replicas: 1},
			},
			backendFramework: BackendFrameworkTRTLLM,
			numberOfNodes:    1,
			expectedDeps: map[string][]string{
				"service": nil,
			},
			expectStartupType: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a PodCliqueSet with cliques matching the roles
			gangSet := &grovev1alpha1.PodCliqueSet{
				Spec: grovev1alpha1.PodCliqueSetSpec{
					Template: grovev1alpha1.PodCliqueSetTemplateSpec{
						Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{},
					},
				},
			}

			// Add cliques for each role
			for _, role := range tt.roles {
				clique := &grovev1alpha1.PodCliqueTemplateSpec{
					Name: strings.ToLower(role.Name),
					Spec: grovev1alpha1.PodCliqueSpec{
						RoleName: strings.ToLower(role.Name),
						Replicas: role.Replicas,
					},
				}
				gangSet.Spec.Template.Cliques = append(gangSet.Spec.Template.Cliques, clique)
			}

			// Apply dependencies
			applyCliqueStartupDependencies(gangSet, tt.roles, tt.backendFramework, tt.numberOfNodes)

			// Verify StartupType
			if tt.expectStartupType {
				if gangSet.Spec.Template.StartupType == nil || *gangSet.Spec.Template.StartupType != grovev1alpha1.CliqueStartupTypeExplicit {
					t.Errorf("Expected StartupType to be CliqueStartupTypeExplicit, got %v", gangSet.Spec.Template.StartupType)
				}
			} else {
				if gangSet.Spec.Template.StartupType != nil {
					t.Errorf("Expected StartupType to be nil, got %v", *gangSet.Spec.Template.StartupType)
				}
			}

			// Verify dependencies for each clique
			for _, clique := range gangSet.Spec.Template.Cliques {
				expectedDeps, exists := tt.expectedDeps[clique.Name]
				if !exists {
					t.Errorf("Unexpected clique %s", clique.Name)
					continue
				}

				if !reflect.DeepEqual(clique.Spec.StartsAfter, expectedDeps) {
					t.Errorf("Clique %s: expected StartsAfter %v, got %v", clique.Name, expectedDeps, clique.Spec.StartsAfter)
				}
			}
		})
	}
}

func TestGetCliqueStartupDependencies(t *testing.T) {
	tests := []struct {
		name              string
		role              Role
		backendFramework  BackendFramework
		leaderCliqueName  string
		workerCliqueNames []string
		expected          []string
	}{
		{
			name:              "trtllm_leader_depends_on_workers",
			role:              RoleLeader,
			backendFramework:  BackendFrameworkTRTLLM,
			leaderCliqueName:  "service-ldr",
			workerCliqueNames: []string{"service-wkr1", "service-wkr2"},
			expected:          []string{"service-wkr1", "service-wkr2"},
		},
		{
			name:              "trtllm_worker_has_no_dependencies",
			role:              RoleWorker,
			backendFramework:  BackendFrameworkTRTLLM,
			leaderCliqueName:  "service-ldr",
			workerCliqueNames: []string{"service-wkr"},
			expected:          nil,
		},
		{
			name:              "leader_with_empty_worker_names",
			role:              RoleLeader,
			backendFramework:  BackendFrameworkTRTLLM,
			leaderCliqueName:  "service-ldr",
			workerCliqueNames: nil,
			expected:          nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getCliqueStartupDependencies(
				tt.role,
				tt.backendFramework,
				tt.leaderCliqueName,
				tt.workerCliqueNames,
			)

			if !reflect.DeepEqual(result, tt.expected) {
				t.Errorf("getCliqueStartupDependencies() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGenerateGrovePodCliqueSet_StartsAfterDependencies(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}

	tests := []struct {
		name              string
		backendFramework  string
		expectedDeps      map[string][]string // clique name -> expected StartsAfter dependencies
		expectStartupType bool
	}{
		{
			name:             "vllm_worker_starts_after_leader",
			backendFramework: string(BackendFrameworkVLLM),
			expectedDeps: map[string][]string{
				"main-wkr": nil, // worker starts after leader
				"main-ldr": nil, // leader has no dependencies
			},
			expectStartupType: false,
		},
		{
			name:             "sglang_worker_starts_after_leader",
			backendFramework: string(BackendFrameworkSGLang),
			expectedDeps: map[string][]string{
				"main-wkr": nil, // worker starts after leader
				"main-ldr": nil, // leader has no dependencies
			},
			expectStartupType: false,
		},
		{
			name:             "trtllm_leader_starts_after_worker",
			backendFramework: string(BackendFrameworkTRTLLM),
			expectedDeps: map[string][]string{
				"main-ldr": {"main-wkr"}, // leader starts after worker
				"main-wkr": nil,          // worker has no dependencies
			},
			expectStartupType: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dynamoDeployment := &v1alpha1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoGraphDeploymentSpec{
					BackendFramework: tt.backendFramework,
					Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
						"main": {
							Multinode: &v1alpha1.MultinodeSpec{
								NodeCount: 2,
							},
							ComponentType: "worker", // Must be worker to trigger backend detection
							Replicas:      ptr.To(int32(1)),
							Resources: &common.Resources{
								Requests: &common.ResourceItem{
									GPU: "1", // 1 GPU per node
								},
							},
						},
					},
				},
			}

			controllerConfig := controller_common.Config{
				EtcdAddress: "etcd-address",
				NatsAddress: "nats-address",
			}

			got, err := GenerateGrovePodCliqueSet(context.Background(), dynamoDeployment, controllerConfig, secretsRetriever)
			if err != nil {
				t.Errorf("GenerateGrovePodCliqueSet() error = %v", err)
				return
			}

			// Verify that StartupType is set to Explicit
			if tt.expectStartupType {
				if got.Spec.Template.StartupType == nil || *got.Spec.Template.StartupType != grovev1alpha1.CliqueStartupTypeExplicit {
					t.Errorf("Expected StartupType to be CliqueStartupTypeExplicit, got %v", got.Spec.Template.StartupType)
				}
			} else {
				if got.Spec.Template.StartupType == nil || *got.Spec.Template.StartupType != grovev1alpha1.CliqueStartupTypeAnyOrder {
					t.Errorf("Expected StartupType to be CliqueStartupTypeAnyOrder, got %v", got.Spec.Template.StartupType)
				}
			}

			// Verify StartsAfter dependencies for each clique
			cliqueMap := make(map[string]*grovev1alpha1.PodCliqueTemplateSpec)
			for _, clique := range got.Spec.Template.Cliques {
				cliqueMap[clique.Name] = clique
			}

			for cliqueName, expectedDeps := range tt.expectedDeps {
				clique, exists := cliqueMap[cliqueName]
				if !exists {
					t.Errorf("Expected clique %s not found", cliqueName)
					continue
				}

				if expectedDeps == nil {
					if len(clique.Spec.StartsAfter) != 0 {
						t.Errorf("Clique %s should have no StartsAfter dependencies, but has %v", cliqueName, clique.Spec.StartsAfter)
					}
				} else {
					if len(clique.Spec.StartsAfter) != len(expectedDeps) {
						t.Errorf("Clique %s expected %d StartsAfter dependencies, got %d", cliqueName, len(expectedDeps), len(clique.Spec.StartsAfter))
						continue
					}

					for i, expectedDep := range expectedDeps {
						if i >= len(clique.Spec.StartsAfter) || clique.Spec.StartsAfter[i] != expectedDep {
							t.Errorf("Clique %s expected StartsAfter[%d] = %s, got %v", cliqueName, i, expectedDep, clique.Spec.StartsAfter)
						}
					}
				}
			}
		})
	}
}

func TestGenerateBasePodSpec_Frontend(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	controllerConfig := controller_common.Config{}
	dynamoDeployment := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-deployment",
			Namespace: "default",
		},
	}

	tests := []struct {
		name             string
		component        *v1alpha1.DynamoComponentDeploymentSharedSpec
		backendFramework BackendFramework
		wantEnvVars      map[string]string
		wantErr          bool
	}{
		{
			name: "frontend with default command",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
			},
			backendFramework: BackendFrameworkVLLM,
			wantEnvVars: map[string]string{
				"DYN_HTTP_PORT": fmt.Sprintf("%d", commonconsts.DynamoServicePort),
			},
		},
		{
			name: "frontend with overriding env var",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				Envs: []corev1.EnvVar{
					{
						Name:  "DYN_HTTP_PORT",
						Value: "3000",
					},
				},
			},
			backendFramework: BackendFrameworkVLLM,
			wantEnvVars: map[string]string{
				"DYN_HTTP_PORT": "3000",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec, err := GenerateBasePodSpec(
				tt.component,
				tt.backendFramework,
				secretsRetriever,
				dynamoDeployment.Name,
				dynamoDeployment.Namespace,
				RoleMain,
				1,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
				"test-service",
			)

			if (err != nil) != tt.wantErr {
				t.Errorf("GenerateBasePodSpec() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr {
				return
			}

			// Check command and args
			wantCommand := []string{"python3"}
			wantArgs := []string{"-m", "dynamo.frontend"}
			if !reflect.DeepEqual(podSpec.Containers[0].Command, wantCommand) {
				t.Errorf("GenerateBasePodSpec() command = %v, want %v",
					podSpec.Containers[0].Command, wantCommand)
			}
			if !reflect.DeepEqual(podSpec.Containers[0].Args, wantArgs) {
				t.Errorf("GenerateBasePodSpec() args = %v, want %v",
					podSpec.Containers[0].Args, wantArgs)
			}

			// Check environment variables
			envVars := make(map[string]string)
			for _, env := range podSpec.Containers[0].Env {
				envVars[env.Name] = env.Value
			}
			for k, v := range tt.wantEnvVars {
				if envVars[k] != v {
					t.Errorf("GenerateBasePodSpec() env var %s = %v, want %v",
						k, envVars[k], v)
				}
			}
		})
	}
}

func TestGenerateBasePodSpec_PlannerServiceAccount(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	controllerConfig := controller_common.Config{}

	tests := []struct {
		name               string
		component          *v1alpha1.DynamoComponentDeploymentSharedSpec
		expectedServiceAcc string
	}{
		{
			name: "Planner component should have planner service account",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypePlanner,
			},
			expectedServiceAcc: commonconsts.PlannerServiceAccountName,
		},
		{
			name: "Planner service account should not be set for non-planner components",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
			},
			expectedServiceAcc: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec, err := GenerateBasePodSpec(
				tt.component,
				BackendFrameworkSGLang,
				secretsRetriever,
				"test-deployment",
				"default",
				RoleMain,
				1,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
				"test-service",
			)

			if err != nil {
				t.Errorf("GenerateBasePodSpec() error = %v", err)
				return
			}

			if podSpec.ServiceAccountName != tt.expectedServiceAcc {
				t.Errorf("GenerateBasePodSpec() serviceAccountName = %v, want %v",
					podSpec.ServiceAccountName, tt.expectedServiceAcc)
			}
		})
	}
}

func TestGenerateBasePodSpec_DisableImagePullSecretDiscovery(t *testing.T) {
	tests := []struct {
		name                     string
		component                *v1alpha1.DynamoComponentDeploymentSharedSpec
		secretsRetriever         SecretsRetriever
		expectedImagePullSecrets []corev1.LocalObjectReference
	}{
		{
			name: "disable docker secrets annotation set to true",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDisableImagePullSecretDiscovery: commonconsts.KubeLabelValueTrue,
				},
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Image: "test-registry/test-image:latest",
					},
				},
			},
			secretsRetriever:         &mockSecretsRetrieverWithSecrets{},
			expectedImagePullSecrets: nil, // Should be nil when disabled
		},
		{
			name: "disable docker secrets annotation set to false",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDisableImagePullSecretDiscovery: commonconsts.KubeLabelValueFalse,
				},
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Image: "test-registry/test-image:latest",
					},
				},
			},
			secretsRetriever: &mockSecretsRetrieverWithSecrets{},
			expectedImagePullSecrets: []corev1.LocalObjectReference{
				{Name: "test-docker-secret"},
			}, // Should be present when enabled
		},
		{
			name: "disable docker secrets annotation not set (default behavior)",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Image: "test-registry/test-image:latest",
					},
				},
			},
			secretsRetriever: &mockSecretsRetrieverWithSecrets{},
			expectedImagePullSecrets: []corev1.LocalObjectReference{
				{Name: "test-docker-secret"},
			}, // Should be present by default
		},
		{
			name: "disable docker secrets annotation set to invalid value",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDisableImagePullSecretDiscovery: "invalid",
				},
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Image: "test-registry/test-image:latest",
					},
				},
			},
			secretsRetriever: &mockSecretsRetrieverWithSecrets{},
			expectedImagePullSecrets: []corev1.LocalObjectReference{
				{Name: "test-docker-secret"},
			}, // Should be present when annotation is not "true"
		},
		{
			name: "disable docker secrets but no secrets retriever",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDisableImagePullSecretDiscovery: commonconsts.KubeLabelValueFalse,
				},
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Image: "test-registry/test-image:latest",
					},
				},
			},
			secretsRetriever:         nil,
			expectedImagePullSecrets: nil, // Should be nil when no retriever
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			controllerConfig := controller_common.Config{}

			podSpec, err := GenerateBasePodSpec(
				tt.component,
				BackendFrameworkNoop,
				tt.secretsRetriever,
				"test-deployment",
				"default",
				RoleMain,
				1,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
				"test-service",
			)

			if err != nil {
				t.Errorf("GenerateBasePodSpec() error = %v", err)
				return
			}

			if !reflect.DeepEqual(podSpec.ImagePullSecrets, tt.expectedImagePullSecrets) {
				t.Errorf("GenerateBasePodSpec() ImagePullSecrets = %v, want %v",
					podSpec.ImagePullSecrets, tt.expectedImagePullSecrets)
			}
		})
	}
}

func TestGenerateBasePodSpec_DiscoverBackend(t *testing.T) {
	tests := []struct {
		name             string
		component        *v1alpha1.DynamoComponentDeploymentSharedSpec
		controllerConfig controller_common.Config
		wantEnvVar       string
	}{
		{
			name: "Discover backend should be set",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoDiscoveryBackend: "kubernetes",
				},
			},
			wantEnvVar: "kubernetes",
		},
		{
			name: "Discover backend should override the controller config",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoDiscoveryBackend: "test",
				},
			},
			controllerConfig: controller_common.Config{
				DiscoveryBackend: "etcd",
			},
			wantEnvVar: "test",
		},
		{
			name: "Discover backend should be set by the controller config",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				Annotations: map[string]string{},
			},
			controllerConfig: controller_common.Config{
				DiscoveryBackend: "etcd",
			},
			wantEnvVar: "etcd",
		},
		{
			name: "Discover backend empty string",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoDiscoveryBackend: "",
				},
			},
			controllerConfig: controller_common.Config{
				DiscoveryBackend: "",
			},
		},
		{
			name:      "Discover backend not set",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{},
		},
	}
	secretsRetriever := &mockSecretsRetriever{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec, err := GenerateBasePodSpec(
				tt.component,
				BackendFrameworkSGLang,
				secretsRetriever,
				"test-deployment",
				"default",
				RoleMain,
				1,
				tt.controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
				"test-service",
			)
			if !assert.NoError(t, err) {
				return
			}
			if tt.wantEnvVar != "" {
				assert.Contains(t, podSpec.Containers[0].Env, corev1.EnvVar{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: tt.wantEnvVar})
			} else {
				for _, env := range podSpec.Containers[0].Env {
					if env.Name == commonconsts.DynamoDiscoveryBackendEnvVar {
						t.Errorf("GenerateBasePodSpec() Discover backend env var should not be set, got %s", env.Value)
					}
				}
			}
		})
	}
}

func TestGenerateBasePodSpec_Worker(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	controllerConfig := controller_common.Config{}

	tests := []struct {
		name            string
		component       *v1alpha1.DynamoComponentDeploymentSharedSpec
		expectedPodSpec *corev1.PodSpec
	}{
		{
			name: "Planner component should have planner service account",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				Envs: []corev1.EnvVar{
					{Name: "ANOTHER_COMPONENTENV", Value: "true"},
				},
				ComponentType: commonconsts.ComponentTypeWorker,
				ExtraPodSpec: &common.ExtraPodSpec{
					MainContainer: &corev1.Container{
						Command: []string{"python3"},
						Args:    []string{"-m", "dynamo.worker"},
						Env: []corev1.EnvVar{
							{Name: "ANOTHER_CONTAINER_ENV", Value: "true"},
						},
					},
				},
			},
			expectedPodSpec: &corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name:    commonconsts.MainContainerName,
						Command: []string{"python3"},
						Args:    []string{"-m", "dynamo.worker"},
						Env: []corev1.EnvVar{
							{Name: "ANOTHER_COMPONENTENV", Value: "true"},
							{Name: "ANOTHER_CONTAINER_ENV", Value: "true"},
							{Name: commonconsts.DynamoComponentEnvVar, Value: "worker"},
							{Name: commonconsts.DynamoNamespaceEnvVar, Value: ""},
							{Name: "DYN_PARENT_DGD_K8S_NAME", Value: "test-deployment"},
							{Name: "DYN_PARENT_DGD_K8S_NAMESPACE", Value: "default"},
							{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
							{Name: "DYN_SYSTEM_PORT", Value: "9090"},
							{Name: "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", Value: "[\"generate\"]"},
							{Name: "POD_NAME", ValueFrom: &corev1.EnvVarSource{
								FieldRef: &corev1.ObjectFieldSelector{
									FieldPath: "metadata.name",
								},
							}},
							{Name: "POD_NAMESPACE", ValueFrom: &corev1.EnvVarSource{
								FieldRef: &corev1.ObjectFieldSelector{
									FieldPath: "metadata.namespace",
								},
							}},
						},
						VolumeMounts: []corev1.VolumeMount{
							{
								Name:      "shared-memory",
								MountPath: "/dev/shm",
							},
						},
						LivenessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{
									Path: "/live",
									Port: intstr.FromString(commonconsts.DynamoSystemPortName),
								},
							},
							PeriodSeconds:    5,
							TimeoutSeconds:   4,
							FailureThreshold: 1,
						},
						ReadinessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{
									Path: "/health",
									Port: intstr.FromString(commonconsts.DynamoSystemPortName),
								},
							},
							PeriodSeconds:    10,
							TimeoutSeconds:   4,
							FailureThreshold: 3,
						},
						StartupProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{
									Path: "/live",
									Port: intstr.FromString(commonconsts.DynamoSystemPortName),
								},
							},
							PeriodSeconds:    10,
							TimeoutSeconds:   5,
							FailureThreshold: 720,
						},
						Ports: []corev1.ContainerPort{
							{
								Name:          commonconsts.DynamoSystemPortName,
								ContainerPort: int32(commonconsts.DynamoSystemPort),
								Protocol:      corev1.ProtocolTCP,
							},
						},
					},
				},
				RestartPolicy:                 corev1.RestartPolicyAlways,
				TerminationGracePeriodSeconds: ptr.To(int64(60)),
				SecurityContext: &corev1.PodSecurityContext{
					// Only fsGroup is injected by default for volume permissions
					FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
				},
				Volumes: []corev1.Volume{
					{
						Name: "shared-memory",
						VolumeSource: corev1.VolumeSource{
							EmptyDir: &corev1.EmptyDirVolumeSource{
								Medium:    corev1.StorageMediumMemory,
								SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec, err := GenerateBasePodSpec(
				tt.component,
				BackendFrameworkSGLang,
				secretsRetriever,
				"test-deployment",
				"default",
				RoleMain,
				1,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
				"test-service",
			)

			if err != nil {
				t.Errorf("GenerateBasePodSpec() error = %v", err)
				return
			}

			diff := cmp.Diff(tt.expectedPodSpec, podSpec)
			if diff != "" {
				t.Errorf("GenerateBasePodSpec() podSpec = %v, want %v, diff = %v", podSpec, tt.expectedPodSpec, diff)
			}
		})
	}
}

func TestGenerateBasePodSpec_VolumeMounts(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	controllerConfig := controller_common.Config{}

	tests := []struct {
		name           string
		component      *v1alpha1.DynamoComponentDeploymentSharedSpec
		expectError    bool
		expectedPVCs   []string
		expectedMounts []corev1.VolumeMount
	}{
		{
			name: "valid volumeMounts",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:       "test-pvc",
						MountPoint: "/data",
					},
				},
			},
			expectError:  false,
			expectedPVCs: []string{"test-pvc"},
			expectedMounts: []corev1.VolumeMount{
				{Name: "test-pvc", MountPath: "/data"},
				{Name: "shared-memory", MountPath: "/dev/shm"},
			},
		},
		{
			name: "multiple volumeMounts",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{Name: "pvc1", MountPoint: "/data1"},
					{Name: "pvc2", MountPoint: "/data2"},
				},
			},
			expectError:  false,
			expectedPVCs: []string{"pvc1", "pvc2"},
			expectedMounts: []corev1.VolumeMount{
				{Name: "pvc1", MountPath: "/data1"},
				{Name: "pvc2", MountPath: "/data2"},
				{Name: "shared-memory", MountPath: "/dev/shm"},
			},
		},
		{
			name: "empty volumeMount name",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{Name: "", MountPoint: "/data"},
				},
			},
			expectError: true,
		},
		{
			name: "empty volumeMount mountPoint",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{Name: "test-pvc", MountPoint: ""},
				},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec, err := GenerateBasePodSpec(
				tt.component,
				BackendFrameworkVLLM,
				secretsRetriever,
				"test-deployment",
				"default",
				RoleMain,
				1,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
				"test-service",
			)

			if tt.expectError {
				if err == nil {
					t.Errorf("GenerateBasePodSpec() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("GenerateBasePodSpec() unexpected error: %v", err)
				return
			}

			// Check expected PVCs are present in volumes
			for _, expectedPVC := range tt.expectedPVCs {
				found := false
				for _, volume := range podSpec.Volumes {
					if volume.Name == expectedPVC && volume.PersistentVolumeClaim != nil {
						if volume.PersistentVolumeClaim.ClaimName == expectedPVC {
							found = true
							break
						}
					}
				}
				if !found {
					t.Errorf("GenerateBasePodSpec() expected PVC volume %s not found", expectedPVC)
				}
			}

			// Check expected mounts are present
			if len(podSpec.Containers) == 0 {
				t.Errorf("GenerateBasePodSpec() no containers found")
				return
			}

			container := podSpec.Containers[0]
			for _, expectedMount := range tt.expectedMounts {
				found := false
				for _, mount := range container.VolumeMounts {
					if mount.Name == expectedMount.Name && mount.MountPath == expectedMount.MountPath {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("GenerateBasePodSpec() expected volume mount %+v not found", expectedMount)
				}
			}
		})
	}
}

func TestGenerateBasePodSpec_ResourceClaims(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	controllerConfig := controller_common.Config{}

	tests := []struct {
		name                   string
		component              *v1alpha1.DynamoComponentDeploymentSharedSpec
		expectError            bool
		expectedResourceClaims []corev1.ResourceClaim
		expectedPodClaims      []corev1.PodResourceClaim
		expectedVolumes        []corev1.Volume
	}{
		{
			name: "component with resource claims",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				Resources: &common.Resources{
					Requests: &common.ResourceItem{
						CPU:    "130",
						Memory: "800Gi",
					},
					Limits: &common.ResourceItem{
						CPU:    "130",
						Memory: "800Gi",
						GPU:    "4",
					},
					Claims: []corev1.ResourceClaim{
						{
							Name: "compute-domain-channel",
						},
					},
				},
				ExtraPodSpec: &common.ExtraPodSpec{
					PodSpec: &corev1.PodSpec{
						ResourceClaims: []corev1.PodResourceClaim{
							{
								Name:                      "compute-domain-channel",
								ResourceClaimTemplateName: ptr.To("trtllm-test-compute-domain-channel"),
							},
						},
						Volumes: []corev1.Volume{
							{
								Name: "model-storage",
								VolumeSource: corev1.VolumeSource{
									PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
										ClaimName: "dynamo-pvc",
									},
								},
							},
						},
					},
					MainContainer: &corev1.Container{
						Image: "rohanv672/dynamo:v0.5.1-trtllm",
						Args: []string{
							"python3 -m dynamo.trtllm --model-path /data/deepseek-r1 --served-model-name deepseek-ai/DeepSeek-R1 --extra-engine-args /data/engine_configs/wide_ep_agg.yaml",
						},
						Command: []string{"/bin/sh", "-c"},
						VolumeMounts: []corev1.VolumeMount{
							{
								Name:      "model-storage",
								MountPath: "/data",
							},
						},
					},
				},
			},
			expectError: false,
			expectedResourceClaims: []corev1.ResourceClaim{
				{
					Name: "compute-domain-channel",
				},
			},
			expectedPodClaims: []corev1.PodResourceClaim{
				{
					Name:                      "compute-domain-channel",
					ResourceClaimTemplateName: ptr.To("trtllm-test-compute-domain-channel"),
				},
			},
			expectedVolumes: []corev1.Volume{
				{
					Name: "model-storage",
					VolumeSource: corev1.VolumeSource{
						PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
							ClaimName: "dynamo-pvc",
						},
					},
				},
				{
					Name: "shared-memory",
					VolumeSource: corev1.VolumeSource{
						EmptyDir: &corev1.EmptyDirVolumeSource{
							Medium:    corev1.StorageMediumMemory,
							SizeLimit: func() *resource.Quantity { q := resource.MustParse(commonconsts.DefaultSharedMemorySize); return &q }(),
						},
					},
				},
			},
		},
		{
			name: "component with multiple resource claims",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeWorker,
				Resources: &common.Resources{
					Claims: []corev1.ResourceClaim{
						{
							Name: "compute-domain-channel",
						},
						{
							Name: "network-domain-channel",
						},
					},
				},
				ExtraPodSpec: &common.ExtraPodSpec{
					PodSpec: &corev1.PodSpec{
						ResourceClaims: []corev1.PodResourceClaim{
							{
								Name:                      "compute-domain-channel",
								ResourceClaimTemplateName: ptr.To("compute-template"),
							},
							{
								Name:                      "network-domain-channel",
								ResourceClaimTemplateName: ptr.To("network-template"),
							},
						},
					},
					MainContainer: &corev1.Container{
						Image:   "test-image",
						Command: []string{"python3"},
						Args:    []string{"-m", "dynamo.worker"},
					},
				},
			},
			expectError: false,
			expectedResourceClaims: []corev1.ResourceClaim{
				{
					Name: "compute-domain-channel",
				},
				{
					Name: "network-domain-channel",
				},
			},
			expectedPodClaims: []corev1.PodResourceClaim{
				{
					Name:                      "compute-domain-channel",
					ResourceClaimTemplateName: ptr.To("compute-template"),
				},
				{
					Name:                      "network-domain-channel",
					ResourceClaimTemplateName: ptr.To("network-template"),
				},
			},
		},
		{
			name: "component without resource claims",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				Resources: &common.Resources{
					Requests: &common.ResourceItem{
						CPU:    "1",
						Memory: "1Gi",
					},
				},
			},
			expectError:            false,
			expectedResourceClaims: nil,
			expectedPodClaims:      nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec, err := GenerateBasePodSpec(
				tt.component,
				BackendFrameworkTRTLLM,
				secretsRetriever,
				"test-deployment",
				"default",
				RoleMain,
				1,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
				"test-service",
			)

			if tt.expectError {
				if err == nil {
					t.Errorf("GenerateBasePodSpec() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("GenerateBasePodSpec() unexpected error: %v", err)
				return
			}

			// Check containers exist
			if len(podSpec.Containers) == 0 {
				t.Errorf("GenerateBasePodSpec() no containers found")
				return
			}

			container := podSpec.Containers[0]

			// Check resource claims in container resources using reflect.DeepEqual
			if !reflect.DeepEqual(container.Resources.Claims, tt.expectedResourceClaims) {
				t.Errorf("GenerateBasePodSpec() resource claims mismatch:\ngot:  %+v\nwant: %+v",
					container.Resources.Claims, tt.expectedResourceClaims)
			}

			// Check pod resource claims using reflect.DeepEqual
			if !reflect.DeepEqual(podSpec.ResourceClaims, tt.expectedPodClaims) {
				t.Errorf("GenerateBasePodSpec() pod resource claims mismatch:\ngot:  %+v\nwant: %+v",
					podSpec.ResourceClaims, tt.expectedPodClaims)
			}

			// Check expected volumes if specified using reflect.DeepEqual
			if tt.expectedVolumes != nil {
				if !reflect.DeepEqual(podSpec.Volumes, tt.expectedVolumes) {
					t.Errorf("GenerateBasePodSpec() volumes mismatch:\ngot:  %+v\nwant: %+v",
						podSpec.Volumes, tt.expectedVolumes)
				}
			}

			// Verify resource requests and limits are properly set when claims are present
			if len(tt.expectedResourceClaims) > 0 {
				// Check that standard resources are still processed correctly
				if tt.component.Resources != nil {
					if tt.component.Resources.Requests != nil {
						if tt.component.Resources.Requests.CPU != "" {
							if container.Resources.Requests == nil {
								t.Errorf("GenerateBasePodSpec() expected CPU request to be set")
							} else if cpu, exists := container.Resources.Requests[corev1.ResourceCPU]; !exists || cpu.IsZero() {
								t.Errorf("GenerateBasePodSpec() expected CPU request to be set")
							}
						}
						if tt.component.Resources.Requests.Memory != "" {
							if container.Resources.Requests == nil {
								t.Errorf("GenerateBasePodSpec() expected Memory request to be set")
							} else if memory, exists := container.Resources.Requests[corev1.ResourceMemory]; !exists || memory.IsZero() {
								t.Errorf("GenerateBasePodSpec() expected Memory request to be set")
							}
						}
					}
					if tt.component.Resources.Limits != nil {
						if tt.component.Resources.Limits.GPU != "" {
							if container.Resources.Limits == nil {
								t.Errorf("GenerateBasePodSpec() expected GPU limit to be set")
							} else if gpu, exists := container.Resources.Limits[corev1.ResourceName("nvidia.com/gpu")]; !exists || gpu.IsZero() {
								t.Errorf("GenerateBasePodSpec() expected GPU limit to be set")
							}
						}
					}
				}
			}
		})
	}
}

func TestGenerateBasePodSpec_UseAsCompilationCache_BackendSupport(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	controllerConfig := controller_common.Config{}

	tests := []struct {
		name             string
		component        *v1alpha1.DynamoComponentDeploymentSharedSpec
		backendFramework BackendFramework
		expectError      bool
		expectedMount    *corev1.VolumeMount
	}{
		{
			name: "useAsCompilationCache with custom mount point",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "cache-pvc",
						MountPoint:            "/custom/cache",
						UseAsCompilationCache: true,
					},
				},
			},
			backendFramework: BackendFrameworkVLLM,
			expectError:      false,
			expectedMount:    &corev1.VolumeMount{Name: "cache-pvc", MountPath: "/custom/cache"},
		},
		{
			name: "useAsCompilationCache with default mount point for VLLM",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "cache-pvc",
						UseAsCompilationCache: true,
					},
				},
			},
			backendFramework: BackendFrameworkVLLM,
			expectError:      false,
			expectedMount:    &corev1.VolumeMount{Name: "cache-pvc", MountPath: commonconsts.DefaultVLLMCacheMountPoint},
		},
		{
			name: "useAsCompilationCache without mount point for SGLang - should error",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "cache-pvc",
						UseAsCompilationCache: true,
					},
				},
			},
			backendFramework: BackendFrameworkSGLang,
			expectError:      true, // SGLang doesn't support compilation cache, requires explicit mount point
			expectedMount:    nil,
		},
		{
			name: "useAsCompilationCache with explicit mount point for SGLang - should work",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "cache-pvc",
						MountPoint:            "/custom/sglang/cache",
						UseAsCompilationCache: true,
					},
				},
			},
			backendFramework: BackendFrameworkSGLang,
			expectError:      false,
			expectedMount:    &corev1.VolumeMount{Name: "cache-pvc", MountPath: "/custom/sglang/cache"},
		},
		{
			name: "useAsCompilationCache without mount point for TensorRT-LLM - should error",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "cache-pvc",
						UseAsCompilationCache: true,
					},
				},
			},
			backendFramework: BackendFrameworkTRTLLM,
			expectError:      true, // TensorRT-LLM doesn't support compilation cache, requires explicit mount point
			expectedMount:    nil,
		},
		{
			name: "useAsCompilationCache with explicit mount point for TensorRT-LLM - should work",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "cache-pvc",
						MountPoint:            "/custom/trtllm/cache",
						UseAsCompilationCache: true,
					},
				},
			},
			backendFramework: BackendFrameworkTRTLLM,
			expectError:      false,
			expectedMount:    &corev1.VolumeMount{Name: "cache-pvc", MountPath: "/custom/trtllm/cache"},
		},
		{
			name: "no useAsCompilationCache volumes - should be ignored",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:       "regular-pvc",
						MountPoint: "/data",
					},
				},
			},
			backendFramework: BackendFrameworkVLLM,
			expectError:      false,
			expectedMount:    nil, // Should be ignored, not error
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec, err := GenerateBasePodSpec(
				tt.component,
				tt.backendFramework,
				secretsRetriever,
				"test-deployment",
				"default",
				RoleMain,
				1,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
				"test-service",
			)

			if tt.expectError {
				if err == nil {
					t.Errorf("GenerateBasePodSpec() expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("GenerateBasePodSpec() unexpected error: %v", err)
				return
			}

			if tt.expectedMount != nil {
				// Check PVC volume exists
				found := false
				for _, volume := range podSpec.Volumes {
					if volume.Name == tt.expectedMount.Name && volume.PersistentVolumeClaim != nil {
						if volume.PersistentVolumeClaim.ClaimName == tt.expectedMount.Name {
							found = true
							break
						}
					}
				}
				if !found {
					t.Errorf("GenerateBasePodSpec() expected PVC volume %s not found", tt.expectedMount.Name)
				}

				// Check volume mount exists
				if len(podSpec.Containers) == 0 {
					t.Errorf("GenerateBasePodSpec() no containers found")
					return
				}

				container := podSpec.Containers[0]
				found = false
				for _, mount := range container.VolumeMounts {
					if mount.Name == tt.expectedMount.Name && mount.MountPath == tt.expectedMount.MountPath {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("GenerateBasePodSpec() expected volume mount %+v not found", tt.expectedMount)
				}
			}
		})
	}
}

func TestGenerateBasePodSpec_SecurityContext(t *testing.T) {
	secretsRetriever := &mockSecretsRetriever{}
	controllerConfig := controller_common.Config{}

	tests := []struct {
		name                    string
		component               *v1alpha1.DynamoComponentDeploymentSharedSpec
		expectedSecurityContext *corev1.PodSecurityContext
		description             string
	}{
		{
			name: "no security context provided - should apply fsGroup default only",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
			},
			expectedSecurityContext: &corev1.PodSecurityContext{
				FSGroup: ptr.To(int64(commonconsts.DefaultSecurityContextFSGroup)),
			},
			description: "Operator should only inject fsGroup for volume permissions, not UID/GID (backward compatible)",
		},
		{
			name: "full security context override - should use user values",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				ExtraPodSpec: &common.ExtraPodSpec{
					PodSpec: &corev1.PodSpec{
						SecurityContext: &corev1.PodSecurityContext{
							RunAsNonRoot: ptr.To(true),
							RunAsUser:    ptr.To(int64(5000)),
							RunAsGroup:   ptr.To(int64(5000)),
							FSGroup:      ptr.To(int64(5000)),
						},
					},
				},
			},
			expectedSecurityContext: &corev1.PodSecurityContext{
				RunAsNonRoot: ptr.To(true),
				RunAsUser:    ptr.To(int64(5000)),
				RunAsGroup:   ptr.To(int64(5000)),
				FSGroup:      ptr.To(int64(5000)),
			},
			description: "User-provided security context should completely override defaults",
		},
		{
			name: "partial security context override - user gets full control",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				ExtraPodSpec: &common.ExtraPodSpec{
					PodSpec: &corev1.PodSpec{
						SecurityContext: &corev1.PodSecurityContext{
							RunAsUser:  ptr.To(int64(2000)),
							RunAsGroup: ptr.To(int64(3000)),
						},
					},
				},
			},
			expectedSecurityContext: &corev1.PodSecurityContext{
				RunAsUser:  ptr.To(int64(2000)),
				RunAsGroup: ptr.To(int64(3000)),
			},
			description: "Partial user override gets full control - no defaults injected",
		},
		{
			name: "only fsGroup override - user gets full control",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				ExtraPodSpec: &common.ExtraPodSpec{
					PodSpec: &corev1.PodSpec{
						SecurityContext: &corev1.PodSecurityContext{
							FSGroup: ptr.To(int64(7000)),
						},
					},
				},
			},
			expectedSecurityContext: &corev1.PodSecurityContext{
				FSGroup: ptr.To(int64(7000)),
			},
			description: "Only fsGroup override - user gets full control, no defaults injected",
		},
		{
			name: "fsGroup 2000 example - exactly what user requested",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				ExtraPodSpec: &common.ExtraPodSpec{
					PodSpec: &corev1.PodSpec{
						SecurityContext: &corev1.PodSecurityContext{
							FSGroup: ptr.To(int64(2000)),
						},
					},
				},
			},
			expectedSecurityContext: &corev1.PodSecurityContext{
				FSGroup: ptr.To(int64(2000)),
			},
			description: "User sets fsGroup:2000, gets ONLY that - critical for allowing root users",
		},
		{
			name: "OpenShift-style namespace range - should use user values",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				ComponentType: commonconsts.ComponentTypeFrontend,
				ExtraPodSpec: &common.ExtraPodSpec{
					PodSpec: &corev1.PodSpec{
						SecurityContext: &corev1.PodSecurityContext{
							RunAsNonRoot: ptr.To(true),
							RunAsUser:    ptr.To(int64(1000700001)),
							RunAsGroup:   ptr.To(int64(1000700001)),
							FSGroup:      ptr.To(int64(1000700001)),
						},
					},
				},
			},
			expectedSecurityContext: &corev1.PodSecurityContext{
				RunAsNonRoot: ptr.To(true),
				RunAsUser:    ptr.To(int64(1000700001)),
				RunAsGroup:   ptr.To(int64(1000700001)),
				FSGroup:      ptr.To(int64(1000700001)),
			},
			description: "OpenShift namespace UID/GID ranges should be respected",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			podSpec, err := GenerateBasePodSpec(
				tt.component,
				BackendFrameworkNoop,
				secretsRetriever,
				"test-deployment",
				"default",
				RoleMain,
				1,
				controllerConfig,
				commonconsts.MultinodeDeploymentTypeGrove,
				"test-service",
			)

			if err != nil {
				t.Errorf("GenerateBasePodSpec() unexpected error: %v", err)
				return
			}

			// Compare the entire SecurityContext using cmp.Diff
			if diff := cmp.Diff(tt.expectedSecurityContext, podSpec.SecurityContext); diff != "" {
				t.Errorf("GenerateBasePodSpec() SecurityContext mismatch (-want +got):\n%s\nDescription: %s", diff, tt.description)
			}
		})
	}
}
