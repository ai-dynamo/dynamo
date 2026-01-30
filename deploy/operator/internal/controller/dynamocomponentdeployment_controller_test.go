/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
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
 * Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES
 */

package controller

import (
	"context"
	"fmt"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/google/go-cmp/cmp"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/format"
	istioNetworking "istio.io/api/networking/v1beta1"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
	volcanov1beta1 "volcano.sh/apis/pkg/apis/scheduling/v1beta1"
)

func TestIsDeploymentReady(t *testing.T) {
	type args struct {
		deployment *appsv1.Deployment
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{
			name: "deployment is nil",
			args: args{
				deployment: nil,
			},
			want: false,
		},
		{
			name: "not ready",
			args: args{
				deployment: &appsv1.Deployment{
					Spec: appsv1.DeploymentSpec{},
					Status: appsv1.DeploymentStatus{
						Conditions: []appsv1.DeploymentCondition{
							{
								Type:   appsv1.DeploymentAvailable,
								Status: corev1.ConditionFalse,
							},
						},
					},
				},
			},
			want: false,
		},
		{
			name: "not ready (paused)",
			args: args{
				deployment: &appsv1.Deployment{
					Spec: appsv1.DeploymentSpec{
						Paused: true,
					},
				},
			},
			want: false,
		},
		{
			name: "not ready (surging)",
			args: args{
				deployment: &appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Generation: 1,
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &[]int32{2}[0],
					},
					Status: appsv1.DeploymentStatus{
						ObservedGeneration: 1,
						UpdatedReplicas:    1,
						AvailableReplicas:  1,
						Replicas:           2,
					},
				},
			},
			want: false,
		},
		{
			name: "ready",
			args: args{
				deployment: &appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Generation: 1,
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &[]int32{1}[0],
					},
					Status: appsv1.DeploymentStatus{
						ObservedGeneration: 1,
						UpdatedReplicas:    1,
						AvailableReplicas:  1,
						Replicas:           1,
						Conditions: []appsv1.DeploymentCondition{
							{
								Type:   appsv1.DeploymentAvailable,
								Status: corev1.ConditionTrue,
							},
						},
					},
				},
			},
			want: true,
		},
		{
			name: "ready (no desired replicas)",
			args: args{
				deployment: &appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Generation: 1,
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &[]int32{0}[0],
					},
				},
			},
			want: true,
		},
		{
			name: "not ready (condition false)",
			args: args{
				deployment: &appsv1.Deployment{
					ObjectMeta: metav1.ObjectMeta{
						Generation: 1,
					},
					Spec: appsv1.DeploymentSpec{
						Replicas: &[]int32{1}[0],
					},
					Status: appsv1.DeploymentStatus{
						ObservedGeneration: 1,
						UpdatedReplicas:    1,
						AvailableReplicas:  1,
						Conditions: []appsv1.DeploymentCondition{
							{
								Type:   appsv1.DeploymentAvailable,
								Status: corev1.ConditionFalse,
							},
						},
					},
				},
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := IsDeploymentReady(tt.args.deployment); got != tt.want {
				t.Errorf("IsDeploymentReady() = %v, want %v", got, tt.want)
			}
		})
	}
}

type mockEtcdStorage struct {
	deleteKeysFunc func(ctx context.Context, prefix string) error
}

func (m *mockEtcdStorage) DeleteKeys(ctx context.Context, prefix string) error {
	return m.deleteKeysFunc(ctx, prefix)
}

func TestDynamoComponentDeploymentReconciler_FinalizeResource(t *testing.T) {
	type fields struct {
		EtcdStorage etcdStorage
		Config      controller_common.Config
	}
	type args struct {
		ctx                       context.Context
		dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		{
			name: "delete etcd keys",
			fields: fields{
				EtcdStorage: &mockEtcdStorage{
					deleteKeysFunc: func(ctx context.Context, prefix string) error {
						if prefix == "/default/components/service1" {
							return nil
						}
						return fmt.Errorf("invalid prefix: %s", prefix)
					},
				},
				Config: controller_common.Config{DiscoveryBackend: "etcd"},
			},
			args: args{
				ctx: context.Background(),
				dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default"}[0],
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "delete etcd keys (error)",
			fields: fields{
				EtcdStorage: &mockEtcdStorage{
					deleteKeysFunc: func(ctx context.Context, prefix string) error {
						return fmt.Errorf("invalid prefix: %s", prefix)
					},
				},
				Config: controller_common.Config{DiscoveryBackend: "etcd"},
			},
			args: args{
				ctx: context.Background(),
				dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
					Spec: v1alpha1.DynamoComponentDeploymentSpec{
						DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
							ServiceName:     "service1",
							DynamoNamespace: &[]string{"default"}[0],
						},
					},
				},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &DynamoComponentDeploymentReconciler{
				EtcdStorage: tt.fields.EtcdStorage,
				Config:      tt.fields.Config,
			}
			if err := r.FinalizeResource(tt.args.ctx, tt.args.dynamoComponentDeployment); (err != nil) != tt.wantErr {
				t.Errorf("DynamoComponentDeploymentReconciler.FinalizeResource() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDynamoComponentDeploymentReconciler_generateIngress(t *testing.T) {
	type fields struct {
	}
	type args struct {
		ctx context.Context
		opt generateResourceOption
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *networkingv1.Ingress
		want1   bool
		wantErr bool
	}{
		{
			name:   "generate ingress",
			fields: fields{},
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service1",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service1",
								DynamoNamespace: &[]string{"default"}[0],
								Ingress: &v1alpha1.IngressSpec{
									Enabled:                    true,
									Host:                       "someservice",
									IngressControllerClassName: &[]string{"nginx"}[0],
									UseVirtualService:          false,
								},
							},
						},
					},
				},
			},
			want: &networkingv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "service1",
					Namespace: "default",
				},
				Spec: networkingv1.IngressSpec{
					IngressClassName: &[]string{"nginx"}[0],
					Rules: []networkingv1.IngressRule{
						{
							Host: "someservice.local",
							IngressRuleValue: networkingv1.IngressRuleValue{
								HTTP: &networkingv1.HTTPIngressRuleValue{
									Paths: []networkingv1.HTTPIngressPath{
										{
											Path:     "/",
											PathType: &[]networkingv1.PathType{networkingv1.PathTypePrefix}[0],
											Backend: networkingv1.IngressBackend{
												Service: &networkingv1.IngressServiceBackend{
													Name: "service1",
													Port: networkingv1.ServiceBackendPort{Number: commonconsts.DynamoServicePort},
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
			want1:   false,
			wantErr: false,
		},
		{
			name:   "generate ingress, disabled",
			fields: fields{},
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service1",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service1",
								DynamoNamespace: &[]string{"default"}[0],
								Ingress: &v1alpha1.IngressSpec{
									Enabled: false,
								},
							},
						},
					},
				},
			},
			want: &networkingv1.Ingress{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "service1",
					Namespace: "default",
				},
			},
			want1:   true,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			r := &DynamoComponentDeploymentReconciler{}
			got, got1, err := r.generateIngress(tt.args.ctx, tt.args.opt)
			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoComponentDeploymentReconciler.generateIngress() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			g.Expect(got).To(gomega.Equal(tt.want))
			g.Expect(got1).To(gomega.Equal(tt.want1))
		})
	}
}

func TestDynamoComponentDeploymentReconciler_generateVirtualService(t *testing.T) {
	type fields struct {
	}
	type args struct {
		ctx context.Context
		opt generateResourceOption
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *networkingv1beta1.VirtualService
		want1   bool
		wantErr bool
	}{
		{
			name:   "generate virtual service, disabled in operator config",
			fields: fields{},
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service1",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service1",
								DynamoNamespace: &[]string{"default"}[0],
								Ingress: &v1alpha1.IngressSpec{
									Enabled: true,
								},
							},
						},
					},
				},
			},
			want: &networkingv1beta1.VirtualService{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "service1",
					Namespace: "default",
				},
			},
			want1:   true,
			wantErr: false,
		},
		{
			name:   "generate virtual service, enabled in operator config",
			fields: fields{},
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service1",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service1",
								DynamoNamespace: &[]string{"default"}[0],
								Ingress: &v1alpha1.IngressSpec{
									Enabled:               true,
									Host:                  "someservice",
									UseVirtualService:     true,
									VirtualServiceGateway: &[]string{"istio-system/ingress-alb"}[0],
								},
							},
						},
					},
				},
			},
			want: &networkingv1beta1.VirtualService{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "service1",
					Namespace: "default",
				},
				Spec: istioNetworking.VirtualService{
					Hosts:    []string{"someservice.local"},
					Gateways: []string{"istio-system/ingress-alb"},
					Http: []*istioNetworking.HTTPRoute{
						{
							Match: []*istioNetworking.HTTPMatchRequest{
								{
									Uri: &istioNetworking.StringMatch{
										MatchType: &istioNetworking.StringMatch_Prefix{Prefix: "/"},
									},
								},
							},
							Route: []*istioNetworking.HTTPRouteDestination{
								{
									Destination: &istioNetworking.Destination{
										Host: "service1",
										Port: &istioNetworking.PortSelector{
											Number: commonconsts.DynamoServicePort,
										},
									},
								},
							},
						},
					},
				},
			},
			want1:   false,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			r := &DynamoComponentDeploymentReconciler{}
			got, got1, err := r.generateVirtualService(tt.args.ctx, tt.args.opt)
			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoComponentDeploymentReconciler.generateVirtualService() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			g.Expect(got).To(gomega.Equal(tt.want))
			g.Expect(got1).To(gomega.Equal(tt.want1))
		})
	}
}

func TestDynamoComponentDeploymentReconciler_generateVolcanoPodGroup(t *testing.T) {
	type fields struct {
		Client      client.Client
		Recorder    record.EventRecorder
		Config      controller_common.Config
		EtcdStorage etcdStorage
	}
	type args struct {
		ctx context.Context
		opt generateResourceOption
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *volcanov1beta1.PodGroup
		want1   bool
		wantErr bool
	}{
		{
			name: "generate volcano pod group",
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service1",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								Multinode: &v1alpha1.MultinodeSpec{
									NodeCount: 2,
								},
								ServiceName:     "service1",
								DynamoNamespace: &[]string{"default"}[0],
							},
						},
					},
					instanceID: ptr.To(5),
				},
			},
			want: &volcanov1beta1.PodGroup{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "service1-5",
					Namespace: "default",
					Labels: map[string]string{
						"instance-id": "5",
					},
				},
				Spec: volcanov1beta1.PodGroupSpec{
					MinMember: 2,
				},
			},
			want1:   false,
			wantErr: false,
		},
		{
			name: "nil instanceID",
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-nil-instanceid",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service-nil-instanceid",
								DynamoNamespace: &[]string{"default"}[0],
								Multinode: &v1alpha1.MultinodeSpec{
									NodeCount: 2,
								},
							},
						},
					},
					instanceID: nil,
				},
			},
			want:    nil,
			want1:   false,
			wantErr: true,
		},
		{
			name: "negative instanceID",
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-negative-instanceid",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service-negative-instanceid",
								DynamoNamespace: &[]string{"default"}[0],
								Multinode: &v1alpha1.MultinodeSpec{
									NodeCount: 2,
								},
							},
						},
					},
					instanceID: ptr.To(-1),
				},
			},
			want:    nil,
			want1:   false,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			r := &DynamoComponentDeploymentReconciler{
				Client:      tt.fields.Client,
				Recorder:    tt.fields.Recorder,
				Config:      tt.fields.Config,
				EtcdStorage: tt.fields.EtcdStorage,
			}
			got, got1, err := r.generateVolcanoPodGroup(tt.args.ctx, tt.args.opt)
			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoComponentDeploymentReconciler.generateVolcanoPodGroup() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Mismatch (-expected +actual):\n%s", diff)
			}
			g.Expect(got).To(gomega.Equal(tt.want))
			g.Expect(got1).To(gomega.Equal(tt.want1))
		})
	}
}

type mockDockerSecretRetriever struct {
	GetSecretsFunc func(namespace, imageName string) ([]string, error)
}

func (m *mockDockerSecretRetriever) GetSecrets(namespace, imageName string) ([]string, error) {
	return m.GetSecretsFunc(namespace, imageName)
}

func TestDynamoComponentDeploymentReconciler_generateLeaderWorkerSet(t *testing.T) {
	var limit = ptr.To(resource.MustParse("250Mi"))
	limit.SetMilli(ptr.To(resource.MustParse("1Gi")).MilliValue() / 2)
	type fields struct {
		Client                client.Client
		Recorder              record.EventRecorder
		Config                controller_common.Config
		EtcdStorage           etcdStorage
		DockerSecretRetriever *mockDockerSecretRetriever
	}
	type args struct {
		ctx context.Context
		opt generateResourceOption
		// Add expected ServiceAccountName if you want to verify it's picked up
		// For now, we'll ensure a default one exists for the happy path
		mockServiceAccounts []client.Object
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *leaderworkersetv1.LeaderWorkerSet
		want1   bool // toDelete
		wantErr bool
	}{
		{
			name: "generateLeaderWorkerSet - nominal case",
			fields: fields{
				Recorder: record.NewFakeRecorder(100),
				Config:   controller_common.Config{}, // Provide default or test-specific config
				DockerSecretRetriever: &mockDockerSecretRetriever{
					GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
						return []string{}, nil
					},
				},
			},
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "test-lws-deploy",
							Namespace: "default",
							OwnerReferences: []metav1.OwnerReference{
								{
									Kind: "DynamoGraphDeployment",
									Name: "test-lws-deploy",
								},
							},
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							BackendFramework: string(dynamo.BackendFrameworkVLLM),
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								Envs: []corev1.EnvVar{
									{
										Name:  "TEST_ENV_FROM_DYNAMO_COMPONENT_DEPLOYMENT_SPEC",
										Value: "test_value_from_dynamo_component_deployment_spec",
									},
								},
								ComponentType:    string(commonconsts.ComponentTypeWorker),
								SubComponentType: "test-sub-component",
								ServiceName:      "test-lws-deploy-service",
								DynamoNamespace:  &[]string{"default"}[0],
								Multinode: &v1alpha1.MultinodeSpec{
									NodeCount: 2,
								},
								Resources: &v1alpha1.Resources{
									Requests: &v1alpha1.ResourceItem{
										CPU:    "300m",
										Memory: "500Mi",
									},
									Limits: &v1alpha1.ResourceItem{
										GPU:    "1",
										Memory: "20Gi",
										CPU:    "10",
									},
								},
								ExtraPodMetadata: &v1alpha1.ExtraPodMetadata{
									Annotations: map[string]string{
										"nvidia.com/annotation1": "annotation1",
									},
									Labels: map[string]string{
										"nvidia.com/label1": "label1",
									},
								},
								ExtraPodSpec: &v1alpha1.ExtraPodSpec{
									PodSpec: &corev1.PodSpec{
										TerminationGracePeriodSeconds: ptr.To(int64(10)),
										Containers: []corev1.Container{
											{
												Image: "another-image:latest",
											},
										},
									},
									MainContainer: &corev1.Container{
										Image: "test-image:latest",
										Command: []string{
											"some",
											"dynamo",
											"command",
										},
										Args: []string{
											"--tensor-parallel-size",
											"4",
											"--pipeline-parallel-size",
											"1",
										},
										Env: []corev1.EnvVar{
											{
												Name:  "TEST_ENV_FROM_EXTRA_POD_SPEC",
												Value: "test_value_from_extra_pod_spec",
											},
										},
									},
								},
							},
						},
					},
					instanceID: ptr.To(0),
				},
				// Define a mock ServiceAccount that should be found by r.List
				mockServiceAccounts: []client.Object{
					&corev1.ServiceAccount{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "default-test-sa", // Name it will be resolved to
							Namespace: "default",         // Must match dynamoComponentDeployment.Namespace
							Labels: map[string]string{
								commonconsts.KubeLabelDynamoComponentPod: commonconsts.KubeLabelValueTrue,
							},
						},
					},
				},
			},
			want: &leaderworkersetv1.LeaderWorkerSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-lws-deploy-0",
					Namespace: "default",
					Labels: map[string]string{
						"instance-id": "0",
					},
				},
				Spec: leaderworkersetv1.LeaderWorkerSetSpec{
					Replicas:      ptr.To(int32(1)),
					StartupPolicy: leaderworkersetv1.LeaderCreatedStartupPolicy,
					LeaderWorkerTemplate: leaderworkersetv1.LeaderWorkerTemplate{
						Size: ptr.To(int32(2)),
						LeaderTemplate: &corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Labels: map[string]string{
									"instance-id":                        "0",
									commonconsts.KubeLabelMetricsEnabled: commonconsts.KubeLabelValueTrue,
									"role":                               "leader",
									"nvidia.com/label1":                  "label1",
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
									commonconsts.KubeLabelDynamoSubComponentType:    "test-sub-component",
									commonconsts.KubeLabelDynamoGraphDeploymentName: "",
								},
								Annotations: map[string]string{
									"scheduling.k8s.io/group-name": "test-lws-deploy-0",
									"nvidia.com/annotation1":       "annotation1",
								},
							},
							Spec: corev1.PodSpec{
								SchedulerName:                 "volcano",
								TerminationGracePeriodSeconds: ptr.To(int64(10)),
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
										Image: "another-image:latest",
									},
									{
										Name:    commonconsts.MainContainerName,
										Image:   "test-image:latest",
										Command: []string{"/bin/sh", "-c"},
										Args:    []string{"ray start --head --port=6379 && some dynamo command --tensor-parallel-size 4 --pipeline-parallel-size 1 --distributed-executor-backend ray"},
										Env: []corev1.EnvVar{
											{Name: commonconsts.DynamoComponentEnvVar, Value: commonconsts.ComponentTypeWorker},
											{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"},
											{Name: "DYN_HEALTH_CHECK_ENABLED", Value: "false"},
											{Name: commonconsts.DynamoNamespaceEnvVar, Value: "default-test-lws-deploy"},
											{Name: "DYN_PARENT_DGD_K8S_NAME", Value: "test-lws-deploy"},
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
											{Name: "POD_UID", ValueFrom: &corev1.EnvVarSource{
												FieldRef: &corev1.ObjectFieldSelector{
													FieldPath: "metadata.uid",
												},
											}},
											{Name: "TEST_ENV_FROM_DYNAMO_COMPONENT_DEPLOYMENT_SPEC", Value: "test_value_from_dynamo_component_deployment_spec"},
											{Name: "TEST_ENV_FROM_EXTRA_POD_SPEC", Value: "test_value_from_extra_pod_spec"},
										},
										Ports: []corev1.ContainerPort{
											{
												Protocol: corev1.ProtocolTCP, Name: commonconsts.DynamoSystemPortName, ContainerPort: commonconsts.DynamoSystemPort,
											},
										},
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "shared-memory",
												MountPath: commonconsts.DefaultSharedMemoryMountPath,
											},
										},
										Resources: corev1.ResourceRequirements{
											Requests: corev1.ResourceList{
												corev1.ResourceCPU:    resource.MustParse("300m"),
												corev1.ResourceMemory: resource.MustParse("500Mi"),
											},
											Limits: corev1.ResourceList{
												corev1.ResourceMemory: resource.MustParse("20Gi"),
												corev1.ResourceCPU:    resource.MustParse("10"),
												corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
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
								ImagePullSecrets:   nil,               // Assuming default config gives empty secret name
								ServiceAccountName: "default-test-sa", // Updated to reflect mocked SA
							},
						},
						WorkerTemplate: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Labels: map[string]string{
									"instance-id":                        "0",
									commonconsts.KubeLabelMetricsEnabled: commonconsts.KubeLabelValueTrue,
									"role":                               "worker",
									"nvidia.com/label1":                  "label1",
									commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeWorker,
									commonconsts.KubeLabelDynamoSubComponentType:    "test-sub-component",
									commonconsts.KubeLabelDynamoGraphDeploymentName: "",
								},
								Annotations: map[string]string{
									"scheduling.k8s.io/group-name": "test-lws-deploy-0",
									"nvidia.com/annotation1":       "annotation1",
								},
							},
							Spec: corev1.PodSpec{
								TerminationGracePeriodSeconds: ptr.To(int64(10)),
								SchedulerName:                 "volcano",
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
										Image: "another-image:latest",
									},
									{
										Name:    commonconsts.MainContainerName,
										Image:   "test-image:latest",
										Command: []string{"/bin/sh", "-c"},
										Args:    []string{"ray start --address=$LWS_LEADER_ADDRESS:6379 --block"},
										Env: []corev1.EnvVar{
											{Name: commonconsts.DynamoComponentEnvVar, Value: commonconsts.ComponentTypeWorker},
											{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"},
											{Name: "DYN_HEALTH_CHECK_ENABLED", Value: "false"},
											{Name: commonconsts.DynamoNamespaceEnvVar, Value: "default-test-lws-deploy"},
											{Name: "DYN_PARENT_DGD_K8S_NAME", Value: "test-lws-deploy"},
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
											{Name: "POD_UID", ValueFrom: &corev1.EnvVarSource{
												FieldRef: &corev1.ObjectFieldSelector{
													FieldPath: "metadata.uid",
												},
											}},
											{Name: "TEST_ENV_FROM_DYNAMO_COMPONENT_DEPLOYMENT_SPEC", Value: "test_value_from_dynamo_component_deployment_spec"},
											{Name: "TEST_ENV_FROM_EXTRA_POD_SPEC", Value: "test_value_from_extra_pod_spec"},
										},
										Ports: []corev1.ContainerPort{
											{
												Protocol: corev1.ProtocolTCP, Name: commonconsts.DynamoSystemPortName, ContainerPort: commonconsts.DynamoSystemPort,
											},
										},
										VolumeMounts: []corev1.VolumeMount{
											{
												Name:      "shared-memory",
												MountPath: commonconsts.DefaultSharedMemoryMountPath,
											},
										},
										Resources: corev1.ResourceRequirements{
											Limits: corev1.ResourceList{
												corev1.ResourceMemory: resource.MustParse("20Gi"),
												corev1.ResourceCPU:    resource.MustParse("10"),
												"nvidia.com/gpu":      resource.MustParse("1"),
											},
											Requests: corev1.ResourceList{
												corev1.ResourceCPU:    resource.MustParse("300m"),
												corev1.ResourceMemory: resource.MustParse("500Mi"),
											},
										},
									},
								},
								ImagePullSecrets:   nil,
								ServiceAccountName: "default-test-sa", // Updated to reflect mocked SA
							},
						},
					},
				},
			},
			want1:   false,
			wantErr: false,
		},
		{
			name: "nil instanceID", // This test should fail before r.List is called in generatePodTemplateSpec
			fields: fields{
				Recorder: record.NewFakeRecorder(100),
			},
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{Name: "test-lws-nil-id", Namespace: "default"},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								Multinode: &v1alpha1.MultinodeSpec{
									NodeCount: 2,
								},
								Resources: &v1alpha1.Resources{
									Limits: &v1alpha1.ResourceItem{
										GPU: "1",
									},
								},
								ExtraPodSpec: &v1alpha1.ExtraPodSpec{
									MainContainer: &corev1.Container{
										Image: "test-image:latest",
									},
								},
							},
						},
					},
					instanceID: nil,
				},
				mockServiceAccounts: []client.Object{ // Provide a default SA for consistency, though not strictly needed here
					&corev1.ServiceAccount{
						ObjectMeta: metav1.ObjectMeta{
							Name: "default-test-sa", Namespace: "default", // Match namespace
							Labels: map[string]string{commonconsts.KubeLabelDynamoComponentPod: commonconsts.KubeLabelValueTrue},
						},
					},
				},
			},
			want:    nil,
			want1:   false,
			wantErr: true,
		},
		{
			name: "error from generateLeaderPodTemplateSpec", // This case involves an error from generatePodTemplateSpec
			fields: fields{
				Recorder: record.NewFakeRecorder(100),
			},
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{Name: "test-lws-leader-err", Namespace: "default"},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								Multinode: &v1alpha1.MultinodeSpec{
									NodeCount: 2,
								},
								Resources: &v1alpha1.Resources{
									Limits: &v1alpha1.ResourceItem{
										GPU: "1",
									},
								},
								ExtraPodSpec: &v1alpha1.ExtraPodSpec{
									MainContainer: &corev1.Container{
										Image: "", // Image is missing, will cause error in generatePodTemplateSpec
									},
								},
							},
						},
					},
					instanceID: ptr.To(0),
				},
				// No specific SA needed if error is before SA listing, but good to be consistent
				mockServiceAccounts: []client.Object{
					&corev1.ServiceAccount{
						ObjectMeta: metav1.ObjectMeta{
							Name: "default-test-sa", Namespace: "default", // Match namespace
							Labels: map[string]string{commonconsts.KubeLabelDynamoComponentPod: commonconsts.KubeLabelValueTrue},
						},
					},
				},
			},
			want:    nil,
			want1:   false,
			wantErr: true,
		},
	}

	// Initialize scheme & add API types
	s := scheme.Scheme
	if err := v1alpha1.AddToScheme(s); err != nil {
		t.Fatalf("Failed to add v1alpha1 to scheme: %v", err)
	}
	if err := corev1.AddToScheme(s); err != nil {
		t.Fatalf("Failed to add corev1 to scheme: %v", err)
	}
	// Add LeaderWorkerSet to scheme if not already present globally for tests
	if err := leaderworkersetv1.AddToScheme(s); err != nil {
		t.Fatalf("Failed to add leaderworkersetv1 to scheme: %v", err)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			format.MaxLength = 0
			g := gomega.NewGomegaWithT(t)

			// Build initial objects for fake client for this test case
			var initialClientObjects []client.Object
			if tt.args.opt.dynamoComponentDeployment != nil {
				initialClientObjects = append(initialClientObjects, tt.args.opt.dynamoComponentDeployment)
			}
			if len(tt.args.mockServiceAccounts) > 0 {
				initialClientObjects = append(initialClientObjects, tt.args.mockServiceAccounts...)
			}

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(initialClientObjects...).
				Build()

			r := &DynamoComponentDeploymentReconciler{
				Client:                fakeKubeClient, // Use the fake client
				Recorder:              tt.fields.Recorder,
				Config:                tt.fields.Config,
				EtcdStorage:           tt.fields.EtcdStorage,
				DockerSecretRetriever: tt.fields.DockerSecretRetriever,
				// Scheme: s, // Pass scheme if reconciler uses it directly, often client uses it
			}
			got, got1, err := r.generateLeaderWorkerSet(tt.args.ctx, tt.args.opt)
			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoComponentDeploymentReconciler.generateLeaderWorkerSet() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Mismatch (-expected +actual):\n%s", diff)
			}
			// Use gomega.Equal for deep comparison of complex structs
			g.Expect(got).To(gomega.BeEquivalentTo(tt.want))
			g.Expect(got1).To(gomega.BeEquivalentTo(tt.want1))
		})
	}
}

func TestDynamoComponentDeploymentReconciler_createOrUpdateOrDeleteDeployments_ReplicaReconciliation(t *testing.T) {
	ctx := context.Background()
	g := gomega.NewGomegaWithT(t)

	// Create a scheme with necessary types
	s := scheme.Scheme
	err := v1alpha1.AddToScheme(s)
	if err != nil {
		t.Fatalf("Failed to add v1alpha1 to scheme: %v", err)
	}
	err = appsv1.AddToScheme(s)
	if err != nil {
		t.Fatalf("Failed to add appsv1 to scheme: %v", err)
	}
	err = corev1.AddToScheme(s)
	if err != nil {
		t.Fatalf("Failed to add corev1 to scheme: %v", err)
	}

	// Create DynamoComponentDeployment with 1 replica
	replicaCount := int32(1)
	dcd := &v1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-component",
			Namespace: "default",
		},
		Spec: v1alpha1.DynamoComponentDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
			DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
				ServiceName:     "test-service",
				DynamoNamespace: ptr.To("default"),
				ComponentType:   string(commonconsts.ComponentTypeDecode),
				Replicas:        &replicaCount,
			},
		},
	}

	// Set up fake client with the DCD
	fakeKubeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(dcd).
		Build()

	// Set up reconciler
	recorder := record.NewFakeRecorder(100)
	reconciler := &DynamoComponentDeploymentReconciler{
		Client:      fakeKubeClient,
		Recorder:    recorder,
		Config:      controller_common.Config{},
		EtcdStorage: nil,
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
				return []string{}, nil
			},
		},
	}

	opt := generateResourceOption{
		dynamoComponentDeployment: dcd,
	}

	// Step 1: Create the deployment with 1 replica
	modified, deployment, err := reconciler.createOrUpdateOrDeleteDeployments(ctx, opt)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(modified).To(gomega.BeTrue(), "Deployment should have been created")
	g.Expect(deployment).NotTo(gomega.BeNil())

	// Verify deployment was created with 1 replica
	deploymentName := "test-component"
	createdDeployment := &appsv1.Deployment{}
	err = fakeKubeClient.Get(ctx, client.ObjectKey{Name: deploymentName, Namespace: "default"}, createdDeployment)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(createdDeployment.Spec.Replicas).NotTo(gomega.BeNil())
	g.Expect(*createdDeployment.Spec.Replicas).To(gomega.Equal(int32(1)), "Initial deployment should have 1 replica")

	// Step 2: Manually update the deployment to 2 replicas (simulating manual edit)
	// Note: Real Kubernetes API server increments generation on spec changes,
	// but the fake client doesn't, so we simulate it here.
	// The operator sets last-applied-generation=1 on create, so we need generation > 1
	// to trigger manual change detection.
	manualReplicaCount := int32(2)
	createdDeployment.Spec.Replicas = &manualReplicaCount
	createdDeployment.Generation = 2 // Simulate K8s incrementing generation on spec change
	err = fakeKubeClient.Update(ctx, createdDeployment)
	g.Expect(err).NotTo(gomega.HaveOccurred())

	// Verify the manual update
	updatedDeployment := &appsv1.Deployment{}
	err = fakeKubeClient.Get(ctx, client.ObjectKey{Name: deploymentName, Namespace: "default"}, updatedDeployment)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(updatedDeployment.Spec.Replicas).NotTo(gomega.BeNil())
	g.Expect(*updatedDeployment.Spec.Replicas).To(gomega.Equal(int32(2)), "Deployment should have been manually updated to 2 replicas")

	// Step 3: Call createOrUpdateOrDeleteDeployments again - it should reconcile back to 1 replica
	modified2, deployment2, err := reconciler.createOrUpdateOrDeleteDeployments(ctx, opt)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(modified2).To(gomega.BeTrue(), "Deployment should have been updated to reconcile replica count")
	g.Expect(deployment2).NotTo(gomega.BeNil())

	// Step 4: Verify the deployment was reconciled back to 1 replica
	reconciledDeployment := &appsv1.Deployment{}
	err = fakeKubeClient.Get(ctx, client.ObjectKey{Name: deploymentName, Namespace: "default"}, reconciledDeployment)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(reconciledDeployment.Spec.Replicas).NotTo(gomega.BeNil())
	g.Expect(*reconciledDeployment.Spec.Replicas).To(gomega.Equal(int32(1)), "Deployment should have been reconciled back to 1 replica")

	// Step 5: Call createOrUpdateOrDeleteDeployments again - it should not be modified
	modified3, deployment3, err := reconciler.createOrUpdateOrDeleteDeployments(ctx, opt)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(modified3).To(gomega.BeFalse(), "Deployment should have been not modified")
	g.Expect(deployment3).NotTo(gomega.BeNil())
}

func Test_createOrUpdateOrDeleteDeployments_K8sAPIDefaults(t *testing.T) {
	g := gomega.NewGomegaWithT(t)
	ctx := context.Background()

	// Set up scheme
	s := scheme.Scheme
	err := v1alpha1.AddToScheme(s)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	err = appsv1.AddToScheme(s)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	err = corev1.AddToScheme(s)
	g.Expect(err).NotTo(gomega.HaveOccurred())

	name := "test-component"
	namespace := defaultNamespace

	// Create DynamoComponentDeployment
	replicaCount := int32(3)
	dcd := &v1alpha1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1alpha1.DynamoComponentDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
			DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
				ServiceName:     "test-service",
				DynamoNamespace: ptr.To("default"),
				ComponentType:   string(commonconsts.ComponentTypeDecode),
				Replicas:        &replicaCount,
			},
		},
	}

	fakeKubeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(dcd).
		Build()

	recorder := record.NewFakeRecorder(100)
	reconciler := &DynamoComponentDeploymentReconciler{
		Client:      fakeKubeClient,
		Recorder:    recorder,
		Config:      controller_common.Config{},
		EtcdStorage: nil,
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
				return []string{}, nil
			},
		},
	}

	opt := generateResourceOption{
		dynamoComponentDeployment: dcd,
	}

	t.Log("=== Step 1: Create deployment (operator's first apply) ===")

	modified1, deployment1, err := reconciler.createOrUpdateOrDeleteDeployments(ctx, opt)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(modified1).To(gomega.BeTrue(), "First create should report as modified")
	g.Expect(deployment1).NotTo(gomega.BeNil())
	g.Expect(deployment1.Spec.RevisionHistoryLimit).To(gomega.BeNil())

	operatorCreatedDeployment := &appsv1.Deployment{}
	err = fakeKubeClient.Get(ctx, client.ObjectKey{Name: name, Namespace: namespace}, operatorCreatedDeployment)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(*operatorCreatedDeployment.Spec.Replicas).To(gomega.Equal(replicaCount))

	annotations := operatorCreatedDeployment.GetAnnotations()
	g.Expect(annotations).NotTo(gomega.BeNil())
	originalHash, hasHash := annotations[controller_common.NvidiaAnnotationHashKey]
	g.Expect(hasHash).To(gomega.BeTrue(), "Hash annotation should be set")
	t.Logf("Hash annotation after create: %s", originalHash)

	t.Log("\n=== Step 2: Simulate K8s adding defaults ===")

	// Operator does not set RevisionHistoryLimit but the k8s API defaults to 10
	operatorCreatedDeployment.Spec.RevisionHistoryLimit = ptr.To(int32(10))
	err = fakeKubeClient.Update(ctx, operatorCreatedDeployment)
	g.Expect(err).NotTo(gomega.HaveOccurred())

	// The deployment should not be modified because the spec is the same
	modified2, deployment2, err := reconciler.createOrUpdateOrDeleteDeployments(ctx, opt)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(modified2).To(gomega.BeFalse(), "Second create should report as not modified")
	g.Expect(deployment2).NotTo(gomega.BeNil())

	modified3, deployment3, err := reconciler.createOrUpdateOrDeleteDeployments(ctx, opt)
	g.Expect(err).NotTo(gomega.HaveOccurred())
	g.Expect(modified3).To(gomega.BeFalse(), "Third create should report as not modified")
	g.Expect(deployment3).NotTo(gomega.BeNil())
}

func Test_reconcileLeaderWorkerSetResources(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                         string
		replicas                     int32
		existingLeaderWorkerSets     []*leaderworkersetv1.LeaderWorkerSet
		wantComponentReconcileResult ComponentReconcileResult
	}{
		{
			name:     "singular LWS replica ready",
			replicas: 1,
			existingLeaderWorkerSets: []*leaderworkersetv1.LeaderWorkerSet{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-component-0",
						Namespace: "default",
					},
					Spec: leaderworkersetv1.LeaderWorkerSetSpec{
						Replicas: ptr.To(int32(1)),
					},
					Status: leaderworkersetv1.LeaderWorkerSetStatus{
						ReadyReplicas:   1,
						UpdatedReplicas: 1,
						Replicas:        1,
						Conditions: []metav1.Condition{
							{
								Type:   string(leaderworkersetv1.LeaderWorkerSetAvailable),
								Status: metav1.ConditionTrue,
							},
						},
					},
				},
			},
			wantComponentReconcileResult: ComponentReconcileResult{
				modified: true,
				status:   metav1.ConditionTrue,
				reason:   "AllLeaderWorkerSetsReady",
				message:  "All LeaderWorkerSets are ready",
				serviceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
					ComponentKind:   v1alpha1.ComponentKindLeaderWorkerSet,
					ComponentName:   "test-component-0",
					ReadyReplicas:   ptr.To(int32(1)),
					UpdatedReplicas: 1,
					Replicas:        1,
				},
			},
		},
		{
			name:     "multiple LWS replicas - at least one is unready",
			replicas: 3,
			existingLeaderWorkerSets: []*leaderworkersetv1.LeaderWorkerSet{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-component-0",
						Namespace: "default",
					},
					Spec: leaderworkersetv1.LeaderWorkerSetSpec{
						Replicas: ptr.To(int32(1)),
					},
					Status: leaderworkersetv1.LeaderWorkerSetStatus{
						ReadyReplicas:   1,
						Replicas:        1,
						UpdatedReplicas: 1,
						Conditions: []metav1.Condition{
							{
								Type:   string(leaderworkersetv1.LeaderWorkerSetAvailable),
								Status: metav1.ConditionTrue,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-component-1",
						Namespace: "default",
					},
					Spec: leaderworkersetv1.LeaderWorkerSetSpec{
						Replicas: ptr.To(int32(1)),
					},
					Status: leaderworkersetv1.LeaderWorkerSetStatus{
						ReadyReplicas:   0, // Not ready
						Replicas:        1,
						UpdatedReplicas: 0,
						Conditions: []metav1.Condition{
							{
								Type:   string(leaderworkersetv1.LeaderWorkerSetAvailable),
								Status: metav1.ConditionFalse,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-component-2",
						Namespace: "default",
					},
					Spec: leaderworkersetv1.LeaderWorkerSetSpec{
						Replicas: ptr.To(int32(1)),
					},
					Status: leaderworkersetv1.LeaderWorkerSetStatus{
						ReadyReplicas:   1,
						Replicas:        1,
						UpdatedReplicas: 1,
						Conditions: []metav1.Condition{
							{
								Type:   string(leaderworkersetv1.LeaderWorkerSetAvailable),
								Status: metav1.ConditionTrue,
							},
						},
					},
				},
			},
			wantComponentReconcileResult: ComponentReconcileResult{
				modified: true,
				status:   metav1.ConditionFalse,
				reason:   "SomeLeaderWorkerSetsNotReady",
				message:  "Some LeaderWorkerSets are not ready",
				serviceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
					ComponentKind:   v1alpha1.ComponentKindLeaderWorkerSet,
					ComponentName:   "test-component-0",
					ReadyReplicas:   ptr.To(int32(2)),
					UpdatedReplicas: 2,
					Replicas:        3,
				},
			},
		},
		{
			name:     "multiple LWS replicas - all ready",
			replicas: 3,
			existingLeaderWorkerSets: []*leaderworkersetv1.LeaderWorkerSet{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-component-0",
						Namespace: "default",
					},
					Spec: leaderworkersetv1.LeaderWorkerSetSpec{
						Replicas: ptr.To(int32(1)),
					},
					Status: leaderworkersetv1.LeaderWorkerSetStatus{
						ReadyReplicas:   1,
						Replicas:        1,
						UpdatedReplicas: 1,
						Conditions: []metav1.Condition{
							{
								Type:   string(leaderworkersetv1.LeaderWorkerSetAvailable),
								Status: metav1.ConditionTrue,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-component-1",
						Namespace: "default",
					},
					Spec: leaderworkersetv1.LeaderWorkerSetSpec{
						Replicas: ptr.To(int32(1)),
					},
					Status: leaderworkersetv1.LeaderWorkerSetStatus{
						ReadyReplicas:   1,
						Replicas:        1,
						UpdatedReplicas: 1,
						Conditions: []metav1.Condition{
							{
								Type:   string(leaderworkersetv1.LeaderWorkerSetAvailable),
								Status: metav1.ConditionTrue,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Name:      "test-component-2",
						Namespace: "default",
					},
					Spec: leaderworkersetv1.LeaderWorkerSetSpec{
						Replicas: ptr.To(int32(1)),
					},
					Status: leaderworkersetv1.LeaderWorkerSetStatus{
						ReadyReplicas:   1,
						Replicas:        1,
						UpdatedReplicas: 1,
						Conditions: []metav1.Condition{
							{
								Type:   string(leaderworkersetv1.LeaderWorkerSetAvailable),
								Status: metav1.ConditionTrue,
							},
						},
					},
				},
			},
			wantComponentReconcileResult: ComponentReconcileResult{
				modified: true,
				status:   metav1.ConditionTrue,
				reason:   "AllLeaderWorkerSetsReady",
				message:  "All LeaderWorkerSets are ready",
				serviceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
					ComponentKind:   v1alpha1.ComponentKindLeaderWorkerSet,
					ComponentName:   "test-component-0",
					ReadyReplicas:   ptr.To(int32(3)),
					UpdatedReplicas: 3,
					Replicas:        3,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Create a scheme with necessary types
			s := scheme.Scheme
			err := v1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			err = leaderworkersetv1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			err = volcanov1beta1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			// Create DynamoComponentDeployment
			dcd := &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-component",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					BackendFramework: string(dynamo.BackendFrameworkVLLM),
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						ServiceName:     "test-service",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        &tt.replicas,
						Multinode: &v1alpha1.MultinodeSpec{
							NodeCount: 2,
						},
						Resources: &v1alpha1.Resources{
							Limits: &v1alpha1.ResourceItem{
								GPU: "1",
							},
						},
						ExtraPodSpec: &v1alpha1.ExtraPodSpec{
							MainContainer: &corev1.Container{
								Image: "test-image:latest",
								Args: []string{
									"--test-arg",
								},
							},
						},
					},
				},
			}

			// Prepare objects for fake client
			var objects []client.Object
			objects = append(objects, dcd)
			for _, lws := range tt.existingLeaderWorkerSets {
				objects = append(objects, lws)
			}
			// Add a mock ServiceAccount that the generateLeaderWorkerSet function needs
			objects = append(objects, &corev1.ServiceAccount{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "default-test-sa",
					Namespace: "default",
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoComponentPod: commonconsts.KubeLabelValueTrue,
					},
				},
			})

			// Set up fake client with the DCD and existing LWS objects
			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			// Set up reconciler
			recorder := record.NewFakeRecorder(100)
			reconciler := &DynamoComponentDeploymentReconciler{
				Client:      fakeKubeClient,
				Recorder:    recorder,
				Config:      controller_common.Config{},
				EtcdStorage: nil,
				DockerSecretRetriever: &mockDockerSecretRetriever{
					GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
						return []string{}, nil
					},
				},
			}

			// Call the function under test
			result, err := reconciler.reconcileLeaderWorkerSetResources(ctx, dcd)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			// Assert the ComponentReconcileResult
			g.Expect(result).To(gomega.Equal(tt.wantComponentReconcileResult))
		})
	}
}

func Test_reconcileDeploymentResources(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                         string
		replicas                     int32
		existingDeployment           *appsv1.Deployment
		wantComponentReconcileResult ComponentReconcileResult
	}{
		{
			name:     "ready deployment",
			replicas: 2,
			existingDeployment: &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "test-component",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: appsv1.DeploymentSpec{
					Replicas: ptr.To(int32(2)),
				},
				Status: appsv1.DeploymentStatus{
					ObservedGeneration: 1,
					Replicas:           2,
					UpdatedReplicas:    2,
					ReadyReplicas:      2,
					AvailableReplicas:  2,
					Conditions: []appsv1.DeploymentCondition{
						{
							Type:   appsv1.DeploymentAvailable,
							Status: corev1.ConditionTrue,
						},
					},
				},
			},
			wantComponentReconcileResult: ComponentReconcileResult{
				modified: true,
				status:   metav1.ConditionTrue,
				reason:   "DeploymentReady",
				message:  "Deployment is ready",
				serviceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
					ComponentKind:     v1alpha1.ComponentKindDeployment,
					ComponentName:     "test-component",
					Replicas:          2,
					UpdatedReplicas:   2,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(2)),
				},
			},
		},
		{
			name:     "unready deployment",
			replicas: 1,
			existingDeployment: &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "test-component",
					Namespace:  "default",
					Generation: 1,
				},
				Spec: appsv1.DeploymentSpec{
					Replicas: ptr.To(int32(1)),
				},
				Status: appsv1.DeploymentStatus{
					ObservedGeneration: 1,
					Replicas:           1,
					UpdatedReplicas:    1,
					ReadyReplicas:      1,
					AvailableReplicas:  0, // Not available
					Conditions: []appsv1.DeploymentCondition{
						{
							Type:   appsv1.DeploymentAvailable,
							Status: corev1.ConditionFalse,
						},
					},
				},
			},
			wantComponentReconcileResult: ComponentReconcileResult{
				modified: true,
				status:   metav1.ConditionFalse,
				reason:   "DeploymentNotReady",
				message:  "Deployment is not ready",
				serviceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
					ComponentKind:     v1alpha1.ComponentKindDeployment,
					ComponentName:     "test-component",
					Replicas:          1,
					UpdatedReplicas:   1,
					ReadyReplicas:     ptr.To(int32(1)),
					AvailableReplicas: ptr.To(int32(0)),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Create a scheme with necessary types
			s := scheme.Scheme
			err := v1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			err = appsv1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			err = corev1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			// Create DynamoComponentDeployment
			dcd := &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-component",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					BackendFramework: string(dynamo.BackendFrameworkVLLM),
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						ServiceName:     "test-service",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        &tt.replicas,
						ExtraPodSpec: &v1alpha1.ExtraPodSpec{
							MainContainer: &corev1.Container{
								Image: "test-image:latest",
								Args: []string{
									"--test-arg",
								},
							},
						},
					},
				},
			}

			// Prepare objects for fake client
			var objects []client.Object
			objects = append(objects, dcd)
			if tt.existingDeployment != nil {
				objects = append(objects, tt.existingDeployment)
			}

			// Set up fake client with the DCD and existing Deployment
			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(objects...).
				WithStatusSubresource(objects...).
				Build()

			// Set up reconciler
			recorder := record.NewFakeRecorder(100)
			reconciler := &DynamoComponentDeploymentReconciler{
				Client:      fakeKubeClient,
				Recorder:    recorder,
				Config:      controller_common.Config{},
				EtcdStorage: nil,
				DockerSecretRetriever: &mockDockerSecretRetriever{
					GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
						return []string{}, nil
					},
				},
			}

			// Call the function under test
			result, err := reconciler.reconcileDeploymentResources(ctx, dcd)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			// Assert the ComponentReconcileResult
			g.Expect(result).To(gomega.Equal(tt.wantComponentReconcileResult))
		})
	}
}

func Test_setStatusConditionAndServiceReplicaStatus(t *testing.T) {
	ctx := context.Background()

	tests := []struct {
		name                     string
		componentReconcileResult ComponentReconcileResult
		wantConditionStatus      metav1.ConditionStatus
		wantConditionReason      string
		wantConditionMessage     string
		wantServiceReplicaStatus *v1alpha1.ServiceReplicaStatus
		wantObservedGeneration   int64
	}{
		{
			name: "deployment backed DCD that is unready",
			componentReconcileResult: ComponentReconcileResult{
				modified: true,
				status:   metav1.ConditionFalse,
				reason:   "DeploymentNotReady",
				message:  "Deployment is not ready",
				serviceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
					ComponentKind:     v1alpha1.ComponentKindDeployment,
					ComponentName:     "test-component",
					Replicas:          1,
					UpdatedReplicas:   1,
					ReadyReplicas:     ptr.To(int32(1)),
					AvailableReplicas: ptr.To(int32(0)),
				},
			},
			wantConditionStatus:  metav1.ConditionFalse,
			wantConditionReason:  "DeploymentNotReady",
			wantConditionMessage: "Deployment is not ready",
			wantServiceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
				ComponentKind:     v1alpha1.ComponentKindDeployment,
				ComponentName:     "test-component",
				Replicas:          1,
				UpdatedReplicas:   1,
				ReadyReplicas:     ptr.To(int32(1)),
				AvailableReplicas: ptr.To(int32(0)),
			},
		},
		{
			name: "deployment backed DCD that is ready",
			componentReconcileResult: ComponentReconcileResult{
				modified: true,
				status:   metav1.ConditionTrue,
				reason:   "DeploymentReady",
				message:  "Deployment is ready",
				serviceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
					ComponentKind:     v1alpha1.ComponentKindDeployment,
					ComponentName:     "test-component",
					Replicas:          2,
					UpdatedReplicas:   2,
					ReadyReplicas:     ptr.To(int32(2)),
					AvailableReplicas: ptr.To(int32(2)),
				},
			},
			wantConditionStatus:  metav1.ConditionTrue,
			wantConditionReason:  "DeploymentReady",
			wantConditionMessage: "Deployment is ready",
			wantServiceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
				ComponentKind:     v1alpha1.ComponentKindDeployment,
				ComponentName:     "test-component",
				Replicas:          2,
				UpdatedReplicas:   2,
				ReadyReplicas:     ptr.To(int32(2)),
				AvailableReplicas: ptr.To(int32(2)),
			},
		},
		{
			name: "LWS backed DCD that is unready",
			componentReconcileResult: ComponentReconcileResult{
				modified: true,
				status:   metav1.ConditionFalse,
				reason:   "SomeLeaderWorkerSetsNotReady",
				message:  "Some LeaderWorkerSets are not ready",
				serviceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
					ComponentKind:   v1alpha1.ComponentKindLeaderWorkerSet,
					ComponentName:   "test-component-0",
					Replicas:        3,
					UpdatedReplicas: 2,
					ReadyReplicas:   ptr.To(int32(2)),
				},
			},
			wantConditionStatus:  metav1.ConditionFalse,
			wantConditionReason:  "SomeLeaderWorkerSetsNotReady",
			wantConditionMessage: "Some LeaderWorkerSets are not ready",
			wantServiceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
				ComponentKind:   v1alpha1.ComponentKindLeaderWorkerSet,
				ComponentName:   "test-component-0",
				Replicas:        3,
				UpdatedReplicas: 2,
				ReadyReplicas:   ptr.To(int32(2)),
			},
		},
		{
			name: "LWS backed DCD that is ready",
			componentReconcileResult: ComponentReconcileResult{
				modified: true,
				status:   metav1.ConditionTrue,
				reason:   "AllLeaderWorkerSetsReady",
				message:  "All LeaderWorkerSets are ready",
				serviceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
					ComponentKind:   v1alpha1.ComponentKindLeaderWorkerSet,
					ComponentName:   "test-component-0",
					Replicas:        3,
					UpdatedReplicas: 3,
					ReadyReplicas:   ptr.To(int32(3)),
				},
			},
			wantConditionStatus:  metav1.ConditionTrue,
			wantConditionReason:  "AllLeaderWorkerSetsReady",
			wantConditionMessage: "All LeaderWorkerSets are ready",
			wantServiceReplicaStatus: &v1alpha1.ServiceReplicaStatus{
				ComponentKind:   v1alpha1.ComponentKindLeaderWorkerSet,
				ComponentName:   "test-component-0",
				Replicas:        3,
				UpdatedReplicas: 3,
				ReadyReplicas:   ptr.To(int32(3)),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Create a scheme with necessary types
			s := scheme.Scheme
			err := v1alpha1.AddToScheme(s)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			// Create DynamoComponentDeployment
			generation := int64(5)
			dcd := &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "test-component",
					Namespace:  "default",
					Generation: generation,
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					BackendFramework: string(dynamo.BackendFrameworkVLLM),
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						ServiceName:     "test-service",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
					},
				},
			}

			// Set up fake client with the DCD
			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(dcd).
				WithStatusSubresource(dcd).
				Build()

			// Set up reconciler
			recorder := record.NewFakeRecorder(100)
			reconciler := &DynamoComponentDeploymentReconciler{
				Client:   fakeKubeClient,
				Recorder: recorder,
			}

			// Create the request
			req := ctrl.Request{
				NamespacedName: client.ObjectKey{
					Name:      "test-component",
					Namespace: "default",
				},
			}

			err = reconciler.setStatusConditionAndServiceReplicaStatus(ctx, dcd, tt.componentReconcileResult)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			// Fetch the updated DCD to verify status was set
			updatedDCD := &v1alpha1.DynamoComponentDeployment{}
			err = fakeKubeClient.Get(ctx, req.NamespacedName, updatedDCD)
			g.Expect(err).NotTo(gomega.HaveOccurred())

			// Assert the status condition
			g.Expect(updatedDCD.Status.Conditions).To(gomega.HaveLen(1))
			condition := updatedDCD.Status.Conditions[0]
			g.Expect(condition.Type).To(gomega.Equal(v1alpha1.DynamoGraphDeploymentConditionTypeAvailable))
			g.Expect(condition.Status).To(gomega.Equal(tt.wantConditionStatus))
			g.Expect(condition.Reason).To(gomega.Equal(tt.wantConditionReason))
			g.Expect(condition.Message).To(gomega.Equal(tt.wantConditionMessage))

			// Assert the service replica status
			g.Expect(updatedDCD.Status.Service).To(gomega.Equal(tt.wantServiceReplicaStatus))

			// Assert the observed generation
			g.Expect(updatedDCD.Status.ObservedGeneration).To(gomega.Equal(generation))
		})
	}
}

func Test_generateDeployment_Strategy(t *testing.T) {
	type args struct {
		annotations map[string]string
	}
	tests := []struct {
		name         string
		args         args
		wantStrategy appsv1.DeploymentStrategy
	}{
		{
			name: "no annotations - default RollingUpdate with default maxSurge and maxUnavailable",
			args: args{
				annotations: nil,
			},
			wantStrategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       ptr.To(intstr.FromString("25%")),
					MaxUnavailable: ptr.To(intstr.FromString("25%")),
				},
			},
		},
		{
			name: "deployment-strategy annotation with Recreate - strategy is Recreate",
			args: args{
				annotations: map[string]string{
					KubeAnnotationDeploymentStrategy: "Recreate",
				},
			},
			wantStrategy: appsv1.DeploymentStrategy{
				Type: appsv1.RecreateDeploymentStrategyType,
			},
		},
		{
			name: "deployment-strategy Recreate with maxSurge/maxUnavailable - maxSurge/maxUnavailable are ignored",
			args: args{
				annotations: map[string]string{
					KubeAnnotationDeploymentStrategy:                    "Recreate",
					KubeAnnotationDeploymentRollingUpdateMaxSurge:       "50%",
					KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "30%",
				},
			},
			wantStrategy: appsv1.DeploymentStrategy{
				Type: appsv1.RecreateDeploymentStrategyType,
			},
		},
		{
			name: "deployment-strategy RollingUpdate with only maxSurge",
			args: args{
				annotations: map[string]string{
					KubeAnnotationDeploymentStrategy:              "RollingUpdate",
					KubeAnnotationDeploymentRollingUpdateMaxSurge: "50%",
				},
			},
			wantStrategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       ptr.To(intstr.FromString("50%")),
					MaxUnavailable: ptr.To(intstr.FromString("25%")),
				},
			},
		},
		{
			name: "deployment-strategy RollingUpdate with only maxUnavailable",
			args: args{
				annotations: map[string]string{
					KubeAnnotationDeploymentStrategy:                    "RollingUpdate",
					KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "10%",
				},
			},
			wantStrategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       ptr.To(intstr.FromString("25%")),
					MaxUnavailable: ptr.To(intstr.FromString("10%")),
				},
			},
		},
		{
			name: "deployment-strategy RollingUpdate with both maxSurge and maxUnavailable",
			args: args{
				annotations: map[string]string{
					KubeAnnotationDeploymentStrategy:                    "RollingUpdate",
					KubeAnnotationDeploymentRollingUpdateMaxSurge:       "40%",
					KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "20%",
				},
			},
			wantStrategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       ptr.To(intstr.FromString("40%")),
					MaxUnavailable: ptr.To(intstr.FromString("20%")),
				},
			},
		},
		{
			name: "deployment-strategy RollingUpdate with integer maxSurge and maxUnavailable (not percentages)",
			args: args{
				annotations: map[string]string{
					KubeAnnotationDeploymentStrategy:                    "RollingUpdate",
					KubeAnnotationDeploymentRollingUpdateMaxSurge:       "1",
					KubeAnnotationDeploymentRollingUpdateMaxUnavailable: "0",
				},
			},
			wantStrategy: appsv1.DeploymentStrategy{
				Type: appsv1.RollingUpdateDeploymentStrategyType,
				RollingUpdate: &appsv1.RollingUpdateDeployment{
					MaxSurge:       ptr.To(intstr.FromInt(1)),
					MaxUnavailable: ptr.To(intstr.FromInt(0)),
				},
			},
		},
	}

	// Initialize scheme & add API types
	s := scheme.Scheme
	if err := v1alpha1.AddToScheme(s); err != nil {
		t.Fatalf("Failed to add v1alpha1 to scheme: %v", err)
	}
	if err := corev1.AddToScheme(s); err != nil {
		t.Fatalf("Failed to add corev1 to scheme: %v", err)
	}
	if err := appsv1.AddToScheme(s); err != nil {
		t.Fatalf("Failed to add appsv1 to scheme: %v", err)
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			dcd := &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment-strategy",
					Namespace: "default",
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					BackendFramework: string(dynamo.BackendFrameworkVLLM),
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						ServiceName:     "test-service",
						DynamoNamespace: ptr.To("default"),
						ComponentType:   string(commonconsts.ComponentTypeDecode),
						Replicas:        ptr.To(int32(1)),
						Annotations:     tt.args.annotations,
						ExtraPodSpec: &v1alpha1.ExtraPodSpec{
							MainContainer: &corev1.Container{
								Image: "test-image:latest",
							},
						},
					},
				},
			}

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(dcd).
				Build()

			recorder := record.NewFakeRecorder(100)
			reconciler := &DynamoComponentDeploymentReconciler{
				Client:   fakeKubeClient,
				Recorder: recorder,
				Config:   controller_common.Config{},
				DockerSecretRetriever: &mockDockerSecretRetriever{
					GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
						return []string{}, nil
					},
				},
			}

			opt := generateResourceOption{
				dynamoComponentDeployment: dcd,
			}

			deployment, toDelete, err := reconciler.generateDeployment(context.Background(), opt)
			g.Expect(err).NotTo(gomega.HaveOccurred())
			g.Expect(toDelete).To(gomega.BeFalse())
			g.Expect(deployment).NotTo(gomega.BeNil())
			g.Expect(deployment.Spec.Strategy).To(gomega.Equal(tt.wantStrategy))
		})
	}
}
