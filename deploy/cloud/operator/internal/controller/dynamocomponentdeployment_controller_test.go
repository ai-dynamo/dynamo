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

package controller

import (
	"context"
	"fmt"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
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
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
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
								Ingress: v1alpha1.IngressSpec{
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
													Port: networkingv1.ServiceBackendPort{Number: 3000},
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
								Ingress: v1alpha1.IngressSpec{
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
								Ingress: v1alpha1.IngressSpec{
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
								Ingress: v1alpha1.IngressSpec{
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
											Number: 3000,
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
		Client            client.Client
		Recorder          record.EventRecorder
		Config            controller_common.Config
		NatsAddr          string
		EtcdAddr          string
		EtcdStorage       etcdStorage
		UseVirtualService bool
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
								ServiceName:     "service1",
								DynamoNamespace: &[]string{"default"}[0],
								Annotations: map[string]string{
									"nvidia.com/deployment-type": "leader-worker",
									"nvidia.com/lws-size":        "2",
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service1",
							Namespace: "default",
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
			name: "missing lws size annotation",
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-missing-lws-size",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service-missing-lws-size",
								DynamoNamespace: &[]string{"default"}[0],
								Annotations: map[string]string{
									"nvidia.com/deployment-type": "leader-worker",
									// "nvidia.com/lws-size" is missing
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-missing-lws-size",
							Namespace: "default",
						},
					},
					instanceID: ptr.To(0),
				},
			},
			want:    nil,
			want1:   false,
			wantErr: true,
		},
		{
			name: "invalid lws size annotation (non-integer)",
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-invalid-lws-size-non-int",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service-invalid-lws-size-non-int",
								DynamoNamespace: &[]string{"default"}[0],
								Annotations: map[string]string{
									"nvidia.com/deployment-type": "leader-worker",
									"nvidia.com/lws-size":        "abc",
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-invalid-lws-size-non-int",
							Namespace: "default",
						},
					},
					instanceID: ptr.To(1),
				},
			},
			want:    nil,
			want1:   false,
			wantErr: true,
		},
		{
			name: "invalid lws size annotation (zero)",
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-invalid-lws-size-zero",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service-invalid-lws-size-zero",
								DynamoNamespace: &[]string{"default"}[0],
								Annotations: map[string]string{
									"nvidia.com/deployment-type": "leader-worker",
									"nvidia.com/lws-size":        "0",
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-invalid-lws-size-zero",
							Namespace: "default",
						},
					},
					instanceID: ptr.To(2),
				},
			},
			want:    nil,
			want1:   false,
			wantErr: true,
		},
		{
			name: "invalid lws size annotation (negative)",
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-invalid-lws-size-negative",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service-invalid-lws-size-negative",
								DynamoNamespace: &[]string{"default"}[0],
								Annotations: map[string]string{
									"nvidia.com/deployment-type": "leader-worker",
									"nvidia.com/lws-size":        "-1",
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-invalid-lws-size-negative",
							Namespace: "default",
						},
					},
					instanceID: ptr.To(3),
				},
			},
			want:    nil,
			want1:   false,
			wantErr: true,
		},
		{
			name: "lws size of 1 - lws should not be used",
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-valid-lws-size-one",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "service-valid-lws-size-one",
								DynamoNamespace: &[]string{"default"}[0],
								Annotations: map[string]string{
									"nvidia.com/deployment-type": "leader-worker",
									"nvidia.com/lws-size":        "1",
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-valid-lws-size-one",
							Namespace: "default",
						},
					},
					instanceID: ptr.To(4),
				},
			},
			want:    nil,
			want1:   false,
			wantErr: true,
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
								Annotations: map[string]string{
									"nvidia.com/deployment-type": "leader-worker",
									"nvidia.com/lws-size":        "2",
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-nil-instanceid",
							Namespace: "default",
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
								Annotations: map[string]string{
									"nvidia.com/deployment-type": "leader-worker",
									"nvidia.com/lws-size":        "2",
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "service-negative-instanceid",
							Namespace: "default",
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
				Client:            tt.fields.Client,
				Recorder:          tt.fields.Recorder,
				Config:            tt.fields.Config,
				NatsAddr:          tt.fields.NatsAddr,
				EtcdAddr:          tt.fields.EtcdAddr,
				EtcdStorage:       tt.fields.EtcdStorage,
				UseVirtualService: tt.fields.UseVirtualService,
			}
			got, got1, err := r.generateVolcanoPodGroup(tt.args.ctx, tt.args.opt)
			if (err != nil) != tt.wantErr {
				t.Errorf("DynamoComponentDeploymentReconciler.generateVolcanoPodGroup() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			g.Expect(got).To(gomega.Equal(tt.want))
			g.Expect(got1).To(gomega.Equal(tt.want1))
		})
	}
}

func TestDynamoComponentDeploymentReconciler_generateLeaderWorkerSet(t *testing.T) {
	var limit = ptr.To(resource.MustParse("250Mi"))
	limit.SetMilli(ptr.To(resource.MustParse("1Gi")).MilliValue() / 2)
	type fields struct {
		Client            client.Client
		Recorder          record.EventRecorder
		Config            controller_common.Config
		NatsAddr          string
		EtcdAddr          string
		EtcdStorage       etcdStorage
		UseVirtualService bool
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
			},
			args: args{
				ctx: context.Background(),
				opt: generateResourceOption{
					dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
						ObjectMeta: metav1.ObjectMeta{
							Name:      "test-lws-deploy",
							Namespace: "default",
						},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponent: "test-lws-component",
							DynamoTag:       "test-tag",
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								ServiceName:     "test-lws-deploy-service",
								DynamoNamespace: &[]string{"default"}[0],
								Annotations: map[string]string{
									"nvidia.com/deployment-type": "leader-worker",
									"nvidia.com/lws-size":        "2",
								},
								Resources: &common.Resources{
									Limits: &common.ResourceItem{
										GPU: "1",
									},
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{
						ObjectMeta: metav1.ObjectMeta{Name: "test-lws-component", Namespace: "default"},
						Spec:       v1alpha1.DynamoComponentSpec{Image: "test-image:latest"},
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
								commonconsts.KubeLabelDynamoDeploymentPod: commonconsts.KubeLabelValueTrue,
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
						commonconsts.KubeLabelDynamoComponent:     "test-lws-component",
						commonconsts.KubeLabelDynamoComponentType: commonconsts.DynamoApiServerComponentName,
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
									"instance-id":                         "0",
									"role":                                "leader",
									commonconsts.KubeLabelDynamoComponent: "test-lws-component",
									commonconsts.KubeLabelDynamoComponentType: commonconsts.DynamoApiServerComponentName,
								},
								Annotations: map[string]string{
									"scheduling.k8s.io/group-name": "test-lws-deploy-0",
								},
							},
							Spec: corev1.PodSpec{
								SchedulerName: "volcano",
								Containers: []corev1.Container{
									{
										Name:    "main",
										Image:   "test-image:latest",
										Command: []string{"sh", "-c"},
										Args:    []string{"ray start --head --port=6379 && cd src && uv run dynamo serve --service-name test-lws-deploy-service test-tag --test-lws-deploy-service.ServiceArgs.dynamo.namespace=default --test-lws-deploy-service.environment=kubernetes"},
										Env:     []corev1.EnvVar{{Name: "DYNAMO_PORT", Value: "3000"}},
										VolumeMounts: []corev1.VolumeMount{
											{
												Name: "shared-memory", MountPath: "/dev/shm",
											},
										},
										Ports: []corev1.ContainerPort{
											{
												Protocol: corev1.ProtocolTCP, Name: "http", ContainerPort: 3000,
											},
										},
										TTY:   true,
										Stdin: true,
										Resources: corev1.ResourceRequirements{
											Requests: corev1.ResourceList{
												corev1.ResourceCPU:    resource.MustParse("300m"),
												corev1.ResourceMemory: resource.MustParse("500Mi"),
											},
											Limits: corev1.ResourceList{
												corev1.ResourceCPU:    resource.MustParse("500m"),
												corev1.ResourceMemory: resource.MustParse("1Gi"),
												"nvidia.com/gpu":      resource.MustParse("1"),
											},
										},
									},
								},
								Volumes:            []corev1.Volume{{Name: "shared-memory", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{Medium: corev1.StorageMediumMemory, SizeLimit: limit}}}},
								ImagePullSecrets:   []corev1.LocalObjectReference{{Name: ""}}, // Assuming default config gives empty secret name
								ServiceAccountName: "default-test-sa",                         // Updated to reflect mocked SA
							},
						},
						WorkerTemplate: corev1.PodTemplateSpec{
							ObjectMeta: metav1.ObjectMeta{
								Labels: map[string]string{
									"instance-id":                         "0",
									"role":                                "worker",
									commonconsts.KubeLabelDynamoComponent: "test-lws-component",
									commonconsts.KubeLabelDynamoComponentType: commonconsts.DynamoApiServerComponentName,
								},
								Annotations: map[string]string{
									"scheduling.k8s.io/group-name": "test-lws-deploy-0",
								},
							},
							Spec: corev1.PodSpec{
								SchedulerName: "volcano",
								Containers: []corev1.Container{
									{
										Name:         "main",
										Image:        "test-image:latest",
										Command:      []string{"sh", "-c"},
										Args:         []string{"ray start --address=$(LWS_LEADER_ADDRESS):6379 --block"},
										Env:          []corev1.EnvVar{{Name: "DYNAMO_PORT", Value: "3000"}},
										VolumeMounts: []corev1.VolumeMount{{Name: "shared-memory", MountPath: "/dev/shm"}},
										Ports:        []corev1.ContainerPort{{Protocol: corev1.ProtocolTCP, Name: "http", ContainerPort: 3000}},
										TTY:          true,
										Stdin:        true,
										Resources: corev1.ResourceRequirements{
											Requests: corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("300m"), corev1.ResourceMemory: resource.MustParse("500Mi")},
											Limits:   corev1.ResourceList{corev1.ResourceCPU: resource.MustParse("500m"), corev1.ResourceMemory: resource.MustParse("1Gi"), "nvidia.com/gpu": resource.MustParse("1")},
										},
									},
								},
								Volumes:            []corev1.Volume{{Name: "shared-memory", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{Medium: corev1.StorageMediumMemory, SizeLimit: limit}}}},
								ImagePullSecrets:   []corev1.LocalObjectReference{{Name: ""}},
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
						ObjectMeta: metav1.ObjectMeta{Name: "test-lws-nil-id", Namespace: "default", Annotations: map[string]string{KubeAnnotationLWSSize: "2"}},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponent: "test-comp", DynamoTag: "test",
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								Resources: &common.Resources{
									Limits: &common.ResourceItem{
										GPU: "1",
									},
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{
						ObjectMeta: metav1.ObjectMeta{Namespace: "default"},
						Spec:       v1alpha1.DynamoComponentSpec{Image: "test-image:latest"},
					},
					instanceID: nil,
				},
				mockServiceAccounts: []client.Object{ // Provide a default SA for consistency, though not strictly needed here
					&corev1.ServiceAccount{
						ObjectMeta: metav1.ObjectMeta{
							Name: "default-test-sa", Namespace: "default", // Match namespace
							Labels: map[string]string{commonconsts.KubeLabelDynamoDeploymentPod: commonconsts.KubeLabelValueTrue},
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
						ObjectMeta: metav1.ObjectMeta{Name: "test-lws-leader-err", Namespace: "default", Annotations: map[string]string{KubeAnnotationLWSSize: "2"}},
						Spec: v1alpha1.DynamoComponentDeploymentSpec{
							DynamoComponent: "test-comp", DynamoTag: "test",
							DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
								Resources: &common.Resources{
									Limits: &common.ResourceItem{
										GPU: "1",
									},
								},
							},
						},
					},
					dynamoComponent: &v1alpha1.DynamoComponent{ // Image is missing, will cause error in generatePodTemplateSpec
						ObjectMeta: metav1.ObjectMeta{Name: "test-lws-component-leader-err", Namespace: "default"},
						Spec:       v1alpha1.DynamoComponentSpec{Image: ""},
					},
					instanceID: ptr.To(0),
				},
				// No specific SA needed if error is before SA listing, but good to be consistent
				mockServiceAccounts: []client.Object{
					&corev1.ServiceAccount{
						ObjectMeta: metav1.ObjectMeta{
							Name: "default-test-sa", Namespace: "default", // Match namespace
							Labels: map[string]string{commonconsts.KubeLabelDynamoDeploymentPod: commonconsts.KubeLabelValueTrue},
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
			if tt.args.opt.dynamoComponent != nil {
				initialClientObjects = append(initialClientObjects, tt.args.opt.dynamoComponent)
			}
			if len(tt.args.mockServiceAccounts) > 0 {
				initialClientObjects = append(initialClientObjects, tt.args.mockServiceAccounts...)
			}

			fakeKubeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(initialClientObjects...).
				Build()

			r := &DynamoComponentDeploymentReconciler{
				Client:            fakeKubeClient, // Use the fake client
				Recorder:          tt.fields.Recorder,
				Config:            tt.fields.Config,
				NatsAddr:          tt.fields.NatsAddr,
				EtcdAddr:          tt.fields.EtcdAddr,
				EtcdStorage:       tt.fields.EtcdStorage,
				UseVirtualService: tt.fields.UseVirtualService,
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
