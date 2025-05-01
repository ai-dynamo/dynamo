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

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/bsm/gomega"
	istioNetworking "istio.io/api/networking/v1beta1"
	networkingv1beta1 "istio.io/client-go/pkg/apis/networking/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
