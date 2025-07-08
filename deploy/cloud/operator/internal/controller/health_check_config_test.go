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

package controller

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/controller_common"
	"github.com/google/go-cmp/cmp"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestDefaultHealthCheckConfig(t *testing.T) {
	config := DefaultHealthCheckConfig()

	if config.LivenessPath != "/healthz" {
		t.Errorf("DefaultHealthCheckConfig().LivenessPath = %v, want %v", config.LivenessPath, "/healthz")
	}

	if config.ReadinessPath != "/readyz" {
		t.Errorf("DefaultHealthCheckConfig().ReadinessPath = %v, want %v", config.ReadinessPath, "/readyz")
	}
}

func TestGetHealthCheckConfig(t *testing.T) {
	tests := []struct {
		name                      string
		dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment
		ctrlConfig                *controller_common.Config
		want                      HealthCheckConfig
	}{
		{
			name: "default paths",
			dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-deployment",
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{},
			},
			ctrlConfig: nil,
			want: HealthCheckConfig{
				LivenessPath:  "/healthz",
				ReadinessPath: "/readyz",
			},
		},
		{
			name: "custom paths from controller config",
			dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-deployment",
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{},
			},
			ctrlConfig: &controller_common.Config{
				CustomLivenessPath:  "/custom-healthz",
				CustomReadinessPath: "/custom-readyz",
			},
			want: HealthCheckConfig{
				LivenessPath:  "/custom-healthz",
				ReadinessPath: "/custom-readyz",
			},
		},
		{
			name: "custom paths from deployment config",
			dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-deployment",
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						Envs: []corev1.EnvVar{
							{
								Name:  "DYN_DEPLOYMENT_CONFIG",
								Value: `{"healthCheck": {"livenessPath": "/deployment-healthz", "readinessPath": "/deployment-readyz"}}`,
							},
						},
					},
				},
			},
			ctrlConfig: nil,
			want: HealthCheckConfig{
				LivenessPath:  "/deployment-healthz",
				ReadinessPath: "/deployment-readyz",
			},
		},
		{
			name: "custom paths from annotations",
			dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-deployment",
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						Annotations: map[string]string{
							"nvidia.com/liveness-path":  "/annotation-healthz",
							"nvidia.com/readiness-path": "/annotation-readyz",
						},
					},
				},
			},
			ctrlConfig: nil,
			want: HealthCheckConfig{
				LivenessPath:  "/annotation-healthz",
				ReadinessPath: "/annotation-readyz",
			},
		},
		{
			name: "priority order - annotations over deployment config over controller config",
			dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-deployment",
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						Annotations: map[string]string{
							"nvidia.com/liveness-path": "/annotation-healthz",
							// No readiness path in annotations
						},
						Envs: []corev1.EnvVar{
							{
								Name:  "DYN_DEPLOYMENT_CONFIG",
								Value: `{"healthCheck": {"livenessPath": "/deployment-healthz", "readinessPath": "/deployment-readyz"}}`,
							},
						},
					},
				},
			},
			ctrlConfig: &controller_common.Config{
				CustomLivenessPath:  "/custom-healthz",
				CustomReadinessPath: "/custom-readyz",
			},
			want: HealthCheckConfig{
				LivenessPath:  "/annotation-healthz", // From annotations (highest priority)
				ReadinessPath: "/deployment-readyz",  // From deployment config (middle priority)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			got := GetHealthCheckConfig(tt.dynamoComponentDeployment, tt.ctrlConfig)
			g.Expect(got).To(gomega.Equal(tt.want))
		})
	}
}

func TestCreateDefaultLivenessProbe(t *testing.T) {
	path := "/custom-healthz"
	probe := CreateDefaultLivenessProbe(path)

	if probe.HTTPGet.Path != path {
		t.Errorf("CreateDefaultLivenessProbe().HTTPGet.Path = %v, want %v", probe.HTTPGet.Path, path)
	}

	if probe.InitialDelaySeconds != 60 {
		t.Errorf("CreateDefaultLivenessProbe().InitialDelaySeconds = %v, want %v", probe.InitialDelaySeconds, 60)
	}
}

func TestCreateDefaultReadinessProbe(t *testing.T) {
	path := "/custom-readyz"
	probe := CreateDefaultReadinessProbe(path)

	if probe.HTTPGet.Path != path {
		t.Errorf("CreateDefaultReadinessProbe().HTTPGet.Path = %v, want %v", probe.HTTPGet.Path, path)
	}

	if probe.InitialDelaySeconds != 60 {
		t.Errorf("CreateDefaultReadinessProbe().InitialDelaySeconds = %v, want %v", probe.InitialDelaySeconds, 60)
	}
}

func TestUpdateProbeWithCustomPath(t *testing.T) {
	tests := []struct {
		name  string
		probe *corev1.Probe
		path  string
		want  *corev1.Probe
	}{
		{
			name:  "nil probe",
			probe: nil,
			path:  "/custom-path",
			want:  nil,
		},
		{
			name: "http probe",
			probe: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					HTTPGet: &corev1.HTTPGetAction{
						Path: "/original-path",
						Port: intstr.FromString("health"),
					},
				},
				InitialDelaySeconds: 30,
				PeriodSeconds:       10,
			},
			path: "/custom-path",
			want: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					HTTPGet: &corev1.HTTPGetAction{
						Path: "/custom-path",
						Port: intstr.FromString("health"),
					},
				},
				InitialDelaySeconds: 30,
				PeriodSeconds:       10,
			},
		},
		{
			name: "non-http probe",
			probe: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					Exec: &corev1.ExecAction{
						Command: []string{"cat", "/tmp/healthy"},
					},
				},
				InitialDelaySeconds: 30,
				PeriodSeconds:       10,
			},
			path: "/custom-path",
			want: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					Exec: &corev1.ExecAction{
						Command: []string{"cat", "/tmp/healthy"},
					},
				},
				InitialDelaySeconds: 30,
				PeriodSeconds:       10,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := UpdateProbeWithCustomPath(tt.probe, tt.path)
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("UpdateProbeWithCustomPath() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestGetProbesWithCustomPaths(t *testing.T) {
	tests := []struct {
		name                      string
		dynamoComponentDeployment *v1alpha1.DynamoComponentDeployment
		ctrlConfig                *controller_common.Config
		wantLiveness              *corev1.Probe
		wantReadiness             *corev1.Probe
	}{
		{
			name: "default probes",
			dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-deployment",
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{},
			},
			ctrlConfig: nil,
			wantLiveness: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					HTTPGet: &corev1.HTTPGetAction{
						Path: "/healthz",
						Port: intstr.FromString("health"),
					},
				},
				InitialDelaySeconds: 60,
				PeriodSeconds:       60,
				TimeoutSeconds:      5,
				FailureThreshold:    10,
				SuccessThreshold:    1,
			},
			wantReadiness: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					HTTPGet: &corev1.HTTPGetAction{
						Path: "/readyz",
						Port: intstr.FromString("health"),
					},
				},
				InitialDelaySeconds: 60,
				PeriodSeconds:       60,
				TimeoutSeconds:      5,
				FailureThreshold:    10,
				SuccessThreshold:    1,
			},
		},
		{
			name: "custom probes from deployment spec",
			dynamoComponentDeployment: &v1alpha1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-deployment",
				},
				Spec: v1alpha1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: v1alpha1.DynamoComponentDeploymentSharedSpec{
						LivenessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{
									Path: "/original-healthz",
									Port: intstr.FromString("health"),
								},
							},
							InitialDelaySeconds: 30,
							PeriodSeconds:       10,
						},
						ReadinessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								HTTPGet: &corev1.HTTPGetAction{
									Path: "/original-readyz",
									Port: intstr.FromString("health"),
								},
							},
							InitialDelaySeconds: 30,
							PeriodSeconds:       10,
						},
						Annotations: map[string]string{
							"nvidia.com/liveness-path":  "/custom-healthz",
							"nvidia.com/readiness-path": "/custom-readyz",
						},
					},
				},
			},
			ctrlConfig: nil,
			wantLiveness: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					HTTPGet: &corev1.HTTPGetAction{
						Path: "/custom-healthz",
						Port: intstr.FromString("health"),
					},
				},
				InitialDelaySeconds: 30,
				PeriodSeconds:       10,
			},
			wantReadiness: &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					HTTPGet: &corev1.HTTPGetAction{
						Path: "/custom-readyz",
						Port: intstr.FromString("health"),
					},
				},
				InitialDelaySeconds: 30,
				PeriodSeconds:       10,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			gotLiveness, gotReadiness := GetProbesWithCustomPaths(tt.dynamoComponentDeployment, tt.ctrlConfig)

			// Compare only the path for HTTP probes
			if gotLiveness != nil && gotLiveness.HTTPGet != nil && tt.wantLiveness != nil && tt.wantLiveness.HTTPGet != nil {
				g.Expect(gotLiveness.HTTPGet.Path).To(gomega.Equal(tt.wantLiveness.HTTPGet.Path))
			} else {
				g.Expect(gotLiveness).To(gomega.BeEquivalentTo(tt.wantLiveness))
			}

			if gotReadiness != nil && gotReadiness.HTTPGet != nil && tt.wantReadiness != nil && tt.wantReadiness.HTTPGet != nil {
				g.Expect(gotReadiness.HTTPGet.Path).To(gomega.Equal(tt.wantReadiness.HTTPGet.Path))
			} else {
				g.Expect(gotReadiness).To(gomega.BeEquivalentTo(tt.wantReadiness))
			}
		})
	}
}
