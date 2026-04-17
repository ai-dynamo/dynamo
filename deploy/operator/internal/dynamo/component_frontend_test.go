/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestComputeFrontendPreStopSeconds(t *testing.T) {
	tests := []struct {
		name               string
		gracePeriodSeconds int64
		wantSleep          int64
	}{
		{
			name:               "default grace period (60s) caps at 10s",
			gracePeriodSeconds: 60,
			wantSleep:          10,
		},
		{
			name:               "high grace period still caps at 10s",
			gracePeriodSeconds: 120,
			wantSleep:          10,
		},
		{
			name:               "grace period 20s gives exactly 10s",
			gracePeriodSeconds: 20,
			wantSleep:          10,
		},
		{
			name:               "grace period 10s gives 5s (half budget)",
			gracePeriodSeconds: 10,
			wantSleep:          5,
		},
		{
			name:               "grace period 6s gives 3s",
			gracePeriodSeconds: 6,
			wantSleep:          3,
		},
		{
			name:               "grace period 2s gives 1s",
			gracePeriodSeconds: 2,
			wantSleep:          1,
		},
		{
			name:               "grace period 1s gives 0 (integer division)",
			gracePeriodSeconds: 1,
			wantSleep:          0,
		},
		{
			name:               "zero grace period gives 0",
			gracePeriodSeconds: 0,
			wantSleep:          0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ComputeFrontendPreStopSeconds(tt.gracePeriodSeconds)
			if got != tt.wantSleep {
				t.Errorf("ComputeFrontendPreStopSeconds(%d) = %d, want %d", tt.gracePeriodSeconds, got, tt.wantSleep)
			}
		})
	}
}

func TestFrontendDefaults_GetBaseContainer(t *testing.T) {
	type fields struct {
		BaseComponentDefaults *BaseComponentDefaults
	}
	tests := []struct {
		name             string
		fields           fields
		componentContext ComponentContext
		want             corev1.Container
		wantErr          bool
	}{
		{
			name: "default frontend container",
			fields: fields{
				BaseComponentDefaults: &BaseComponentDefaults{},
			},
			componentContext: ComponentContext{
				numberOfNodes:                  1,
				ParentGraphDeploymentName:      "name",
				ParentGraphDeploymentNamespace: "namespace",
				DynamoNamespace:                "dynamo-namespace",
				ComponentType:                  commonconsts.ComponentTypeFrontend,
			},
			want: corev1.Container{
				Name:    commonconsts.MainContainerName,
				Command: []string{"python3"},
				Args:    []string{"-m", "dynamo.frontend"},
				Ports: []corev1.ContainerPort{
					{
						Protocol:      corev1.ProtocolTCP,
						Name:          commonconsts.DynamoContainerPortName,
						ContainerPort: int32(commonconsts.DynamoServicePort),
					},
				},
				LivenessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						HTTPGet: &corev1.HTTPGetAction{
							Path: "/live",
							Port: intstr.FromString(commonconsts.DynamoContainerPortName),
						},
					},
					InitialDelaySeconds: 15,
					PeriodSeconds:       10,
					TimeoutSeconds:      1,
					FailureThreshold:    3,
				},
				ReadinessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						HTTPGet: &corev1.HTTPGetAction{
							Path: "/health",
							Port: intstr.FromString(commonconsts.DynamoContainerPortName),
						},
					},
					InitialDelaySeconds: 10,
					PeriodSeconds:       10,
					TimeoutSeconds:      3,
					FailureThreshold:    3,
				},
				Env: []corev1.EnvVar{
					{Name: commonconsts.DynamoNamespaceEnvVar, Value: "dynamo-namespace"},
					{Name: commonconsts.DynamoComponentEnvVar, Value: commonconsts.ComponentTypeFrontend},
					{Name: "DYN_PARENT_DGD_K8S_NAME", Value: "name"},
					{Name: "DYN_PARENT_DGD_K8S_NAMESPACE", Value: "namespace"},
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
						Name: "POD_UID",
						ValueFrom: &corev1.EnvVarSource{
							FieldRef: &corev1.ObjectFieldSelector{
								FieldPath: "metadata.uid",
							},
						},
					},
					{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"},
					{Name: commonconsts.EnvDynamoServicePort, Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort)},
					{Name: "DYN_HTTP_PORT", Value: fmt.Sprintf("%d", commonconsts.DynamoServicePort)},
					{Name: commonconsts.DynamoNamespacePrefixEnvVar, Value: "dynamo-namespace"},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &FrontendDefaults{
				BaseComponentDefaults: tt.fields.BaseComponentDefaults,
			}
			got, err := f.GetBaseContainer(tt.componentContext)
			if (err != nil) != tt.wantErr {
				t.Errorf("FrontendDefaults.GetBaseContainer() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			diff := cmp.Diff(got, tt.want)
			if diff != "" {
				t.Errorf("FrontendDefaults.GetBaseContainer() mismatch (-got +want):\n%s", diff)
			}
		})
	}
}
