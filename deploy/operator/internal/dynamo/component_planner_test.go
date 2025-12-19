/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/cloud/operator/internal/consts"
	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
)

func TestPlannerDefaults_GetBaseContainer(t *testing.T) {
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
			name: "test",
			fields: fields{
				BaseComponentDefaults: &BaseComponentDefaults{},
			},
			componentContext: ComponentContext{
				numberOfNodes:                  1,
				ParentGraphDeploymentName:      "name",
				ParentGraphDeploymentNamespace: "namespace",
				DynamoNamespace:                "dynamo-namespace",
				ComponentType:                  commonconsts.ComponentTypePlanner,
			},
			want: corev1.Container{
				Name: commonconsts.MainContainerName,
				Command: []string{
					"/bin/sh",
					"-c",
				},
				Ports: []corev1.ContainerPort{
					{Name: commonconsts.DynamoMetricsPortName, ContainerPort: commonconsts.DynamoPlannerMetricsPort, Protocol: corev1.ProtocolTCP},
				},
				Env: []corev1.EnvVar{
					{Name: commonconsts.DynamoNamespaceEnvVar, Value: "dynamo-namespace"},
					{Name: commonconsts.DynamoComponentEnvVar, Value: commonconsts.ComponentTypePlanner},
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
					{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"},
					{Name: "PLANNER_PROMETHEUS_PORT", Value: fmt.Sprintf("%d", commonconsts.DynamoPlannerMetricsPort)},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &PlannerDefaults{
				BaseComponentDefaults: tt.fields.BaseComponentDefaults,
			}
			got, err := p.GetBaseContainer(tt.componentContext)
			if (err != nil) != tt.wantErr {
				t.Errorf("PlannerDefaults.GetBaseContainer() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			diff := cmp.Diff(got, tt.want)
			if diff != "" {
				t.Errorf("PlannerDefaults.GetBaseContainer() = %v, want %v", diff, tt.want)
			}
		})
	}
}
