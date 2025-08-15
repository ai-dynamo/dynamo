/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestPlannerDefaults_GetBaseContainer(t *testing.T) {
	type fields struct {
		BaseComponentDefaults *BaseComponentDefaults
	}
	type args struct {
		numberOfNodes                  int32
		parentGraphDeploymentName      string
		parentGraphDeploymentNamespace string
		dynamoNamespace                string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    corev1.Container
		wantErr bool
	}{
		{
			name: "test",
			fields: fields{
				BaseComponentDefaults: &BaseComponentDefaults{},
			},
			args: args{
				numberOfNodes:                  1,
				parentGraphDeploymentName:      "name",
				parentGraphDeploymentNamespace: "namespace",
				dynamoNamespace:                "dynamo-namespace",
			},
			want: corev1.Container{
				Name: "main",
				Command: []string{
					"/bin/sh",
					"-c",
				},
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("2"),
						corev1.ResourceMemory: resource.MustParse("2Gi"),
					},
					Limits: corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("2"),
						corev1.ResourceMemory: resource.MustParse("2Gi"),
					},
				},
				Env: []corev1.EnvVar{
					{Name: "DYN_NAMESPACE", Value: "dynamo-namespace"},
					{Name: "DYN_PARENT_DGD_K8S_NAME", Value: "name"},
					{Name: "DYN_PARENT_DGD_K8S_NAMESPACE", Value: "namespace"},
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &PlannerDefaults{
				BaseComponentDefaults: tt.fields.BaseComponentDefaults,
			}
			got, err := p.GetBaseContainer(ComponentContext{
				numberOfNodes:                  tt.args.numberOfNodes,
				ParentGraphDeploymentName:      tt.args.parentGraphDeploymentName,
				ParentGraphDeploymentNamespace: tt.args.parentGraphDeploymentNamespace,
				DynamoNamespace:                tt.args.dynamoNamespace,
			})
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
