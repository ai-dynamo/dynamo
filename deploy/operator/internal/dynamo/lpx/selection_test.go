/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestLPXSchedulerSelectionUsesComponents(t *testing.T) {
	t.Log("Select components by their declared type, independently of build or templates")
	canonicalLPX := []dynamov1beta1.DynamoComponentDeploymentSharedSpec{{ComponentType: dynamov1beta1.ComponentTypeLPX}}

	t.Log("Define default-selection scenarios")
	tests := []struct {
		name        string
		annotations map[string]string
		components  []dynamov1beta1.DynamoComponentDeploymentSharedSpec
		want        bool
	}{
		{name: "canonical LPX", components: canonicalLPX, want: true},
		{
			name: "component provider",
			annotations: map[string]string{
				commonconsts.KubeAnnotationWorkloadProvider: commonconsts.WorkloadProviderComponent,
			},
			components: canonicalLPX,
			want:       true,
		},
		{
			name: "Grove opt-out",
			annotations: map[string]string{
				commonconsts.KubeAnnotationEnableGrove: "FALSE",
			},
			components: canonicalLPX,
			want:       true,
		},
		{name: "non-LPX", components: []dynamov1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: dynamov1beta1.ComponentRoleLPXAgent}}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Construct the DGD selection input")
			dgd := &dynamov1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Annotations: test.annotations},
				Spec: dynamov1beta1.DynamoGraphDeploymentSpec{
					Components: test.components,
					Scheduling: &dynamov1beta1.SchedulingSpec{},
				},
			}

			t.Log("Resolve and verify scheduler selection")
			selected := dgd.HasLPXComponent()
			require.Equal(t, test.want, selected)
		})
	}
}

func TestExpandedSelectedModelNames(t *testing.T) {
	t.Log("Expand admitted component replicas into runtime model names")
	require.Equal(t, []string{"draft0", "draft1", "draft2"}, expandedSelectedModelNames(0, 2, 3))
	require.Equal(t, []string{"draft0"}, expandedSelectedModelNames(0, 2, 1))
	require.Equal(t, []string{"target"}, expandedSelectedModelNames(1, 2, 1))
	require.Equal(t, []string{"default"}, expandedSelectedModelNames(0, 1, 2))
}
