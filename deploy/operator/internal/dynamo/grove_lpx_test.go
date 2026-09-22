// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

func TestEvaluateLPXGroveReadinessUsesEachComponentRuntimeNamespace(t *testing.T) {
	draft := v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:         "draft",
		ComponentType:         v1beta1.ComponentTypeLPX,
		GlobalDynamoNamespace: true,
		LPX:                   &v1beta1.LPXConfig{BuildID: "draft-build"},
		Roles:                 []v1beta1.ComponentRoleSpec{{Name: v1beta1.ComponentRoleLPXAgent}},
	}
	serving := v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "serving",
		ComponentType: v1beta1.ComponentTypeLPX,
		LPX:           &v1beta1.LPXConfig{BuildID: "serving-build"},
		Roles: []v1beta1.ComponentRoleSpec{
			{Name: v1beta1.ComponentRoleLPXConductor},
			{Name: v1beta1.ComponentRoleLPXAgent},
		},
	}
	source := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "tenant"},
		Spec:       v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{draft, serving}},
	}

	t.Log("Preserve component namespaces while the PCS or its scaling group is still missing")
	for _, pcs := range []*grovev1alpha1.PodCliqueSet{
		nil,
		{
			ObjectMeta: metav1.ObjectMeta{Generation: 1},
			Status: grovev1alpha1.PodCliqueSetStatus{
				ObservedGeneration: ptr.To(int64(1)), CurrentGenerationHash: ptr.To("accepted"),
			},
		},
	} {
		readiness := EvaluateLPXGroveReadiness(t.Context(), source, "serving", []string{"draft", "serving"}, pcs, nil, nil)
		require.False(t, readiness.Ready)
		require.Equal(t, "dynamo", readiness.ComponentStatuses[draft.ComponentName].RuntimeNamespace)
		require.Equal(t, "tenant-graph", readiness.ComponentStatuses[serving.ComponentName].RuntimeNamespace)
	}
}
