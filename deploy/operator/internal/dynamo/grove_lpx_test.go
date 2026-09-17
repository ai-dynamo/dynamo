// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

	readiness, err := EvaluateLPXGroveReadiness(t.Context(), nil, source, &v1alpha1.LPXGraphDeployment{}, nil)
	require.NoError(t, err)
	require.NotEqual(t, source.GetDynamoNamespaceForComponent(&draft), source.GetDynamoNamespaceForComponent(&serving))
	require.Equal(t, source.GetDynamoNamespaceForComponent(&draft), readiness.ComponentStatuses[draft.ComponentName].RuntimeNamespace)
	require.Equal(t, source.GetDynamoNamespaceForComponent(&serving), readiness.ComponentStatuses[serving.ComponentName].RuntimeNamespace)
}
