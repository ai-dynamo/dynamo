/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func newParallelRestartDGD(
	restartID string,
	status *nvidiacomv1beta1.RestartStatus,
	components ...nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		Name: "parallel-restart", Namespace: "default", UID: "parallel-restart-uid",
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: components,
			Restart: &nvidiacomv1beta1.Restart{
				ID:       restartID,
				Strategy: &nvidiacomv1beta1.RestartStrategy{Type: nvidiacomv1beta1.RestartStrategyTypeParallel},
			},
		},
		Status: nvidiacomv1beta1.DynamoGraphDeploymentStatus{Restart: status},
	}
}

func namedComponents(names ...string) []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	components := make([]nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, 0, len(names))
	for _, name := range names {
		components = append(components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: name})
	}
	return components
}

// dsParallelRestartComponents returns two DisaggregatedSet-eligible worker
// components, listed in unsorted order.
func dsParallelRestartComponents() []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	return []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		{
			ComponentName: "prefill",
			ComponentType: nvidiacomv1beta1.ComponentTypePrefill,
			Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
		},
		{
			ComponentName: "decode",
			ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
			Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
		},
	}
}

func TestParallelRestartDisaggregatedSetEntryPath(t *testing.T) {
	t.Log("An ongoing restart must remain pending while its DisaggregatedSet is missing")
	dgd := newParallelRestartDGD("restart-2", &nvidiacomv1beta1.RestartStatus{
		ObservedID: "restart-2",
		Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
		InProgress: []string{"prefill"},
	}, dsParallelRestartComponents()...)
	scheme := runtime.NewScheme()
	require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
	scheme.AddKnownTypeWithName(disaggregatedSetGVK, &unstructured.Unstructured{})
	scheme.AddKnownTypeWithName(disaggregatedSetGVK.GroupVersion().WithKind("DisaggregatedSetList"), &unstructured.UnstructuredList{})
	k8sClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	r := &disaggregatedSetWorkloadsReconciler{
		reader: k8sClient,
		restartProgress: newDisaggregatedSetRestartProgressResolver(
			k8sClient,
			newDisaggregatedSetReadinessResolver(k8sClient),
			newComponentRestartProgressResolver(k8sClient),
		),
	}

	t.Log("Resolve progress through the DisaggregatedSet entry path")
	got := r.computeRestartStatus(context.Background(), dgd)
	require.Equal(t, &nvidiacomv1beta1.RestartStatus{
		ObservedID: "restart-2",
		Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
		InProgress: []string{"prefill"},
	}, got)
}
