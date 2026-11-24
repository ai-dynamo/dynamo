// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/stretchr/testify/require"
)

func TestRetuneStatusAcknowledgesSemanticTarget(t *testing.T) {
	for _, test := range []struct {
		name                 string
		enabled              bool
		observed             int64
		targetsAcknowledged  bool
		missingPlannerStatus bool
		published            bool
	}{
		{name: "acknowledged targets before readiness or optional projections", enabled: true, observed: 7, targetsAcknowledged: true, published: true},
		{name: "child lowering pending", enabled: true, observed: 7},
		{name: "missing Planner observation", enabled: true, observed: 7, targetsAcknowledged: true, missingPlannerStatus: true},
		{name: "stale graph acknowledgment", enabled: true, observed: 6, targetsAcknowledged: true},
		{name: "disabled clears former capability", observed: 7, targetsAcknowledged: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Publish only the acknowledged semantic contract hash, without workload projections")
			dgd := &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns", Generation: 7, Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend},
				{ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner,
					Retune: &v1beta1.RetuneSpec{Enabled: test.enabled, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "traces"}}},
			}}}
			original := dgd.DeepCopy()
			result := newWorkloadProgramResult(dgd)
			result.applyReconcileResult(test.observed, ReconcileResult{State: v1beta1.DGDStatePending, ComponentStatus: map[string]v1beta1.ComponentReplicaStatus{
				"gateway": {}, "optimizer": {Retune: &v1beta1.RetuneStatus{ConfigHash: "old-target"}},
			}})
			if test.missingPlannerStatus {
				delete(result.Status.Components, "optimizer")
			}
			result.applyRetuneStatus(dgd, test.targetsAcknowledged)
			if test.published {
				_, configuration, err := dynamo.RetuneConfiguration(dgd)
				require.NoError(t, err)
				require.Equal(t, &v1beta1.RetuneStatus{ConfigHash: configuration.ConfigHash}, result.Status.Components["optimizer"].Retune)
			} else {
				require.Nil(t, result.Status.Components["optimizer"].Retune)
			}
			require.Nil(t, result.Status.Components["gateway"].Retune)
			require.Equal(t, original, dgd)
		})
	}
}

func TestRetuneDCDTargetAcknowledgment(t *testing.T) {
	for _, test := range []struct {
		name             string
		missing          bool
		frontendObserved int64
		plannerObserved  int64
		want             bool
	}{
		{name: "both lowerings observed without ready Pods", frontendObserved: 3, plannerObserved: 3, want: true},
		{name: "missing frontend", missing: true, frontendObserved: 3, plannerObserved: 3},
		{name: "frontend child pending", frontendObserved: 2, plannerObserved: 3},
		{name: "Planner child pending", frontendObserved: 3, plannerObserved: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Observe the existing child generation boundary, not health or PodSpec snapshots")
			dgd := &v1beta1.DynamoGraphDeployment{Spec: v1beta1.DynamoGraphDeploymentSpec{Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend},
				{ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner,
					Retune: &v1beta1.RetuneSpec{Enabled: true, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "traces"}}},
			}}}
			children := []*v1beta1.DynamoComponentDeployment{{Generation: 3, Spec: v1beta1.DynamoComponentDeploymentSpec{
				DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "optimizer"},
			}, Status: v1beta1.DynamoComponentDeploymentStatus{ObservedGeneration: test.plannerObserved, Component: &v1beta1.ComponentReplicaStatus{ComponentKind: v1beta1.ComponentKindDeployment}}}}
			if !test.missing {
				children = append(children, &v1beta1.DynamoComponentDeployment{Generation: 3, Spec: v1beta1.DynamoComponentDeploymentSpec{
					DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "gateway"},
				}, Status: v1beta1.DynamoComponentDeploymentStatus{ObservedGeneration: test.frontendObserved, Component: &v1beta1.ComponentReplicaStatus{ComponentKind: v1beta1.ComponentKindDeployment}}})
			}
			require.Equal(t, test.want, retuneDCDTargetsAcknowledged(dgd, children))
		})
	}
}
