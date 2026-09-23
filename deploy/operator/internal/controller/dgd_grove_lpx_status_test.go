// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestLPXStatusUsesOneCurrentChild(t *testing.T) {
	const stale = "stale"
	for _, scenario := range []string{"ready", "no-download", stale, "retiring", "failure"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Create three independent components under one child")
			source := newLPXHandoffSource(t, "node-local-v2-hybrid")
			other := newLPXHandoffSource(t, "single_v2").Spec.Components[0]
			other.ComponentName = "other"
			third := *other.DeepCopy()
			third.ComponentName = "third"
			source.Spec.Components = append(source.Spec.Components, other, third)
			kube := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(source).Build()
			child, err := (&dgdLPXHandoff{client: kube}).Reconcile(t.Context(), source)
			require.NoError(t, err)
			children := &v1alpha1.LPXGraphDeploymentList{}
			require.NoError(t, kube.List(t.Context(), children))
			require.Len(t, children.Items, 1)
			child.Generation = 2

			t.Log("Record one download result for all builds and independent component replica counts")
			checkedAt := metav1.Unix(1, 0)
			child.Status = v1alpha1.LPXGraphDeploymentStatus{
				ObservedGeneration: 2,
				ModelDownload: &v1beta1.ModelDownloadStatus{
					Builds: []string{"gs://models/first", "gs://models/second"}, LastCheckedAt: &checkedAt,
				},
				Components: map[string]v1beta1.ComponentReplicaStatus{},
				Conditions: []metav1.Condition{{Type: "Ready", Status: metav1.ConditionTrue, ObservedGeneration: 2}},
			}
			for i, component := range source.Spec.Components {
				child.Status.Components[component.ComponentName] = v1beta1.ComponentReplicaStatus{Ready: true, Replicas: int32(i + 1)}
			}

			t.Log("Gate the graph on the shared child's current lifecycle")
			want := v1beta1.DGDStateSuccessful
			switch scenario {
			case "no-download":
				child.Status.ModelDownload = nil
			case stale:
				child.Status.ObservedGeneration--
				want = v1beta1.DGDStatePending
			case "retiring":
				now := metav1.Now()
				child.DeletionTimestamp = &now
				want = v1beta1.DGDStatePending
			case "failure":
				child.Status.Conditions = []metav1.Condition{{Type: "Ready", Status: metav1.ConditionFalse, ObservedGeneration: 2, Reason: v1alpha1.LPXReadyReasonFailed}}
				want = v1beta1.DGDStateFailed
			}
			before := child.DeepCopy()
			result := mergeLPXChildStatus(source, child, ReconcileResult{State: v1beta1.DGDStateSuccessful})
			require.Equal(t, want, result.State)

			t.Log("Publish per-component readiness only for a current child")
			if scenario == "ready" || scenario == "failure" || scenario == "no-download" {
				require.Equal(t, before.Status.Components, result.ComponentStatus)
			} else {
				for _, component := range result.ComponentStatus {
					require.False(t, component.Ready)
					require.Zero(t, component.Replicas)
				}
			}
			require.Equal(t, before, child)
		})
	}
}
