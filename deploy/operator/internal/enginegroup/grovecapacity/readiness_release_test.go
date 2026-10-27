/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package grovecapacity

import (
	"slices"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/podcache"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestReadinessReleaseWaitsForKubeletAndAcknowledgesOnlyExpectedUID(t *testing.T) {
	t.Log("prepare two exact allocations and initialize missing retention gates through the compact cache")
	fixture, victim, target := newReadinessReleaseFixture(t)
	require.NoError(t, fixture.adapter.InitializeRetention(t.Context()))
	require.NoError(t, fixture.client.Get(t.Context(), client.ObjectKeyFromObject(victim), victim))
	assert.False(t, retentionClosed(victim))
	projected := podcache.Project(victim.DeepCopy())
	require.True(t, hasRetentionGate(projected))
	require.Len(t, projected.Status.Conditions, 2)

	t.Log("persist the exact expected UIDs, close only the victim gate, and preserve unowned Pod status")
	result, err := fixture.adapter.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.Zero(t, fixture.scale.updates)
	require.NoError(t, fixture.client.Get(t.Context(), client.ObjectKeyFromObject(victim), victim))
	require.True(t, retentionClosed(victim))
	require.True(t, podAvailable(victim), "a status patch cannot stand in for kubelet-observed NotReady")
	require.True(t, slices.ContainsFunc(victim.Status.Conditions, func(condition corev1.PodCondition) bool {
		return condition.Type == corev1.PodScheduled && condition.Status == corev1.ConditionTrue
	}))

	t.Log("replaying an accepted release while PodReady is stale must not scale")
	restarted := *fixture.adapter
	result, err = restarted.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.Zero(t, fixture.scale.updates)
	victim.Status.Conditions = slices.DeleteFunc(victim.Status.Conditions, func(condition corev1.PodCondition) bool {
		return condition.Type == corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate)
	})
	require.NoError(t, fixture.client.Status().Update(t.Context(), victim))
	require.NoError(t, restarted.InitializeRetention(t.Context()))
	require.NoError(t, fixture.client.Get(t.Context(), client.ObjectKeyFromObject(victim), victim))
	require.True(t, retentionClosed(victim), "missing status must not reopen a durably accepted retirement")

	t.Log("once the kubelet reports NotReady, lower the native count exactly once")
	for i := range victim.Status.Conditions {
		if victim.Status.Conditions[i].Type == corev1.PodReady {
			victim.Status.Conditions[i].Status = corev1.ConditionFalse
		}
	}
	require.NoError(t, fixture.client.Status().Update(t.Context(), victim))
	result, err = restarted.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.Equal(t, 1, fixture.scale.updates)
	observation, err := restarted.Observe(t.Context(), "group")
	require.NoError(t, err)
	require.Empty(t, observation.ReleaseFences, "a count write does not prove the victim was removed")

	t.Log("observe removal of the authorized UID before publishing a durable release receipt")
	require.NoError(t, fixture.client.Delete(t.Context(), victim))
	observation, err = restarted.Observe(t.Context(), "group")
	require.NoError(t, err)
	require.Equal(t, target.ReleaseFences, observation.ReleaseFences)
	result, err = restarted.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)
	require.Equal(t, 1, fixture.scale.updates)
}

func TestReadinessReleaseStopsOnAmbiguousOrUnexpectedVictims(t *testing.T) {
	for _, scenario := range []string{"retained-not-ready", "retained-pending", "retained-unassigned", "lost-retained-uid", "replaced-victim-uid", "victim-missing-before-scale"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("accept a UID-pinned release before introducing one unsafe selection condition")
			fixture, victim, target := newReadinessReleaseFixture(t)
			require.NoError(t, fixture.adapter.InitializeRetention(t.Context()))
			result, err := fixture.adapter.Apply(t.Context(), "group", target)
			require.NoError(t, err)
			require.Nil(t, result.Rejection)
			require.NoError(t, fixture.client.Get(t.Context(), client.ObjectKeyFromObject(fixture.primary), fixture.primary))
			switch scenario {
			case "retained-not-ready":
				for i := range fixture.primary.Status.Conditions {
					if fixture.primary.Status.Conditions[i].Type == corev1.PodReady {
						fixture.primary.Status.Conditions[i].Status = corev1.ConditionFalse
					}
				}
				require.NoError(t, fixture.client.Status().Update(t.Context(), fixture.primary))
			case "retained-pending":
				fixture.primary.Status.Phase = corev1.PodPending
				require.NoError(t, fixture.client.Status().Update(t.Context(), fixture.primary))
			case "retained-unassigned":
				fixture.primary.Spec.NodeName = ""
				require.NoError(t, fixture.client.Update(t.Context(), fixture.primary))
			case "lost-retained-uid":
				require.NoError(t, fixture.client.Delete(t.Context(), fixture.primary))
			case "replaced-victim-uid":
				require.NoError(t, fixture.client.Delete(t.Context(), victim))
				replacement := grovePod(fixture.clique, "replacement", "new-uid", "1", true)
				require.NoError(t, fixture.client.Create(t.Context(), replacement))
			case "victim-missing-before-scale":
				require.NoError(t, fixture.client.Delete(t.Context(), victim))
			}

			t.Log("selection ambiguity waits; identity loss requires recovery; neither lowers the count")
			result, err = fixture.adapter.Apply(t.Context(), "group", target)
			if scenario == "lost-retained-uid" || scenario == "replaced-victim-uid" || scenario == "victim-missing-before-scale" {
				require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
			} else {
				require.NoError(t, err)
				require.Nil(t, result.Rejection)
			}
			require.Zero(t, fixture.scale.updates)
		})
	}
}

func TestReadinessReleaseRejectsLossOfRetainedCapacityAfterScale(t *testing.T) {
	t.Log("write a scale-in only after the exact retired victim is observed NotReady")
	fixture, victim, target := newReadinessReleaseFixture(t)
	require.NoError(t, fixture.adapter.InitializeRetention(t.Context()))
	_, err := fixture.adapter.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.NoError(t, fixture.client.Get(t.Context(), client.ObjectKeyFromObject(victim), victim))
	for i := range victim.Status.Conditions {
		if victim.Status.Conditions[i].Type == corev1.PodReady {
			victim.Status.Conditions[i].Status = corev1.ConditionFalse
		}
	}
	require.NoError(t, fixture.client.Status().Update(t.Context(), victim))
	_, err = fixture.adapter.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.Equal(t, 1, fixture.scale.updates)

	t.Log("a different deletion cannot be acknowledged even when the count reaches its target")
	require.NoError(t, fixture.client.Delete(t.Context(), fixture.primary))
	_, err = fixture.adapter.Observe(t.Context(), "group")
	require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
	_, err = fixture.adapter.Apply(t.Context(), "group", target)
	require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
	require.Equal(t, 1, fixture.scale.updates, "never repair unexpected loss by scaling again")
}

func newReadinessReleaseFixture(t *testing.T) (capacityFixture, *corev1.Pod, enginegroup.CapacityTarget) {
	t.Helper()
	fixture := newCapacityFixture(t)
	fixture.adapter.ReadinessScaleIn = true
	fixture.clique.Spec.Replicas = 2
	fixture.clique.Spec.PodSpec.ReadinessGates = []corev1.PodReadinessGate{{ConditionType: corev1.PodConditionType(consts.EngineGroupRetainedReadinessGate)}}
	require.NoError(t, fixture.client.Update(t.Context(), fixture.clique))
	fixture.primary.Spec.ReadinessGates = fixture.clique.Spec.PodSpec.ReadinessGates
	fixture.primary.Spec.NodeName = "node-0"
	require.NoError(t, fixture.client.Update(t.Context(), fixture.primary))
	fixture.primary.Status.Phase = corev1.PodRunning
	require.NoError(t, fixture.client.Status().Update(t.Context(), fixture.primary))
	victim := grovePod(fixture.clique, "victim", "victim-uid", "1", true)
	victim.Spec.NodeName, victim.Status.Phase = "node-1", corev1.PodRunning
	victim.Status.Conditions = append(victim.Status.Conditions, corev1.PodCondition{Type: corev1.PodScheduled, Status: corev1.ConditionTrue})
	require.NoError(t, fixture.client.Create(t.Context(), victim))
	allocation, err := fixture.adapter.allocation(fixture.primary)
	require.NoError(t, err)
	incarnation := physicalAssertion(allocation)
	target := enginegroup.CapacityTarget{
		ControlRevision: 1, TransitionID: "retire", ProfileFingerprint: "profile",
		ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
		Replicas:              []enginegroup.CapacityReplicaTarget{{ReplicaID: incarnation.ReplicaID, SlotID: incarnation.SlotID, Incarnation: &incarnation}},
		ReleaseFences: []enginegroup.ReleaseFence{{ReplicaID: "replica-1", SlotID: "slot-1", TransitionID: "retire", AuthorizingTopologyGeneration: 2,
			CapacityRefs: []enginegroup.CapacityRef{{Name: victim.Name, UID: enginegroup.PodUID(victim.UID)}},
		}},
	}
	return fixture, victim, target
}
