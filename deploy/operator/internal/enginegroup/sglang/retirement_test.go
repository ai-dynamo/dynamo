/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"net/http"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestPackedRetirementRequiresEveryNativeDrainProof(t *testing.T) {
	t.Log("form EP2 as one two-member allocation, without a synthetic resize")
	engine := &nativeGrowthEngine{state: ScaleState{
		InstanceID: "runtime-1", TopologyToken: "topology-1", AllocationWidth: 2, InitialEPSize: 2, EffectiveEPSize: 2,
		MaxCommittedEPSize: 2, RuntimeHealth: "healthy", RetireeLifecycle: "external", SafeToTerminateRanks: []int32{},
	}}
	control, err := NewClient("http://sglang.test", &http.Client{Transport: engine})
	require.NoError(t, err)
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	kube := fake.NewClientBuilder().WithScheme(scheme).Build()
	first, second := testAllocation(0), testAllocation(1)
	first.Incarnation.Members, second.Incarnation.Members = testNativeMembers(0, 2), testNativeMembers(2, 2)
	capacity := &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{first}}
	adapter := &MembershipAdapter{Client: control, Capacity: capacity, MembersPerReplica: 2, InitialEPSize: 2,
		ProfileFingerprint: "profile-v1", Journal: kubejournal.NewStore(kube, "test", "group", "group-uid", "membership")}
	base, err := adapter.Observe(t.Context(), "group-uid", "")
	require.NoError(t, err)

	t.Log("append one packed allocation, preserving the original world and process identities")
	grow := testGrowPlan(1)
	grow.Change.Grow.Replicas[0].NativeMembers = nativeMemberIDs(2, 2)
	capacity.allocations = append(capacity.allocations, second)
	target := validatedTarget(t, adapter, base.CommittedTopology, grow, []enginegroup.JoiningReplica{{
		ReplicaID: "replica-1", Members: second.Incarnation.Members,
	}})
	require.NoError(t, adapter.Apply(t.Context(), "group-uid", target))
	engine.state.Scaling, engine.state.EffectiveEPSize, engine.state.MaxCommittedEPSize = false, 4, 4
	engine.state.OperationSucceeded = new(true)
	grown, err := adapter.Observe(t.Context(), "group-uid", target.TransitionID)
	require.NoError(t, err)
	require.Equal(t, enginegroup.MembershipTransitionPhaseCommitted, grown.Transition.Phase)

	t.Log("persist graceful withdrawal before submitting native suffix retirement")
	retire := testRetirePlan()
	target = validatedTarget(t, adapter, grown.CommittedTopology, retire, nil)
	traffic := &TrafficProjection{Client: control, Capacity: capacity, Membership: adapter,
		Journal: kubejournal.NewStore(kube, "test", "group", "group-uid", "traffic")}
	withdrawal := enginegroup.TrafficTarget{ControlRevision: 2, TransitionID: "retire-packed", TopologyGeneration: 2,
		Drain: []enginegroup.TrafficDrainTarget{{Mode: enginegroup.TrafficDrainModeGraceful,
			Membership: grown.CommittedTopology.Replicas[1]}},
	}
	accepted, err := traffic.Apply(t.Context(), "group-uid", withdrawal)
	require.NoError(t, err)
	require.Nil(t, accepted.Rejection)
	before, err := traffic.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	assert.Empty(t, before.Drained)
	require.NoError(t, adapter.Apply(t.Context(), "group-uid", target))
	require.Len(t, engine.requests, 2)
	assert.Equal(t, int32(2), engine.requests[1].NewEPSize)
	assert.Empty(t, engine.requests[1].ExpectedJoiningAllocationIDs)

	t.Log("neither a pending scale nor a committed smaller size proves terminal drain")
	before, err = traffic.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	assert.Empty(t, before.Drained)
	engine.state.Scaling, engine.state.EffectiveEPSize, engine.state.Phase = false, 2, "serving_shrunk"
	engine.state.OperationSucceeded = new(true)
	shrunk, err := adapter.Observe(t.Context(), "group-uid", target.TransitionID)
	require.NoError(t, err)
	require.Equal(t, enginegroup.MembershipTransitionPhaseCommitted, shrunk.Transition.Phase)
	assert.Equal(t, int64(3), shrunk.CommittedTopology.Generation)
	assert.Equal(t, base.CommittedTopology.Replicas, shrunk.CommittedTopology.Replicas)
	before, err = traffic.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	assert.Empty(t, before.Drained)

	t.Log("one safe member cannot authorize the shared two-member Pod's release")
	engine.state.SafeToTerminateRanks = []int32{2}
	before, err = traffic.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	assert.Empty(t, before.Drained)

	t.Log("persist exact all-member proof, then retain it across adapter restart and physical release")
	engine.state.SafeToTerminateRanks = []int32{2, 3}
	proof, err := traffic.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	require.Equal(t, []enginegroup.ReplicaMembership{grown.CommittedTopology.Replicas[1]}, proof.Drained)
	capacity.allocations = capacity.allocations[:1]
	engine.state.SafeToTerminateRanks = nil
	restarted := *traffic
	after, err := restarted.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	assert.Equal(t, proof, after)

	t.Log("a repeated append into the retired range is rejected before creating capacity")
	regrow := testGrowPlan(1)
	regrow.Change.Grow.Replicas[0].NativeMembers = nativeMemberIDs(2, 2)
	resolution, err := adapter.ValidatePlan(t.Context(), "group-uid", enginegroup.PlanValidationRequest{
		BaseTopology: shrunk.CommittedTopology, Plan: regrow, PlanDigest: "regrow-digest",
	})
	require.NoError(t, err)
	require.NotNil(t, resolution.Rejection)
	assert.Equal(t, "SlotRecoveryUnsupported", resolution.Rejection.Reason)

	t.Log("missing native slot history is inconclusive, never proof that retired slots are fresh")
	engine.state.MaxCommittedEPSize = 0
	resolution, err = adapter.ValidatePlan(t.Context(), "group-uid", enginegroup.PlanValidationRequest{
		BaseTopology: shrunk.CommittedTopology, Plan: regrow, PlanDigest: "regrow-digest",
	})
	require.ErrorContains(t, err, "maximum committed EP size")
	assert.Nil(t, resolution.Rejection)
	engine.state.MaxCommittedEPSize = 4

	t.Log("readmitting a terminally drained incarnation never leaves its old drain proof usable")
	capacity.allocations = append(capacity.allocations, second)
	engine.state.EffectiveEPSize = 4
	_, err = restarted.Observe(t.Context(), "group-uid")
	require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
}

func TestRetirementPreflightRequiresExternalNativeRelease(t *testing.T) {
	for _, tc := range []struct {
		name      string
		lifecycle string
		ranks     []int32
	}{
		{name: "growth-only status", lifecycle: "", ranks: nil},
		{name: "retirees exit independently", lifecycle: "self_exit", ranks: []int32{}},
		{name: "missing native release evidence", lifecycle: "external", ranks: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("reject an incompatible native retirement contract before any capacity or traffic effect")
			engine := &nativeGrowthEngine{state: ScaleState{EffectiveEPSize: 4, RuntimeHealth: "healthy",
				RetireeLifecycle: tc.lifecycle, SafeToTerminateRanks: tc.ranks}}
			control, err := NewClient("http://sglang.test", &http.Client{Transport: engine})
			require.NoError(t, err)
			adapter := &MembershipAdapter{Client: control, ProfileFingerprint: "profile-v1", MembersPerReplica: 2, InitialEPSize: 2}
			base := enginegroup.MembershipTopology{Generation: 1, Replicas: []enginegroup.ReplicaMembership{
				{ReplicaID: "replica-0", Members: testNativeMembers(0, 2)},
				{ReplicaID: "replica-1", Members: testNativeMembers(2, 2)},
			}}
			result, err := adapter.ValidatePlan(t.Context(), "group-uid", enginegroup.PlanValidationRequest{BaseTopology: base, Plan: testRetirePlan(), PlanDigest: "digest"})
			require.NoError(t, err)
			require.NotNil(t, result.Rejection)
			assert.Equal(t, "NativeRetirementUnsupported", result.Rejection.Reason)
			assert.Empty(t, engine.requests)
		})
	}
}

func validatedTarget(t *testing.T, adapter *MembershipAdapter, base enginegroup.MembershipTopology, plan enginegroup.ResolvedPlan, joining []enginegroup.JoiningReplica) enginegroup.MembershipTarget {
	t.Helper()
	preflight, err := adapter.ValidatePlan(t.Context(), "group-uid", enginegroup.PlanValidationRequest{BaseTopology: base, Plan: plan, PlanDigest: plan.ID + "-plan"})
	require.NoError(t, err)
	require.Nil(t, preflight.Rejection)
	require.NotNil(t, preflight.Evidence)
	target := enginegroup.MembershipTarget{ControlRevision: base.Generation, TransitionID: plan.ID, TargetDigest: plan.ID + "-target",
		Validation: *preflight.Evidence, BaseTopology: base, Plan: plan, Joining: joining}
	preflight, err = adapter.ValidateTarget(t.Context(), "group-uid", target)
	require.NoError(t, err)
	require.Nil(t, preflight.Rejection)
	require.NotNil(t, preflight.Evidence)
	target.Validation = *preflight.Evidence
	return target
}

func testRetirePlan() enginegroup.ResolvedPlan {
	return enginegroup.ResolvedPlan{ID: "retire-packed", ProfileFingerprint: "profile-v1",
		ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:    enginegroup.TrafficRequirementEngineManagedDrain, VerificationRequirement: enginegroup.VerificationRequirementRequired,
		Change: enginegroup.ResolvedChange{Kind: enginegroup.PlanKindRetire, Retire: &enginegroup.RetireChange{Replicas: []enginegroup.ReplicaID{"replica-1"}}},
	}
}
