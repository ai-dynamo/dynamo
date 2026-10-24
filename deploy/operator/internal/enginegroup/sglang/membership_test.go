/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestGrowthAdapterNativeCorrelation(t *testing.T) {
	for _, width := range []int32{1, 4} {
		t.Run(fmt.Sprintf("members per allocation %d", width), func(t *testing.T) {
			t.Log("observe a formed native world with a stable runtime ID and allocation width")
			engine := &nativeGrowthEngine{state: ScaleState{InstanceID: "runtime-1", AllocationWidth: width,
				InitialEPSize: 2 * width, EffectiveEPSize: 2 * width, RuntimeHealth: "healthy"}}
			capacity := &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{testAllocation(0), testAllocation(1)}}
			for index := range capacity.allocations {
				capacity.allocations[index].Incarnation.Members = testNativeMembers(int32(index)*width, width)
			}
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			kube := fake.NewClientBuilder().WithScheme(scheme).Build()
			control, err := NewClient("http://sglang.test", &http.Client{Transport: engine})
			require.NoError(t, err)
			adapter := &GrowthAdapter{Client: control, Capacity: capacity, MembersPerReplica: width, ProfileFingerprint: "profile-v1",
				Journal: kubejournal.NewStore(kube, "test", "group", "group-uid", "membership")}
			initial, err := adapter.Observe(t.Context(), "group-uid", "")
			require.NoError(t, err)

			t.Log("validate and freeze one complete joining allocation before dispatch")
			plan := testGrowPlan(2)
			joining := testAllocation(2)
			joining.Incarnation.Members = testNativeMembers(2*width, width)
			plan.Change.Grow.Replicas[0].NativeMembers = nil
			for _, member := range joining.Incarnation.Members {
				plan.Change.Grow.Replicas[0].NativeMembers = append(plan.Change.Grow.Replicas[0].NativeMembers, member.ID)
			}
			preflight, err := adapter.ValidatePlan(t.Context(), "group-uid", enginegroup.PlanValidationRequest{
				BaseTopology: initial.CommittedTopology, Plan: plan, PlanDigest: "plan-digest",
			})
			require.NoError(t, err)
			require.NotNil(t, preflight.Evidence)
			capacity.allocations = append(capacity.allocations, joining)
			target := enginegroup.MembershipTarget{ControlRevision: 1, TransitionID: plan.ID, TargetDigest: "target-digest",
				Validation: *preflight.Evidence, BaseTopology: initial.CommittedTopology, Plan: plan,
				Joining: []enginegroup.JoiningReplica{{ReplicaID: "replica-2", Members: joining.Incarnation.Members}}}
			validated, err := adapter.ValidateTarget(t.Context(), "group-uid", target)
			require.NoError(t, err)
			require.NotNil(t, validated.Evidence)
			target.Validation = *validated.Evidence

			t.Log("a transport timeout retains an ambiguous transition and the exact replay payload")
			engine.failSubmission = true
			require.ErrorContains(t, adapter.Apply(t.Context(), "group-uid", target), "transport timeout")
			unknown, err := adapter.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)
			require.Equal(t, enginegroup.MembershipTransitionPhaseUnknown, unknown.Transition.Phase)

			t.Log("a restarted adapter replays the same operation ID, runtime ID, target, and joining Pod UID")
			restarted := *adapter
			engine.failSubmission = false
			require.NoError(t, restarted.Apply(t.Context(), "group-uid", target))
			require.Len(t, engine.requests, 2)
			assert.Equal(t, engine.requests[0], engine.requests[1])
			assert.Equal(t, ScaleRequest{NewEPSize: 3 * width, OperationID: target.TransitionID, ExpectedInstanceID: "runtime-1",
				ExpectedJoiningAllocationIDs: []string{"pod-2"}}, engine.requests[0])
			pending, err := restarted.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)
			assert.Equal(t, enginegroup.MembershipTransitionPhasePending, pending.Transition.Phase)
			assert.Nil(t, pending.Transition.Failure)

			t.Log("commit requires the native terminal result, not merely the requested EP size")
			engine.state.Scaling = false
			engine.state.EffectiveEPSize = 3 * width
			succeeded := true
			engine.state.OperationSucceeded = &succeeded
			committed, err := restarted.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)
			require.Equal(t, enginegroup.MembershipTransitionPhaseCommitted, committed.Transition.Phase)
			assert.Equal(t, int64(2), committed.CommittedTopology.Generation)
			assert.Equal(t, committed.CommittedTopology, *committed.Transition.ResultTopology)
			assert.Len(t, committed.CommittedTopology.Replicas, 3)

			t.Log("terminal correlation and result survive controller outages without re-dispatch")
			require.NoError(t, restarted.Apply(t.Context(), "group-uid", target))
			assert.Len(t, engine.requests, 2)
			reobserved, err := restarted.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)
			assert.Equal(t, committed, reobserved)

			t.Log("a later runtime failure blocks reconciliation without rewriting the committed resize")
			beforeFailure := membershipJournal{}
			_, err = adapter.Journal.Load(t.Context(), &beforeFailure)
			require.NoError(t, err)
			engine.state.RuntimeHealth = "recovery_unsupported"
			engine.state.RuntimeError = "append rank recovery is unsupported"
			failed, err := restarted.Observe(t.Context(), "group-uid", target.TransitionID)
			require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
			require.ErrorContains(t, err, engine.state.RuntimeError)
			assert.Empty(t, failed.CommittedTopology.Replicas)
			require.ErrorIs(t, restarted.Apply(t.Context(), "group-uid", target), enginegroup.ErrRecoveryRequired)
			assert.Len(t, engine.requests, 2)
			afterFailure := membershipJournal{}
			_, err = adapter.Journal.Load(t.Context(), &afterFailure)
			require.NoError(t, err)
			assert.Equal(t, beforeFailure, afterFailure)
			assert.True(t, *engine.state.OperationSucceeded)

			t.Log("traffic projection also refuses the failed runtime's historical effective size")
			traffic := &TrafficProjection{Client: control, Capacity: capacity,
				Journal: kubejournal.NewStore(kube, "test", "group", "group-uid", "traffic")}
			routing, err := traffic.Observe(t.Context(), "group-uid")
			require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
			assert.Empty(t, routing.Admitted)

			t.Log("a replaced native runtime cannot inherit the old world's journal")
			engine.state.RuntimeHealth = "healthy"
			engine.state.RuntimeError = ""
			engine.state.InstanceID = "runtime-2"
			_, err = restarted.Observe(t.Context(), "group-uid", target.TransitionID)
			require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
		})
	}
}

func TestGrowthAdapterRejectsUnsupportedPlan(t *testing.T) {
	for _, kind := range []enginegroup.PlanKind{enginegroup.PlanKindRetire, enginegroup.PlanKindReduceToSurvivors, enginegroup.PlanKindRestore} {
		t.Run(string(kind), func(t *testing.T) {
			t.Log("reject unsupported operation shapes before allocating or draining capacity")
			plan := testGrowPlan(2)
			plan.Change.Kind = kind
			result, err := (&GrowthAdapter{ProfileFingerprint: "profile-v1"}).ValidatePlan(t.Context(), "group", enginegroup.PlanValidationRequest{Plan: plan})
			require.NoError(t, err)
			require.NotNil(t, result.Rejection)
			assert.Equal(t, "UnsupportedOperation", result.Rejection.Reason)
		})
	}
}

func TestGrowthAdapterStaleReplayCannotOverwriteNewerControl(t *testing.T) {
	t.Log("persist one native operation and retain a stale cached copy of its journal")
	engine := &nativeGrowthEngine{state: ScaleState{InstanceID: "runtime-1", AllocationWidth: 1,
		InitialEPSize: 2, EffectiveEPSize: 2, RuntimeHealth: "healthy"}}
	control, err := NewClient("http://sglang.test", &http.Client{Transport: engine})
	require.NoError(t, err)
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	kube := fake.NewClientBuilder().WithScheme(scheme).Build()
	capacity := &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{testAllocation(0), testAllocation(1), testAllocation(2)}}
	adapter := &GrowthAdapter{Client: control, Capacity: capacity, MembersPerReplica: 1, ProfileFingerprint: "profile-v1",
		Journal: kubejournal.NewStore(kube, "test", "group", "group-uid", "membership")}
	base, err := adapter.Observe(t.Context(), "group-uid", "")
	require.NoError(t, err)
	target := enginegroup.MembershipTarget{ControlRevision: 1, TransitionID: "grow", TargetDigest: "digest", BaseTopology: base.CommittedTopology,
		Plan: testGrowPlan(2), Joining: []enginegroup.JoiningReplica{{ReplicaID: "replica-2", Members: testAllocation(2).Incarnation.Members}}}
	require.NoError(t, adapter.Apply(t.Context(), "group-uid", target))
	stale := &corev1.ConfigMap{}
	require.NoError(t, kube.Get(t.Context(), client.ObjectKey{Namespace: adapter.Journal.Namespace, Name: adapter.Journal.Name}, stale))

	t.Log("a newer reconcile updates the durable journal while the old reader still sees its prior snapshot")
	state := membershipJournal{}
	snapshot, err := adapter.Journal.Load(t.Context(), &state)
	require.NoError(t, err)
	state.Transition.ControlRevision++
	_, err = adapter.Journal.Save(t.Context(), snapshot, state)
	require.NoError(t, err)
	adapter.Journal.Client = &staleMembershipReader{Client: kube, cached: stale}

	t.Log("compare-and-swap rejects the stale replay before a second native request")
	err = adapter.Apply(t.Context(), "group-uid", target)
	require.True(t, apierrors.IsConflict(err), "%v", err)
	require.Len(t, engine.requests, 1)
}

type staleMembershipReader struct {
	client.Client
	cached *corev1.ConfigMap
}

func (c *staleMembershipReader) Get(_ context.Context, _ client.ObjectKey, out client.Object, _ ...client.GetOption) error {
	*out.(*corev1.ConfigMap) = *c.cached.DeepCopy()
	return nil
}

type nativeGrowthEngine struct {
	state          ScaleState
	requests       []ScaleRequest
	failSubmission bool
}

func (e *nativeGrowthEngine) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.Method == http.MethodGet {
		encoded, err := json.Marshal(e.state)
		if err != nil {
			return nil, err
		}
		return jsonResponse(string(encoded)), nil
	}
	var target ScaleRequest
	if err := json.NewDecoder(request.Body).Decode(&target); err != nil {
		return nil, err
	}
	e.requests = append(e.requests, target)
	e.state.OperationID = target.OperationID
	e.state.TargetEPSize = target.NewEPSize
	e.state.JoiningAllocationIDs = target.ExpectedJoiningAllocationIDs
	if e.failSubmission {
		e.state.Phase = "submission_unknown"
		return nil, fmt.Errorf("transport timeout")
	}
	e.state.Scaling = true
	e.state.Phase = "joining"
	return jsonResponse(`{"status":"ok"}`), nil
}

type testCapacityObserver struct {
	allocations []enginegroup.CapacityAllocation
}

func (o *testCapacityObserver) Observe(context.Context, enginegroup.GroupID) (enginegroup.CapacityObservation, error) {
	return enginegroup.CapacityObservation{Allocations: append([]enginegroup.CapacityAllocation(nil), o.allocations...)}, nil
}

func testNativeMembers(start, width int32) []enginegroup.NativeMemberIncarnation {
	members := make([]enginegroup.NativeMemberIncarnation, width)
	for local := range members {
		members[local] = enginegroup.NativeMemberIncarnation{ID: enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", start+int32(local))),
			RuntimeIncarnation: enginegroup.RuntimeIncarnationID(fmt.Sprintf("process-%d", start+int32(local)))}
	}
	return members
}

func testAllocation(rank int) enginegroup.CapacityAllocation {
	uid := fmt.Sprintf("pod-%d", rank)
	return enginegroup.CapacityAllocation{Available: true, Incarnation: enginegroup.ReplicaIncarnation{
		ReplicaID: enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank)), SlotID: enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", rank)),
		Members:      []enginegroup.NativeMemberIncarnation{{ID: enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", rank)), RuntimeIncarnation: enginegroup.RuntimeIncarnationID(uid)}},
		CapacityRefs: []enginegroup.CapacityRef{{Name: uid, UID: enginegroup.PodUID(uid)}},
	}}
}

func testGrowPlan(rank int) enginegroup.ResolvedPlan {
	return enginegroup.ResolvedPlan{ID: fmt.Sprintf("grow-%d-%d", rank, rank+1), ProfileFingerprint: "profile-v1", ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement: enginegroup.TrafficRequirementKeepServing, VerificationRequirement: enginegroup.VerificationRequirementRequired,
		Change: enginegroup.ResolvedChange{Kind: enginegroup.PlanKindGrow, Grow: &enginegroup.GrowChange{Replicas: []enginegroup.ReplicaTarget{{
			ReplicaID: enginegroup.ReplicaID(fmt.Sprintf("replica-%d", rank)), SlotID: enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", rank)),
			Bootstrap: enginegroup.BootstrapModeJoin, NativeMembers: []enginegroup.NativeMemberID{enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", rank))},
		}}}},
	}
}
