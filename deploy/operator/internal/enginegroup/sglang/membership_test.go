/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"slices"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestMembershipAdapterNativeCorrelation(t *testing.T) {
	for _, width := range []int32{1, 4} {
		t.Run(fmt.Sprintf("members per allocation %d", width), func(t *testing.T) {
			t.Log("observe a formed native world with a stable runtime ID and allocation width")
			engine := &nativeGrowthEngine{state: ScaleState{InstanceID: "runtime-1", TopologyToken: "topology-1", AllocationWidth: width,
				InitialEPSize: 2 * width, EffectiveEPSize: 2 * width, MaxCommittedEPSize: 2 * width, RuntimeHealth: "healthy"}}
			capacity := &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{testAllocation(0), testAllocation(1)}}
			for index := range capacity.allocations {
				capacity.allocations[index].Incarnation.Members = testNativeMembers(int32(index)*width, width)
			}
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			kube := fake.NewClientBuilder().WithScheme(scheme).Build()
			control, err := NewClient("http://sglang.test", &http.Client{Transport: engine})
			require.NoError(t, err)
			adapter := &MembershipAdapter{Client: control, Capacity: capacity, MembersPerReplica: width, ProfileFingerprint: "profile-v1",
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
			t.Log("use canonical string ordering, including dp-10 before dp-8, without changing physical observations")
			slices.Sort(plan.Change.Grow.Replicas[0].NativeMembers)
			t.Log("refuse unsettled, uncorrelated or stale native baselines before creating joining capacity")
			settled := engine.state
			for _, changed := range []struct {
				name, token, operation string
				scaling                bool
			}{
				{name: "active collective", token: settled.TopologyToken, scaling: true},
				{name: "same size newer topology", token: "later-token"},
				{name: "unrelated operation", token: settled.TopologyToken, operation: "another-writer"},
			} {
				engine.state = settled
				engine.state.Scaling, engine.state.TopologyToken, engine.state.OperationID = changed.scaling, changed.token, changed.operation
				refused, err := adapter.ValidatePlan(t.Context(), "group-uid", enginegroup.PlanValidationRequest{
					BaseTopology: initial.CommittedTopology, Plan: plan, PlanDigest: "plan-digest",
				})
				require.Error(t, err, changed.name)
				assert.Nil(t, refused.Evidence)
				assert.Empty(t, engine.requests)
				assert.Len(t, capacity.allocations, 2)
			}
			engine.state = settled

			t.Log("preflight the unchanged baseline and then create the one joining allocation")
			preflight, err := adapter.ValidatePlan(t.Context(), "group-uid", enginegroup.PlanValidationRequest{
				BaseTopology: initial.CommittedTopology, Plan: plan, PlanDigest: "plan-digest",
			})
			require.NoError(t, err)
			require.NotNil(t, preflight.Evidence)
			capacity.allocations = append(capacity.allocations, joining)
			target := enginegroup.MembershipTarget{ControlRevision: 1, TransitionID: plan.ID, TargetDigest: "target-digest",
				Validation: *preflight.Evidence, BaseTopology: initial.CommittedTopology, Plan: plan,
				Joining: []enginegroup.JoiningReplica{{ReplicaID: "replica-2", Members: joining.Incarnation.Members}}}
			target.Joining[0].Members = slices.Clone(target.Joining[0].Members)
			slices.SortFunc(target.Joining[0].Members, func(left, right enginegroup.NativeMemberIncarnation) int {
				return cmp.Compare(left.ID, right.ID)
			})
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
			assert.Equal(t, ScaleRequest{NewEPSize: 3 * width, OperationID: target.TransitionID, ExpectedInstanceID: "runtime-1", ExpectedTopologyToken: "topology-1",
				ExpectedJoiningAllocationIDs: []string{"pod-2"}}, engine.requests[0])
			pending, err := restarted.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)
			assert.Equal(t, enginegroup.MembershipTransitionPhasePending, pending.Transition.Phase)
			assert.Nil(t, pending.Transition.Failure)

			t.Log("commit requires the native terminal result, not merely the requested EP size")
			engine.state.Scaling = false
			engine.state.EffectiveEPSize = 3 * width
			for _, succeeded := range []*bool{nil, new(false)} {
				engine.state.OperationSucceeded = succeeded
				partial, observeErr := restarted.Observe(t.Context(), "group-uid", target.TransitionID)
				require.ErrorContains(t, observeErr, "without a correlated successful commit")
				assert.Empty(t, partial.CommittedTopology.Replicas)
			}
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

func TestMembershipAdapterRejectsUnsupportedPlan(t *testing.T) {
	for _, kind := range []enginegroup.PlanKind{enginegroup.PlanKindRetire, enginegroup.PlanKindReduceToSurvivors, enginegroup.PlanKindRestore} {
		t.Run(string(kind), func(t *testing.T) {
			t.Log("reject unsupported operation shapes before allocating or draining capacity")
			plan := testGrowPlan(2)
			plan.Change.Kind = kind
			result, err := (&MembershipAdapter{ProfileFingerprint: "profile-v1"}).ValidatePlan(t.Context(), "group", enginegroup.PlanValidationRequest{Plan: plan})
			require.NoError(t, err)
			require.NotNil(t, result.Rejection)
			assert.Equal(t, "UnsupportedOperation", result.Rejection.Reason)
		})
	}
}

func TestMembershipAdapterRejectsPartialAllocationBeforeNativePreflight(t *testing.T) {
	t.Log("describe one surviving member of a fixed two-member allocation")
	base := enginegroup.MembershipTopology{Generation: 1, Replicas: []enginegroup.ReplicaMembership{{
		ReplicaID: "replica-0", Members: testNativeMembers(0, 1),
	}}}
	adapter := &MembershipAdapter{MembersPerReplica: 2, ProfileFingerprint: "profile-v1"}

	t.Log("reject healthy growth without contacting an engine, creating capacity, or withdrawing traffic")
	result, err := adapter.ValidatePlan(t.Context(), "group", enginegroup.PlanValidationRequest{
		BaseTopology: base, Plan: testGrowPlan(1), PlanDigest: "plan-digest",
	})
	require.NoError(t, err)
	require.NotNil(t, result.Rejection)
	assert.Equal(t, "PartialMembershipUnsupported", result.Rejection.Reason)
	assert.Nil(t, result.Evidence)
}

func TestMembershipAdapterStaleReplayCannotOverwriteNewerControl(t *testing.T) {
	t.Log("persist one native operation and retain a stale cached copy of its journal")
	engine := &nativeGrowthEngine{state: ScaleState{InstanceID: "runtime-1", TopologyToken: "topology-1", AllocationWidth: 1,
		InitialEPSize: 2, EffectiveEPSize: 2, MaxCommittedEPSize: 2, RuntimeHealth: "healthy"}}
	control, err := NewClient("http://sglang.test", &http.Client{Transport: engine})
	require.NoError(t, err)
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	kube := fake.NewClientBuilder().WithScheme(scheme).Build()
	capacity := &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{testAllocation(0), testAllocation(1), testAllocation(2)}}
	adapter := &MembershipAdapter{Client: control, Capacity: capacity, MembersPerReplica: 1, ProfileFingerprint: "profile-v1",
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
	if c.cached == nil {
		return apierrors.NewNotFound(schema.GroupResource{Resource: "configmaps"}, "membership")
	}
	*out.(*corev1.ConfigMap) = *c.cached.DeepCopy()
	return nil
}

func TestNativeAcceptanceAheadOfCachedMembershipJournal(t *testing.T) {
	for _, missing := range []bool{false, true} {
		t.Run(fmt.Sprintf("cached absence=%t", missing), func(t *testing.T) {
			t.Log("retain a cached baseline before the engine accepts the recorded growth")
			engine := &nativeGrowthEngine{state: ScaleState{InstanceID: "runtime-1", TopologyToken: "topology-1", AllocationWidth: 1,
				InitialEPSize: 2, EffectiveEPSize: 2, MaxCommittedEPSize: 2, RuntimeHealth: "healthy"}}
			control, err := NewClient("http://sglang.test", &http.Client{Transport: engine})
			require.NoError(t, err)
			scheme := runtime.NewScheme()
			require.NoError(t, corev1.AddToScheme(scheme))
			kube := fake.NewClientBuilder().WithScheme(scheme).Build()
			capacity := &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{testAllocation(0), testAllocation(1), testAllocation(2)}}
			adapter := &MembershipAdapter{Client: control, Capacity: capacity, MembersPerReplica: 1, ProfileFingerprint: "profile-v1",
				Journal: kubejournal.NewStore(kube, "test", "group", "group-uid", "membership")}
			base, err := adapter.Observe(t.Context(), "group-uid", "")
			require.NoError(t, err)
			var cached *corev1.ConfigMap
			if !missing {
				cached = &corev1.ConfigMap{}
				require.NoError(t, kube.Get(t.Context(), client.ObjectKey{Namespace: adapter.Journal.Namespace, Name: adapter.Journal.Name}, cached))
			}
			target := enginegroup.MembershipTarget{ControlRevision: 1, TransitionID: "grow", TargetDigest: "digest", BaseTopology: base.CommittedTopology,
				Plan: testGrowPlan(2), Joining: []enginegroup.JoiningReplica{{ReplicaID: "replica-2", Members: testAllocation(2).Incarnation.Members}}}
			require.NoError(t, adapter.Apply(t.Context(), "group-uid", target))
			adapter.Journal.Client = &staleMembershipReader{Client: kube, cached: cached}

			t.Log("do not claim authoritative absence or permanent recovery from cache lag")
			observed, err := adapter.Observe(t.Context(), "group-uid", target.TransitionID)
			require.ErrorIs(t, err, enginegroup.ErrObservationPending)
			assert.NotErrorIs(t, err, enginegroup.ErrRecoveryRequired)
			assert.Equal(t, enginegroup.MembershipObservation{}, observed)
			assert.Len(t, engine.requests, 1)

			t.Log("converge after the watch catches up, without issuing another native resize")
			adapter.Journal.Client = kube
			engine.state.Scaling = false
			engine.state.EffectiveEPSize = 3
			engine.state.TopologyToken = "committed-grow"
			engine.state.OperationSucceeded = new(true)
			observed, err = adapter.Observe(t.Context(), "group-uid", target.TransitionID)
			require.NoError(t, err)
			require.NotNil(t, observed.Transition)
			assert.Equal(t, enginegroup.MembershipTransitionPhaseCommitted, observed.Transition.Phase)
			assert.Equal(t, int64(2), observed.CommittedTopology.Generation)
			assert.Len(t, engine.requests, 1)
		})
	}
}

func TestUnexplainedSGLangTokenDriftRequiresRecovery(t *testing.T) {
	t.Log("bind a settled world and retain its unchanged size and native identity")
	engine := &nativeGrowthEngine{state: ScaleState{InstanceID: "runtime-1", TopologyToken: "topology-1", AllocationWidth: 1,
		InitialEPSize: 2, EffectiveEPSize: 2, MaxCommittedEPSize: 2, RuntimeHealth: "healthy"}}
	control, err := NewClient("http://sglang.test", &http.Client{Transport: engine})
	require.NoError(t, err)
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	kube := fake.NewClientBuilder().WithScheme(scheme).Build()
	capacity := &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{testAllocation(0), testAllocation(1)}}
	adapter := &MembershipAdapter{Client: control, Capacity: capacity, MembersPerReplica: 1, ProfileFingerprint: "profile-v1",
		Journal: kubejournal.NewStore(kube, "test", "group", "group-uid", "membership")}
	base, err := adapter.Observe(t.Context(), "group-uid", "")
	require.NoError(t, err)

	t.Log("a new unexplained token is a confirmed fault even when member count is unchanged")
	engine.state.TopologyToken = "unrelated-epoch"
	_, err = adapter.Observe(t.Context(), "group-uid", "")
	require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
	_, err = adapter.ValidatePlan(t.Context(), "group-uid", enginegroup.PlanValidationRequest{BaseTopology: base.CommittedTopology,
		Plan: testGrowPlan(2), PlanDigest: "digest"})
	require.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
	assert.Empty(t, engine.requests)

	t.Log("do not silently adopt the new token or manufacture a rejection")
	state := membershipJournal{}
	_, err = adapter.Journal.Load(t.Context(), &state)
	require.NoError(t, err)
	assert.Equal(t, "topology-1", state.TopologyToken)
	assert.Nil(t, state.Transition)
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
	if e.state.OperationID == target.OperationID {
		if len(e.requests) == 0 || !reflect.DeepEqual(e.requests[len(e.requests)-1], target) {
			return jsonResponse(`{"status":"error","message":"conflicting replay"}`), nil
		}
	} else if target.ExpectedTopologyToken != e.state.TopologyToken {
		return jsonResponse(`{"status":"error","message":"stale base"}`), nil
	}
	e.requests = append(e.requests, target)
	e.state.TopologyToken = "accepted-" + target.OperationID
	e.state.OperationID = target.OperationID
	e.state.TargetEPSize = target.NewEPSize
	e.state.OperationSucceeded = nil
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
