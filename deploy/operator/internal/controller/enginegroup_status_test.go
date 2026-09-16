/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestEngineGroupStatusAPIRoundTrip(t *testing.T) {
	t.Log("build a fully correlated durable coordinator journal")
	status := engineGroupTestStatus()

	t.Log("serialize the journal into Kubernetes API status")
	encoded := engineGroupStatusToAPI(status)
	require.NotNil(t, encoded)

	t.Log("restore the coordinator journal after a simulated controller restart")
	restored := engineGroupStatusFromAPI(encoded)

	t.Log("verify the restored journal reproduces the exact persisted API representation")
	assert.Equal(t, encoded, engineGroupStatusToAPI(restored))
}

func TestEngineGroupPlanAPIRoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		change enginegroup.ResolvedChange
	}{
		{
			name: "grow",
			change: enginegroup.ResolvedChange{Kind: enginegroup.PlanKindGrow, Grow: &enginegroup.GrowChange{
				Replicas: []enginegroup.ReplicaTarget{engineGroupTestReplicaTarget("replica-2", "slot-2", "dp-2")},
			}},
		},
		{
			name: "retire",
			change: enginegroup.ResolvedChange{Kind: enginegroup.PlanKindRetire, Retire: &enginegroup.RetireChange{
				Replicas: []enginegroup.ReplicaID{"replica-1"},
			}},
		},
		{
			name: "reduce to survivors",
			change: enginegroup.ResolvedChange{
				Kind: enginegroup.PlanKindReduceToSurvivors,
				ReduceToSurvivors: &enginegroup.ReduceToSurvivorsChange{
					Survivors: []enginegroup.ReplicaID{"replica-0"},
				},
			},
		},
		{
			name: "restore",
			change: enginegroup.ResolvedChange{Kind: enginegroup.PlanKindRestore, Restore: &enginegroup.RestoreChange{
				Replicas: []enginegroup.RestorationTarget{{
					ReplicaTarget: engineGroupTestReplicaTarget("replica-1", "slot-1", "dp-1"),
				}},
			}},
		},
		{
			name: "remap",
			change: enginegroup.ResolvedChange{Kind: enginegroup.PlanKindRemap, Remap: &enginegroup.RemapChange{
				Membership: []enginegroup.ReplicaNativeMembership{{
					ReplicaID:     "replica-0",
					SlotID:        "slot-0",
					NativeMembers: []enginegroup.NativeMemberID{"dp-4"},
				}},
			}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build one immutable resolved operation shape")
			plan := engineGroupTestPlan(test.change)

			t.Log("round-trip the plan through its Kubernetes representation")
			restored := engineGroupPlanFromAPI(engineGroupPlanToAPI(plan))

			t.Log("verify exact tagged-union semantics survive persistence")
			assert.Equal(t, plan, restored)
		})
	}
}

func TestProjectEngineGroupStatusRequiresExactActiveAllocations(t *testing.T) {
	active := engineGroupTestMembership("replica-0", "runtime-0", "dp-0")
	activeIncarnation := engineGroupTestIncarnation(
		"replica-0", "slot-0", "runtime-0", "worker-0", "uid-0",
	)
	joiningIncarnation := engineGroupTestIncarnation(
		"replica-1", "slot-1", "runtime-1", "worker-1", "uid-1",
	)
	status := enginegroup.GroupStatus{
		Registry: enginegroup.ReplicaRegistry{Replicas: []enginegroup.ReplicaRecord{
			{ReplicaID: "replica-0", SlotID: "slot-0", Current: &activeIncarnation},
			{ReplicaID: "replica-1", SlotID: "slot-1", Current: &joiningIncarnation},
		}},
		Topologies: enginegroup.TopologyHistory{
			CurrentGeneration: 1,
			Snapshots: []enginegroup.MembershipTopology{{
				Generation: 1,
				Replicas:   []enginegroup.ReplicaMembership{active},
			}},
		},
		Capacity: enginegroup.CapacityStatus{Observed: enginegroup.CapacityObservation{
			Allocations: []enginegroup.CapacityAllocation{
				{Incarnation: activeIncarnation, Available: false},
				{Incarnation: joiningIncarnation, Available: true},
			},
		}},
		Traffic: enginegroup.TrafficStatus{Observed: enginegroup.TrafficObservation{
			Admitted: []enginegroup.ReplicaMembership{active},
		}},
		Membership: enginegroup.MembershipStatus{Observed: enginegroup.MembershipObservation{
			CommittedTopology: enginegroup.MembershipTopology{
				Generation: 1,
				Replicas:   []enginegroup.ReplicaMembership{active},
			},
		}},
	}
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{
		Fingerprint:            "profile-v1",
		MinSafeServingReplicas: 1,
	}
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Generation: 1},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}

	t.Log("project one unavailable active replica beside an available uncommitted joiner")
	reconciler := &DynamoGraphDeploymentEngineGroupReconciler{}
	reconciler.projectEngineGroupStatus(group, profile, status, nil, nil)

	t.Log("verify aggregate capacity cannot hide that the exact committed incarnation is unavailable")
	available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
	require.NotNil(t, available)
	assert.Equal(t, metav1.ConditionFalse, available.Status)
	assert.Equal(t, int32(1), group.Status.AvailableReplicas)
	assert.Equal(t, int32(1), group.Status.ActiveReplicas)
	assert.Equal(t, int32(0), group.Status.LastStableReplicas)
}

func TestProjectEngineGroupStatusDistinguishesPlannedAndUnplannedMembershipLoss(t *testing.T) {
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{
		Fingerprint:            "profile-v1",
		MinSafeServingReplicas: 1,
	}

	tests := []struct {
		name              string
		status            enginegroup.GroupStatus
		targetReplicas    int32
		lastStable        int32
		wantLastStable    int32
		wantAvailable     metav1.ConditionStatus
		wantTargetReached metav1.ConditionStatus
		wantDegraded      metav1.ConditionStatus
	}{
		{
			name:              "initial healthy topology becomes stable",
			status:            healthyEngineGroupProjectionStatus(2),
			targetReplicas:    2,
			wantLastStable:    2,
			wantAvailable:     metav1.ConditionTrue,
			wantTargetReached: metav1.ConditionTrue,
			wantDegraded:      metav1.ConditionFalse,
		},
		{
			name:              "exact planned retirement is not degradation",
			status:            plannedRetirementProjectionStatus(t),
			targetReplicas:    1,
			lastStable:        2,
			wantLastStable:    2,
			wantAvailable:     metav1.ConditionTrue,
			wantTargetReached: metav1.ConditionTrue,
			wantDegraded:      metav1.ConditionFalse,
		},
		{
			name:              "same-size but wrong retirement survivor is degradation",
			status:            wrongPlannedRetirementProjectionStatus(t),
			targetReplicas:    1,
			lastStable:        2,
			wantLastStable:    2,
			wantAvailable:     metav1.ConditionTrue,
			wantTargetReached: metav1.ConditionTrue,
			wantDegraded:      metav1.ConditionTrue,
		},
		{
			name:              "survivor reduction is degradation",
			status:            survivorReductionProjectionStatus(t),
			targetReplicas:    1,
			lastStable:        2,
			wantLastStable:    2,
			wantAvailable:     metav1.ConditionTrue,
			wantTargetReached: metav1.ConditionTrue,
			wantDegraded:      metav1.ConditionTrue,
		},
		{
			name:              "unfinished growth preserves the stable baseline",
			status:            unfinishedGrowthProjectionStatus(),
			targetReplicas:    3,
			lastStable:        2,
			wantLastStable:    2,
			wantAvailable:     metav1.ConditionTrue,
			wantTargetReached: metav1.ConditionFalse,
			wantDegraded:      metav1.ConditionFalse,
		},
		{
			name:              "blocked post-commit verification is unavailable but not membership degradation",
			status:            blockedPostCommitProjectionStatus(),
			targetReplicas:    3,
			lastStable:        2,
			wantLastStable:    2,
			wantAvailable:     metav1.ConditionFalse,
			wantTargetReached: metav1.ConditionTrue,
			wantDegraded:      metav1.ConditionFalse,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
				ObjectMeta: metav1.ObjectMeta{Generation: 1},
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{
					Replicas: test.targetReplicas,
				},
				Status: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupStatus{
					LastStableReplicas: test.lastStable,
				},
			}

			t.Log("project health from exact committed identities and transition intent")
			(&DynamoGraphDeploymentEngineGroupReconciler{}).
				projectEngineGroupStatus(group, profile, test.status, nil, nil)

			assert.Equal(t, test.wantLastStable, group.Status.LastStableReplicas)
			assert.Equal(t, test.wantAvailable,
				meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable).Status)
			assert.Equal(t, test.wantTargetReached,
				meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTargetReached).Status)
			assert.Equal(t, test.wantDegraded,
				meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionDegraded).Status)
		})
	}
}

func TestProjectEngineGroupReplicaStatesIncludesPreviousIncarnations(t *testing.T) {
	status := engineGroupTestStatus()

	t.Log("project the durable registry into the concise user-facing replica state")
	replicas := projectEngineGroupReplicaStates(status)

	t.Log("verify recovery history retains the exact previous runtime and Pod incarnation")
	require.Len(t, replicas, 2)
	require.Len(t, replicas[1].PreviousIncarnations, 1)
	assert.Equal(t, "runtime-old", replicas[1].PreviousIncarnations[0].RuntimeIncarnation)
	require.Len(t, replicas[1].PreviousIncarnations[0].CapacityRefs, 1)
	assert.Equal(t, "uid-old", string(replicas[1].PreviousIncarnations[0].CapacityRefs[0].UID))
}

func TestResolveDesiredEngineGroupPlanDoesNotTreatCardinalityAsConvergence(t *testing.T) {
	status := engineGroupTestStatus()
	status.Transition = nil
	current, found := status.Topologies.Current()
	require.True(t, found)
	plan := engineGroupTestPlan(enginegroup.ResolvedChange{
		Kind: enginegroup.PlanKindRemap,
		Remap: &enginegroup.RemapChange{Membership: []enginegroup.ReplicaNativeMembership{
			{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []enginegroup.NativeMemberID{"dp-2"}},
			{ReplicaID: "replica-1", SlotID: "slot-1", NativeMembers: []enginegroup.NativeMemberID{"dp-3"}},
		}},
	})
	planner := &engineGroupControllerTestPlanResolver{resolution: ScalePlanResolution{Plan: &plan}}
	runtime := EngineGroupRuntime{
		Profile: nvidiacomv1beta1.EngineGroupProfileStatus{
			MinReplicas: 1,
			MaxReplicas: 8,
		},
		Planner: planner,
	}
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: current.ReplicaCount()},
	}

	t.Log("resolve desired work when target and active cardinality are already equal")
	resolved, validation, err := (&DynamoGraphDeploymentEngineGroupReconciler{}).
		resolveDesiredEngineGroupPlan(context.Background(), group, runtime, status)

	t.Log("verify the planner can request recovery or remap without changing cardinality")
	require.Nil(t, validation)
	require.NoError(t, err)
	require.NotNil(t, resolved)
	assert.Equal(t, enginegroup.PlanKindRemap, resolved.Change.Kind)
	assert.Equal(t, 1, planner.calls)
}

func TestResolveDesiredEngineGroupPlanDistinguishesEveryPlannerOutcome(t *testing.T) {
	status := engineGroupTestStatus()
	status.Transition = nil
	current, found := status.Topologies.Current()
	require.True(t, found)
	plan := engineGroupTestPlan(enginegroup.ResolvedChange{
		Kind: enginegroup.PlanKindRemap,
		Remap: &enginegroup.RemapChange{Membership: []enginegroup.ReplicaNativeMembership{
			{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []enginegroup.NativeMemberID{"dp-2"}},
			{ReplicaID: "replica-1", SlotID: "slot-1", NativeMembers: []enginegroup.NativeMemberID{"dp-3"}},
		}},
	})
	plannerFailure := errors.New("planner authority unavailable")

	tests := []struct {
		name               string
		resolution         ScalePlanResolution
		err                error
		wantPlan           bool
		wantValidation     bool
		wantResolutionFail bool
	}{
		{name: "resolved plan", resolution: ScalePlanResolution{Plan: &plan}, wantPlan: true},
		{
			name: "definitive rejection",
			resolution: ScalePlanResolution{Rejection: &enginegroup.Failure{
				Classification: enginegroup.FailureClassificationTerminal,
				Reason:         "UnsupportedShape",
				Message:        "the backend cannot express this target",
			}},
			wantValidation: true,
		},
		{
			name: "definitive rejection with optional empty message",
			resolution: ScalePlanResolution{Rejection: &enginegroup.Failure{
				Classification: enginegroup.FailureClassificationTerminal,
				Reason:         "UnsupportedShape",
			}},
			wantValidation: true,
		},
		{name: "no operation required"},
		{name: "inconclusive planner failure", err: plannerFailure, wantResolutionFail: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			planner := &engineGroupControllerTestPlanResolver{resolution: test.resolution, err: test.err}
			runtime := EngineGroupRuntime{
				Profile: nvidiacomv1beta1.EngineGroupProfileStatus{MinReplicas: 1, MaxReplicas: 8},
				Planner: planner,
			}
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: current.ReplicaCount()},
			}

			t.Log("resolve one explicit planner outcome")
			resolved, validation, err := (&DynamoGraphDeploymentEngineGroupReconciler{}).
				resolveDesiredEngineGroupPlan(context.Background(), group, runtime, status)

			assert.Equal(t, test.wantPlan, resolved != nil)
			assert.Equal(t, test.wantValidation, validation != nil)
			assert.Equal(t, test.wantResolutionFail, err != nil)
			if test.wantResolutionFail {
				assert.ErrorIs(t, err, plannerFailure)
			}
		})
	}
}

func TestProjectEngineGroupStatusMarksHealthUnknownOnlyForObservationFailures(t *testing.T) {
	status := healthyEngineGroupProjectionStatus(2)
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{
		Fingerprint:            "profile-v1",
		MinSafeServingReplicas: 1,
	}
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Generation: 2},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
		Status:     nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupStatus{LastStableReplicas: 2},
	}
	reconciler := &DynamoGraphDeploymentEngineGroupReconciler{}

	t.Log("project stale durable evidence after a typed capacity observation failure")
	reconciler.projectEngineGroupStatus(group, profile, status, &enginegroup.ObservationError{
		Authority: enginegroup.ObservationAuthorityCapacity,
		Err:       errors.New("capacity unavailable"),
	}, nil)
	for _, conditionType := range []string{
		engineGroupConditionTopologyKnown,
		engineGroupConditionAvailable,
		engineGroupConditionTargetReached,
		engineGroupConditionDegraded,
	} {
		condition := meta.FindStatusCondition(group.Status.Conditions, conditionType)
		require.NotNil(t, condition)
		assert.Equal(t, metav1.ConditionUnknown, condition.Status, conditionType)
	}
	assert.Equal(t, int32(2), group.Status.LastStableReplicas)

	t.Log("project the same fresh observations with an unrelated post-observation error")
	reconciler.projectEngineGroupStatus(group, profile, status, nil, errors.New("planner unavailable"))
	assert.Equal(t, metav1.ConditionTrue,
		meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTopologyKnown).Status)
	assert.Equal(t, metav1.ConditionTrue,
		meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable).Status)
	assert.Equal(t, metav1.ConditionFalse,
		meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionDegraded).Status)
	assert.Equal(t, metav1.ConditionUnknown,
		meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTargetValid).Status)
}

func TestProjectEngineGroupStatusUsesObservedTopologyWithoutAcceptingItAsStable(t *testing.T) {
	status := healthyEngineGroupProjectionStatus(2)
	committed := enginegroup.MembershipTopology{
		Generation: 2,
		Replicas:   []enginegroup.ReplicaMembership{status.Membership.Observed.CommittedTopology.Replicas[0]},
	}
	status.Membership.Observed.CommittedTopology = committed
	status.Capacity.Observed.Allocations = status.Capacity.Observed.Allocations[:1]
	status.Traffic.Observed.Admitted = committed.Replicas
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Generation: 2},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
		Status:     nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupStatus{LastStableReplicas: 2},
	}
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{
		Fingerprint:            "profile-v1",
		MinSafeServingReplicas: 1,
	}

	t.Log("project an authoritative survivor topology that has no correlated accepted transition")
	(&DynamoGraphDeploymentEngineGroupReconciler{}).
		projectEngineGroupStatus(group, profile, status, nil, nil)

	t.Log("expose fresh engine truth without advancing the coordinator's stable baseline")
	require.NotNil(t, group.Status.Topology)
	assert.Equal(t, int64(2), group.Status.Topology.Generation)
	assert.Len(t, group.Status.Topology.Replicas, 1)
	assert.Equal(t, int32(1), group.Status.ActiveReplicas)
	assert.Equal(t, int32(2), group.Status.LastStableReplicas)
	assert.Equal(t, metav1.ConditionTrue,
		meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTopologyKnown).Status)
	assert.Equal(t, metav1.ConditionFalse,
		meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionTargetReached).Status)
	assert.Equal(t, metav1.ConditionTrue,
		meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionDegraded).Status)
}

func TestProjectEngineGroupStatusRequiresExactAdmittedMembership(t *testing.T) {
	status := healthyEngineGroupProjectionStatus(1)
	status.Traffic.Observed.Admitted = append(status.Traffic.Observed.Admitted, enginegroup.ReplicaMembership{
		ReplicaID:          "replica-joining",
		RuntimeIncarnation: "runtime-joining",
		NativeMembers:      []enginegroup.NativeMemberID{"dp-joining"},
	})
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Generation: 1},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{
		Fingerprint:            "profile-v1",
		MinSafeServingReplicas: 1,
	}

	t.Log("project a committed member together with an incorrectly admitted uncommitted joiner")
	(&DynamoGraphDeploymentEngineGroupReconciler{}).
		projectEngineGroupStatus(group, profile, status, nil, nil)

	t.Log("treat traffic as unavailable until its admitted set exactly matches committed membership")
	available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
	require.NotNil(t, available)
	assert.Equal(t, metav1.ConditionFalse, available.Status)
}

func TestProjectEngineGroupTrafficStatusUsesAcceptedTargetCorrelation(t *testing.T) {
	status := healthyEngineGroupProjectionStatus(2)
	accepted := &enginegroup.TrafficTarget{
		ControlRevision:    4,
		TransitionID:       "previous-operation",
		TopologyGeneration: 1,
		Admitted:           status.Traffic.Observed.Admitted,
	}
	status.ControlRevision = 4
	status.Traffic.Desired = accepted
	status.Traffic.Accepted = accepted
	status.Traffic.Observed.AppliedRevision = 4
	status.Transition = &enginegroup.TransitionStatus{
		Spec:    enginegroup.TransitionSpec{ID: "new-operation"},
		Outcome: enginegroup.TransitionOutcomeProgressing,
	}
	committed := status.Membership.Observed.CommittedTopology
	committed.Generation = 2
	status.Membership.Observed.CommittedTopology = committed
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Generation: 2},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
	}
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{Fingerprint: "profile-v1", MinSafeServingReplicas: 1}

	t.Log("project traffic while a newer membership operation exists but its routing target is not accepted")
	(&DynamoGraphDeploymentEngineGroupReconciler{}).
		projectEngineGroupStatus(group, profile, status, nil, nil)

	require.NotNil(t, group.Status.Traffic)
	assert.Equal(t, "previous-operation", group.Status.Traffic.OperationID)
	assert.Equal(t, int64(1), group.Status.Traffic.TopologyGeneration)

	t.Log("fall back to initialized observed topology before any traffic target has been accepted")
	status.ControlRevision = 0
	status.Traffic.Desired = nil
	status.Traffic.Accepted = nil
	status.Traffic.Observed.AppliedRevision = 0
	(&DynamoGraphDeploymentEngineGroupReconciler{}).
		projectEngineGroupStatus(group, profile, status, nil, nil)
	assert.Empty(t, group.Status.Traffic.OperationID)
	assert.Equal(t, int64(2), group.Status.Traffic.TopologyGeneration)
}

type engineGroupControllerTestPlanResolver struct {
	resolution ScalePlanResolution
	err        error
	calls      int
}

func (p *engineGroupControllerTestPlanResolver) ResolveScalePlan(
	context.Context,
	enginegroup.GroupID,
	int32,
	enginegroup.GroupStatus,
) (ScalePlanResolution, error) {
	p.calls++
	return p.resolution, p.err
}

func engineGroupTestStatus() enginegroup.GroupStatus {
	now := time.Date(2026, time.September, 15, 12, 0, 0, 0, time.UTC)
	baseMember := engineGroupTestMembership("replica-0", "runtime-0", "dp-0")
	joiningMember := engineGroupTestMembership("replica-1", "runtime-1", "dp-1")
	baseTopology := enginegroup.MembershipTopology{Generation: 4, Replicas: []enginegroup.ReplicaMembership{baseMember}}
	resultTopology := enginegroup.MembershipTopology{
		Generation: 5,
		Replicas:   []enginegroup.ReplicaMembership{baseMember, joiningMember},
	}
	baseIncarnation := engineGroupTestIncarnation("replica-0", "slot-0", "runtime-0", "worker-0", "uid-0")
	joiningIncarnation := engineGroupTestIncarnation("replica-1", "slot-1", "runtime-1", "worker-1", "uid-1")
	plan := engineGroupTestPlan(enginegroup.ResolvedChange{
		Kind: enginegroup.PlanKindGrow,
		Grow: &enginegroup.GrowChange{Replicas: []enginegroup.ReplicaTarget{
			engineGroupTestReplicaTarget("replica-1", "slot-1", "dp-1"),
		}},
	})
	planEvidence := &enginegroup.ValidationEvidence{
		PlanDigest:           "plan-digest",
		ProfileFingerprint:   "profile-v1",
		CapabilityGeneration: "capabilities-v2",
	}
	targetEvidence := &enginegroup.ValidationEvidence{
		PlanDigest:           "plan-digest",
		TargetDigest:         "target-digest",
		ProfileFingerprint:   "profile-v1",
		CapabilityGeneration: "capabilities-v2",
	}
	target := &enginegroup.MembershipTarget{
		ControlRevision: 3,
		TransitionID:    "transition-4-grow-1",
		TargetDigest:    "target-digest",
		Validation:      *targetEvidence,
		BaseTopology:    baseTopology,
		Plan:            plan,
		Joining: []enginegroup.JoiningReplica{{
			ReplicaID:          "replica-1",
			RuntimeIncarnation: "runtime-1",
		}},
	}
	capacityTarget := &enginegroup.CapacityTarget{
		ControlRevision:       1,
		TransitionID:          "grow-1",
		ProfileFingerprint:    "profile-v1",
		ProcessLifecycleOwner: enginegroup.ProcessLifecycleOwnerOrchestrator,
		Replicas: []enginegroup.CapacityReplicaTarget{
			{ReplicaID: "replica-0", SlotID: "slot-0", Incarnation: &baseIncarnation},
			{ReplicaID: "replica-1", SlotID: "slot-1", Incarnation: &joiningIncarnation},
		},
	}
	trafficTarget := &enginegroup.TrafficTarget{
		ControlRevision:    2,
		TransitionID:       "grow-1",
		TopologyGeneration: 4,
		Admitted:           []enginegroup.ReplicaMembership{baseMember},
		Drain: []enginegroup.TrafficDrainTarget{{
			Membership: joiningMember,
			Mode:       enginegroup.TrafficDrainModeConfirmInactive,
		}},
	}

	return enginegroup.GroupStatus{
		ControlRevision: 3,
		Registry: enginegroup.ReplicaRegistry{Replicas: []enginegroup.ReplicaRecord{
			{ReplicaID: "replica-0", SlotID: "slot-0", Current: &baseIncarnation},
			{
				ReplicaID: "replica-1",
				SlotID:    "slot-1",
				Current:   &joiningIncarnation,
				History: []enginegroup.ReplicaHistoryEntry{{
					TopologyGeneration: 3,
					Incarnation: engineGroupTestIncarnation(
						"replica-1", "slot-1", "runtime-old", "worker-1", "uid-old",
					),
					NativeMembers: []enginegroup.NativeMemberID{"dp-1"},
				}},
			},
		}},
		Topologies: enginegroup.TopologyHistory{
			CurrentGeneration: 5,
			Snapshots:         []enginegroup.MembershipTopology{baseTopology, resultTopology},
		},
		Capacity: enginegroup.CapacityStatus{
			Desired:  capacityTarget,
			Accepted: capacityTarget,
			Observed: enginegroup.CapacityObservation{
				AppliedRevision: 1,
				Allocations: []enginegroup.CapacityAllocation{
					{Incarnation: baseIncarnation, Available: true},
					{Incarnation: joiningIncarnation, Available: true},
				},
			},
		},
		Traffic: enginegroup.TrafficStatus{
			Desired:  trafficTarget,
			Accepted: trafficTarget,
			Observed: enginegroup.TrafficObservation{
				AppliedRevision: 2,
				Admitted:        []enginegroup.ReplicaMembership{baseMember},
				Drained:         []enginegroup.ReplicaMembership{joiningMember},
			},
		},
		Membership: enginegroup.MembershipStatus{
			Desired: target,
			Observed: enginegroup.MembershipObservation{
				CommittedTopology:     resultTopology,
				RequestedTransitionID: "transition-4-grow-1",
				Transition: &enginegroup.MembershipTransitionObservation{
					TransitionID:    "transition-4-grow-1",
					ControlRevision: 3,
					TargetDigest:    "target-digest",
					Phase:           enginegroup.MembershipTransitionPhaseCommitted,
					ResultTopology:  &resultTopology,
				},
			},
		},
		Transition: &enginegroup.TransitionStatus{
			Spec: enginegroup.TransitionSpec{ID: "grow-1", BaseTopologyGeneration: 4, Plan: plan},
			PlanPreflight: enginegroup.PreflightStatus{
				TransitionID:  "grow-1",
				SubjectDigest: "plan-digest",
				Evidence:      planEvidence,
			},
			TargetPreflight: enginegroup.PreflightStatus{
				TransitionID:    "transition-4-grow-1",
				ControlRevision: 3,
				SubjectDigest:   "target-digest",
				Evidence:        targetEvidence,
			},
			Verification: enginegroup.VerificationStatus{
				Phase: enginegroup.VerificationPhasePassed,
				Proof: &enginegroup.ServingProof{
					TopologyGeneration: 5,
					RuntimeDigest:      "runtime-digest",
					ObservedAt:         now,
				},
			},
			Outcome:   enginegroup.TransitionOutcomeCompleted,
			StartedAt: now.Add(-time.Minute),
			UpdatedAt: now,
		},
	}
}

func healthyEngineGroupProjectionStatus(replicas int) enginegroup.GroupStatus {
	topology := enginegroup.MembershipTopology{Generation: 1}
	status := enginegroup.GroupStatus{}
	for index := 0; index < replicas; index++ {
		replicaID := enginegroup.ReplicaID(fmt.Sprintf("replica-%d", index))
		slotID := enginegroup.CapacitySlotID(fmt.Sprintf("slot-%d", index))
		runtimeID := enginegroup.RuntimeIncarnationID(fmt.Sprintf("runtime-%d", index))
		member := enginegroup.ReplicaMembership{
			ReplicaID:          replicaID,
			RuntimeIncarnation: runtimeID,
			NativeMembers:      []enginegroup.NativeMemberID{enginegroup.NativeMemberID(fmt.Sprintf("dp-%d", index))},
		}
		incarnation := enginegroup.ReplicaIncarnation{
			ReplicaID:          replicaID,
			SlotID:             slotID,
			RuntimeIncarnation: runtimeID,
			CapacityRefs: []enginegroup.CapacityRef{{
				Name: fmt.Sprintf("worker-%d", index),
				UID:  enginegroup.PodUID(fmt.Sprintf("uid-%d", index)),
			}},
		}
		topology.Replicas = append(topology.Replicas, member)
		status.Registry.Replicas = append(status.Registry.Replicas, enginegroup.ReplicaRecord{
			ReplicaID: replicaID,
			SlotID:    slotID,
			Current:   &incarnation,
		})
		status.Capacity.Observed.Allocations = append(status.Capacity.Observed.Allocations,
			enginegroup.CapacityAllocation{Incarnation: incarnation, Available: true})
		status.Traffic.Observed.Admitted = append(status.Traffic.Observed.Admitted, member)
	}
	status.Topologies = enginegroup.TopologyHistory{CurrentGeneration: 1, Snapshots: []enginegroup.MembershipTopology{topology}}
	status.Membership.Observed.CommittedTopology = topology
	return status
}

func plannedRetirementProjectionStatus(t *testing.T) enginegroup.GroupStatus {
	t.Helper()
	status := healthyEngineGroupProjectionStatus(2)
	base, found := status.Topologies.Current()
	require.True(t, found)
	survivor := base.Replicas[0]
	committed := enginegroup.MembershipTopology{Generation: 2, Replicas: []enginegroup.ReplicaMembership{survivor}}
	status.Topologies = enginegroup.TopologyHistory{
		CurrentGeneration: 2,
		Snapshots:         []enginegroup.MembershipTopology{base, committed},
	}
	status.Membership.Observed.CommittedTopology = committed
	status.Capacity.Observed.Allocations = status.Capacity.Observed.Allocations[:1]
	status.Traffic.Observed.Admitted = []enginegroup.ReplicaMembership{survivor}
	status.Transition = &enginegroup.TransitionStatus{
		Spec: enginegroup.TransitionSpec{
			ID:                     "retire-one",
			BaseTopologyGeneration: 1,
			Plan: enginegroup.ResolvedPlan{Change: enginegroup.ResolvedChange{
				Kind: enginegroup.PlanKindRetire,
				Retire: &enginegroup.RetireChange{
					Replicas: []enginegroup.ReplicaID{base.Replicas[1].ReplicaID},
				},
			}},
		},
		Outcome: enginegroup.TransitionOutcomeProgressing,
	}
	return status
}

func survivorReductionProjectionStatus(t *testing.T) enginegroup.GroupStatus {
	t.Helper()
	status := plannedRetirementProjectionStatus(t)
	status.Transition.Spec.Plan.Change = enginegroup.ResolvedChange{
		Kind: enginegroup.PlanKindReduceToSurvivors,
		ReduceToSurvivors: &enginegroup.ReduceToSurvivorsChange{
			Survivors: []enginegroup.ReplicaID{status.Membership.Observed.CommittedTopology.Replicas[0].ReplicaID},
		},
	}
	return status
}

func wrongPlannedRetirementProjectionStatus(t *testing.T) enginegroup.GroupStatus {
	t.Helper()
	status := plannedRetirementProjectionStatus(t)
	base, found := status.Topologies.Snapshot(1)
	require.True(t, found)
	wrongSurvivor := base.Replicas[1]
	committed := enginegroup.MembershipTopology{
		Generation: 2,
		Replicas:   []enginegroup.ReplicaMembership{wrongSurvivor},
	}
	status.Topologies = enginegroup.TopologyHistory{
		CurrentGeneration: 2,
		Snapshots:         []enginegroup.MembershipTopology{base, committed},
	}
	status.Membership.Observed.CommittedTopology = committed
	record, found := engineGroupReplicaRecord(status.Registry, wrongSurvivor.ReplicaID)
	require.True(t, found)
	require.NotNil(t, record.Current)
	status.Capacity.Observed.Allocations = []enginegroup.CapacityAllocation{{
		Incarnation: *record.Current,
		Available:   true,
	}}
	status.Traffic.Observed.Admitted = []enginegroup.ReplicaMembership{wrongSurvivor}
	return status
}

func unfinishedGrowthProjectionStatus() enginegroup.GroupStatus {
	status := healthyEngineGroupProjectionStatus(2)
	status.Transition = &enginegroup.TransitionStatus{
		Spec: enginegroup.TransitionSpec{
			ID:                     "grow-one",
			BaseTopologyGeneration: 1,
			Plan: enginegroup.ResolvedPlan{
				Change: enginegroup.ResolvedChange{
					Kind: enginegroup.PlanKindGrow,
					Grow: &enginegroup.GrowChange{
						Replicas: []enginegroup.ReplicaTarget{{
							ReplicaID: "replica-2",
							SlotID:    "slot-2",
						}},
					},
				},
			},
		},
		Outcome: enginegroup.TransitionOutcomeProgressing,
	}
	return status
}

func blockedPostCommitProjectionStatus() enginegroup.GroupStatus {
	status := healthyEngineGroupProjectionStatus(3)
	status.Traffic.Observed.Admitted = status.Traffic.Observed.Admitted[:2]
	status.Transition = &enginegroup.TransitionStatus{
		Spec: enginegroup.TransitionSpec{
			ID:                     "grow-one",
			BaseTopologyGeneration: 1,
			Plan: enginegroup.ResolvedPlan{
				Change: enginegroup.ResolvedChange{
					Kind: enginegroup.PlanKindGrow,
					Grow: &enginegroup.GrowChange{
						Replicas: []enginegroup.ReplicaTarget{{
							ReplicaID: "replica-2",
							SlotID:    "slot-2",
						}},
					},
				},
			},
		},
		Outcome: enginegroup.TransitionOutcomeBlocked,
	}
	return status
}

func engineGroupTestPlan(change enginegroup.ResolvedChange) enginegroup.ResolvedPlan {
	return enginegroup.ResolvedPlan{
		ID:                      "plan-1",
		ProfileFingerprint:      "profile-v1",
		ProcessLifecycleOwner:   enginegroup.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      enginegroup.TrafficRequirementQuiesceGroup,
		VerificationRequirement: enginegroup.VerificationRequirementRequired,
		Change:                  change,
	}
}

func engineGroupTestReplicaTarget(replicaID, slotID, nativeMember string) enginegroup.ReplicaTarget {
	return enginegroup.ReplicaTarget{
		ReplicaID:     enginegroup.ReplicaID(replicaID),
		SlotID:        enginegroup.CapacitySlotID(slotID),
		Bootstrap:     enginegroup.BootstrapModeJoin,
		NativeMembers: []enginegroup.NativeMemberID{enginegroup.NativeMemberID(nativeMember)},
	}
}

func engineGroupTestMembership(replicaID, runtimeID, nativeMember string) enginegroup.ReplicaMembership {
	return enginegroup.ReplicaMembership{
		ReplicaID:          enginegroup.ReplicaID(replicaID),
		RuntimeIncarnation: enginegroup.RuntimeIncarnationID(runtimeID),
		NativeMembers:      []enginegroup.NativeMemberID{enginegroup.NativeMemberID(nativeMember)},
	}
}

func engineGroupTestIncarnation(
	replicaID string,
	slotID string,
	runtimeID string,
	podName string,
	podUID string,
) enginegroup.ReplicaIncarnation {
	return enginegroup.ReplicaIncarnation{
		ReplicaID:          enginegroup.ReplicaID(replicaID),
		SlotID:             enginegroup.CapacitySlotID(slotID),
		RuntimeIncarnation: enginegroup.RuntimeIncarnationID(runtimeID),
		CapacityRefs:       []enginegroup.CapacityRef{{Name: podName, UID: enginegroup.PodUID(podUID)}},
	}
}
