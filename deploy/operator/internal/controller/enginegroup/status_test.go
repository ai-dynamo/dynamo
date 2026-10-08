/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestEngineGroupCheckpointStateRoundTrip(t *testing.T) {
	t.Log("build a fully correlated durable coordinator journal")
	status := engineGroupTestStatus()

	t.Log("serialize the coordinator journal directly, without a public API mirror")
	encoded, err := json.Marshal(status)
	require.NoError(t, err)

	t.Log("restore the coordinator journal after a simulated controller restart")
	var restored domain.GroupStatus
	require.NoError(t, json.Unmarshal(encoded, &restored))

	t.Log("verify every desired level, digest, incarnation, and observation survives persistence")
	assert.Equal(t, status, restored)
}

func TestEngineGroupCheckpointPlanRoundTrip(t *testing.T) {
	tests := []struct {
		name   string
		change domain.ResolvedChange
	}{
		{
			name: "grow",
			change: domain.ResolvedChange{Kind: domain.PlanKindGrow, Grow: &domain.GrowChange{
				Replicas: []domain.ReplicaTarget{engineGroupTestReplicaTarget("replica-2", "slot-2", "dp-2")},
			}},
		},
		{
			name: "retire",
			change: domain.ResolvedChange{Kind: domain.PlanKindRetire, Retire: &domain.RetireChange{
				Replicas: []domain.ReplicaID{"replica-1"},
			}},
		},
		{
			name: "reduce to survivors",
			change: domain.ResolvedChange{
				Kind: domain.PlanKindReduceToSurvivors,
				ReduceToSurvivors: &domain.ReduceToSurvivorsChange{
					Survivors: []domain.ReplicaNativeMembership{{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []domain.NativeMemberID{"dp-0"}}},
				},
			},
		},
		{
			name: "restore",
			change: domain.ResolvedChange{Kind: domain.PlanKindRestore, Restore: &domain.RestoreChange{
				Replicas: []domain.RestorationTarget{{
					ReplicaTarget: engineGroupTestReplicaTarget("replica-1", "slot-1", "dp-1"),
				}},
			}},
		},
		{
			name: "remap",
			change: domain.ResolvedChange{Kind: domain.PlanKindRemap, Remap: &domain.RemapChange{
				Membership: []domain.ReplicaNativeMembership{{
					ReplicaID:     "replica-0",
					SlotID:        "slot-0",
					NativeMembers: []domain.NativeMemberID{"dp-4"},
				}},
			}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build one immutable resolved operation shape")
			plan := engineGroupTestPlan(test.change)

			t.Log("round-trip the tagged plan directly through private checkpoint JSON")
			encoded, err := json.Marshal(plan)
			require.NoError(t, err)
			var restored domain.ResolvedPlan
			require.NoError(t, json.Unmarshal(encoded, &restored))

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
	status := domain.GroupStatus{
		Registry: domain.ReplicaRegistry{Replicas: []domain.ReplicaRecord{
			{ReplicaID: "replica-0", SlotID: "slot-0", Current: &activeIncarnation},
			{ReplicaID: "replica-1", SlotID: "slot-1", Current: &joiningIncarnation},
		}},
		Topologies: domain.TopologyHistory{
			CurrentGeneration: 1,
			Snapshots: []domain.MembershipTopology{{
				Generation: 1,
				Replicas:   []domain.ReplicaMembership{active},
			}},
		},
		Capacity: domain.CapacityStatus{Observed: domain.CapacityObservation{
			Allocations: []domain.CapacityAllocation{
				{Incarnation: activeIncarnation, Available: false},
				{Incarnation: joiningIncarnation, Available: true},
			},
		}},
		Traffic: domain.TrafficStatus{Observed: domain.TrafficObservation{
			Admitted: []domain.ReplicaMembership{active},
		}},
		Membership: domain.MembershipStatus{Observed: domain.MembershipObservation{
			CommittedTopology: domain.MembershipTopology{
				Generation: 1,
				Replicas:   []domain.ReplicaMembership{active},
			},
		}},
	}
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{
		Fingerprint:                 "profile-v1",
		NativeMembersPerReplica:     1,
		MinSafeServingNativeMembers: 1,
	}
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Generation: 1},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}

	t.Log("project one unavailable active replica beside an available uncommitted joiner")
	reconciler := &Reconciler{}
	reconciler.projectEngineGroupStatus(group, profile, status, nil, nil)

	t.Log("verify aggregate capacity cannot hide that the exact committed incarnation is unavailable")
	available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
	require.NotNil(t, available)
	assert.Equal(t, metav1.ConditionFalse, available.Status)
	assert.Equal(t, int32(0), group.Status.AvailableReplicas)
	assert.Equal(t, int32(1), group.Status.ActiveNativeMemberCount)
	assert.Equal(t, int32(0), group.Status.LastStableReplicas)
}

func TestProjectEngineGroupStatusDistinguishesPlannedAndUnplannedMembershipLoss(t *testing.T) {
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{
		Fingerprint:                 "profile-v1",
		NativeMembersPerReplica:     1,
		MinSafeServingNativeMembers: 1,
	}

	tests := []struct {
		name              string
		status            domain.GroupStatus
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
			wantTargetReached: metav1.ConditionFalse,
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
			var desiredPlan *domain.ResolvedPlan
			if test.status.Transition != nil {
				desiredPlan = &test.status.Transition.Spec.Plan
			}
			reconcileEngineGroupDesiredAssignment(group, profile, test.status, desiredPlan)
			(&Reconciler{}).
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

func TestProjectEngineGroupReplicaStatesKeepsHistoryInTheDurableJournal(t *testing.T) {
	status := engineGroupTestStatus()

	t.Log("project the durable registry into the concise user-facing replica state")
	replicas := projectEngineGroupReplicaStates(status, nil)

	t.Log("verify recovery history retains the exact previous runtime and Pod incarnation")
	require.Len(t, replicas, 2)
	require.Len(t, status.Registry.Replicas[1].History, 1)
	assert.Equal(t, domain.RuntimeIncarnationID("runtime-old"), status.Registry.Replicas[1].History[0].Incarnation.Members[0].RuntimeIncarnation)
	assert.Equal(t, domain.PodUID("uid-old"), status.Registry.Replicas[1].History[0].Incarnation.CapacityRefs[0].UID)
}

func TestResolveDesiredEngineGroupPlanDoesNotTreatCardinalityAsConvergence(t *testing.T) {
	status := engineGroupTestStatus()
	status.Transition = nil
	current, found := status.Topologies.Current()
	require.True(t, found)
	plan := engineGroupTestPlan(domain.ResolvedChange{
		Kind: domain.PlanKindRemap,
		Remap: &domain.RemapChange{Membership: []domain.ReplicaNativeMembership{
			{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []domain.NativeMemberID{"dp-2"}},
			{ReplicaID: "replica-1", SlotID: "slot-1", NativeMembers: []domain.NativeMemberID{"dp-3"}},
		}},
	})
	planner := &engineGroupControllerTestPlanResolver{resolution: ScalePlanResolution{Plan: &plan}}
	runtime := Runtime{
		Profile: nvidiacomv1beta1.EngineGroupProfileStatus{
			MinSupportedReplicas: 1,
			MaxSupportedReplicas: 8,
		},
		Planner: planner,
	}
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: current.ReplicaCount()},
	}

	t.Log("resolve desired work when target and active cardinality are already equal")
	resolved, validation, err := (&Reconciler{}).
		resolveDesiredEngineGroupPlan(context.Background(), group, runtime, status)

	t.Log("verify the planner can request recovery or remap without changing cardinality")
	require.Nil(t, validation)
	require.NoError(t, err)
	require.NotNil(t, resolved)
	assert.Equal(t, domain.PlanKindRemap, resolved.Change.Kind)
	assert.Equal(t, 1, planner.calls)
}

func TestResolveDesiredEngineGroupPlanDistinguishesEveryPlannerOutcome(t *testing.T) {
	status := engineGroupTestStatus()
	status.Transition = nil
	current, found := status.Topologies.Current()
	require.True(t, found)
	plan := engineGroupTestPlan(domain.ResolvedChange{
		Kind: domain.PlanKindRemap,
		Remap: &domain.RemapChange{Membership: []domain.ReplicaNativeMembership{
			{ReplicaID: "replica-0", SlotID: "slot-0", NativeMembers: []domain.NativeMemberID{"dp-2"}},
			{ReplicaID: "replica-1", SlotID: "slot-1", NativeMembers: []domain.NativeMemberID{"dp-3"}},
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
			resolution: ScalePlanResolution{Rejection: &domain.Failure{
				Classification: domain.FailureClassificationTerminal,
				Reason:         "UnsupportedShape",
				Message:        "the backend cannot express this target",
			}},
			wantValidation: true,
		},
		{
			name: "definitive rejection with optional empty message",
			resolution: ScalePlanResolution{Rejection: &domain.Failure{
				Classification: domain.FailureClassificationTerminal,
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
			runtime := Runtime{
				Profile: nvidiacomv1beta1.EngineGroupProfileStatus{MinSupportedReplicas: 1, MaxSupportedReplicas: 8},
				Planner: planner,
			}
			group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
				Spec: nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: current.ReplicaCount()},
			}

			t.Log("resolve one explicit planner outcome")
			resolved, validation, err := (&Reconciler{}).
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
		Fingerprint:                 "profile-v1",
		NativeMembersPerReplica:     1,
		MinSafeServingNativeMembers: 1,
	}
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Generation: 2},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
		Status:     nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupStatus{LastStableReplicas: 2},
	}
	reconciler := &Reconciler{}

	t.Log("project stale durable evidence after a typed capacity observation failure")
	reconciler.projectEngineGroupStatus(group, profile, status, &domain.ObservationError{
		Authority: domain.ObservationAuthorityCapacity,
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
	committed := domain.MembershipTopology{
		Generation: 2,
		Replicas:   []domain.ReplicaMembership{status.Membership.Observed.CommittedTopology.Replicas[0]},
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
		Fingerprint:                 "profile-v1",
		NativeMembersPerReplica:     1,
		MinSafeServingNativeMembers: 1,
	}

	t.Log("project an authoritative survivor topology that has no correlated accepted transition")
	(&Reconciler{}).
		projectEngineGroupStatus(group, profile, status, nil, nil)

	t.Log("expose fresh engine truth without advancing the coordinator's stable baseline")
	require.NotNil(t, group.Status.Topology)
	assert.Equal(t, int64(2), group.Status.Topology.Generation)
	assert.Len(t, group.Status.Topology.Replicas, 1)
	assert.Equal(t, int32(1), group.Status.ActiveNativeMemberCount)
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
	status.Traffic.Observed.Admitted = append(status.Traffic.Observed.Admitted, domain.ReplicaMembership{
		ReplicaID: "replica-joining",
		Members:   []domain.NativeMemberIncarnation{{ID: "dp-joining", RuntimeIncarnation: "runtime-joining"}},
	})
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Generation: 1},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 1},
	}
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{
		Fingerprint:                 "profile-v1",
		NativeMembersPerReplica:     1,
		MinSafeServingNativeMembers: 1,
	}

	t.Log("project a committed member together with an incorrectly admitted uncommitted joiner")
	(&Reconciler{}).
		projectEngineGroupStatus(group, profile, status, nil, nil)

	t.Log("treat traffic as unavailable until its admitted set exactly matches committed membership")
	available := meta.FindStatusCondition(group.Status.Conditions, engineGroupConditionAvailable)
	require.NotNil(t, available)
	assert.Equal(t, metav1.ConditionFalse, available.Status)
}

func TestProjectEngineGroupTrafficStatusUsesAcceptedTargetCorrelation(t *testing.T) {
	status := healthyEngineGroupProjectionStatus(2)
	accepted := &domain.TrafficTarget{
		ControlRevision:    4,
		TransitionID:       "previous-operation",
		TopologyGeneration: 1,
		Admitted:           status.Traffic.Observed.Admitted,
	}
	status.ControlRevision = 4
	status.Traffic.Desired = accepted
	status.Traffic.Accepted = accepted
	status.Traffic.Observed.AppliedRevision = 4
	status.Transition = &domain.TransitionStatus{
		Spec:    domain.TransitionSpec{ID: "new-operation"},
		Outcome: domain.TransitionOutcomeProgressing,
	}
	committed := status.Membership.Observed.CommittedTopology
	committed.Generation = 2
	status.Membership.Observed.CommittedTopology = committed
	group := &nvidiacomv1beta1.DynamoGraphDeploymentEngineGroup{
		ObjectMeta: metav1.ObjectMeta{Generation: 2},
		Spec:       nvidiacomv1beta1.DynamoGraphDeploymentEngineGroupSpec{Replicas: 2},
	}
	profile := nvidiacomv1beta1.EngineGroupProfileStatus{Fingerprint: "profile-v1", NativeMembersPerReplica: 1,
		MinSafeServingNativeMembers: 1}

	t.Log("project traffic while a newer membership operation exists but its routing target is not accepted")
	(&Reconciler{}).
		projectEngineGroupStatus(group, profile, status, nil, nil)

	require.NotNil(t, group.Status.Traffic)
	assert.Equal(t, "previous-operation", group.Status.Traffic.OperationID)
	assert.Equal(t, int64(1), group.Status.Traffic.TopologyGeneration)

	t.Log("fall back to initialized observed topology before any traffic target has been accepted")
	status.ControlRevision = 0
	status.Traffic.Desired = nil
	status.Traffic.Accepted = nil
	status.Traffic.Observed.AppliedRevision = 0
	(&Reconciler{}).
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
	domain.GroupID,
	int32,
	domain.GroupStatus,
) (ScalePlanResolution, error) {
	p.calls++
	return p.resolution, p.err
}

func engineGroupTestStatus() domain.GroupStatus {
	now := time.Date(2026, time.September, 15, 12, 0, 0, 0, time.UTC)
	baseMember := engineGroupTestMembership("replica-0", "runtime-0", "dp-0")
	joiningMember := engineGroupTestMembership("replica-1", "runtime-1", "dp-1")
	baseTopology := domain.MembershipTopology{Generation: 4, Replicas: []domain.ReplicaMembership{baseMember}}
	resultTopology := domain.MembershipTopology{
		Generation: 5,
		Replicas:   []domain.ReplicaMembership{baseMember, joiningMember},
	}
	baseIncarnation := engineGroupTestIncarnation("replica-0", "slot-0", "runtime-0", "worker-0", "uid-0")
	joiningIncarnation := engineGroupTestIncarnation("replica-1", "slot-1", "runtime-1", "worker-1", "uid-1")
	plan := engineGroupTestPlan(domain.ResolvedChange{
		Kind: domain.PlanKindGrow,
		Grow: &domain.GrowChange{Replicas: []domain.ReplicaTarget{
			engineGroupTestReplicaTarget("replica-1", "slot-1", "dp-1"),
		}},
	})
	planEvidence := &domain.ValidationEvidence{
		PlanDigest:           "plan-digest",
		ProfileFingerprint:   "profile-v1",
		CapabilityGeneration: "capabilities-v2",
	}
	targetEvidence := &domain.ValidationEvidence{
		PlanDigest:           "plan-digest",
		TargetDigest:         "target-digest",
		ProfileFingerprint:   "profile-v1",
		CapabilityGeneration: "capabilities-v2",
	}
	target := &domain.MembershipTarget{
		ControlRevision: 3,
		TransitionID:    "transition-4-grow-1",
		TargetDigest:    "target-digest",
		Validation:      *targetEvidence,
		BaseTopology:    baseTopology,
		Plan:            plan,
		Joining: []domain.JoiningReplica{{
			ReplicaID: "replica-1",
			Members:   []domain.NativeMemberIncarnation{{ID: "dp-1", RuntimeIncarnation: "runtime-1"}},
		}},
	}
	capacityTarget := &domain.CapacityTarget{
		ControlRevision:       1,
		TransitionID:          "grow-1",
		ProfileFingerprint:    "profile-v1",
		ProcessLifecycleOwner: domain.ProcessLifecycleOwnerOrchestrator,
		Replicas: []domain.CapacityReplicaTarget{
			{ReplicaID: "replica-0", SlotID: "slot-0", Incarnation: &baseIncarnation},
			{ReplicaID: "replica-1", SlotID: "slot-1", Incarnation: &joiningIncarnation},
		},
	}
	trafficTarget := &domain.TrafficTarget{
		ControlRevision:    2,
		TransitionID:       "grow-1",
		TopologyGeneration: 4,
		Admitted:           []domain.ReplicaMembership{baseMember},
		Drain: []domain.TrafficDrainTarget{{
			Membership: joiningMember,
			Mode:       domain.TrafficDrainModeConfirmInactive,
		}},
	}

	return domain.GroupStatus{
		ControlRevision: 3,
		Registry: domain.ReplicaRegistry{Replicas: []domain.ReplicaRecord{
			{ReplicaID: "replica-0", SlotID: "slot-0", Current: &baseIncarnation},
			{
				ReplicaID: "replica-1",
				SlotID:    "slot-1",
				Current:   &joiningIncarnation,
				History: []domain.ReplicaHistoryEntry{{
					TopologyGeneration: 3,
					Incarnation: engineGroupTestIncarnation(
						"replica-1", "slot-1", "runtime-old", "worker-1", "uid-old",
					),
					NativeMembers: []domain.NativeMemberID{"dp-1"},
				}},
			},
		}},
		Topologies: domain.TopologyHistory{
			CurrentGeneration: 5,
			Snapshots:         []domain.MembershipTopology{baseTopology, resultTopology},
		},
		Capacity: domain.CapacityStatus{
			Desired:  capacityTarget,
			Accepted: capacityTarget,
			Observed: domain.CapacityObservation{
				AppliedRevision: 1,
				Allocations: []domain.CapacityAllocation{
					{Incarnation: baseIncarnation, Available: true},
					{Incarnation: joiningIncarnation, Available: true},
				},
			},
		},
		Traffic: domain.TrafficStatus{
			Desired:  trafficTarget,
			Accepted: trafficTarget,
			Observed: domain.TrafficObservation{
				AppliedRevision: 2,
				Admitted:        []domain.ReplicaMembership{baseMember},
				Drained:         []domain.ReplicaMembership{joiningMember},
			},
		},
		Membership: domain.MembershipStatus{
			Desired: target,
			Observed: domain.MembershipObservation{
				CommittedTopology:     resultTopology,
				RequestedTransitionID: "transition-4-grow-1",
				Transition: &domain.MembershipTransitionObservation{
					TransitionID:    "transition-4-grow-1",
					ControlRevision: 3,
					TargetDigest:    "target-digest",
					Phase:           domain.MembershipTransitionPhaseCommitted,
					ResultTopology:  &resultTopology,
				},
			},
		},
		Transition: &domain.TransitionStatus{
			Spec: domain.TransitionSpec{ID: "grow-1", BaseTopologyGeneration: 4, Plan: plan},
			PlanPreflight: domain.PreflightStatus{
				TransitionID:  "grow-1",
				SubjectDigest: "plan-digest",
				Evidence:      planEvidence,
			},
			TargetPreflight: domain.PreflightStatus{
				TransitionID:    "transition-4-grow-1",
				ControlRevision: 3,
				SubjectDigest:   "target-digest",
				Evidence:        targetEvidence,
			},
			Verification: domain.VerificationStatus{
				Phase: domain.VerificationPhasePassed,
				Proof: &domain.ServingProof{
					TopologyGeneration: 5,
					RuntimeDigest:      "runtime-digest",
					ObservedAt:         now,
				},
			},
			Outcome:   domain.TransitionOutcomeCompleted,
			StartedAt: now.Add(-time.Minute),
			UpdatedAt: now,
		},
	}
}

func healthyEngineGroupProjectionStatus(replicas int) domain.GroupStatus {
	topology := domain.MembershipTopology{Generation: 1}
	status := domain.GroupStatus{}
	for index := 0; index < replicas; index++ {
		replicaID := domain.ReplicaID(fmt.Sprintf("replica-%d", index))
		slotID := domain.CapacitySlotID(fmt.Sprintf("slot-%d", index))
		runtimeID := domain.RuntimeIncarnationID(fmt.Sprintf("runtime-%d", index))
		member := domain.ReplicaMembership{
			ReplicaID: replicaID,
			Members:   []domain.NativeMemberIncarnation{{ID: domain.NativeMemberID(fmt.Sprintf("dp-%d", index)), RuntimeIncarnation: runtimeID}},
		}
		incarnation := domain.ReplicaIncarnation{
			ReplicaID: replicaID,
			SlotID:    slotID,
			Members:   append([]domain.NativeMemberIncarnation(nil), member.Members...),
			CapacityRefs: []domain.CapacityRef{{
				Name: fmt.Sprintf("worker-%d", index),
				UID:  domain.PodUID(fmt.Sprintf("uid-%d", index)),
			}},
		}
		topology.Replicas = append(topology.Replicas, member)
		status.Registry.Replicas = append(status.Registry.Replicas, domain.ReplicaRecord{
			ReplicaID:            replicaID,
			SlotID:               slotID,
			DesiredNativeMembers: []domain.NativeMemberID{member.Members[0].ID},
			Current:              &incarnation,
		})
		status.Capacity.Observed.Allocations = append(status.Capacity.Observed.Allocations,
			domain.CapacityAllocation{Incarnation: incarnation, Available: true})
		status.Traffic.Observed.Admitted = append(status.Traffic.Observed.Admitted, member)
	}
	status.Topologies = domain.TopologyHistory{CurrentGeneration: 1, Snapshots: []domain.MembershipTopology{topology}}
	status.Membership.Observed.CommittedTopology = topology
	return status
}

func plannedRetirementProjectionStatus(t *testing.T) domain.GroupStatus {
	t.Helper()
	status := healthyEngineGroupProjectionStatus(2)
	base, found := status.Topologies.Current()
	require.True(t, found)
	survivor := base.Replicas[0]
	committed := domain.MembershipTopology{Generation: 2, Replicas: []domain.ReplicaMembership{survivor}}
	status.Topologies = domain.TopologyHistory{
		CurrentGeneration: 2,
		Snapshots:         []domain.MembershipTopology{base, committed},
	}
	status.Membership.Observed.CommittedTopology = committed
	status.Capacity.Observed.Allocations = status.Capacity.Observed.Allocations[:1]
	status.Traffic.Observed.Admitted = []domain.ReplicaMembership{survivor}
	status.Transition = &domain.TransitionStatus{
		Spec: domain.TransitionSpec{
			ID:                     "retire-one",
			BaseTopologyGeneration: 1,
			Plan: domain.ResolvedPlan{Change: domain.ResolvedChange{
				Kind: domain.PlanKindRetire,
				Retire: &domain.RetireChange{
					Replicas: []domain.ReplicaID{base.Replicas[1].ReplicaID},
				},
			}},
		},
		Outcome: domain.TransitionOutcomeProgressing,
	}
	return status
}

func survivorReductionProjectionStatus(t *testing.T) domain.GroupStatus {
	t.Helper()
	status := plannedRetirementProjectionStatus(t)
	status.Transition.Spec.Plan.Change = domain.ResolvedChange{
		Kind: domain.PlanKindReduceToSurvivors,
		ReduceToSurvivors: &domain.ReduceToSurvivorsChange{
			Survivors: []domain.ReplicaNativeMembership{{ReplicaID: status.Membership.Observed.CommittedTopology.Replicas[0].ReplicaID, SlotID: "slot-0", NativeMembers: []domain.NativeMemberID{"dp-0"}}},
		},
	}
	return status
}

func wrongPlannedRetirementProjectionStatus(t *testing.T) domain.GroupStatus {
	t.Helper()
	status := plannedRetirementProjectionStatus(t)
	base, found := status.Topologies.Snapshot(1)
	require.True(t, found)
	wrongSurvivor := base.Replicas[1]
	committed := domain.MembershipTopology{
		Generation: 2,
		Replicas:   []domain.ReplicaMembership{wrongSurvivor},
	}
	status.Topologies = domain.TopologyHistory{
		CurrentGeneration: 2,
		Snapshots:         []domain.MembershipTopology{base, committed},
	}
	status.Membership.Observed.CommittedTopology = committed
	record, found := engineGroupReplicaRecord(status.Registry, wrongSurvivor.ReplicaID)
	require.True(t, found)
	require.NotNil(t, record.Current)
	status.Capacity.Observed.Allocations = []domain.CapacityAllocation{{
		Incarnation: *record.Current,
		Available:   true,
	}}
	status.Traffic.Observed.Admitted = []domain.ReplicaMembership{wrongSurvivor}
	return status
}

func unfinishedGrowthProjectionStatus() domain.GroupStatus {
	status := healthyEngineGroupProjectionStatus(2)
	status.Transition = &domain.TransitionStatus{
		Spec: domain.TransitionSpec{
			ID:                     "grow-one",
			BaseTopologyGeneration: 1,
			Plan: domain.ResolvedPlan{
				Change: domain.ResolvedChange{
					Kind: domain.PlanKindGrow,
					Grow: &domain.GrowChange{
						Replicas: []domain.ReplicaTarget{{
							ReplicaID: "replica-2",
							SlotID:    "slot-2",
						}},
					},
				},
			},
		},
		Outcome: domain.TransitionOutcomeProgressing,
	}
	return status
}

func blockedPostCommitProjectionStatus() domain.GroupStatus {
	status := healthyEngineGroupProjectionStatus(3)
	status.Traffic.Observed.Admitted = status.Traffic.Observed.Admitted[:2]
	status.Transition = &domain.TransitionStatus{
		Spec: domain.TransitionSpec{
			ID:                     "grow-one",
			BaseTopologyGeneration: 1,
			Plan: domain.ResolvedPlan{
				Change: domain.ResolvedChange{
					Kind: domain.PlanKindGrow,
					Grow: &domain.GrowChange{
						Replicas: []domain.ReplicaTarget{{
							ReplicaID: "replica-2",
							SlotID:    "slot-2",
						}},
					},
				},
			},
		},
		Outcome: domain.TransitionOutcomeBlocked,
	}
	return status
}

func engineGroupTestPlan(change domain.ResolvedChange) domain.ResolvedPlan {
	return domain.ResolvedPlan{
		ID:                      "plan-1",
		ProfileFingerprint:      "profile-v1",
		ProcessLifecycleOwner:   domain.ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      domain.TrafficRequirementQuiesceGroup,
		VerificationRequirement: domain.VerificationRequirementRequired,
		Change:                  change,
	}
}

func engineGroupTestReplicaTarget(replicaID, slotID, nativeMember string) domain.ReplicaTarget {
	return domain.ReplicaTarget{
		ReplicaID:     domain.ReplicaID(replicaID),
		SlotID:        domain.CapacitySlotID(slotID),
		Bootstrap:     domain.BootstrapModeJoin,
		NativeMembers: []domain.NativeMemberID{domain.NativeMemberID(nativeMember)},
	}
}

func engineGroupTestMembership(replicaID, runtimeID, nativeMember string) domain.ReplicaMembership {
	return domain.ReplicaMembership{
		ReplicaID: domain.ReplicaID(replicaID),
		Members:   []domain.NativeMemberIncarnation{{ID: domain.NativeMemberID(nativeMember), RuntimeIncarnation: domain.RuntimeIncarnationID(runtimeID)}},
	}
}

func engineGroupTestIncarnation(
	replicaID string,
	slotID string,
	runtimeID string,
	podName string,
	podUID string,
) domain.ReplicaIncarnation {
	return domain.ReplicaIncarnation{
		ReplicaID:    domain.ReplicaID(replicaID),
		SlotID:       domain.CapacitySlotID(slotID),
		Members:      []domain.NativeMemberIncarnation{{ID: domain.NativeMemberID("dp-" + strings.TrimPrefix(replicaID, "replica-")), RuntimeIncarnation: domain.RuntimeIncarnationID(runtimeID)}},
		CapacityRefs: []domain.CapacityRef{{Name: podName, UID: domain.PodUID(podUID)}},
	}
}
