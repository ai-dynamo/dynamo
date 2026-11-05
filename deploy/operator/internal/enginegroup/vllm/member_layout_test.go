/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"fmt"
	"net/http"
	"testing"

	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/stretchr/testify/require"
)

func TestReplicaLayoutKeepsAllocationAndNativeScaleUnitsSeparate(t *testing.T) {
	for _, width := range []int32{1, 4} {
		t.Run(fmt.Sprintf("width-%d", width), func(t *testing.T) {
			t.Log("convert three allocations to their native target without rounding or changing scale units")
			layout := ReplicaLayout{NativeMembersPerReplica: width}
			size, err := layout.NativeSize(3)
			require.NoError(t, err)
			require.Equal(t, 3*width, size)

			t.Log("assign the third allocation its complete immutable rank range")
			members, err := layout.Members(2)
			require.NoError(t, err)
			require.Len(t, members, int(width))
			require.Equal(t, domain.NativeMemberID(fmt.Sprintf("dp-%d", 2*width)), members[0])
			require.Equal(t, domain.NativeMemberID(fmt.Sprintf("dp-%d", 3*width-1)), members[len(members)-1])
		})
	}
}

func TestPackedNativeResizeAndDrainArePerMemberButReleaseIsPerAllocation(t *testing.T) {
	t.Log("represent two allocations as eight independent native processes sharing two Pod UIDs")
	fixture := newMembershipFixture(t, 8, 12)
	fixture.adapter.Layout = ReplicaLayout{NativeMembersPerReplica: 4}
	fixture.adapter.ManagedDrainValidator = ValidateWorkerManagedDrain
	workers := &workerControlFixture{
		engine: fixture.engine, registered: make(map[int32]bool), podUIDs: make(map[int32]domain.PodUID),
		pendingWithdrawals: make(map[int32]bool),
	}
	for i := range fixture.adapter.Processes {
		process := &fixture.adapter.Processes[i]
		process.Membership.ReplicaID = domain.ReplicaID(fmt.Sprintf("replica-%d", i/4))
		process.PodUID = domain.PodUID(fmt.Sprintf("pod-%d", i/4))
		workers.podUIDs[int32(i)] = process.PodUID
		workers.registered[int32(i)] = i < 8
		control, err := NewWorkerClient(fmt.Sprintf("http://rank-%d.test", i), &http.Client{Transport: workers})
		require.NoError(t, err)
		process.Client = control
	}
	groupID := domain.GroupID("group-uid")
	base, err := fixture.adapter.Observe(t.Context(), groupID, "")
	require.NoError(t, err)
	require.Equal(t, int32(2), base.CommittedTopology.ReplicaCount())
	require.Equal(t, int32(8), base.CommittedTopology.NativeMemberCount())
	capacity := &workerTestCapacity{processes: fixture.adapter.Processes, count: 2, width: 4}
	physical, err := capacity.Observe(t.Context(), groupID)
	require.NoError(t, err)
	traffic := &WorkerTraffic{Membership: fixture.adapter, Journal: kubejournal.NewStore(fixture.adapter.Journal.Client, "test", "group", "group-uid", "traffic")}
	registered, err := traffic.Observe(t.Context(), groupID)
	require.NoError(t, err)
	status, err := domain.NewGroupStatus(base.CommittedTopology, physical, registered)
	require.NoError(t, err)
	grow := fixture.target(t, base.CommittedTopology, "grow-packed", 10, 12, domain.PlanKindGrow).Plan
	require.Len(t, grow.Change.Grow.Replicas, 1)
	require.Len(t, grow.Change.Grow.Replicas[0].NativeMembers, 4)
	verifier := &WorkerVerifier{Membership: fixture.adapter}

	t.Log("one allocation increment submits DP12 rather than DP9")
	for range 40 {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &grow, status)
		require.NoError(t, err)
		status = result.Status
		if len(fixture.engine.requests[0]) != 0 {
			break
		}
	}
	require.Equal(t, int32(3), capacity.count)
	require.NotEmpty(t, fixture.engine.requests[0])
	require.Equal(t, int32(12), fixture.engine.requests[0][0].NewDataParallelSize)
	fixture.engine.complete(12)
	for rank := int32(8); rank < 12; rank++ {
		workers.registered[rank] = true
	}
	for range 30 {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &grow, status)
		require.NoError(t, err)
		status = result.Status
		if status.Transition.Outcome == domain.TransitionOutcomeCompleted {
			break
		}
	}
	require.Equal(t, domain.TransitionOutcomeCompleted, status.Transition.Outcome)

	t.Log("retire one allocation by shrinking four native members, retaining every process's drain identity")
	current, found := status.Topologies.Current()
	require.True(t, found)
	retire := fixture.target(t, current, "retire-packed", 100, 8, domain.PlanKindRetire).Plan
	retire.TrafficRequirement = domain.TrafficRequirementEngineManagedDrain
	for range 35 {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &retire, status)
		require.NoError(t, err)
		status = result.Status
		if len(fixture.engine.requests[11]) != 0 {
			break
		}
	}
	require.NotEmpty(t, fixture.engine.requests[11])
	fixture.engine.complete(8)
	for rank := int32(8); rank < 12; rank++ {
		workers.registered[rank] = false
	}
	workers.pendingWithdrawals[11] = true
	for range 10 {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &retire, status)
		require.NoError(t, err)
		status = result.Status
	}
	require.Equal(t, int32(3), capacity.count, "three drained members cannot authorize deleting their shared Pod")
	require.Len(t, status.Traffic.Observed.Drained, 1)
	require.Len(t, status.Traffic.Observed.Drained[0].Members, 3)
	require.Len(t, status.Traffic.Observed.Draining, 1)
	require.Len(t, status.Traffic.Observed.Draining[0].Members, 1)

	t.Log("a restart retains the first three proofs; the fourth permits exactly one Pod release")
	workers.pendingWithdrawals[11] = false
	traffic = &WorkerTraffic{Membership: fixture.adapter, Journal: traffic.Journal}
	for range 30 {
		result, err := domain.NewCoordinator(capacity, fixture.adapter, traffic, verifier).Reconcile(t.Context(), groupID, &retire, status)
		require.NoError(t, err)
		status = result.Status
		if status.Transition.Outcome == domain.TransitionOutcomeCompleted {
			break
		}
	}
	require.Equal(t, domain.TransitionOutcomeCompleted, status.Transition.Outcome)
	require.Equal(t, int32(2), capacity.count)
	require.Len(t, capacity.releases, 1)
	require.Equal(t, []domain.CapacityRef{{Name: "pod-2", UID: "pod-2"}}, capacity.releases[0].CapacityRefs)

	t.Log("survivor masking remains a separate, explicitly unsupported native API capability")
	current, _ = status.Topologies.Current()
	retire.Change = domain.ResolvedChange{Kind: domain.PlanKindReduceToSurvivors, ReduceToSurvivors: &domain.ReduceToSurvivorsChange{}}
	retire.TrafficRequirement = domain.TrafficRequirementKeepServing
	preflight, err := fixture.adapter.ValidatePlan(t.Context(), groupID, domain.PlanValidationRequest{BaseTopology: current, Plan: retire, PlanDigest: "recover-packed"})
	require.NoError(t, err)
	require.NotNil(t, preflight.Rejection)
	require.Equal(t, "UnsupportedOperation", preflight.Rejection.Reason)
}
