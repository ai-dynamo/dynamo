/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"fmt"
	"reflect"
	"slices"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"k8s.io/utils/ptr"
)

// WorkerTraffic observes autonomous registration and native graceful retirement.
// The journal pins requested withdrawals and native API lifetimes before membership submission.
// vLLM owns admission closure and drain; this adapter never grants serving permission.
type WorkerTraffic struct {
	Membership *MembershipAdapter
	Journal    kubejournal.Store
}

type workerTrafficState struct {
	Target    enginegroup.TrafficTarget
	Instances map[enginegroup.NativeMemberID]string
	Drained   []enginegroup.ReplicaMembership
}

// Apply durably accepts withdrawal intent without running a second engine drain protocol.
func (a *WorkerTraffic) Apply(ctx context.Context, groupID enginegroup.GroupID, target enginegroup.TrafficTarget) (enginegroup.ApplyResult, error) {
	// Fence the complete absolute target before observing or retaining any process identity.
	if groupID == "" || groupID != enginegroup.GroupID(a.Journal.Owner.UID) || target.ControlRevision <= 0 || target.TopologyGeneration <= 0 {
		return enginegroup.ApplyResult{Rejection: terminalFailure("InvalidTrafficTarget", "traffic requires the exact group and positive revision and topology generation")}, nil
	}
	state := workerTrafficState{}
	snapshot, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	if snapshot.Exists() && (target.ControlRevision < state.Target.ControlRevision ||
		target.ControlRevision == state.Target.ControlRevision && !reflect.DeepEqual(target, state.Target)) {
		return enginegroup.ApplyResult{Rejection: terminalFailure("ConflictingTrafficRevision", "traffic revision is stale or has another payload")}, nil
	}
	if snapshot.Exists() && target.ControlRevision == state.Target.ControlRevision {
		return enginegroup.ApplyResult{}, nil
	}

	// A continuing withdrawal keeps its original native identity even after the member leaves topology.
	binding, _, err := a.Membership.load(ctx)
	if err != nil {
		return enginegroup.ApplyResult{}, err
	}
	drains, err := memberDrainTargets(target.Drain)
	if err != nil {
		return enginegroup.ApplyResult{Rejection: terminalFailure("DrainUnsupported", err.Error())}, nil
	}
	next := workerTrafficState{Target: target, Instances: make(map[enginegroup.NativeMemberID]string)}
	for _, drain := range drains {
		membership := drain.Membership
		id := membership.Members[0].ID
		prior := slices.ContainsFunc(state.Target.Drain, func(previous enginegroup.TrafficDrainTarget) bool {
			return previous.Mode == drain.Mode && containsMemberSubset([]enginegroup.ReplicaMembership{previous.Membership}, membership)
		})
		if prior && state.Target.TransitionID == target.TransitionID && state.Instances[id] != "" {
			next.Instances[id] = state.Instances[id]
			if containsMemberSubset(state.Drained, membership) {
				next.Drained = appendMemberSubset(next.Drained, membership)
			}
			continue
		}
		rank, err := nativeRank(membership.Members[0].ID)
		if err != nil || int(rank) >= len(binding.Instances) || !containsMemberSubset(binding.Topology.Replicas, membership) {
			return enginegroup.ApplyResult{Rejection: terminalFailure("UnboundWithdrawal", "withdrawal does not identify a current native process")}, nil
		}
		next.Instances[id] = binding.Instances[rank]
	}

	// Accepted intent survives an operator restart; only correlated native completion creates drain proof.
	_, err = a.Journal.Save(ctx, snapshot, next)
	return enginegroup.ApplyResult{}, err
}

// Observe reports real registration and retains terminal retirement evidence until the exact allocation is released.
func (a *WorkerTraffic) Observe(ctx context.Context, groupID enginegroup.GroupID) (enginegroup.TrafficObservation, error) {
	// Native membership remains authoritative independently of traffic-target acceptance.
	membership, err := a.Membership.Observe(ctx, groupID, "")
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	binding, _, err := a.Membership.load(ctx)
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	state := workerTrafficState{}
	snapshot, err := a.Journal.Load(ctx, &state)
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	observed := enginegroup.TrafficObservation{}
	if snapshot.Exists() {
		observed.AppliedRevision = state.Target.ControlRevision
	}
	observed.Admitted, err = a.observeAdmitted(ctx, membership.CommittedTopology, binding)
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}

	// A missing endpoint is never new drain evidence, but it cannot erase a previously proven retirement.
	changed := false
	drains, err := memberDrainTargets(state.Target.Drain)
	if err != nil {
		return enginegroup.TrafficObservation{}, err
	}
	for _, drain := range drains {
		member := drain.Membership
		id := member.Members[0].ID
		proven := containsMemberSubset(state.Drained, member)
		index := slices.IndexFunc(a.Membership.Processes, func(process Process) bool {
			return slices.ContainsFunc(process.Membership.Members, func(candidate enginegroup.NativeMemberIncarnation) bool { return candidate.ID == id })
		})
		if index < 0 {
			if !proven {
				return enginegroup.TrafficObservation{}, fmt.Errorf("withdrawn process disappeared without terminal drain evidence")
			}
			observed.Drained = appendMemberSubset(observed.Drained, member)
			continue
		}
		process := a.Membership.Processes[index]
		if !containsMemberSubset([]enginegroup.ReplicaMembership{process.Membership}, member) {
			return enginegroup.TrafficObservation{}, fmt.Errorf("withdrawn member changed its process incarnation")
		}
		worker, err := observeWorker(ctx, process)
		if err != nil {
			return enginegroup.TrafficObservation{}, err
		}
		if worker.InstanceID != state.Instances[id] {
			return enginegroup.TrafficObservation{}, fmt.Errorf("withdrawn worker changed its native API lifetime")
		}
		if worker.Admitted {
			if proven {
				return enginegroup.TrafficObservation{}, fmt.Errorf("retired process became admitted after terminal drain")
			}
			observed.Admitted = appendMemberSubset(observed.Admitted, member)
			continue
		}
		// A terminal record belongs to this incarnation, not to the world's next resize operation.
		if proven {
			if !worker.Retired || !worker.Withdrawn || !worker.NativeDrainEnabled {
				return enginegroup.TrafficObservation{}, fmt.Errorf("retired process lost its terminal native evidence")
			}
			observed.Drained = appendMemberSubset(observed.Drained, member)
			continue
		}
		complete, err := a.retirementComplete(ctx, state, binding, process, worker)
		if err != nil {
			return enginegroup.TrafficObservation{}, err
		}
		if !complete {
			observed.Draining = appendMemberSubset(observed.Draining, member)
			continue
		}
		observed.Drained = appendMemberSubset(observed.Drained, member)
		state.Drained = appendMemberSubset(state.Drained, member)
		changed = true
	}
	if changed {
		if _, err := a.Journal.Save(ctx, snapshot, state); err != nil {
			return enginegroup.TrafficObservation{}, err
		}
	}
	return observed, nil
}

func (a *WorkerTraffic) observeAdmitted(ctx context.Context, topology enginegroup.MembershipTopology, binding membershipBinding) ([]enginegroup.ReplicaMembership, error) {
	// Registration requires fresh serving evidence for the exact committed native topology.
	members, err := membersByRank(topology)
	if err != nil {
		return nil, err
	}
	var admitted []enginegroup.ReplicaMembership
	for _, member := range members {
		process, err := a.Membership.processForMember(member.Members[0].ID)
		if err != nil {
			return nil, err
		}
		worker, err := observeWorker(ctx, process)
		if err != nil {
			return nil, err
		}
		if !worker.Admitted {
			continue
		}
		if !worker.Verified || !worker.ListenerReady || worker.Binding == nil ||
			worker.Binding.InstanceID != binding.Instances[process.Rank] ||
			ptr.Deref(worker.Binding.OperationID, "") != binding.NativeOperationID ||
			ptr.Deref(worker.Binding.Epoch, "") != binding.NativeEpoch ||
			worker.Binding.DataParallelSize != topology.NativeMemberCount() {
			return nil, fmt.Errorf("registered worker lacks current native serving evidence")
		}
		admitted = appendMemberSubset(admitted, member)
	}
	return admitted, nil
}

func (a *WorkerTraffic) retirementComplete(ctx context.Context, state workerTrafficState, binding membershipBinding, process Process, worker workerStatus) (bool, error) {
	// The completion barrier is meaningful only for this exact planned retirement with draining enabled.
	if binding.Target == nil || binding.Result == nil || binding.Result.Phase != enginegroup.MembershipTransitionPhaseCommitted ||
		binding.Target.Plan.Change.Kind != enginegroup.PlanKindRetire ||
		state.Target.TransitionID != fmt.Sprintf("%s@%d", binding.Target.Plan.ID, binding.Target.BaseTopology.Generation) ||
		!worker.Retired || !worker.Withdrawn || !worker.NativeDrainEnabled {
		return false, nil
	}
	native, err := process.Client.Observe(ctx)
	if err != nil {
		return false, err
	}
	return native.InstanceID == worker.InstanceID &&
		native.Phase == PhaseCompleted && native.TopologyKnown &&
		ptr.Deref(native.OperationID, "") == binding.NativeOperationID &&
		ptr.Deref(native.Epoch, "") == binding.NativeEpoch &&
		native.EffectiveDataParallelSize == binding.Topology.NativeMemberCount() &&
		process.Rank >= native.EffectiveDataParallelSize, nil
}

func memberDrainTargets(drains []enginegroup.TrafficDrainTarget) ([]enginegroup.TrafficDrainTarget, error) {
	// Native retirement evidence is per process even when one allocation contains several members.
	var targets []enginegroup.TrafficDrainTarget
	seen := make(map[enginegroup.NativeMemberID]bool)
	for _, drain := range drains {
		if drain.Mode != enginegroup.TrafficDrainModeGraceful || drain.Membership.ReplicaID == "" || len(drain.Membership.Members) == 0 {
			return nil, fmt.Errorf("vLLM planned retirement requires graceful withdrawals of exact native members")
		}
		for _, member := range drain.Membership.Members {
			if member.ID == "" || member.RuntimeIncarnation == "" || seen[member.ID] {
				return nil, fmt.Errorf("vLLM withdrawal contains an invalid or duplicate process identity")
			}
			seen[member.ID] = true
			targets = append(targets, enginegroup.TrafficDrainTarget{Mode: drain.Mode, Membership: enginegroup.ReplicaMembership{
				ReplicaID: drain.Membership.ReplicaID, Members: []enginegroup.NativeMemberIncarnation{member},
			}})
		}
	}
	return targets, nil
}
