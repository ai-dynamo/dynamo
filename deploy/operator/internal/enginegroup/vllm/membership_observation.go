/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
)

// ErrTopologyUnknown means native status cannot currently establish authoritative committed membership.
// The coordinator must retain capacity and retry observation, never publish the last-known size as current health.
var ErrTopologyUnknown = errors.New("vLLM committed topology is unknown")

// Observe reads native state independently of one requested transition and durably correlates native completion.
// It never dispatches a resize. Temporary cross-rank observation skew is inconclusive, not a commit or rejection.
func (a *MembershipAdapter) Observe(ctx context.Context, groupID enginegroup.GroupID, transitionID string) (enginegroup.MembershipObservation, error) {
	// Initialize only an already-formed, idle world; a missing binding cannot adopt an uncorrelated resize.
	if err := a.validate(groupID); err != nil {
		return enginegroup.MembershipObservation{}, err
	}
	state, snapshot, err := a.load(ctx)
	if err != nil {
		return enginegroup.MembershipObservation{}, err
	}
	if !snapshot.Exists() {
		state, snapshot, err = a.adoptInitial(ctx, snapshot)
		if err != nil {
			return enginegroup.MembershipObservation{}, err
		}
	}
	statuses, err := a.readTopology(ctx, state.Topology, state.Instances)
	if err != nil {
		return enginegroup.MembershipObservation{}, err
	}
	native := statuses[0]
	if state.Target != nil && state.Result == nil {
		state, err = a.observeTransition(ctx, state, snapshot, native)
		if err != nil {
			return enginegroup.MembershipObservation{}, err
		}
	} else if err := validateNativeBaseline(state, native); err != nil {
		return enginegroup.MembershipObservation{}, err
	}
	observation := enginegroup.MembershipObservation{
		CommittedTopology: state.Topology, RequestedTransitionID: transitionID,
	}
	if transitionID == "" || state.Target == nil || state.Target.TransitionID != transitionID {
		return observation, nil
	}
	if state.Result != nil {
		observation.Transition = state.Result
		return observation, nil
	}

	// A durable target remains Pending even if the first fan-out never reached rank zero.
	// Reassertion can then repair missing deliveries without inventing a competing native operation.
	observation.Transition = transitionResult(*state.Target, enginegroup.MembershipTransitionPhasePending)
	if native.Phase == PhaseFailed && native.OperationID != nil && *native.OperationID == transitionID {
		observation.Transition.Phase = enginegroup.MembershipTransitionPhaseUnknown
		observation.Transition.Failure = terminalFailure("NativeResizeFailed", *native.Error)
	}
	return observation, nil
}

func (a *MembershipAdapter) observeTransition(
	ctx context.Context,
	state membershipBinding,
	snapshot kubejournal.Snapshot,
	native Status,
) (membershipBinding, error) {
	target := *state.Target
	if native.OperationID == nil || *native.OperationID != target.TransitionID {
		if state.Epoch != "" {
			return state, fmt.Errorf("vLLM forgot or superseded a previously observed native operation")
		}
		return state, validateNativeBaseline(state, native)
	}
	size, failure := a.resolvePlan(target.BaseTopology, target.Plan)
	if failure != nil || native.RequestedDataParallelSize == nil || *native.RequestedDataParallelSize != size {
		return state, fmt.Errorf("vLLM native operation does not match the durable target")
	}
	if state.Epoch != "" && state.Epoch != *native.Epoch {
		return state, fmt.Errorf("vLLM changed the native epoch of the accepted operation")
	}
	if state.Epoch == "" {
		state.Epoch = *native.Epoch
		var err error
		snapshot, err = a.Journal.Save(ctx, snapshot, state)
		if err != nil {
			return state, err
		}
	}
	if !native.TopologyKnown || native.Phase == PhaseCommitting {
		return state, ErrTopologyUnknown
	}
	if native.Phase != PhaseCompleted {
		if native.EffectiveDataParallelSize != target.BaseTopology.NativeMemberCount() {
			return state, fmt.Errorf("vLLM pending or failed operation no longer reports its exact base")
		}
		return state, nil
	}

	// Completion must match the exact frozen result and every surviving/joining API process before promotion.
	result := resultTopology(target)
	statuses, err := a.readTopology(ctx, result, nil)
	if err != nil {
		return state, err
	}
	if !sameNativeStatus(native, statuses[0]) {
		return state, fmt.Errorf("vLLM result is not yet observed consistently by all committed ranks")
	}
	for rank := 0; rank < min(len(state.Instances), len(statuses)); rank++ {
		if statuses[rank].InstanceID != state.Instances[rank] {
			return state, fmt.Errorf("vLLM retained DP rank %d changed API process during commit", rank)
		}
	}
	state.Topology, state.Instances = result, nativeInstances(statuses)
	state.NativeOperationID, state.NativeEpoch = target.TransitionID, state.Epoch
	state.Result = transitionResult(target, enginegroup.MembershipTransitionPhaseCommitted)
	state.Result.ResultTopology = &result
	if _, err := a.Journal.Save(ctx, snapshot, state); err != nil {
		return state, err
	}
	return state, nil
}

func (a *MembershipAdapter) adoptInitial(ctx context.Context, snapshot kubejournal.Snapshot) (membershipBinding, kubejournal.Snapshot, error) {
	primary, err := a.process(0)
	if err != nil {
		return membershipBinding{}, snapshot, err
	}
	native, err := primary.Client.Observe(ctx)
	if err != nil {
		return membershipBinding{}, snapshot, err
	}
	if native.Phase != PhaseIdle || !native.TopologyKnown {
		return membershipBinding{}, snapshot, fmt.Errorf("cannot initialize a vLLM binding from non-idle native state")
	}
	size, failure := a.validateSize(native.EffectiveDataParallelSize)
	if failure != nil {
		return membershipBinding{}, snapshot, fmt.Errorf("cannot adopt vLLM topology: %s", failure.Message)
	}
	topology := enginegroup.MembershipTopology{Generation: 1}
	for rank := int32(0); rank < size; rank++ {
		process, err := a.process(rank)
		if err != nil {
			return membershipBinding{}, snapshot, err
		}
		topology.Replicas = appendMemberSubset(topology.Replicas, process.Membership)
	}
	for _, allocation := range topology.Replicas {
		if len(allocation.Members) != int(a.Layout.NativeMembersPerReplica) {
			return membershipBinding{}, snapshot, fmt.Errorf("initial vLLM world does not match the profile's complete allocation width")
		}
	}
	statuses, err := a.readTopology(ctx, topology, nil)
	if err != nil {
		return membershipBinding{}, snapshot, err
	}
	if !sameNativeStatus(native, statuses[0]) {
		return membershipBinding{}, snapshot, fmt.Errorf("vLLM topology changed during initial adoption")
	}
	state := membershipBinding{
		Version: membershipBindingVersion, ProfileFingerprint: a.ProfileFingerprint,
		Topology: topology, Instances: nativeInstances(statuses),
	}
	snapshot, err = a.Journal.Save(ctx, snapshot, state)
	return state, snapshot, err
}

func (a *MembershipAdapter) validateBase(ctx context.Context, base enginegroup.MembershipTopology) error {
	state, snapshot, err := a.load(ctx)
	if err != nil {
		return err
	}
	if !snapshot.Exists() || !enginegroup.SameTopology(base, state.Topology) || (state.Target != nil && state.Result == nil) {
		return fmt.Errorf("vLLM preflight requires the current bound topology without another unresolved target")
	}
	statuses, err := a.readTopology(ctx, state.Topology, state.Instances)
	if err != nil {
		return err
	}
	return validateNativeBaseline(state, statuses[0])
}

func (a *MembershipAdapter) readTopology(ctx context.Context, topology enginegroup.MembershipTopology, instances []string) ([]Status, error) {
	byRank, err := membersByRank(topology)
	if err != nil {
		return nil, err
	}
	processes := make([]Process, len(byRank))
	for rank, membership := range byRank {
		process, err := a.process(int32(rank))
		if err != nil || !reflect.DeepEqual(process.Membership, membership) {
			return nil, fmt.Errorf("vLLM DP rank %d no longer matches its frozen process incarnation", rank)
		}
		processes[rank] = process
	}

	// Bound the complete observation by one native read budget, rather than serializing a timeout per rank.
	statuses := make([]Status, len(processes))
	failures := make([]error, len(processes))
	var pending sync.WaitGroup
	for rank, process := range processes {
		pending.Go(func() {
			statuses[rank], failures[rank] = process.Client.Observe(ctx)
		})
	}
	pending.Wait()
	if err := errors.Join(failures...); err != nil {
		return nil, err
	}
	seen := make(map[string]bool, len(statuses))
	for rank, status := range statuses {
		if seen[status.InstanceID] || (instances != nil && status.InstanceID != instances[rank]) {
			return nil, fmt.Errorf("vLLM DP rank %d changed or duplicated a pinned API process", rank)
		}
		if !sameNativeStatus(statuses[0], status) {
			return nil, fmt.Errorf("vLLM rank observations do not yet agree on the same native operation and topology")
		}
		seen[status.InstanceID] = true
	}
	return statuses, nil
}

func validateNativeBaseline(state membershipBinding, native Status) error {
	if !native.TopologyKnown {
		return ErrTopologyUnknown
	}
	if native.EffectiveDataParallelSize != state.Topology.NativeMemberCount() {
		return fmt.Errorf("vLLM effective size differs from the committed binding")
	}
	if state.NativeOperationID == "" && native.Phase == PhaseIdle {
		return nil
	}
	if native.Phase != PhaseCompleted || native.OperationID == nil || *native.OperationID != state.NativeOperationID ||
		native.Epoch == nil || *native.Epoch != state.NativeEpoch {
		return fmt.Errorf("vLLM reports an unrelated or unresolved native operation")
	}
	return nil
}

func sameNativeStatus(left, right Status) bool {
	left.InstanceID, right.InstanceID = "", ""
	return reflect.DeepEqual(left, right)
}

func nativeInstances(statuses []Status) []string {
	instances := make([]string, len(statuses))
	for rank, status := range statuses {
		instances[rank] = status.InstanceID
	}
	return instances
}

func resultTopology(target enginegroup.MembershipTarget) enginegroup.MembershipTopology {
	result := enginegroup.MembershipTopology{Generation: target.BaseTopology.Generation + 1}
	retiring := make(map[enginegroup.ReplicaID]bool)
	if retire := target.Plan.Change.Retire; retire != nil {
		for _, id := range retire.Replicas {
			retiring[id] = true
		}
	}
	for _, membership := range target.BaseTopology.Replicas {
		if !retiring[membership.ReplicaID] {
			membership.Members = slices.Clone(membership.Members)
			result.Replicas = append(result.Replicas, membership)
		}
	}
	for _, membership := range target.Joining {
		membership.Members = slices.Clone(membership.Members)
		result.Replicas = append(result.Replicas, membership)
	}
	return result
}
