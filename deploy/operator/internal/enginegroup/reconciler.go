/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package enginegroup

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"time"
)

// Coordinator reconciles one Engine Group through explicit capacity, traffic, membership, and verification steps.
type Coordinator struct {
	capacity   CapacityAdapter
	membership MembershipAdapter
	traffic    TrafficAdapter
	verifier   ServingVerifier
	now        func() time.Time
}

// NewCoordinator constructs an Engine Group coordinator. Every adapter must be non-nil.
func NewCoordinator(
	capacity CapacityAdapter,
	membership MembershipAdapter,
	traffic TrafficAdapter,
	verifier ServingVerifier,
) *Coordinator {
	return &Coordinator{
		capacity:   capacity,
		membership: membership,
		traffic:    traffic,
		verifier:   verifier,
		now:        time.Now,
	}
}

// Reconcile derives each next action from the complete durable status and fresh external observations.
// A changed absolute target or membership request is returned before its adapter is called, so the caller can persist
// the returned status first. The returned status remains authoritative when a later external call returns an error.
// A nil desiredPlan means no new transition is requested; an already-running transition still converges.
func (c *Coordinator) Reconcile(
	ctx context.Context,
	groupID GroupID,
	desiredPlan *ResolvedPlan,
	status GroupStatus,
) (ReconcileResult, error) {
	next := cloneStatus(status)
	if groupID == "" {
		return ReconcileResult{Status: next}, errors.New("group ID must not be empty")
	}
	if err := validateGroupStatus(next); err != nil {
		return ReconcileResult{Status: next}, fmt.Errorf("validate durable status: %w", err)
	}

	// Observe each independently owned state dimension before deriving the next level-based action.
	topology, err := c.observeGroup(ctx, groupID, &next)
	if err != nil {
		return ReconcileResult{Status: next, Requeue: true}, err
	}

	// Start, retain, or replace the one durable resolved transition before executing its explicit steps.
	handled, result, err := c.reconcileDesiredPlan(next, desiredPlan, topology)
	if err != nil || handled {
		return result, err
	}
	return c.reconcileActiveTransition(ctx, groupID, result.Status, topology)
}

func (c *Coordinator) observeGroup(
	ctx context.Context,
	groupID GroupID,
	status *GroupStatus,
) (MembershipTopology, error) {
	capacity, err := c.capacity.Observe(ctx, groupID)
	if err != nil {
		return MembershipTopology{}, fmt.Errorf("observe capacity: %w", err)
	}
	if err := validateCapacityObservation(capacity); err != nil {
		return MembershipTopology{}, fmt.Errorf("validate capacity observation: %w", err)
	}
	traffic, err := c.traffic.Observe(ctx, groupID)
	if err != nil {
		return MembershipTopology{}, fmt.Errorf("observe traffic: %w", err)
	}
	if err := validateTrafficObservation(traffic); err != nil {
		return MembershipTopology{}, fmt.Errorf("validate traffic observation: %w", err)
	}
	topology, err := c.membership.ObserveTopology(ctx, groupID)
	if err != nil {
		return MembershipTopology{}, fmt.Errorf("observe membership topology: %w", err)
	}
	if err := validateTopology(topology); err != nil {
		return MembershipTopology{}, fmt.Errorf("validate membership topology: %w", err)
	}

	// Surface fresh physical and traffic truth without making either observation the authority for membership.
	status.Capacity.Observed = cloneCapacityObservation(capacity)
	status.Traffic.Observed = cloneTrafficObservation(traffic)
	if err := validateObservedRevisions(*status); err != nil {
		return MembershipTopology{}, err
	}
	return topology, nil
}

func (c *Coordinator) reconcileDesiredPlan(
	status GroupStatus,
	desiredPlan *ResolvedPlan,
	observedTopology MembershipTopology,
) (handled bool, result ReconcileResult, err error) {
	// A completed or safely blocked transition may be replaced only by a distinct explicit plan.
	if status.Transition != nil &&
		(status.Transition.Outcome == TransitionOutcomeCompleted ||
			status.Transition.Outcome == TransitionOutcomeRolledBack ||
			canReplaceBlockedTransition(status)) &&
		desiredPlan != nil && desiredPlan.ID != status.Transition.Spec.Plan.ID {
		status.Transition = nil
		status.Topologies = compactTopologyHistory(status.Topologies)
		status.Registry = compactRegistryHistory(status.Registry)
	}

	// Freeze a new resolved plan and its stable logical slots before creating capacity or changing traffic.
	if status.Transition == nil {
		if desiredPlan == nil {
			if !sameTopologyWithCurrent(status.Topologies, observedTopology) {
				return true, ReconcileResult{Status: status}, errors.New(
					"engine topology changed without a resolved transition",
				)
			}
			return true, ReconcileResult{Status: status}, nil
		}
		started, startErr := c.startTransition(status, observedTopology, *desiredPlan)
		if startErr != nil {
			return true, ReconcileResult{Status: status}, startErr
		}
		return true, ReconcileResult{Status: started, Requeue: true}, nil
	}
	if desiredPlan != nil && desiredPlan.ID == status.Transition.Spec.Plan.ID &&
		!sameResolvedPlan(*desiredPlan, status.Transition.Spec.Plan) {
		return true, ReconcileResult{Status: status}, fmt.Errorf(
			"plan %q changed after becoming durable",
			desiredPlan.ID,
		)
	}
	if status.Transition.Outcome == TransitionOutcomeBlocked ||
		status.Transition.Outcome == TransitionOutcomeCompleted ||
		status.Transition.Outcome == TransitionOutcomeRolledBack {
		return true, ReconcileResult{Status: status}, nil
	}
	return false, ReconcileResult{Status: status}, nil
}

func (c *Coordinator) reconcileActiveTransition(
	ctx context.Context,
	groupID GroupID,
	next GroupStatus,
	topology MembershipTopology,
) (ReconcileResult, error) {

	base, found := next.Topologies.Snapshot(next.Transition.Spec.BaseTopologyGeneration)
	if !found {
		return ReconcileResult{Status: next}, fmt.Errorf(
			"base topology generation %d is absent",
			next.Transition.Spec.BaseTopologyGeneration,
		)
	}
	resolution, err := validateResolvedPlan(base, next.Registry, next.Transition.Spec.Plan)
	if err != nil {
		return ReconcileResult{Status: next}, fmt.Errorf("revalidate durable plan: %w", err)
	}
	if next.Transition.Outcome == TransitionOutcomeReverting {
		return c.reconcileRollback(ctx, groupID, next, base, topology, resolution)
	}

	// Growth and restoration first converge exact named capacity, then persist the observed incarnations in the registry.
	ready, persist, err := c.reconcileJoiningCapacity(ctx, groupID, &next, resolution)
	if err != nil || persist || !ready {
		return ReconcileResult{Status: next, Requeue: err == nil}, err
	}

	// Operation-sensitive traffic withdrawal and drain complete before any membership mutation.
	ready, persist, err = c.reconcilePreMembershipTraffic(ctx, groupID, &next, base, resolution)
	if err != nil || persist || !ready {
		return ReconcileResult{Status: next, Requeue: err == nil}, err
	}

	// Membership is the only external subsystem retaining an asynchronous idempotent operation identity.
	ready, persist, err = c.reconcileMembership(ctx, groupID, &next, base, topology, resolution)
	if err != nil || persist || !ready {
		return ReconcileResult{Status: next, Requeue: err == nil}, err
	}

	committed, found := next.Topologies.Snapshot(next.Transition.Membership.CommittedTopologyGeneration)
	if !found {
		return ReconcileResult{Status: next}, errors.New("committed membership topology is absent from history")
	}

	// Exact UID-bound release may begin only after the engine has excluded the selected logical replicas.
	ready, persist, err = c.reconcileRetiredCapacity(ctx, groupID, &next, base, committed, resolution)
	if err != nil || persist || !ready {
		return ReconcileResult{Status: next, Requeue: err == nil}, err
	}

	// Serving verification is rerunnable and records only a topology-bound result, never another operation journal.
	ready, persist, err = c.reconcileServingVerification(ctx, groupID, &next, committed)
	if err != nil || persist || !ready {
		return ReconcileResult{Status: next, Requeue: err == nil}, err
	}

	// The committed and verified topology becomes routable only through a new revisioned absolute traffic projection.
	ready, persist, err = c.reconcileCommittedTraffic(ctx, groupID, &next, base, committed, resolution)
	if err != nil || persist || !ready {
		return ReconcileResult{Status: next, Requeue: err == nil}, err
	}

	// Completion summarizes the cross-subsystem outcome without changing the independent membership result.
	next.Transition.Outcome = TransitionOutcomeCompleted
	next.Transition.Failure = nil
	next.Transition.UpdatedAt = c.now()
	return ReconcileResult{Status: next}, nil
}

func (c *Coordinator) startTransition(
	status GroupStatus,
	observedTopology MembershipTopology,
	plan ResolvedPlan,
) (GroupStatus, error) {
	base, found := status.Topologies.Current()
	if !found {
		return status, errors.New("current topology is absent")
	}
	if !sameTopology(base, observedTopology) {
		return status, errors.New("cannot start a transition from an unrecognized engine topology")
	}
	resolution, err := validateResolvedPlan(base, status.Registry, plan)
	if err != nil {
		return status, fmt.Errorf("validate resolved plan: %w", err)
	}
	if err := validatePlanCanStart(status.Registry, resolution); err != nil {
		return status, err
	}

	// Reserve fresh or restored stable slots in the canonical registry before asking for physical capacity.
	for _, target := range resolution.joiningTargets {
		if _, found := status.Registry.Find(target.ReplicaID); found {
			continue
		}
		status.Registry.Replicas = append(status.Registry.Replicas, ReplicaRecord{
			ReplicaID: target.ReplicaID,
			SlotID:    target.SlotID,
		})
	}
	if err := validateRegistry(status.Registry); err != nil {
		return status, fmt.Errorf("reserve replica identities: %w", err)
	}

	now := c.now()
	status.Transition = &TransitionStatus{
		Spec: TransitionSpec{
			ID:                     transitionID(plan.ID, base.Generation),
			BaseTopologyGeneration: base.Generation,
			Plan:                   cloneResolvedPlan(plan),
		},
		Membership: MembershipOperationStatus{
			ID:    membershipOperationID(plan.ID, base.Generation),
			Phase: MembershipOperationPhaseNotStarted,
		},
		Outcome:   TransitionOutcomeProgressing,
		StartedAt: now,
		UpdatedAt: now,
	}
	return status, nil
}

func transitionID(planID string, baseGeneration int64) string {
	return fmt.Sprintf("%s@%d", planID, baseGeneration)
}

func membershipOperationID(planID string, baseGeneration int64) string {
	return fmt.Sprintf("%s@%d/membership", planID, baseGeneration)
}

func canReplaceBlockedTransition(status GroupStatus) bool {
	if status.Transition.Outcome != TransitionOutcomeBlocked ||
		status.Transition.Membership.Phase == MembershipOperationPhaseUnknown {
		return false
	}
	current, found := status.Topologies.Current()
	if !found || len(status.Capacity.Observed.Allocations) != len(current.Replicas) {
		return false
	}
	currentRecords := 0
	for _, record := range status.Registry.Replicas {
		if record.Current != nil {
			currentRecords++
		}
	}
	if currentRecords != len(current.Replicas) {
		return false
	}
	for _, membership := range current.Replicas {
		allocation, found := allocationByID(status.Capacity.Observed, membership.ReplicaID)
		record, recorded := status.Registry.Find(membership.ReplicaID)
		if !found || !recorded || record.Current == nil ||
			!sameIncarnation(allocation.Incarnation, *record.Current) ||
			!membershipMatchesIncarnation(membership, *record.Current) {
			return false
		}
	}
	return true
}

func (c *Coordinator) nextControlRevision(status *GroupStatus) (int64, error) {
	if status.ControlRevision == math.MaxInt64 {
		return 0, errors.New("Engine Group control revision exhausted")
	}
	status.ControlRevision++
	return status.ControlRevision, nil
}

func (c *Coordinator) blockTransition(status *GroupStatus, failure Failure) {
	status.Transition.Outcome = TransitionOutcomeBlocked
	status.Transition.Failure = cloneFailure(&failure)
	status.Transition.UpdatedAt = c.now()
}

func (c *Coordinator) beginRollback(status *GroupStatus, failure Failure) {
	status.Transition.Outcome = TransitionOutcomeReverting
	status.Transition.Failure = cloneFailure(&failure)
	status.Transition.UpdatedAt = c.now()
}

func validateObservedRevisions(status GroupStatus) error {
	if status.Capacity.Observed.AppliedRevision > status.ControlRevision {
		return fmt.Errorf(
			"capacity revision %d is ahead of durable control revision %d",
			status.Capacity.Observed.AppliedRevision,
			status.ControlRevision,
		)
	}
	if status.Traffic.Observed.AppliedRevision > status.ControlRevision {
		return fmt.Errorf(
			"traffic revision %d is ahead of durable control revision %d",
			status.Traffic.Observed.AppliedRevision,
			status.ControlRevision,
		)
	}
	return nil
}

func sameTopologyWithCurrent(history TopologyHistory, topology MembershipTopology) bool {
	current, found := history.Current()
	return found && sameTopology(current, topology)
}

func compactTopologyHistory(history TopologyHistory) TopologyHistory {
	current, found := history.Current()
	if !found {
		return history
	}
	return TopologyHistory{
		CurrentGeneration: current.Generation,
		Snapshots:         []MembershipTopology{current},
	}
}

func compactRegistryHistory(registry ReplicaRegistry) ReplicaRegistry {
	registry = cloneRegistry(registry)
	for index := range registry.Replicas {
		history := registry.Replicas[index].History
		if len(history) > 1 {
			registry.Replicas[index].History = history[len(history)-1:]
		}
	}
	return registry
}

func sameResolvedPlan(left, right ResolvedPlan) bool {
	if left.ID != right.ID ||
		left.ProfileFingerprint != right.ProfileFingerprint ||
		left.ProcessLifecycleOwner != right.ProcessLifecycleOwner ||
		left.TrafficRequirement != right.TrafficRequirement ||
		left.RetirementSafety != right.RetirementSafety ||
		left.VerificationRequirement != right.VerificationRequirement {
		return false
	}

	switch leftChange := left.Change.(type) {
	case *GrowChange:
		rightChange, ok := right.Change.(*GrowChange)
		return ok && slices.Equal(leftChange.Replicas, rightChange.Replicas)
	case *RetireChange:
		rightChange, ok := right.Change.(*RetireChange)
		return ok && sameReplicaIDs(leftChange.Replicas, rightChange.Replicas)
	case *ReduceToSurvivorsChange:
		rightChange, ok := right.Change.(*ReduceToSurvivorsChange)
		return ok && sameReplicaIDs(leftChange.Survivors, rightChange.Survivors)
	case *RestoreChange:
		rightChange, ok := right.Change.(*RestoreChange)
		return ok && sameRestorationTargets(leftChange.Replicas, rightChange.Replicas)
	case *RemapChange:
		rightChange, ok := right.Change.(*RemapChange)
		return ok && sameNativeMemberships(leftChange.Membership, rightChange.Membership)
	default:
		return false
	}
}

func sameRestorationTargets(left, right []RestorationTarget) bool {
	if len(left) != len(right) {
		return false
	}
	leftByID := make(map[ReplicaID]RestorationTarget, len(left))
	for _, target := range left {
		leftByID[target.ReplicaID] = target
	}
	for _, target := range right {
		other, found := leftByID[target.ReplicaID]
		if !found || other.ReplicaTarget != target.ReplicaTarget ||
			!slices.Equal(normalizeNativeMembers(other.NativeMembers), normalizeNativeMembers(target.NativeMembers)) {
			return false
		}
	}
	return true
}

func sameNativeMemberships(left, right []ReplicaNativeMembership) bool {
	if len(left) != len(right) {
		return false
	}
	leftByID := make(map[ReplicaID]ReplicaNativeMembership, len(left))
	for _, membership := range left {
		leftByID[membership.ReplicaID] = membership
	}
	for _, membership := range right {
		other, found := leftByID[membership.ReplicaID]
		if !found || other.SlotID != membership.SlotID ||
			!slices.Equal(
				normalizeNativeMembers(other.NativeMembers),
				normalizeNativeMembers(membership.NativeMembers),
			) {
			return false
		}
	}
	return true
}
