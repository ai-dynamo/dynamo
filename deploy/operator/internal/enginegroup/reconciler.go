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
// A changed absolute target or membership target is returned before its adapter is called, so the caller can persist
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

	// Recover an acknowledgment already present in durable observations before validating terminal state after restart.
	promoteAcceptedTargets(&next)
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
	if err != nil {
		return result, err
	}
	if handled {
		if maintainsAcceptedTargets(result.Status) {
			return c.reconcileMaintainedTargets(ctx, groupID, result.Status)
		}
		return result, nil
	}
	return c.reconcileActiveTransition(ctx, groupID, result.Status, topology)
}

func maintainsAcceptedTargets(status GroupStatus) bool {
	return status.Transition != nil &&
		(status.Transition.Outcome == TransitionOutcomeCompleted ||
			status.Transition.Outcome == TransitionOutcomeRolledBack ||
			status.Transition.Outcome == TransitionOutcomeBlocked)
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
	transitionID := ""
	if status.Membership.Desired != nil {
		transitionID = status.Membership.Desired.TransitionID
	}
	membership, err := c.membership.Observe(ctx, groupID, transitionID)
	if err != nil {
		return MembershipTopology{}, fmt.Errorf("observe membership: %w", err)
	}
	if err := validateMembershipObservation(membership, status.Membership.Desired); err != nil {
		return MembershipTopology{}, fmt.Errorf("validate membership observation: %w", err)
	}
	if err := validateMembershipObservationEvolution(status.Membership.Observed, membership); err != nil {
		return MembershipTopology{}, fmt.Errorf("validate membership observation evolution: %w", err)
	}

	// Surface fresh physical and traffic truth without making either observation the authority for membership.
	status.Capacity.Observed = cloneCapacityObservation(capacity)
	status.Traffic.Observed = cloneTrafficObservation(traffic)
	status.Membership.Observed = cloneMembershipObservation(membership)
	if err := validateObservedRevisions(*status); err != nil {
		return MembershipTopology{}, err
	}

	// Retain the payload behind each adapter-acknowledged revision before a newer desired target can replace it.
	promoteAcceptedTargets(status)
	if err := validateAcceptedTargets(*status); err != nil {
		return MembershipTopology{}, err
	}
	return cloneTopology(membership.CommittedTopology), nil
}

func promoteAcceptedTargets(status *GroupStatus) {
	if status.Capacity.Desired != nil &&
		status.Capacity.Observed.AppliedRevision == status.Capacity.Desired.ControlRevision {
		status.Capacity.Accepted = cloneCapacityTarget(status.Capacity.Desired)
	}
	if status.Traffic.Desired != nil &&
		status.Traffic.Observed.AppliedRevision == status.Traffic.Desired.ControlRevision {
		status.Traffic.Accepted = cloneTrafficTarget(status.Traffic.Desired)
	}
}

func (c *Coordinator) reconcileDesiredPlan(
	status GroupStatus,
	desiredPlan *ResolvedPlan,
	observedTopology MembershipTopology,
) (handled bool, result ReconcileResult, err error) {
	// Finish interpreting a recovered membership result before considering a replacement plan. In particular, a
	// correlated rejection must restore preparatory capacity and traffic state even when a newer plan is already desired.
	if recovered, recoveredStatus := c.recoverMembershipAuthority(status, observedTopology); recovered {
		return true, ReconcileResult{Status: recoveredStatus, Requeue: true}, nil
	}

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

func (c *Coordinator) recoverMembershipAuthority(
	status GroupStatus,
	observedTopology MembershipTopology,
) (bool, GroupStatus) {
	if status.Transition == nil || status.Transition.Outcome != TransitionOutcomeBlocked ||
		status.Transition.Failure == nil ||
		status.Transition.Failure.Reason != "UnknownMembershipOutcome" ||
		status.Membership.Observed.Transition == nil {
		return false, status
	}

	observed := status.Membership.Observed.Transition
	switch observed.Phase {
	case MembershipTransitionPhasePending, MembershipTransitionPhaseCommitted:
		status.Transition.Outcome = TransitionOutcomeProgressing
		status.Transition.Failure = nil
		status.Transition.UpdatedAt = c.now()
		return true, status
	case MembershipTransitionPhaseRejected:
		base, found := status.Topologies.Snapshot(status.Transition.Spec.BaseTopologyGeneration)
		if !found || !sameTopology(base, observedTopology) {
			return false, status
		}
		c.beginRollback(&status, *observed.Failure)
		return true, status
	default:
		return false, status
	}
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

	// Obtain durable adapter approval for the complete resolved semantics before any external prework.
	ready, persist, err := c.reconcilePlanPreflight(ctx, groupID, &next, base)
	if err != nil || persist || !ready {
		return ReconcileResult{Status: next, Requeue: err == nil}, err
	}

	// Reserve stable logical slots only after authoritative plan validation succeeds.
	ready, persist, err = c.reconcileReplicaReservations(&next, resolution)
	if err != nil || persist || !ready {
		return ReconcileResult{Status: next, Requeue: err == nil}, err
	}
	return c.reconcilePreparedTransition(ctx, groupID, next, base, topology, resolution)
}

func (c *Coordinator) reconcilePreparedTransition(
	ctx context.Context,
	groupID GroupID,
	next GroupStatus,
	base MembershipTopology,
	topology MembershipTopology,
	resolution planResolution,
) (ReconcileResult, error) {
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

	membershipTransition := next.Membership.Observed.Transition
	if membershipTransition == nil || membershipTransition.ResultTopology == nil {
		return ReconcileResult{Status: next}, errors.New("committed membership result is absent")
	}
	committed, found := next.Topologies.Snapshot(membershipTransition.ResultTopology.Generation)
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
	plan = normalizeResolvedPlan(plan)
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

	now := c.now()
	status.Transition = &TransitionStatus{
		Spec: TransitionSpec{
			ID:                     transitionID(plan.ID, base.Generation),
			BaseTopologyGeneration: base.Generation,
			Plan:                   cloneResolvedPlan(plan),
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

func membershipTransitionID(planID string, baseGeneration int64) string {
	return fmt.Sprintf("%s@%d/membership", planID, baseGeneration)
}

func canReplaceBlockedTransition(status GroupStatus) bool {
	if status.Transition.Outcome != TransitionOutcomeBlocked {
		return false
	}
	expectedTransitionID := membershipTransitionID(
		status.Transition.Spec.Plan.ID,
		status.Transition.Spec.BaseTopologyGeneration,
	)
	if status.Membership.Desired != nil &&
		status.Membership.Desired.TransitionID == expectedTransitionID &&
		status.Membership.Observed.Transition != nil &&
		(status.Membership.Observed.Transition.Phase == MembershipTransitionPhasePending ||
			status.Membership.Observed.Transition.Phase == MembershipTransitionPhaseUnknown) {
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
	next, err := nextControlRevisionValue(*status)
	if err != nil {
		return 0, err
	}
	status.ControlRevision = next
	return next, nil
}

func nextControlRevisionValue(status GroupStatus) (int64, error) {
	if status.ControlRevision == math.MaxInt64 {
		return 0, errors.New("Engine Group control revision exhausted")
	}
	return status.ControlRevision + 1, nil
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
	left = normalizeResolvedPlan(left)
	right = normalizeResolvedPlan(right)
	if left.ID != right.ID ||
		left.ProfileFingerprint != right.ProfileFingerprint ||
		left.ProcessLifecycleOwner != right.ProcessLifecycleOwner ||
		left.TrafficRequirement != right.TrafficRequirement ||
		left.VerificationRequirement != right.VerificationRequirement {
		return false
	}

	if left.Change.Kind != right.Change.Kind {
		return false
	}
	switch left.Change.Kind {
	case PlanKindGrow:
		return left.Change.Grow != nil && right.Change.Grow != nil &&
			sameReplicaTargets(left.Change.Grow.Replicas, right.Change.Grow.Replicas)
	case PlanKindRetire:
		return left.Change.Retire != nil && right.Change.Retire != nil &&
			slices.Equal(left.Change.Retire.Replicas, right.Change.Retire.Replicas)
	case PlanKindReduceToSurvivors:
		return left.Change.ReduceToSurvivors != nil && right.Change.ReduceToSurvivors != nil &&
			slices.Equal(left.Change.ReduceToSurvivors.Survivors, right.Change.ReduceToSurvivors.Survivors)
	case PlanKindRestore:
		return left.Change.Restore != nil && right.Change.Restore != nil &&
			sameRestorationTargets(left.Change.Restore.Replicas, right.Change.Restore.Replicas)
	case PlanKindRemap:
		return left.Change.Remap != nil && right.Change.Remap != nil &&
			sameNativeMemberships(left.Change.Remap.Membership, right.Change.Remap.Membership)
	default:
		return false
	}
}

func sameReplicaTargets(left, right []ReplicaTarget) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index].ReplicaID != right[index].ReplicaID ||
			left[index].SlotID != right[index].SlotID ||
			left[index].Bootstrap != right[index].Bootstrap ||
			!slices.Equal(
				normalizeNativeMembers(left[index].NativeMembers),
				normalizeNativeMembers(right[index].NativeMembers),
			) {
			return false
		}
	}
	return true
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
		if !found || !sameReplicaTargets(
			[]ReplicaTarget{other.ReplicaTarget},
			[]ReplicaTarget{target.ReplicaTarget},
		) ||
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
