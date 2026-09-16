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
	"slices"
	"strings"
	"testing"
	"time"
)

const (
	testReplacementRuntime = RuntimeIncarnationID("runtime-1-v2")
	testReplacementPodUID  = PodUID("pod-uid-1-v2")
)

type testCapacityAdapter struct {
	observation CapacityObservation
	observeErr  error
	planned     map[ReplicaID]ReplicaIncarnation
	firstTarget *CapacityTarget
	lastTarget  *CapacityTarget
	applyCalls  int
	available   bool
	rejectNext  *Failure
	partial     []CapacityAllocation
	events      *[]string
}

func (a *testCapacityAdapter) Observe(context.Context, GroupID) (CapacityObservation, error) {
	if a.observeErr != nil {
		return CapacityObservation{}, a.observeErr
	}
	return cloneCapacityObservation(a.observation), nil
}

func (a *testCapacityAdapter) Apply(
	_ context.Context,
	_ GroupID,
	target CapacityTarget,
) (ApplyResult, error) {
	a.applyCalls++
	*a.events = append(*a.events, fmt.Sprintf("capacity:%d", target.ControlRevision))
	if target.ControlRevision < a.observation.AppliedRevision {
		return rejected("StaleCapacityRevision", "capacity revision is stale"), nil
	}
	sameRevision := target.ControlRevision == a.observation.AppliedRevision
	if sameRevision {
		if a.lastTarget == nil || !sameCapacityTargetIntent(*a.lastTarget, target) {
			return rejected("ConflictingCapacityRevision", "capacity revision payload changed"), nil
		}
	}
	if !sameRevision && a.rejectNext != nil {
		rejection := cloneFailure(a.rejectNext)
		a.rejectNext = nil
		a.observation.Allocations = append(
			a.observation.Allocations,
			cloneCapacityObservation(CapacityObservation{Allocations: a.partial}).Allocations...,
		)
		return ApplyResult{Rejection: rejection}, nil
	}

	// Exact release fences may remove only the concrete Pod UIDs captured by the controller.
	for _, fence := range target.ReleaseFences {
		allocation, found := allocationByID(a.observation, fence.ReplicaID)
		if found && !sameCapacityRefs(allocation.Incarnation.CapacityRefs, fence.CapacityRefs) {
			return rejected("StaleReleaseFence", "release fence no longer identifies current capacity"), nil
		}
	}

	// The fake converges the accepted absolute target immediately; the next reconcile observes the result.
	allocations := make([]CapacityAllocation, 0, len(target.Replicas))
	for _, replica := range target.Replicas {
		if replica.Incarnation != nil {
			allocations = append(allocations, CapacityAllocation{
				Incarnation: cloneReplicaIncarnation(*replica.Incarnation),
				Available:   true,
			})
			continue
		}
		incarnation, found := a.planned[replica.ReplicaID]
		if !found {
			return ApplyResult{}, fmt.Errorf("no planned allocation for replica %q", replica.ReplicaID)
		}
		allocations = append(allocations, CapacityAllocation{
			Incarnation: cloneReplicaIncarnation(incarnation),
			Available:   a.available,
		})
	}
	a.observation.AppliedRevision = target.ControlRevision
	a.observation.Allocations = allocations
	a.observation.ReleaseFences = convergeReleaseFences(a.observation.ReleaseFences, target)
	if a.firstTarget == nil {
		a.firstTarget = cloneCapacityTarget(&target)
	}
	a.lastTarget = cloneCapacityTarget(&target)
	return ApplyResult{}, nil
}

func convergeReleaseFences(existing []ReleaseFence, target CapacityTarget) []ReleaseFence {
	byReplica := make(map[ReplicaID]ReleaseFence, len(existing)+len(target.ReleaseFences))
	for _, fence := range existing {
		byReplica[fence.ReplicaID] = fence
	}
	for _, replica := range target.Replicas {
		delete(byReplica, replica.ReplicaID)
	}
	for _, fence := range target.ReleaseFences {
		byReplica[fence.ReplicaID] = fence
	}

	result := make([]ReleaseFence, 0, len(byReplica))
	for _, fence := range byReplica {
		result = append(result, fence)
	}
	return normalizeReleaseFences(result)
}

type testTrafficAdapter struct {
	observation     TrafficObservation
	observeErr      error
	lastTarget      *TrafficTarget
	applyCalls      int
	autoDrain       bool
	confirmInactive bool
	rejectNext      *Failure
	events          *[]string
}

func (a *testTrafficAdapter) Observe(context.Context, GroupID) (TrafficObservation, error) {
	if a.observeErr != nil {
		return TrafficObservation{}, a.observeErr
	}
	return cloneTrafficObservation(a.observation), nil
}

func (a *testTrafficAdapter) Apply(
	_ context.Context,
	_ GroupID,
	target TrafficTarget,
) (ApplyResult, error) {
	a.applyCalls++
	*a.events = append(*a.events, fmt.Sprintf("traffic:%d", target.ControlRevision))
	if target.ControlRevision < a.observation.AppliedRevision {
		return rejected("StaleTrafficRevision", "traffic revision is stale"), nil
	}
	if target.ControlRevision == a.observation.AppliedRevision {
		if a.lastTarget == nil || !sameTrafficTargetIntent(*a.lastTarget, target) {
			return rejected("ConflictingTrafficRevision", "traffic revision payload changed"), nil
		}
	}
	if target.ControlRevision > a.observation.AppliedRevision && a.rejectNext != nil {
		rejection := cloneFailure(a.rejectNext)
		a.rejectNext = nil
		return ApplyResult{Rejection: rejection}, nil
	}

	// Applying an absolute traffic set withdraws everything else without implicitly admitting engine members.
	a.observation.AppliedRevision = target.ControlRevision
	a.observation.Admitted = cloneReplicaMemberships(target.Admitted)
	a.observation.Draining = nil
	for _, drain := range target.Drain {
		complete := a.autoDrain
		if drain.Mode == TrafficDrainModeConfirmInactive {
			complete = a.confirmInactive
		}
		if complete {
			if !containsMembership(a.observation.Drained, drain.Membership) {
				a.observation.Drained = append(
					a.observation.Drained,
					cloneReplicaMembership(drain.Membership),
				)
			}
			continue
		}
		if drain.Mode == TrafficDrainModeGraceful {
			a.observation.Draining = append(a.observation.Draining, cloneReplicaMembership(drain.Membership))
		}
	}
	a.lastTarget = cloneTrafficTarget(&target)
	return ApplyResult{}, nil
}

type testMembershipAdapter struct {
	topology              MembershipTopology
	observeErr            error
	transitions           map[string]MembershipTransitionObservation
	targets               map[string]MembershipTarget
	lastPlanValidation    *PlanValidationRequest
	planValidationCalls   int
	targetValidationCalls int
	applyCalls            int
	failBeforeFirstAccept bool
	failAfterFirstAccept  bool
	planRejection         *Failure
	targetRejection       *Failure
	applyRejection        *Failure
	events                *[]string
}

func (a *testMembershipAdapter) ValidatePlan(
	_ context.Context,
	_ GroupID,
	request PlanValidationRequest,
) (PreflightResult, error) {
	a.planValidationCalls++
	request.BaseTopology = cloneTopology(request.BaseTopology)
	request.Plan = cloneResolvedPlan(request.Plan)
	a.lastPlanValidation = &request
	if a.planRejection != nil {
		return PreflightResult{Rejection: cloneFailure(a.planRejection)}, nil
	}
	return PreflightResult{Evidence: &ValidationEvidence{
		PlanDigest:           request.PlanDigest,
		ProfileFingerprint:   request.Plan.ProfileFingerprint,
		CapabilityGeneration: "capabilities-v1",
	}}, nil
}

func (a *testMembershipAdapter) ValidateTarget(
	_ context.Context,
	_ GroupID,
	target MembershipTarget,
) (PreflightResult, error) {
	a.targetValidationCalls++
	if a.targetRejection != nil {
		return PreflightResult{Rejection: cloneFailure(a.targetRejection)}, nil
	}
	evidence := target.Validation
	evidence.TargetDigest = target.TargetDigest
	return PreflightResult{Evidence: &evidence}, nil
}

func (a *testMembershipAdapter) Observe(
	_ context.Context,
	_ GroupID,
	transitionID string,
) (MembershipObservation, error) {
	if a.observeErr != nil {
		return MembershipObservation{}, a.observeErr
	}
	observation := MembershipObservation{
		CommittedTopology:     cloneTopology(a.topology),
		RequestedTransitionID: transitionID,
	}
	transition, found := a.transitions[transitionID]
	if !found || transitionID == "" {
		return observation, nil
	}
	observation.Transition = cloneMembershipTransitionObservation(&transition)
	return observation, nil
}

func (a *testMembershipAdapter) Apply(
	_ context.Context,
	_ GroupID,
	target MembershipTarget,
) error {
	a.applyCalls++
	*a.events = append(*a.events, "membership:"+target.TransitionID)
	if a.failBeforeFirstAccept && a.applyCalls == 1 {
		return errors.New("request timed out before acceptance")
	}
	if existing, found := a.targets[target.TransitionID]; found {
		if !sameMembershipTarget(existing, target) {
			return errors.New("membership transition payload changed")
		}
		return nil
	}

	a.targets[target.TransitionID] = *cloneMembershipTarget(&target)
	transition := MembershipTransitionObservation{
		TransitionID:    target.TransitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           MembershipTransitionPhasePending,
	}
	if a.applyRejection != nil {
		transition.Phase = MembershipTransitionPhaseRejected
		transition.Failure = cloneFailure(a.applyRejection)
	}
	a.transitions[target.TransitionID] = transition
	if a.failAfterFirstAccept && a.applyCalls == 1 {
		return errors.New("request timed out after acceptance")
	}
	return nil
}

func (a *testMembershipAdapter) commit(transitionID string, topology MembershipTopology) {
	topology = cloneTopology(topology)
	a.topology = topology
	a.reportCommit(transitionID, topology)
}

func (a *testMembershipAdapter) reportCommit(transitionID string, topology MembershipTopology) {
	topology = cloneTopology(topology)
	target := a.targets[transitionID]
	a.transitions[transitionID] = MembershipTransitionObservation{
		TransitionID:    transitionID,
		ControlRevision: target.ControlRevision,
		TargetDigest:    target.TargetDigest,
		Phase:           MembershipTransitionPhaseCommitted,
		ResultTopology:  &topology,
	}
}

type testServingVerifier struct {
	failure *Failure
	calls   int
	events  *[]string
}

func (v *testServingVerifier) Verify(
	_ context.Context,
	_ GroupID,
	topology MembershipTopology,
) (VerificationResult, error) {
	v.calls++
	*v.events = append(*v.events, fmt.Sprintf("verify:%d", topology.Generation))
	if v.failure != nil {
		return VerificationResult{Failure: cloneFailure(v.failure)}, nil
	}
	return VerificationResult{Proof: &ServingProof{
		TopologyGeneration: topology.Generation,
		RuntimeDigest:      TopologyRuntimeDigest(topology),
		ObservedAt:         time.Unix(100, 0),
	}}, nil
}

type coordinatorScenario struct {
	t           *testing.T
	groupID     GroupID
	status      GroupStatus
	desired     *ResolvedPlan
	capacity    *testCapacityAdapter
	traffic     *testTrafficAdapter
	membership  *testMembershipAdapter
	verifier    *testServingVerifier
	coordinator *Coordinator
	events      []string
}

func newCoordinatorScenario(t *testing.T, topology MembershipTopology) *coordinatorScenario {
	t.Helper()
	events := make([]string, 0)
	capacityObservation := capacityForTopology(topology)
	trafficObservation := TrafficObservation{Admitted: cloneReplicaMemberships(topology.Replicas)}
	status, err := NewGroupStatus(topology, capacityObservation, trafficObservation)
	if err != nil {
		t.Fatalf("construct initial Engine Group status: %v", err)
	}

	scenario := &coordinatorScenario{
		t:       t,
		groupID: "test-group",
		status:  status,
		events:  events,
	}
	scenario.capacity = &testCapacityAdapter{
		observation: capacityObservation,
		planned:     make(map[ReplicaID]ReplicaIncarnation),
		available:   true,
		events:      &scenario.events,
	}
	scenario.traffic = &testTrafficAdapter{
		observation:     trafficObservation,
		autoDrain:       true,
		confirmInactive: true,
		events:          &scenario.events,
	}
	scenario.membership = &testMembershipAdapter{
		topology:    cloneTopology(topology),
		transitions: make(map[string]MembershipTransitionObservation),
		targets:     make(map[string]MembershipTarget),
		events:      &scenario.events,
	}
	scenario.verifier = &testServingVerifier{events: &scenario.events}
	scenario.rebuildCoordinator()
	return scenario
}

func (s *coordinatorScenario) rebuildCoordinator() {
	s.coordinator = NewCoordinator(s.capacity, s.membership, s.traffic, s.verifier)
}

func (s *coordinatorScenario) reconcile(step string) error {
	s.t.Helper()
	s.t.Log(step)
	result, err := s.coordinator.Reconcile(
		context.Background(),
		s.groupID,
		s.desired,
		s.status,
	)
	s.status = result.Status
	return err
}

func (s *coordinatorScenario) mustReconcile(step string) {
	s.t.Helper()
	if err := s.reconcile(step); err != nil {
		s.t.Fatalf("%s: %v", step, err)
	}
}

func (s *coordinatorScenario) runUntil(
	step string,
	predicate func(*coordinatorScenario) bool,
) {
	s.t.Helper()
	for iteration := 1; iteration <= 30; iteration++ {
		if predicate(s) {
			return
		}
		s.mustReconcile(fmt.Sprintf("%s (reconcile %d)", step, iteration))
	}
	s.t.Fatalf("%s did not converge after 30 reconciles: status=%#v events=%v", step, s.status, s.events)
}

func engineTopology(generation int64, replicaCount int) MembershipTopology {
	replicas := make([]ReplicaMembership, 0, replicaCount)
	for index := 0; index < replicaCount; index++ {
		replicas = append(replicas, engineReplica(index))
	}
	return MembershipTopology{Generation: generation, Replicas: replicas}
}

func engineReplica(index int) ReplicaMembership {
	incarnation := replicaIncarnation(index)
	return ReplicaMembership{
		ReplicaID:          incarnation.ReplicaID,
		RuntimeIncarnation: incarnation.RuntimeIncarnation,
		NativeMembers:      []NativeMemberID{NativeMemberID(fmt.Sprintf("dp-%d", index))},
	}
}

func replicaIncarnation(index int) ReplicaIncarnation {
	return ReplicaIncarnation{
		ReplicaID:          ReplicaID(fmt.Sprintf("replica-%d", index)),
		SlotID:             CapacitySlotID(fmt.Sprintf("slot-%d", index)),
		RuntimeIncarnation: RuntimeIncarnationID(fmt.Sprintf("runtime-%d-v1", index)),
		CapacityRefs: []CapacityRef{{
			Name: fmt.Sprintf("worker-%d", index),
			UID:  PodUID(fmt.Sprintf("pod-uid-%d-v1", index)),
		}},
	}
}

func capacityForTopology(topology MembershipTopology) CapacityObservation {
	allocations := make([]CapacityAllocation, 0, len(topology.Replicas))
	for _, membership := range topology.Replicas {
		allocations = append(allocations, CapacityAllocation{
			Incarnation: physicalIncarnationForMembership(membership),
			Available:   true,
		})
	}
	return CapacityObservation{Allocations: allocations}
}

func physicalIncarnationForMembership(membership ReplicaMembership) ReplicaIncarnation {
	suffix := strings.TrimPrefix(string(membership.ReplicaID), "replica-")
	return ReplicaIncarnation{
		ReplicaID:          membership.ReplicaID,
		SlotID:             CapacitySlotID("slot-" + suffix),
		RuntimeIncarnation: membership.RuntimeIncarnation,
		CapacityRefs: []CapacityRef{{
			Name: "worker-" + suffix,
			UID:  PodUID("pod-uid-" + suffix + "-v1"),
		}},
	}
}

func growPlan(planID string, target ReplicaTarget, verification VerificationRequirement) ResolvedPlan {
	if len(target.NativeMembers) == 0 {
		suffix := strings.TrimPrefix(string(target.ReplicaID), "replica-")
		target.NativeMembers = []NativeMemberID{NativeMemberID("dp-" + suffix)}
	}
	return ResolvedPlan{
		ID:                      planID,
		ProfileFingerprint:      "profile-v1",
		ProcessLifecycleOwner:   ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      TrafficRequirementKeepServing,
		VerificationRequirement: verification,
		Change: ResolvedChange{
			Kind: PlanKindGrow,
			Grow: &GrowChange{Replicas: []ReplicaTarget{target}},
		},
	}
}

func retirePlan(
	planID string,
	replicaID ReplicaID,
	trafficRequirement TrafficRequirement,
	verification VerificationRequirement,
) ResolvedPlan {
	return ResolvedPlan{
		ID:                      planID,
		ProfileFingerprint:      "profile-v1",
		ProcessLifecycleOwner:   ProcessLifecycleOwnerOrchestrator,
		TrafficRequirement:      trafficRequirement,
		VerificationRequirement: verification,
		Change: ResolvedChange{
			Kind:   PlanKindRetire,
			Retire: &RetireChange{Replicas: []ReplicaID{replicaID}},
		},
	}
}

func rejected(reason string, message string) ApplyResult {
	return ApplyResult{Rejection: &Failure{
		Classification: FailureClassificationTerminal,
		Reason:         reason,
		Message:        message,
	}}
}

func sameMembershipTarget(left, right MembershipTarget) bool {
	return left.ControlRevision == right.ControlRevision &&
		left.TransitionID == right.TransitionID &&
		left.TargetDigest == right.TargetDigest &&
		left.Validation == right.Validation &&
		sameTopology(left.BaseTopology, right.BaseTopology) &&
		sameResolvedPlan(left.Plan, right.Plan) &&
		sameJoiningReplicas(left.Joining, right.Joining)
}

func sameJoiningReplicas(left, right []JoiningReplica) bool {
	if len(left) != len(right) {
		return false
	}
	left = slices.Clone(left)
	right = slices.Clone(right)
	slices.SortFunc(left, func(a, b JoiningReplica) int {
		return strings.Compare(string(a.ReplicaID), string(b.ReplicaID))
	})
	slices.SortFunc(right, func(a, b JoiningReplica) int {
		return strings.Compare(string(a.ReplicaID), string(b.ReplicaID))
	})
	return slices.Equal(left, right)
}

func eventIndex(events []string, prefix string) int {
	for index, event := range events {
		if len(event) >= len(prefix) && event[:len(prefix)] == prefix {
			return index
		}
	}
	return -1
}
