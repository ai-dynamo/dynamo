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

import "context"

// CapacityReplicaTarget describes one logical allocation required by an absolute capacity target.
type CapacityReplicaTarget struct {
	ReplicaID   ReplicaID
	SlotID      CapacitySlotID
	Incarnation *ReplicaIncarnation
	Bootstrap   *CapacityBootstrap
}

// CapacityBootstrap is the profile-resolved process intent for one new orchestrator-owned allocation. After the
// allocation becomes available, the coordinator replaces this intent with its exact Incarnation before membership.
type CapacityBootstrap struct {
	Mode                   BootstrapMode
	BaseTopologyGeneration int64
	NativeMembers          []NativeMemberID
}

// ReleaseFence authorizes removal of only the named concrete Pod incarnations from one stable slot.
type ReleaseFence struct {
	TransitionID                  string
	AuthorizingTopologyGeneration int64
	ReplicaID                     ReplicaID
	SlotID                        CapacitySlotID
	CapacityRefs                  []CapacityRef
}

// CapacityTarget is one durable, group-ordered absolute physical-capacity projection.
type CapacityTarget struct {
	ControlRevision       int64
	TransitionID          string
	ProfileFingerprint    string
	ProcessLifecycleOwner ProcessLifecycleOwner
	Replicas              []CapacityReplicaTarget
	ReleaseFences         []ReleaseFence
}

// CapacityAllocation is one workload-manager allocation and its complete availability state.
type CapacityAllocation struct {
	Incarnation ReplicaIncarnation
	Available   bool
}

// CapacityObservation is the workload manager's current allocation state.
type CapacityObservation struct {
	// AppliedRevision is the last target revision durably accepted by the adapter, whether or not it has converged.
	AppliedRevision int64
	Allocations     []CapacityAllocation
	ReleaseFences   []ReleaseFence
}

// TrafficTarget is one durable, group-ordered absolute routing projection.
type TrafficTarget struct {
	ControlRevision    int64
	TransitionID       string
	TopologyGeneration int64
	Admitted           []ReplicaMembership
	Drain              []TrafficDrainTarget
}

// TrafficDrainMode distinguishes participating graceful drain from confirmation that a failed member is inactive.
type TrafficDrainMode string

const (
	// TrafficDrainModeGraceful waits for a reachable member's in-flight work to finish.
	TrafficDrainModeGraceful TrafficDrainMode = "Graceful"
	// TrafficDrainModeConfirmInactive proves an unreachable member is no longer routable without its participation.
	TrafficDrainModeConfirmInactive TrafficDrainMode = "ConfirmInactive"
)

// TrafficDrainTarget requests terminal non-serving evidence for one exact member incarnation.
type TrafficDrainTarget struct {
	Membership ReplicaMembership
	Mode       TrafficDrainMode
}

// TrafficObservation is the runtime's exact routing and drain state.
type TrafficObservation struct {
	// AppliedRevision is the last target revision durably accepted by the adapter, whether or not it has converged.
	AppliedRevision int64
	Admitted        []ReplicaMembership
	Draining        []ReplicaMembership
	Drained         []ReplicaMembership
}

// MembershipTransitionPhase is the adapter's durable state for one exact desired membership transition.
type MembershipTransitionPhase string

const (
	// MembershipTransitionPhasePending means the adapter accepted the target and is changing membership.
	MembershipTransitionPhasePending MembershipTransitionPhase = "Pending"
	// MembershipTransitionPhaseCommitted means the target produced the reported immutable result topology.
	MembershipTransitionPhaseCommitted MembershipTransitionPhase = "Committed"
	// MembershipTransitionPhaseRejected means the target definitively cannot mutate membership.
	MembershipTransitionPhaseRejected MembershipTransitionPhase = "Rejected"
	// MembershipTransitionPhaseUnknown means the adapter cannot establish whether the target mutated membership.
	MembershipTransitionPhaseUnknown MembershipTransitionPhase = "Unknown"
)

// MembershipTarget is one exact immutable topology transition desired by the coordinator.
type MembershipTarget struct {
	ControlRevision int64
	TransitionID    string
	TargetDigest    string
	Validation      ValidationEvidence
	BaseTopology    MembershipTopology
	Plan            ResolvedPlan
	Joining         []JoiningReplica
}

// PlanValidationRequest carries the coordinator-owned canonical digest alongside the exact normalized plan.
type PlanValidationRequest struct {
	BaseTopology MembershipTopology
	Plan         ResolvedPlan
	PlanDigest   string
}

// ValidationEvidence binds adapter approval to immutable plan, profile, capability, and optional target state.
type ValidationEvidence struct {
	PlanDigest           string
	TargetDigest         string
	ProfileFingerprint   string
	CapabilityGeneration string
}

// PreflightResult is either durable validation evidence or a definitive rejection of the validated subject. With a
// nil method error exactly one field is set. With a non-nil method error neither field is authoritative. A rejection is
// terminal for the correlated request and guarantees that the adapter cannot later commit it.
type PreflightResult struct {
	Evidence  *ValidationEvidence
	Rejection *Failure
}

// MembershipTransitionObservation is the durable adapter result for one exact target identity.
type MembershipTransitionObservation struct {
	TransitionID    string
	ControlRevision int64
	TargetDigest    string
	Phase           MembershipTransitionPhase
	ResultTopology  *MembershipTopology
	Failure         *Failure
}

// MembershipObservation reports current engine topology independently from one correlated transition result.
type MembershipObservation struct {
	CommittedTopology     MembershipTopology
	RequestedTransitionID string
	Transition            *MembershipTransitionObservation
}

// ApplyResult reports a definitive, replay-stable rejection of an adapter request.
// For capacity and traffic, rejection guarantees that no further work for the rejected revision can occur; any partial
// effect is returned by the next level-based observation. For membership submission, rejection guarantees no mutation.
// A nil Rejection means the adapter accepted the request or had already applied it.
type ApplyResult struct {
	Rejection *Failure
}

// VerificationResult is one safely rerunnable serving check against an immutable topology.
type VerificationResult struct {
	Proof   *ServingProof
	Failure *Failure
}

// CapacityAdapter converges physical capacity to revisioned absolute targets.
type CapacityAdapter interface {
	// Observe returns stable allocation identities, exact Pod UIDs, availability, and durable release fences.
	Observe(ctx context.Context, groupID GroupID) (CapacityObservation, error)
	// Apply converges to target without deleting capacity absent from an exact UID-bound release fence.
	// Revisions are group-global and monotonic. Repeating an equal revision and payload is idempotent; an equal revision
	// must repair observable drift, while an equal revision with another payload or a lower revision is definitively
	// rejected. Deletion uses Pod UID preconditions, and an applied fence remains observable until a later target
	// explicitly reopens the stable replica slot. An exact Incarnation is an identity assertion, not permission to
	// recreate a missing process or Pod UID.
	Apply(ctx context.Context, groupID GroupID, target CapacityTarget) (ApplyResult, error)
}

// MembershipAdapter converges membership to one desired level while owning the serialized compare-and-apply protocol.
type MembershipAdapter interface {
	// ValidatePlan authoritatively checks complete resolved semantics before capacity or traffic prework. It is
	// side-effect-free and returns evidence bound to the normalized plan and current adapter capabilities.
	ValidatePlan(
		ctx context.Context,
		groupID GroupID,
		request PlanValidationRequest,
	) (PreflightResult, error)
	// ValidateTarget revalidates the exact target after joining runtime identities are frozen. It is side-effect-free
	// and either adds the canonical target digest to matching plan evidence or definitively rejects that target.
	ValidateTarget(ctx context.Context, groupID GroupID, target MembershipTarget) (PreflightResult, error)
	// Observe always returns the authoritative committed topology. When transitionID is non-empty, a nil Transition
	// authoritatively means the adapter has no record of that exact identity. Stale or inconclusive reads return an
	// error or an Unknown transition instead. Terminal results remain observable until a newer revision is accepted.
	Observe(
		ctx context.Context,
		groupID GroupID,
		transitionID string,
	) (MembershipObservation, error)
	// Apply atomically compares BaseTopology and applies target. It is idempotent by transition identity and exact
	// payload, permits only one non-terminal mutation, and never supersedes Pending or Unknown work. Any returned error
	// leaves acceptance ambiguous; definitive rejection is reported durably by Observe.
	Apply(ctx context.Context, groupID GroupID, target MembershipTarget) error
}

// TrafficAdapter converges runtime discovery and routing to revisioned absolute identity sets.
type TrafficAdapter interface {
	// Observe returns exact admitted, draining, and durably drained replica incarnations.
	Observe(ctx context.Context, groupID GroupID) (TrafficObservation, error)
	// Apply converges toward the exact admitted set and preserves drain tombstones for every target in Drain. Graceful
	// drain waits for in-flight work; ConfirmInactive proves a failed member non-routable without its participation.
	// Revisions are group-global and monotonic. Repeating an equal revision and payload must repair observable drift; a
	// lower revision or conflicting equal revision is definitively rejected. No engine membership or discovery event may
	// implicitly admit an incarnation absent from the latest target.
	Apply(ctx context.Context, groupID GroupID, target TrafficTarget) (ApplyResult, error)
}

// ServingVerifier performs a safely repeatable serving-progress check against one exact committed topology.
type ServingVerifier interface {
	// Verify returns a positive proof or a conclusive failure. A non-nil error is inconclusive and may be retried against
	// the same snapshot. Repeating a successful check is safe; a crash before proof persistence therefore needs no
	// separate verification-operation journal.
	Verify(ctx context.Context, groupID GroupID, topology MembershipTopology) (VerificationResult, error)
}
