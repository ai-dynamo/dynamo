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
	"time"

	"k8s.io/apimachinery/pkg/util/uuid"
)

var errMembershipOperationUnsupported = errors.New("membership operation is unsupported")

// OperationCoordinator advances one durable membership operation without owning its persistence.
type OperationCoordinator struct {
	membership     MembershipAdapter
	now            func() time.Time
	newOperationID func() string
}

// NewOperationCoordinator constructs a membership coordinator. membership must be non-nil.
func NewOperationCoordinator(membership MembershipAdapter) *OperationCoordinator {
	return &OperationCoordinator{
		membership:     membership,
		now:            time.Now,
		newOperationID: func() string { return string(uuid.NewUUID()) },
	}
}

// Step advances at most one durable operation transition. A nil input Operation is supported and means no operation exists yet.
func (c *OperationCoordinator) Step(ctx context.Context, input OperationInput) (OperationResult, error) {
	// Reject invalid desired or durable state before observing or mutating the backend.
	if err := validateOperationInput(input); err != nil {
		return OperationResult{Operation: cloneOperation(input.Operation)}, err
	}

	// Observe the authoritative topology used to plan or verify this transition.
	topology, err := c.membership.ObserveTopology(ctx, input.GroupID)
	if err != nil {
		return OperationResult{Operation: cloneOperation(input.Operation)}, fmt.Errorf("observe membership topology: %w", err)
	}
	if err := validateTopology(topology); err != nil {
		return OperationResult{Operation: cloneOperation(input.Operation)}, fmt.Errorf("validate membership topology: %w", err)
	}

	result := OperationResult{
		Operation: cloneOperation(input.Operation),
		Topology:  cloneTopology(topology),
	}

	// A new explicit recovery plan may adopt a later survivor topology after durable prior commit proof.
	if result.Operation != nil &&
		result.Operation.Phase == OperationPhaseUnknown &&
		result.Operation.CommittedTopologyGeneration != 0 &&
		input.Plan != nil &&
		input.Plan.ID != result.Operation.PlanID {
		if err := c.requireCapability(ctx, input.GroupID, input.Plan.Intent); err != nil {
			return result, err
		}
		adopted, err := c.adoptObservedRecovery(input, topology, *result.Operation, *input.Plan)
		if err != nil {
			return result, err
		}
		result.Operation = adopted
		result.OperationChanged = true
		return result, nil
	}

	// Start a durable pending operation only when current observations require one.
	if result.Operation == nil {
		plan, err := desiredPlan(input, topology)
		if err != nil {
			return result, err
		}
		if plan == nil {
			return result, nil
		}
		if err := c.requireCapability(ctx, input.GroupID, plan.Intent); err != nil {
			return result, err
		}

		operation, err := c.newPendingOperation(input, topology, *plan)
		if err != nil {
			return result, err
		}
		result.Operation = operation
		result.OperationChanged = true
		return result, nil
	}

	// Dispatch by durable phase so every external call is preceded by a persisted marker.
	switch result.Operation.Phase {
	case OperationPhasePending, OperationPhaseFailed:
		return c.advanceResolved(input, result)
	case OperationPhaseSubmitting:
		return c.advanceSubmitting(ctx, input, result)
	case OperationPhaseAccepted, OperationPhaseCommitting, OperationPhaseUnknown:
		return c.observeInFlight(ctx, input, result)
	case OperationPhaseCommitted, OperationPhaseAborting, OperationPhaseAborted:
		return result, nil
	default:
		return result, fmt.Errorf("unsupported operation phase %q", result.Operation.Phase)
	}
}

func (c *OperationCoordinator) requireCapability(
	ctx context.Context,
	groupID GroupID,
	intent OperationIntent,
) error {
	capabilities, err := c.membership.ObserveCapabilities(ctx, groupID)
	if err != nil {
		return fmt.Errorf("observe membership capabilities: %w", err)
	}
	if err := validateMembershipCapabilities(capabilities); err != nil {
		return fmt.Errorf("validate membership capabilities: %w", err)
	}
	if !slices.Contains(capabilities.Intents, intent) {
		return fmt.Errorf("membership operation %q is not supported: %w", intent, errMembershipOperationUnsupported)
	}
	return nil
}

// PrepareSubmission returns a durable Submitting transition without calling the backend.
// operation must be non-nil. Callers must persist the returned operation and recheck external
// prerequisites before calling Submit.
func (c *OperationCoordinator) PrepareSubmission(
	operation *Operation,
	joiningReplicas []ReplicaID,
) (*Operation, error) {
	// Validate the caller-owned record before deriving a transition from it.
	if err := validateOperation(*operation); err != nil {
		return nil, err
	}

	// Only an unsubmitted operation or an explicitly retryable failure may be prepared.
	if operation.Phase != OperationPhasePending &&
		(operation.Phase != OperationPhaseFailed ||
			operation.Failure.Classification != FailureClassificationRetryable) {
		return nil, fmt.Errorf("operation phase %q cannot transition to Submitting", operation.Phase)
	}

	// Freeze exact joining identities into the request that may be replayed after persistence.
	prepared := cloneOperation(operation)
	if prepared.Phase == OperationPhasePending {
		prepared.JoiningReplicas = normalizeReplicaIDs(joiningReplicas)
	} else {
		if len(joiningReplicas) > 0 && !sameReplicaIDs(joiningReplicas, prepared.JoiningReplicas) {
			return nil, errors.New("retry cannot change joining replica identities")
		}
		prepared.Attempt++
		prepared.BackendOperationID = ""
	}
	prepared.Failure = nil
	prepared = transitionOperation(prepared, OperationPhaseSubmitting, c.now())

	// Reject an incomplete or inconsistent request before it can become externally visible.
	if err := validateOperation(*prepared); err != nil {
		return nil, err
	}
	return prepared, nil
}

// Submit idempotently applies one previously persisted Submitting operation. operation must be non-nil.
func (c *OperationCoordinator) Submit(
	ctx context.Context,
	groupID GroupID,
	operation *Operation,
) (OperationResult, error) {
	// Validate the exact durable request before observing or mutating the backend.
	if groupID == "" {
		return OperationResult{Operation: cloneOperation(operation)}, errors.New("group ID must not be empty")
	}
	if err := validateOperation(*operation); err != nil {
		return OperationResult{Operation: cloneOperation(operation)}, err
	}
	if operation.Phase != OperationPhaseSubmitting {
		return OperationResult{Operation: cloneOperation(operation)},
			fmt.Errorf("operation phase %q cannot be submitted", operation.Phase)
	}

	// Recheck that the committed base topology has not moved since the request was prepared.
	topology, err := c.membership.ObserveTopology(ctx, groupID)
	if err != nil {
		return OperationResult{Operation: cloneOperation(operation)}, fmt.Errorf("observe membership topology: %w", err)
	}
	if err := validateTopology(topology); err != nil {
		return OperationResult{Operation: cloneOperation(operation)}, fmt.Errorf("validate membership topology: %w", err)
	}
	result := OperationResult{
		Operation: cloneOperation(operation),
		Topology:  cloneTopology(topology),
	}
	if !operationMatchesBaseTopology(*operation, topology) {
		return c.resolveSubmissionBaseDrift(ctx, groupID, result)
	}

	// Recheck semantic support immediately before every first or replayed backend mutation.
	if err := c.requireCapability(ctx, groupID, operation.Intent); err != nil {
		return result, err
	}

	// Submit the exact persisted request; the backend contract makes replay idempotent by operation ID.
	observed, err := c.membership.SubmitOperation(ctx, groupID, membershipRequest(*operation))
	if err != nil {
		return result, fmt.Errorf("submit membership operation %q: %w", operation.ID, err)
	}
	return c.applyBackendObservation(ctx, groupID, result, observed)
}

func (c *OperationCoordinator) resolveSubmissionBaseDrift(
	ctx context.Context,
	groupID GroupID,
	result OperationResult,
) (OperationResult, error) {
	// Re-observe after the newer topology so a commit racing the earlier Absent observation cannot be discarded.
	observed, err := c.membership.ObserveOperation(
		ctx,
		groupID,
		result.Operation.ID,
		result.Operation.Attempt,
	)
	if err != nil {
		return result, fmt.Errorf(
			"observe membership operation %q after base drift: %w",
			result.Operation.ID,
			err,
		)
	}
	if observed.Phase == BackendOperationPhaseAbsent {
		return c.abortForBaseDrift(result), nil
	}
	return c.applyBackendObservation(ctx, groupID, result, observed)
}

func (c *OperationCoordinator) advanceResolved(input OperationInput, result OperationResult) (OperationResult, error) {
	// No membership call can precede Pending, so a changed base makes compensation safe and necessary.
	if !operationMatchesBaseTopology(*result.Operation, result.Topology) {
		return c.abortForBaseDrift(result), nil
	}

	// Pending and Failed requests remain frozen until completion or explicit future compensation.
	result.OperationChanged = setQueuedTarget(result.Operation, input.DesiredReplicas) || result.OperationChanged
	return result, nil
}

func (c *OperationCoordinator) advanceSubmitting(
	ctx context.Context,
	input OperationInput,
	result OperationResult,
) (OperationResult, error) {
	// Record a superseding desired target without mutating the submitted request.
	result.OperationChanged = setQueuedTarget(result.Operation, input.DesiredReplicas) || result.OperationChanged

	// Observe before replaying so a restart does not create a competing operation.
	observed, err := c.membership.ObserveOperation(
		ctx,
		input.GroupID,
		result.Operation.ID,
		result.Operation.Attempt,
	)
	if err != nil {
		return result, fmt.Errorf("observe membership operation %q: %w", result.Operation.ID, err)
	}
	if observed.Phase != BackendOperationPhaseAbsent {
		return c.applyBackendObservation(ctx, input.GroupID, result, observed)
	}

	// Base drift makes the request stale after the backend proves this attempt absent. A racing replay is also
	// harmless because SubmitOperation atomically refuses its now-stale base generation and identity set.
	if !operationMatchesBaseTopology(*result.Operation, result.Topology) {
		return c.abortForBaseDrift(result), nil
	}

	// Report that the exact persisted request may need submission after caller-owned prerequisites are checked.
	result.SubmissionNeeded = true
	return result, nil
}

func (c *OperationCoordinator) observeInFlight(
	ctx context.Context,
	input OperationInput,
	result OperationResult,
) (OperationResult, error) {
	// Preserve a later desired target while the current operation remains authoritative.
	result.OperationChanged = setQueuedTarget(result.Operation, input.DesiredReplicas) || result.OperationChanged

	// Continue observing only the persisted operation identity.
	observed, err := c.membership.ObserveOperation(
		ctx,
		input.GroupID,
		result.Operation.ID,
		result.Operation.Attempt,
	)
	if err != nil {
		return result, fmt.Errorf("observe membership operation %q: %w", result.Operation.ID, err)
	}
	if observed.Phase == BackendOperationPhaseAbsent {
		if result.Operation.Phase == OperationPhaseUnknown {
			return result, nil
		}

		result.Operation = transitionOperation(result.Operation, OperationPhaseUnknown, c.now())
		result.OperationChanged = true
		return result, nil
	}
	return c.applyBackendObservation(ctx, input.GroupID, result, observed)
}

func (c *OperationCoordinator) applyBackendObservation(
	ctx context.Context,
	groupID GroupID,
	result OperationResult,
	observed BackendOperation,
) (OperationResult, error) {
	// An accepted-but-not-yet-observable submission remains ambiguous and safely replayable.
	if observed.Phase == BackendOperationPhaseAbsent {
		return result, nil
	}

	// Reject an observation that is not correlated to the exact persisted request.
	if observed.ID != result.Operation.ID ||
		observed.Attempt != result.Operation.Attempt ||
		observed.TargetReplicas != result.Operation.TargetReplicas {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}

	// Once commit was durable, stale progress or failure cannot overwrite it; only exact commit revalidation can recover.
	if result.Operation.Phase == OperationPhaseUnknown &&
		result.Operation.CommittedTopologyGeneration != 0 &&
		observed.Phase != BackendOperationPhaseCommitted {
		return result, nil
	}

	// A backend correlation identity is immutable once one non-empty value has been observed.
	if observed.BackendID != "" &&
		result.Operation.BackendOperationID != "" &&
		result.Operation.BackendOperationID != observed.BackendID {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}
	if result.Operation.BackendOperationID == "" && observed.BackendID != "" {
		result.Operation.BackendOperationID = observed.BackendID
		result.OperationChanged = true
	}

	// Translate backend progress into the durable controller-facing phase model.
	switch observed.Phase {
	case BackendOperationPhaseAccepted:
		if result.Operation.Phase == OperationPhaseCommitting {
			return result, nil
		}
		return c.setObservedPhase(result, OperationPhaseAccepted), nil
	case BackendOperationPhaseCommitting:
		return c.setObservedPhase(result, OperationPhaseCommitting), nil
	case BackendOperationPhaseCommitted:
		return c.verifyCommitted(ctx, groupID, result, observed)
	case BackendOperationPhaseFailed:
		if err := validateFailure(observed.Failure); err != nil {
			return c.setObservedPhase(result, OperationPhaseUnknown), nil
		}

		result.Operation.Failure = cloneFailure(observed.Failure)
		return c.setObservedPhase(result, OperationPhaseFailed), nil
	case BackendOperationPhaseUnknown:
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	default:
		return result, fmt.Errorf("unsupported backend operation phase %q", observed.Phase)
	}
}

func (c *OperationCoordinator) verifyCommitted(
	ctx context.Context,
	groupID GroupID,
	result OperationResult,
	observed BackendOperation,
) (OperationResult, error) {
	// A successful membership transaction must identify a newer committed topology.
	if observed.CommittedTopologyGeneration <= result.Operation.BaseTopologyGeneration {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}

	// Re-observe topology after the commit observation to avoid validating against a stale pre-submit snapshot.
	topology, err := c.membership.ObserveTopology(ctx, groupID)
	if err != nil {
		return result, fmt.Errorf("observe committed membership topology: %w", err)
	}
	if err := validateTopology(topology); err != nil {
		return result, fmt.Errorf("validate committed membership topology: %w", err)
	}
	result.Topology = cloneTopology(topology)

	// Durable commit proof is immutable; only the exact same generation may recover an Unknown operation.
	if result.Operation.CommittedTopologyGeneration != 0 &&
		observed.CommittedTopologyGeneration != result.Operation.CommittedTopologyGeneration {
		return result, nil
	}

	// Wait for an eventually consistent topology view to reach the committed generation.
	if topology.Generation < observed.CommittedTopologyGeneration {
		if result.Operation.CommittedTopologyGeneration != 0 {
			return result, nil
		}
		return c.setObservedPhase(result, OperationPhaseCommitting), nil
	}

	// Preserve skipped commit proof when a later authoritative topology contains only expected survivors.
	if topology.Generation > observed.CommittedTopologyGeneration {
		if err := requireReplicaSubset(
			topologyReplicaIDs(topology),
			expectedCommittedReplicaIDs(*result.Operation),
			"later survivor",
		); err != nil {
			return c.setObservedPhase(result, OperationPhaseUnknown), nil
		}
		if result.Operation.CommittedTopologyGeneration != observed.CommittedTopologyGeneration {
			result.Operation.CommittedTopologyGeneration = observed.CommittedTopologyGeneration
			result.OperationChanged = true
		}
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}

	// An identity-different topology at the claimed commit generation cannot prove the operation outcome.
	if !operationMatchesCommittedTopology(*result.Operation, topology) {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}

	// Record commit only after the exact authoritative topology is visible.
	if result.Operation.CommittedTopologyGeneration != topology.Generation {
		result.Operation.CommittedTopologyGeneration = topology.Generation
		result.OperationChanged = true
	}
	return c.setObservedPhase(result, OperationPhaseCommitted), nil
}

func (c *OperationCoordinator) setObservedPhase(result OperationResult, phase OperationPhase) OperationResult {
	if result.Operation.Phase == phase {
		return result
	}

	result.Operation = transitionOperation(result.Operation, phase, c.now())
	result.OperationChanged = true
	return result
}

func (c *OperationCoordinator) abortForBaseDrift(result OperationResult) OperationResult {
	result.Operation.Failure = &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "BaseTopologyChanged",
		Message: fmt.Sprintf(
			"membership topology changed from generation %d before the operation committed",
			result.Operation.BaseTopologyGeneration,
		),
	}
	return c.setObservedPhase(result, OperationPhaseAborting)
}

func (c *OperationCoordinator) newPendingOperation(
	input OperationInput,
	topology MembershipTopology,
	plan OperationPlan,
) (*Operation, error) {
	operationID := c.newOperationID()
	if operationID == "" {
		return nil, errors.New("generate membership operation ID: empty value")
	}

	now := c.now()
	operation := &Operation{
		ID:                     operationID,
		Attempt:                1,
		PlanID:                 plan.ID,
		Intent:                 plan.Intent,
		SpecGeneration:         input.SpecGeneration,
		BaseTopologyGeneration: topology.Generation,
		BaseReplicas:           topologyReplicaIDs(topology),
		TargetReplicas:         plan.TargetReplicas,
		NominatedReplicas:      slices.Clone(plan.NominatedReplicas),
		Phase:                  OperationPhasePending,
		StartedAt:              now,
		LastTransitionTime:     now,
	}
	setQueuedTarget(operation, input.DesiredReplicas)
	if err := validateOperation(*operation); err != nil {
		return nil, fmt.Errorf("validate planned membership operation: %w", err)
	}
	return operation, nil
}

func (c *OperationCoordinator) adoptObservedRecovery(
	input OperationInput,
	topology MembershipTopology,
	previous Operation,
	plan OperationPlan,
) (*Operation, error) {
	// Adoption is restricted to an explicit recovery of a strictly later authoritative topology.
	normalizedPlan := normalizePlan(plan)
	if normalizedPlan.Intent != OperationIntentRecover {
		return nil, fmt.Errorf("only a recovery plan may adopt observed membership, got %q", normalizedPlan.Intent)
	}
	if topology.Generation <= previous.CommittedTopologyGeneration {
		return nil, fmt.Errorf(
			"adopted topology generation %d must exceed last committed generation %d",
			topology.Generation,
			previous.CommittedTopologyGeneration,
		)
	}
	if normalizedPlan.TargetReplicas != topology.ReplicaCount() {
		return nil, fmt.Errorf(
			"recovery plan target %d does not match observed topology size %d",
			normalizedPlan.TargetReplicas,
			topology.ReplicaCount(),
		)
	}

	// The observed survivors must be a subset of the last fully completed logical topology.
	baseReplicas := expectedCommittedReplicaIDs(previous)
	observedReplicas := topologyReplicaIDs(topology)
	if err := requireReplicaSubset(observedReplicas, baseReplicas, "observed survivor"); err != nil {
		return nil, err
	}
	nominatedReplicas := differenceReplicaIDs(baseReplicas, observedReplicas)
	if !sameReplicaIDs(normalizedPlan.NominatedReplicas, nominatedReplicas) {
		return nil, fmt.Errorf(
			"recovery plan nominations %v do not match missing replicas %v",
			normalizedPlan.NominatedReplicas,
			nominatedReplicas,
		)
	}

	// Mint a durable adopted record without implying that Dynamo submitted the engine-side transition.
	operationID := c.newOperationID()
	if operationID == "" {
		return nil, errors.New("generate adopted recovery operation ID: empty value")
	}
	now := c.now()
	operation := &Operation{
		ID:                          operationID,
		Attempt:                     1,
		PlanID:                      normalizedPlan.ID,
		Intent:                      OperationIntentRecover,
		SpecGeneration:              input.SpecGeneration,
		BaseTopologyGeneration:      previous.CommittedTopologyGeneration,
		BaseReplicas:                baseReplicas,
		TargetReplicas:              normalizedPlan.TargetReplicas,
		NominatedReplicas:           nominatedReplicas,
		Phase:                       OperationPhaseCommitted,
		CommittedTopologyGeneration: topology.Generation,
		Adopted:                     true,
		StartedAt:                   now,
		LastTransitionTime:          now,
	}
	setQueuedTarget(operation, input.DesiredReplicas)
	if err := validateOperation(*operation); err != nil {
		return nil, fmt.Errorf("validate adopted recovery operation: %w", err)
	}
	return operation, nil
}

func desiredPlan(input OperationInput, topology MembershipTopology) (*OperationPlan, error) {
	// An explicit higher-level plan carries recovery or selected-retirement semantics that counts cannot express.
	if input.Plan != nil {
		plan := normalizePlan(*input.Plan)
		if err := validatePlan(plan, topology); err != nil {
			return nil, err
		}
		return &plan, nil
	}

	// Equal desired and active counts require no cardinal membership operation.
	activeReplicas := topology.ReplicaCount()
	if input.DesiredReplicas == activeReplicas {
		return nil, nil
	}

	if input.DesiredReplicas < activeReplicas {
		return nil, errors.New("shrinking membership requires an identity-aware operation plan")
	}
	return &OperationPlan{Intent: OperationIntentGrow, TargetReplicas: input.DesiredReplicas}, nil
}

func validateOperationInput(input OperationInput) error {
	if input.GroupID == "" {
		return errors.New("group ID must not be empty")
	}
	if input.SpecGeneration < 0 {
		return fmt.Errorf("spec generation must not be negative: %d", input.SpecGeneration)
	}
	if input.DesiredReplicas < 0 {
		return fmt.Errorf("desired replicas must not be negative: %d", input.DesiredReplicas)
	}
	if input.Plan != nil && input.Plan.ID == "" {
		return errors.New("explicit operation plan ID must not be empty")
	}
	if input.Plan != nil && !validOperationIntent(input.Plan.Intent) {
		return fmt.Errorf("invalid operation plan intent %q", input.Plan.Intent)
	}
	if input.Operation != nil &&
		input.Plan != nil &&
		input.Operation.PlanID == input.Plan.ID &&
		!operationCarriesPlan(*input.Operation, *input.Plan) {
		return fmt.Errorf("operation plan ID %q was reused with a different payload", input.Plan.ID)
	}
	if input.Operation != nil {
		if err := validateOperation(*input.Operation); err != nil {
			return err
		}
	}
	return nil
}

func validateOperation(operation Operation) error {
	// Validate the immutable request payload independently from lifecycle evidence.
	if err := validateOperationRequest(operation); err != nil {
		return err
	}

	// Validate phase-dependent timestamps, failures, and commit provenance.
	return validateOperationLifecycle(operation)
}

func validateOperationRequest(operation Operation) error {
	if operation.ID == "" {
		return errors.New("operation ID must not be empty")
	}
	if operation.Attempt < 1 {
		return fmt.Errorf("operation attempt must be positive: %d", operation.Attempt)
	}
	if operation.Intent != OperationIntentGrow && operation.PlanID == "" {
		return fmt.Errorf("%s operation must carry its explicit plan ID", operation.Intent)
	}
	if !validOperationIntent(operation.Intent) {
		return fmt.Errorf("invalid operation intent %q", operation.Intent)
	}
	if operation.SpecGeneration < 0 {
		return fmt.Errorf("operation spec generation must not be negative: %d", operation.SpecGeneration)
	}
	if operation.BaseTopologyGeneration < 0 {
		return fmt.Errorf("base topology generation must not be negative: %d", operation.BaseTopologyGeneration)
	}
	if err := validateReplicaIDs("base", operation.BaseReplicas); err != nil {
		return err
	}
	if operation.TargetReplicas < 0 {
		return fmt.Errorf("operation target replicas must not be negative: %d", operation.TargetReplicas)
	}
	if err := validateReplicaIDs("joining", operation.JoiningReplicas); err != nil {
		return err
	}
	if err := validateReplicaIDs("nominated", operation.NominatedReplicas); err != nil {
		return err
	}
	if err := validateReplicaIDs("cleanup", operation.CleanupReplicas); err != nil {
		return err
	}
	if len(operation.CleanupReplicas) != 0 &&
		operation.Phase != OperationPhaseAborting &&
		operation.Phase != OperationPhaseAborted {
		return fmt.Errorf("operation phase %q cannot carry abort cleanup replicas", operation.Phase)
	}
	if operation.CapacityTargetReplicas < 0 {
		return fmt.Errorf("capacity target replicas must not be negative: %d", operation.CapacityTargetReplicas)
	}
	if operation.CapacityTopologyGeneration < 0 {
		return fmt.Errorf(
			"capacity topology generation must not be negative: %d",
			operation.CapacityTopologyGeneration,
		)
	}
	if operation.CapacityTargetApplied &&
		operation.Phase != OperationPhaseCommitted &&
		operation.Phase != OperationPhaseUnknown &&
		operation.Phase != OperationPhaseAborting &&
		operation.Phase != OperationPhaseAborted {
		return fmt.Errorf("operation phase %q cannot carry applied capacity target proof", operation.Phase)
	}
	if !operation.CapacityTargetApplied &&
		(operation.CapacityTargetReplicas != 0 || operation.CapacityTopologyGeneration != 0) {
		return errors.New("capacity target proof values require an applied capacity target")
	}
	if err := validateCapacityTargetProof(operation); err != nil {
		return err
	}
	if err := validateOperationReplicaGeometry(operation); err != nil {
		return err
	}
	if operation.QueuedTargetReplicas != nil && *operation.QueuedTargetReplicas < 0 {
		return fmt.Errorf("queued target replicas must not be negative: %d", *operation.QueuedTargetReplicas)
	}
	return nil
}

func validateCapacityTargetProof(operation Operation) error {
	if !operation.CapacityTargetApplied ||
		operation.Phase == OperationPhaseAborting ||
		operation.Phase == OperationPhaseAborted {
		return nil
	}
	if operation.CapacityTopologyGeneration < operation.CommittedTopologyGeneration {
		return fmt.Errorf(
			"capacity target topology generation %d precedes committed generation %d",
			operation.CapacityTopologyGeneration,
			operation.CommittedTopologyGeneration,
		)
	}
	if operation.CapacityTopologyGeneration == operation.CommittedTopologyGeneration &&
		operation.CapacityTargetReplicas != operation.TargetReplicas {
		return fmt.Errorf(
			"capacity target %d does not match committed target %d",
			operation.CapacityTargetReplicas,
			operation.TargetReplicas,
		)
	}
	if operation.CapacityTopologyGeneration > operation.CommittedTopologyGeneration &&
		operation.CapacityTargetReplicas > operation.TargetReplicas {
		return fmt.Errorf(
			"later capacity target %d exceeds committed target %d",
			operation.CapacityTargetReplicas,
			operation.TargetReplicas,
		)
	}
	return nil
}

func validateOperationLifecycle(operation Operation) error {
	if operation.StartedAt.IsZero() || operation.LastTransitionTime.IsZero() {
		return errors.New("operation timestamps must not be zero")
	}
	if operation.LastTransitionTime.Before(operation.StartedAt) {
		return errors.New("operation last transition time must not precede start time")
	}
	if !validOperationPhase(operation.Phase) {
		return fmt.Errorf("invalid operation phase %q", operation.Phase)
	}
	requiresFailure := operation.Phase == OperationPhaseFailed ||
		operation.Phase == OperationPhaseAborting ||
		operation.Phase == OperationPhaseAborted
	if requiresFailure {
		if err := validateFailure(operation.Failure); err != nil {
			return fmt.Errorf("validate %s operation: %w", operation.Phase, err)
		}
		if (operation.Phase == OperationPhaseAborting || operation.Phase == OperationPhaseAborted) &&
			operation.Failure.Classification != FailureClassificationTerminal {
			return fmt.Errorf(
				"operation phase %q requires a terminal failure, got %q",
				operation.Phase,
				operation.Failure.Classification,
			)
		}
	} else if operation.Failure != nil {
		return fmt.Errorf("operation phase %q must not carry a failure", operation.Phase)
	}
	if operation.CommittedTopologyGeneration != 0 &&
		operation.CommittedTopologyGeneration <= operation.BaseTopologyGeneration {
		return fmt.Errorf(
			"committed topology generation %d must exceed base generation %d",
			operation.CommittedTopologyGeneration,
			operation.BaseTopologyGeneration,
		)
	}
	if operation.CommittedTopologyGeneration != 0 &&
		operation.Phase != OperationPhaseCommitted &&
		operation.Phase != OperationPhaseUnknown {
		return fmt.Errorf(
			"operation phase %q cannot carry committed topology generation %d",
			operation.Phase,
			operation.CommittedTopologyGeneration,
		)
	}
	if operation.Phase == OperationPhaseCommitted && operation.CommittedTopologyGeneration == 0 {
		return errors.New("committed operation must carry a committed topology generation")
	}
	if operation.PostCommitComplete && operation.CommittedTopologyGeneration == 0 {
		return errors.New("post-commit completion requires a committed topology generation")
	}
	if operation.Adopted {
		if operation.Intent != OperationIntentRecover {
			return fmt.Errorf("adopted operation must have Recover intent, got %q", operation.Intent)
		}
		if operation.CommittedTopologyGeneration == 0 {
			return errors.New("adopted operation must carry a committed topology generation")
		}
		if operation.BackendOperationID != "" {
			return errors.New("adopted operation must not claim a backend operation ID")
		}
		if operation.Phase != OperationPhaseCommitted && operation.Phase != OperationPhaseUnknown {
			return fmt.Errorf("adopted operation cannot have phase %q", operation.Phase)
		}
	}
	return nil
}

func validatePlan(plan OperationPlan, topology MembershipTopology) error {
	if plan.TargetReplicas < 0 {
		return fmt.Errorf("plan target replicas must not be negative: %d", plan.TargetReplicas)
	}
	if err := validateReplicaIDs("nominated", plan.NominatedReplicas); err != nil {
		return err
	}

	activeReplicas := topology.ReplicaCount()

	switch plan.Intent {
	case OperationIntentGrow:
		if plan.TargetReplicas <= activeReplicas {
			return fmt.Errorf("grow target %d must exceed active replicas %d", plan.TargetReplicas, activeReplicas)
		}
		if len(plan.NominatedReplicas) != 0 {
			return errors.New("grow plan must not nominate retiring replicas")
		}
	case OperationIntentShrink:
		if plan.TargetReplicas >= activeReplicas {
			return fmt.Errorf("shrink target %d must be below active replicas %d", plan.TargetReplicas, activeReplicas)
		}
	case OperationIntentRecover:
		// Recovery may preserve, reduce, or restore cardinality while changing native identities.
	case OperationIntentRetire:
		if plan.TargetReplicas != 0 {
			return fmt.Errorf("retire target must be zero: %d", plan.TargetReplicas)
		}
	default:
		return fmt.Errorf("invalid operation intent %q", plan.Intent)
	}

	// Reductions require the complete exact victim set before traffic withdrawal may begin.
	removedReplicas := activeReplicas - plan.TargetReplicas
	if removedReplicas > 0 {
		if int32(len(plan.NominatedReplicas)) != removedReplicas {
			return fmt.Errorf(
				"target %d from %d active replicas requires %d nominated replicas, got %d",
				plan.TargetReplicas,
				activeReplicas,
				removedReplicas,
				len(plan.NominatedReplicas),
			)
		}
		if err := requireReplicaSubset(plan.NominatedReplicas, topologyReplicaIDs(topology), "nominated"); err != nil {
			return err
		}
	} else if len(plan.NominatedReplicas) != 0 {
		return errors.New("non-reducing plan must not nominate retiring replicas")
	}
	return nil
}

func validateTopology(topology MembershipTopology) error {
	if topology.Generation < 0 {
		return fmt.Errorf("topology generation must not be negative: %d", topology.Generation)
	}

	replicas := make(map[ReplicaID]struct{}, len(topology.Replicas))
	nativeMembers := make(map[NativeMemberID]struct{})
	for _, replica := range topology.Replicas {
		if replica.ReplicaID == "" {
			return errors.New("topology replica ID must not be empty")
		}
		if _, exists := replicas[replica.ReplicaID]; exists {
			return fmt.Errorf("duplicate topology replica ID %q", replica.ReplicaID)
		}
		replicas[replica.ReplicaID] = struct{}{}

		if len(replica.NativeMembers) == 0 {
			return fmt.Errorf("topology replica %q has no native members", replica.ReplicaID)
		}
		for _, member := range replica.NativeMembers {
			if member == "" {
				return fmt.Errorf("topology replica %q has an empty native member ID", replica.ReplicaID)
			}
			if _, exists := nativeMembers[member]; exists {
				return fmt.Errorf("duplicate native member ID %q", member)
			}
			nativeMembers[member] = struct{}{}
		}
	}
	return nil
}

func validateMembershipCapabilities(capabilities MembershipCapabilities) error {
	seen := make(map[OperationIntent]struct{}, len(capabilities.Intents))
	for _, intent := range capabilities.Intents {
		if !validOperationIntent(intent) {
			return fmt.Errorf("invalid supported membership intent %q", intent)
		}
		if _, exists := seen[intent]; exists {
			return fmt.Errorf("duplicate supported membership intent %q", intent)
		}
		seen[intent] = struct{}{}
	}
	return nil
}

func validateOperationReplicaGeometry(operation Operation) error {
	baseReplicas := int32(len(operation.BaseReplicas))
	joiningReplicas := int32(len(operation.JoiningReplicas))
	nominatedReplicas := int32(len(operation.NominatedReplicas))

	// The declared intent constrains geometry even when durable state did not originate from this coordinator.
	switch operation.Intent {
	case OperationIntentGrow:
		if operation.TargetReplicas <= baseReplicas {
			return fmt.Errorf(
				"grow target %d must exceed %d base replicas",
				operation.TargetReplicas,
				baseReplicas,
			)
		}
	case OperationIntentShrink:
		if operation.TargetReplicas >= baseReplicas {
			return fmt.Errorf(
				"shrink target %d must be below %d base replicas",
				operation.TargetReplicas,
				baseReplicas,
			)
		}
	case OperationIntentRetire:
		if operation.TargetReplicas != 0 || baseReplicas == 0 {
			return fmt.Errorf(
				"retire operation requires a non-empty base topology and target zero, got base %d and target %d",
				baseReplicas,
				operation.TargetReplicas,
			)
		}
	case OperationIntentRecover:
		// Recovery may reduce, restore, or preserve cardinality while correlating exact identities.
	default:
		return fmt.Errorf("invalid operation intent %q", operation.Intent)
	}

	// Joining and retiring identities always describe disjoint logical replicas.
	if overlap := intersectReplicaIDs(operation.JoiningReplicas, operation.NominatedReplicas); len(overlap) != 0 {
		return fmt.Errorf("replicas cannot be both joining and nominated: %v", overlap)
	}

	// A reduction must name every removed base replica and cannot introduce a joiner.
	if operation.TargetReplicas < baseReplicas {
		expectedNominations := baseReplicas - operation.TargetReplicas
		if nominatedReplicas != expectedNominations {
			return fmt.Errorf(
				"target %d from %d base replicas requires %d nominated replicas, got %d",
				operation.TargetReplicas,
				baseReplicas,
				expectedNominations,
				nominatedReplicas,
			)
		}
		if joiningReplicas != 0 {
			return errors.New("reducing operation must not contain joining replicas")
		}
		return requireReplicaSubset(operation.NominatedReplicas, operation.BaseReplicas, "nominated")
	}

	// An expansion freezes every added identity before submission and cannot retire a base replica.
	if operation.TargetReplicas > baseReplicas {
		if nominatedReplicas != 0 {
			return errors.New("expanding operation must not nominate retiring replicas")
		}
		expectedJoiners := operation.TargetReplicas - baseReplicas
		joinersMayBeUnfrozen := operation.Phase == OperationPhasePending ||
			((operation.Phase == OperationPhaseUnknown ||
				operation.Phase == OperationPhaseAborting ||
				operation.Phase == OperationPhaseAborted) &&
				operation.CommittedTopologyGeneration == 0 &&
				joiningReplicas == 0)
		if !joinersMayBeUnfrozen && joiningReplicas != expectedJoiners {
			return fmt.Errorf(
				"target %d from %d base replicas requires %d joining replicas, got %d",
				operation.TargetReplicas,
				baseReplicas,
				expectedJoiners,
				joiningReplicas,
			)
		}
		if overlap := intersectReplicaIDs(operation.JoiningReplicas, operation.BaseReplicas); len(overlap) != 0 {
			return fmt.Errorf("joining replicas are already active in the base topology: %v", overlap)
		}
		return nil
	}

	// A cardinally stable recovery may remap native members but not logical replica identities.
	if joiningReplicas != 0 || nominatedReplicas != 0 {
		return errors.New("cardinally stable operation must not add or nominate logical replicas")
	}
	return nil
}

func validateReplicaIDs(label string, replicaIDs []ReplicaID) error {
	seen := make(map[ReplicaID]struct{}, len(replicaIDs))
	for _, replicaID := range replicaIDs {
		if replicaID == "" {
			return fmt.Errorf("%s replica ID must not be empty", label)
		}
		if _, exists := seen[replicaID]; exists {
			return fmt.Errorf("duplicate %s replica ID %q", label, replicaID)
		}
		seen[replicaID] = struct{}{}
	}
	return nil
}

func requireReplicaSubset(subset []ReplicaID, superset []ReplicaID, label string) error {
	allowed := make(map[ReplicaID]struct{}, len(superset))
	for _, replicaID := range superset {
		allowed[replicaID] = struct{}{}
	}

	for _, replicaID := range subset {
		if _, exists := allowed[replicaID]; !exists {
			return fmt.Errorf("%s replica %q is not in the permitted identity set", label, replicaID)
		}
	}
	return nil
}

func validateFailure(failure *OperationFailure) error {
	if failure == nil {
		return errors.New("failed operation must carry a structured failure")
	}
	if failure.Classification != FailureClassificationRetryable &&
		failure.Classification != FailureClassificationTerminal {
		return fmt.Errorf("invalid failure classification %q", failure.Classification)
	}
	if failure.Reason == "" {
		return errors.New("failure reason must not be empty")
	}
	return nil
}

func validOperationPhase(phase OperationPhase) bool {
	switch phase {
	case OperationPhasePending,
		OperationPhaseSubmitting,
		OperationPhaseAccepted,
		OperationPhaseCommitting,
		OperationPhaseCommitted,
		OperationPhaseFailed,
		OperationPhaseUnknown,
		OperationPhaseAborting,
		OperationPhaseAborted:
		return true
	default:
		return false
	}
}

func validOperationIntent(intent OperationIntent) bool {
	switch intent {
	case OperationIntentGrow,
		OperationIntentShrink,
		OperationIntentRecover,
		OperationIntentRetire:
		return true
	default:
		return false
	}
}

func normalizePlan(plan OperationPlan) OperationPlan {
	plan.NominatedReplicas = normalizeReplicaIDs(plan.NominatedReplicas)
	return plan
}

func operationCarriesPlan(operation Operation, plan OperationPlan) bool {
	normalized := normalizePlan(plan)

	return operation.PlanID == normalized.ID &&
		operation.Intent == normalized.Intent &&
		operation.TargetReplicas == normalized.TargetReplicas &&
		sameReplicaIDs(operation.NominatedReplicas, normalized.NominatedReplicas)
}

func operationMatchesBaseTopology(operation Operation, topology MembershipTopology) bool {
	return operation.BaseTopologyGeneration == topology.Generation &&
		sameReplicaIDs(operation.BaseReplicas, topologyReplicaIDs(topology))
}

func operationMatchesCommittedTopology(operation Operation, topology MembershipTopology) bool {
	return topology.ReplicaCount() == operation.TargetReplicas &&
		sameReplicaIDs(expectedCommittedReplicaIDs(operation), topologyReplicaIDs(topology))
}

func expectedCommittedReplicaIDs(operation Operation) []ReplicaID {
	expected := make(map[ReplicaID]struct{}, len(operation.BaseReplicas)+len(operation.JoiningReplicas))
	for _, replicaID := range operation.BaseReplicas {
		expected[replicaID] = struct{}{}
	}
	for _, replicaID := range operation.NominatedReplicas {
		delete(expected, replicaID)
	}
	for _, replicaID := range operation.JoiningReplicas {
		expected[replicaID] = struct{}{}
	}

	replicaIDs := make([]ReplicaID, 0, len(expected))
	for replicaID := range expected {
		replicaIDs = append(replicaIDs, replicaID)
	}
	slices.Sort(replicaIDs)
	return replicaIDs
}

func topologyReplicaIDs(topology MembershipTopology) []ReplicaID {
	replicaIDs := make([]ReplicaID, len(topology.Replicas))
	for i, replica := range topology.Replicas {
		replicaIDs[i] = replica.ReplicaID
	}
	return normalizeReplicaIDs(replicaIDs)
}

func normalizeReplicaIDs(replicaIDs []ReplicaID) []ReplicaID {
	normalized := slices.Clone(replicaIDs)
	slices.Sort(normalized)
	return normalized
}

func normalizeUniqueReplicaIDs(replicaIDs []ReplicaID) []ReplicaID {
	if len(replicaIDs) == 0 {
		return nil
	}

	seen := make(map[ReplicaID]struct{}, len(replicaIDs))
	normalized := make([]ReplicaID, 0, len(replicaIDs))
	for _, replicaID := range replicaIDs {
		if _, exists := seen[replicaID]; exists {
			continue
		}
		seen[replicaID] = struct{}{}
		normalized = append(normalized, replicaID)
	}
	slices.Sort(normalized)
	return normalized
}

func sameReplicaIDs(left []ReplicaID, right []ReplicaID) bool {
	return slices.Equal(normalizeReplicaIDs(left), normalizeReplicaIDs(right))
}

func intersectReplicaIDs(left []ReplicaID, right []ReplicaID) []ReplicaID {
	rightSet := make(map[ReplicaID]struct{}, len(right))
	for _, replicaID := range right {
		rightSet[replicaID] = struct{}{}
	}

	intersection := make([]ReplicaID, 0)
	for _, replicaID := range left {
		if _, exists := rightSet[replicaID]; exists {
			intersection = append(intersection, replicaID)
		}
	}
	return normalizeReplicaIDs(intersection)
}

func differenceReplicaIDs(left []ReplicaID, right []ReplicaID) []ReplicaID {
	rightSet := make(map[ReplicaID]struct{}, len(right))
	for _, replicaID := range right {
		rightSet[replicaID] = struct{}{}
	}

	difference := make([]ReplicaID, 0)
	for _, replicaID := range left {
		if _, exists := rightSet[replicaID]; !exists {
			difference = append(difference, replicaID)
		}
	}
	return normalizeReplicaIDs(difference)
}

func membershipRequest(operation Operation) MembershipRequest {
	return MembershipRequest{
		ID:                     operation.ID,
		Attempt:                operation.Attempt,
		Intent:                 operation.Intent,
		BaseTopologyGeneration: operation.BaseTopologyGeneration,
		BaseReplicas:           slices.Clone(operation.BaseReplicas),
		TargetReplicas:         operation.TargetReplicas,
		JoiningReplicas:        slices.Clone(operation.JoiningReplicas),
		NominatedReplicas:      slices.Clone(operation.NominatedReplicas),
	}
}

func setQueuedTarget(operation *Operation, desiredReplicas int32) bool {
	if desiredReplicas == operation.TargetReplicas {
		if operation.QueuedTargetReplicas == nil {
			return false
		}
		operation.QueuedTargetReplicas = nil
		return true
	}
	if operation.QueuedTargetReplicas != nil && *operation.QueuedTargetReplicas == desiredReplicas {
		return false
	}

	queuedTarget := desiredReplicas
	operation.QueuedTargetReplicas = &queuedTarget
	return true
}

func transitionOperation(operation *Operation, phase OperationPhase, now time.Time) *Operation {
	transitioned := cloneOperation(operation)
	transitioned.Phase = phase
	transitioned.LastTransitionTime = now
	return transitioned
}

func cloneOperation(operation *Operation) *Operation {
	if operation == nil {
		return nil
	}

	cloned := *operation
	cloned.BaseReplicas = slices.Clone(operation.BaseReplicas)
	cloned.JoiningReplicas = slices.Clone(operation.JoiningReplicas)
	cloned.NominatedReplicas = slices.Clone(operation.NominatedReplicas)
	cloned.CleanupReplicas = slices.Clone(operation.CleanupReplicas)
	cloned.Failure = cloneFailure(operation.Failure)
	if operation.QueuedTargetReplicas != nil {
		queuedTarget := *operation.QueuedTargetReplicas
		cloned.QueuedTargetReplicas = &queuedTarget
	}
	return &cloned
}

func cloneFailure(failure *OperationFailure) *OperationFailure {
	if failure == nil {
		return nil
	}

	cloned := *failure
	return &cloned
}

func cloneTopology(topology MembershipTopology) MembershipTopology {
	cloned := MembershipTopology{
		Generation: topology.Generation,
		Replicas:   make([]ReplicaMembership, len(topology.Replicas)),
	}
	for i, replica := range topology.Replicas {
		cloned.Replicas[i] = ReplicaMembership{
			ReplicaID:     replica.ReplicaID,
			NativeMembers: slices.Clone(replica.NativeMembers),
		}
	}
	return cloned
}
