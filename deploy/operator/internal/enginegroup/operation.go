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

// ErrMembershipOperationUnsupported identifies a definitive, side-effect-free rejection of a semantic plan, exact
// request, or observed transition. MembershipAdapter validation methods must wrap this error for every rejection
// that cannot succeed unchanged; callers treat every other validation error as transient and retryable.
var ErrMembershipOperationUnsupported = errors.New("membership operation is unsupported")

// submissionPreflightError distinguishes side-effect-free exact-request rejection from an ambiguous submit error.
type submissionPreflightError struct {
	cause error
}

func (e *submissionPreflightError) Error() string {
	return e.cause.Error()
}

func (e *submissionPreflightError) Unwrap() error {
	return e.cause
}

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
	// Reject invalid desired or durable state before observing or mutating the backend. A queued candidate plan is
	// validated only at the branch that can consume it, so malformed future intent cannot suppress observation and
	// safety reconciliation for the current durable operation.
	if err := validateOperationStateInput(input); err != nil {
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
		Operation:        cloneOperation(input.Operation),
		Topology:         cloneTopology(topology),
		TopologyObserved: true,
	}

	// Capacity launch may trigger a backend-native recovery before a Pending operation is submitted. Adopt only the
	// exact already-durable recovery plan; the workflow fences the observed topology before persisting this result.
	if operationNeedsObservedRecoveryAdoption(result.Operation, topology) {
		return c.adoptObservedPlan(
			ctx,
			input,
			result,
			result.Operation.BaseTopology,
			operationPlan(*result.Operation),
		)
	}

	// A new explicit recovery plan may adopt a later topology after durable prior serving-state proof.
	if adopted, handled, err := c.adoptSupersedingObservedRecovery(ctx, input, result); handled {
		return adopted, err
	}

	// Start a durable pending operation only when current observations require one.
	if result.Operation == nil {
		return c.beginOperation(ctx, input, result)
	}

	// Dispatch by durable phase so every external call is preceded by a persisted marker.
	return c.advanceOperation(ctx, input, result)
}

func operationNeedsObservedRecoveryAdoption(
	operation *Operation,
	topology MembershipTopology,
) bool {
	if operation == nil || operation.Intent != OperationIntentRecover {
		return false
	}
	unsubmitted := operation.Phase == OperationPhasePending ||
		(operation.Phase == OperationPhaseFailed && operation.CommittedTopology == nil)
	return unsubmitted && !servingVerificationTopologiesEqual(topology, operation.BaseTopology)
}

func (c *OperationCoordinator) adoptSupersedingObservedRecovery(
	ctx context.Context,
	input OperationInput,
	result OperationResult,
) (OperationResult, bool, error) {
	operation := result.Operation
	if operation == nil ||
		!operationAllowsObservedAdoption(*operation) ||
		input.Plan == nil ||
		input.Plan.ID == operation.PlanID {
		return result, false, nil
	}

	baseline := recoveryBaselineTopology(*operation)
	if baseline == nil || servingVerificationTopologiesEqual(result.Topology, *baseline) {
		return result, false, nil
	}
	if err := validateOperationCandidate(input.Operation, input.Plan); err != nil {
		return result, true, err
	}
	adopted, err := c.adoptObservedPlan(ctx, input, result, *baseline, *input.Plan)
	return adopted, true, err
}

func (c *OperationCoordinator) beginOperation(
	ctx context.Context,
	input OperationInput,
	result OperationResult,
) (OperationResult, error) {
	if err := validateOperationCandidate(nil, input.Plan); err != nil {
		return result, err
	}
	plan, err := desiredPlan(input, result.Topology)
	if err != nil || plan == nil {
		return result, err
	}
	capability, err := c.validatePlan(ctx, input.GroupID, result.Topology, *plan)
	if err != nil {
		return result, err
	}
	operation, err := c.newPendingOperation(input, result.Topology, *plan, capability)
	if err != nil {
		return result, err
	}
	result.Operation = operation
	result.OperationChanged = true
	return result, nil
}

func (c *OperationCoordinator) advanceOperation(
	ctx context.Context,
	input OperationInput,
	result OperationResult,
) (OperationResult, error) {
	switch result.Operation.Phase {
	case OperationPhasePending:
		return c.advanceResolved(input, result)
	case OperationPhaseFailed:
		if result.Operation.CommittedTopology != nil {
			return result, nil
		}
		return c.advanceResolved(input, result)
	case OperationPhaseSubmitting:
		return c.advanceSubmitting(ctx, input, result)
	case OperationPhaseAccepted, OperationPhaseCommitting:
		return c.observeInFlight(ctx, input, result)
	case OperationPhaseUnknown:
		if result.Operation.CompensationTopology != nil {
			return result, nil
		}
		return c.observeInFlight(ctx, input, result)
	case OperationPhaseCommitted, OperationPhaseAborting, OperationPhaseAborted:
		return result, nil
	default:
		return result, fmt.Errorf("unsupported operation phase %q", result.Operation.Phase)
	}
}

func (c *OperationCoordinator) validatePlan(
	ctx context.Context,
	groupID GroupID,
	topology MembershipTopology,
	plan OperationPlan,
) (ResolvedOperationCapability, error) {
	capabilities, err := c.membership.ObserveCapabilities(ctx, groupID)
	if err != nil {
		return ResolvedOperationCapability{}, fmt.Errorf("observe membership capabilities: %w", err)
	}
	if err := validateMembershipCapabilities(capabilities); err != nil {
		return ResolvedOperationCapability{}, fmt.Errorf("validate membership capabilities: %w", err)
	}

	// Resolve semantic support only from the complete capability set advertised for planning and status.
	capability, err := c.membership.ValidatePlan(ctx, groupID, cloneTopology(topology), normalizePlan(plan))
	if err != nil {
		return ResolvedOperationCapability{}, fmt.Errorf("validate membership operation plan: %w", err)
	}
	if err := validateCapabilityForPlan(capability, plan, topology); err != nil {
		return ResolvedOperationCapability{}, fmt.Errorf("validate resolved membership capability: %w", err)
	}
	if !slices.Contains(capabilities.OperationShapes, capability.Shape) {
		return ResolvedOperationCapability{}, fmt.Errorf(
			"resolved membership operation shape %q was not advertised: %w",
			capability.Shape,
			ErrMembershipOperationUnsupported,
		)
	}
	return capability, nil
}

func (c *OperationCoordinator) adoptObservedPlan(
	ctx context.Context,
	input OperationInput,
	result OperationResult,
	previousTopology MembershipTopology,
	plan OperationPlan,
) (OperationResult, error) {
	transition, err := newObservedMembershipTransition(previousTopology, result.Topology, plan)
	if err != nil {
		return result, err
	}
	capability, err := c.validateObservedTransition(ctx, input.GroupID, transition)
	if err != nil {
		return result, err
	}
	adopted, err := c.adoptObservedRecovery(input, transition, capability)
	if err != nil {
		return result, err
	}
	result.Operation = adopted
	result.OperationChanged = true
	return result, nil
}

func (c *OperationCoordinator) validateObservedTransition(
	ctx context.Context,
	groupID GroupID,
	transition ObservedMembershipTransition,
) (ResolvedOperationCapability, error) {
	capabilities, err := c.membership.ObserveCapabilities(ctx, groupID)
	if err != nil {
		return ResolvedOperationCapability{}, fmt.Errorf("observe membership capabilities: %w", err)
	}
	if err := validateMembershipCapabilities(capabilities); err != nil {
		return ResolvedOperationCapability{}, fmt.Errorf("validate membership capabilities: %w", err)
	}
	capability, err := c.membership.ValidateObservedTransition(
		ctx,
		groupID,
		cloneObservedMembershipTransition(transition),
	)
	if err != nil {
		return ResolvedOperationCapability{}, fmt.Errorf("validate observed membership transition: %w", err)
	}
	if err := validateCapabilityForObservedTransition(capability, transition); err != nil {
		return ResolvedOperationCapability{}, fmt.Errorf("validate observed transition capability: %w", err)
	}
	if !slices.Contains(capabilities.OperationShapes, capability.Shape) {
		return ResolvedOperationCapability{}, fmt.Errorf(
			"resolved observed membership operation shape %q was not advertised: %w",
			capability.Shape,
			ErrMembershipOperationUnsupported,
		)
	}

	// The controller first learns about an adopted transition after the engine has committed it. Without durable
	// pre-transition drain evidence, whole-group quiescence cannot be established retroactively.
	if capability.TrafficRequirement == ReconfigurationTrafficQuiesceGroup {
		return ResolvedOperationCapability{}, fmt.Errorf(
			"cannot adopt observed membership transition requiring historical whole-group quiescence: %w",
			ErrMembershipOperationUnsupported,
		)
	}
	if capability.VerificationRequirement != ServingVerificationRequired {
		return ResolvedOperationCapability{}, fmt.Errorf(
			"observed membership transition requires post-adoption serving verification: %w",
			ErrMembershipOperationUnsupported,
		)
	}
	return capability, nil
}

// PrepareSubmission returns a durable Submitting transition without calling the backend.
// operation must be non-nil. Callers must persist the returned operation and recheck external
// prerequisites before calling Submit.
func (c *OperationCoordinator) PrepareSubmission(
	ctx context.Context,
	groupID GroupID,
	operation *Operation,
	joiningReplicas []ReplicaIncarnation,
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
		resolvedJoiners := normalizeReplicaIncarnations(joiningReplicas)
		if len(prepared.JoiningReplicas) != 0 &&
			!sameReplicaIncarnations(prepared.JoiningReplicas, resolvedJoiners) {
			return nil, errors.New("submission cannot replace planned joining replica identities")
		}
		if len(prepared.JoiningReplicas) == 0 {
			prepared.JoiningReplicas = resolvedJoiners
		}
	} else {
		if len(joiningReplicas) > 0 && !sameReplicaIncarnations(joiningReplicas, prepared.JoiningReplicas) {
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
	if err := c.membership.ValidateRequest(ctx, groupID, membershipRequest(*prepared)); err != nil {
		return nil, fmt.Errorf("validate exact membership request: %w", err)
	}
	return prepared, nil
}

// ResolvePreparationRejection re-observes membership after a definitive exact-request preflight rejection. It
// distinguishes true capability loss on the unchanged base from a backend-native transition that raced preflight.
func (c *OperationCoordinator) ResolvePreparationRejection(
	ctx context.Context,
	groupID GroupID,
	operation *Operation,
) (OperationResult, error) {
	if groupID == "" {
		return OperationResult{Operation: cloneOperation(operation)}, errors.New("group ID must not be empty")
	}
	if operation == nil {
		return OperationResult{}, errors.New("preparation rejection requires a durable operation")
	}
	if err := validateOperation(*operation); err != nil {
		return OperationResult{Operation: cloneOperation(operation)}, err
	}

	topology, err := c.membership.ObserveTopology(ctx, groupID)
	if err != nil {
		return OperationResult{Operation: cloneOperation(operation)}, fmt.Errorf(
			"observe membership topology after preparation rejection: %w",
			err,
		)
	}
	if err := validateTopology(topology); err != nil {
		return OperationResult{Operation: cloneOperation(operation)}, fmt.Errorf(
			"validate membership topology after preparation rejection: %w",
			err,
		)
	}
	result := OperationResult{
		Operation:        cloneOperation(operation),
		Topology:         cloneTopology(topology),
		TopologyObserved: true,
	}
	if operationMatchesBaseTopology(*operation, topology) {
		return result, nil
	}
	if operation.Intent == OperationIntentRecover {
		plan := operationPlan(*operation)
		input := OperationInput{
			GroupID:         groupID,
			SpecGeneration:  operation.SpecGeneration,
			DesiredReplicas: operation.TargetReplicas,
			Plan:            &plan,
			Operation:       operation,
		}
		return c.adoptObservedPlan(
			ctx,
			input,
			result,
			operation.BaseTopology,
			plan,
		)
	}
	return c.markBaseDriftUnknown(result), nil
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
		Operation:        cloneOperation(operation),
		Topology:         cloneTopology(topology),
		TopologyObserved: true,
	}
	if !operationMatchesBaseTopology(*operation, topology) {
		return c.resolveSubmissionBaseDrift(ctx, groupID, result)
	}

	// Revalidate the exact frozen request, while leaving atomic validation authoritative at submission.
	request := membershipRequest(*operation)
	if err := c.membership.ValidateRequest(ctx, groupID, request); err != nil {
		if errors.Is(err, ErrMembershipOperationUnsupported) {
			refreshed, observeErr := c.membership.ObserveTopology(ctx, groupID)
			if observeErr != nil {
				return result, errors.Join(
					&submissionPreflightError{cause: fmt.Errorf("validate exact membership request: %w", err)},
					fmt.Errorf("re-observe membership topology after rejection: %w", observeErr),
				)
			}
			if validateErr := validateTopology(refreshed); validateErr != nil {
				return result, errors.Join(
					&submissionPreflightError{cause: fmt.Errorf("validate exact membership request: %w", err)},
					fmt.Errorf("validate re-observed membership topology after rejection: %w", validateErr),
				)
			}
			if !operationMatchesBaseTopology(*operation, refreshed) {
				result.Topology = cloneTopology(refreshed)
				return c.resolveSubmissionBaseDrift(ctx, groupID, result)
			}
		}
		return result, &submissionPreflightError{
			cause: fmt.Errorf("validate exact membership request: %w", err),
		}
	}

	// Submit the exact persisted request; the backend contract makes replay idempotent by operation ID.
	observed, err := c.membership.SubmitOperation(ctx, groupID, request)
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
	if err := validateBackendOperation(observed); err != nil {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}
	if observed.Phase == BackendOperationPhaseAbsent {
		if result.Operation.Intent == OperationIntentRecover {
			plan := operationPlan(*result.Operation)
			return c.adoptObservedPlan(
				ctx,
				OperationInput{
					GroupID:         groupID,
					SpecGeneration:  result.Operation.SpecGeneration,
					DesiredReplicas: result.Operation.TargetReplicas,
					Plan:            &plan,
					Operation:       result.Operation,
				},
				result,
				result.Operation.BaseTopology,
				plan,
			)
		}
		return c.markBaseDriftUnknown(result), nil
	}
	return c.applyBackendObservation(ctx, groupID, result, observed)
}

func (c *OperationCoordinator) advanceResolved(input OperationInput, result OperationResult) (OperationResult, error) {
	// A changed base was not caused by this unsubmitted operation. Do not reinterpret or compensate for it.
	if !operationMatchesBaseTopology(*result.Operation, result.Topology) {
		// A definitive backend failure cannot later commit. Preserve it for workflow-owned fail-closed compensation.
		if result.Operation.Phase == OperationPhaseFailed {
			return result, nil
		}
		return c.markBaseDriftUnknown(result), nil
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
	if err := validateBackendOperation(observed); err != nil {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}
	if observed.Phase != BackendOperationPhaseAbsent {
		return c.applyBackendObservation(ctx, input.GroupID, result, observed)
	}

	// Base drift makes the request stale after the backend proves this attempt absent. A racing replay is also
	// harmless because SubmitOperation atomically refuses its now-stale base generation and identity set.
	if !operationMatchesBaseTopology(*result.Operation, result.Topology) {
		if result.Operation.Intent == OperationIntentRecover {
			return c.adoptObservedPlan(
				ctx,
				input,
				result,
				result.Operation.BaseTopology,
				operationPlan(*result.Operation),
			)
		}
		return c.markBaseDriftUnknown(result), nil
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
	if err := validateBackendOperation(observed); err != nil {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}
	if observed.Phase == BackendOperationPhaseAbsent {
		if result.Operation.Phase == OperationPhaseUnknown {
			if operationMatchesBaseTopology(*result.Operation, result.Topology) {
				result.Operation.Failure = &OperationFailure{
					Classification: FailureClassificationTerminal,
					Reason:         "OperationAbsentAfterUncertainty",
					Message:        "the backend definitively reports no operation and the base topology is unchanged",
				}
				return c.setObservedPhase(result, OperationPhaseFailed), nil
			}

			// A definitively absent attempt cannot race a later commit. Preserve the old base as the cleanup boundary so
			// workflow compensation can fence the independently changed topology and reach explicit recovery.
			result.Operation.CompensationTopology = topologyPointer(result.Operation.BaseTopology)
			result.Operation.Failure = &OperationFailure{
				Classification: FailureClassificationTerminal,
				Reason:         "OperationAbsentAfterTopologyDrift",
				Message:        "the backend definitively reports no operation after authoritative topology changed",
			}
			result.Operation = transitionOperation(result.Operation, OperationPhaseAborting, c.now())
			result.OperationChanged = true
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
	if err := validateBackendOperation(observed); err != nil {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}

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
		result.Operation.CommittedTopology != nil &&
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
	if (observed.Phase == BackendOperationPhaseAccepted ||
		observed.Phase == BackendOperationPhaseCommitting) &&
		!operationMatchesBaseTopology(*result.Operation, result.Topology) {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
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
	// A successful membership transaction must carry an exact, newer committed topology.
	if observed.CommittedTopology == nil {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}
	committedTopology := cloneTopology(*observed.CommittedTopology)
	if err := validateTopology(committedTopology); err != nil ||
		committedTopology.Generation <= result.Operation.BaseTopology.Generation ||
		committedTopology.ReplicaCount() != observed.TargetReplicas ||
		!operationMatchesCommittedTopology(*result.Operation, committedTopology) {
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

	// Durable commit proof is immutable; only the exact same generation and member mapping may recover Unknown.
	if result.Operation.CommittedTopology != nil &&
		!servingVerificationTopologiesEqual(committedTopology, *result.Operation.CommittedTopology) {
		return result, nil
	}

	// Wait for an eventually consistent topology view to reach the committed generation.
	if topology.Generation < committedTopology.Generation {
		if result.Operation.CommittedTopology != nil {
			return result, nil
		}
		return c.setObservedPhase(result, OperationPhaseCommitting), nil
	}

	// Preserve skipped exact commit proof when a later authoritative topology contains only expected survivors.
	if topology.Generation > committedTopology.Generation {
		if err := requireReplicaSubset(
			topologyReplicaIDs(topology),
			topologyReplicaIDs(committedTopology),
			"later survivor",
		); err != nil {
			return c.setObservedPhase(result, OperationPhaseUnknown), nil
		}
		if result.Operation.CommittedTopology == nil {
			result.Operation.CommittedTopology = topologyPointer(committedTopology)
			result.OperationChanged = true
		}
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}

	// An identity-different topology at the claimed commit generation cannot prove the operation outcome.
	if !servingVerificationTopologiesEqual(committedTopology, topology) {
		return c.setObservedPhase(result, OperationPhaseUnknown), nil
	}

	// Record commit only after the exact authoritative topology is visible.
	if result.Operation.CommittedTopology == nil {
		result.Operation.CommittedTopology = topologyPointer(topology)
		result.OperationChanged = true
	}
	if topology.ReplicaCount() > 0 &&
		result.Operation.Capability.VerificationRequirement == ServingVerificationRequired &&
		result.Operation.ServingVerificationAttempt == 0 {
		result.Operation.ServingVerificationAttempt = 1
		result.Operation.ServingVerificationTarget = topologyPointer(topology)
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

func (c *OperationCoordinator) markBaseDriftUnknown(result OperationResult) OperationResult {
	// An independently changed topology has not been validated as a recovery transition. The old operation may no
	// longer mutate membership, traffic, or capacity; a higher-level controller must recreate the group explicitly.
	result.Operation.Failure = nil
	return c.setObservedPhase(result, OperationPhaseUnknown)
}

func (c *OperationCoordinator) newPendingOperation(
	input OperationInput,
	topology MembershipTopology,
	plan OperationPlan,
	capability ResolvedOperationCapability,
) (*Operation, error) {
	operationID := c.newOperationID()
	if operationID == "" {
		return nil, errors.New("generate membership operation ID: empty value")
	}

	now := c.now()
	operation := &Operation{
		ID:                 operationID,
		Attempt:            1,
		PlanID:             plan.ID,
		Intent:             plan.Intent,
		Capability:         capability,
		SpecGeneration:     input.SpecGeneration,
		BaseTopology:       cloneTopology(topology),
		TargetReplicas:     plan.TargetReplicas,
		RestoredMembership: cloneReplicaNativeMemberships(plan.RestoredMembership),
		NominatedReplicas:  slices.Clone(plan.NominatedReplicas),
		TargetMembership:   cloneReplicaMemberships(plan.TargetMembership),
		Phase:              OperationPhasePending,
		StartedAt:          now,
		LastTransitionTime: now,
	}
	setQueuedTarget(operation, input.DesiredReplicas)
	if err := validateOperation(*operation); err != nil {
		return nil, fmt.Errorf("validate planned membership operation: %w", err)
	}
	return operation, nil
}

func (c *OperationCoordinator) adoptObservedRecovery(
	input OperationInput,
	transition ObservedMembershipTransition,
	capability ResolvedOperationCapability,
) (*Operation, error) {
	// The exact observed transition was validated before choosing a capability; persist that same immutable payload.
	plan := transition.Plan

	// Mint a durable adopted record without implying that Dynamo submitted the engine-side transition.
	operationID := c.newOperationID()
	if operationID == "" {
		return nil, errors.New("generate adopted recovery operation ID: empty value")
	}
	now := c.now()
	operation := &Operation{
		ID:                 operationID,
		Attempt:            1,
		PlanID:             plan.ID,
		Intent:             OperationIntentRecover,
		Capability:         capability,
		SpecGeneration:     input.SpecGeneration,
		BaseTopology:       cloneTopology(transition.PreviousTopology),
		TargetReplicas:     plan.TargetReplicas,
		RestoredMembership: cloneReplicaNativeMemberships(plan.RestoredMembership),
		JoiningReplicas: topologyReplicaIncarnationsForIDs(
			transition.ObservedTopology,
			replicaNativeMembershipIDs(plan.RestoredMembership),
		),
		NominatedReplicas:  slices.Clone(plan.NominatedReplicas),
		TargetMembership:   cloneReplicaMemberships(plan.TargetMembership),
		Phase:              OperationPhaseCommitted,
		CommittedTopology:  topologyPointer(transition.ObservedTopology),
		Adopted:            true,
		StartedAt:          now,
		LastTransitionTime: now,
	}
	if transition.ObservedTopology.ReplicaCount() > 0 &&
		operation.Capability.VerificationRequirement == ServingVerificationRequired {
		operation.ServingVerificationAttempt = 1
		operation.ServingVerificationTarget = topologyPointer(transition.ObservedTopology)
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

func validateOperationStateInput(input OperationInput) error {
	if input.GroupID == "" {
		return errors.New("group ID must not be empty")
	}
	if input.SpecGeneration < 0 {
		return fmt.Errorf("spec generation must not be negative: %d", input.SpecGeneration)
	}
	if input.DesiredReplicas < 0 {
		return fmt.Errorf("desired replicas must not be negative: %d", input.DesiredReplicas)
	}
	if input.Operation != nil {
		if err := validateOperation(*input.Operation); err != nil {
			return err
		}
	}
	return nil
}

func validateOperationCandidate(operation *Operation, plan *OperationPlan) error {
	if plan == nil {
		return nil
	}
	if plan.ID == "" {
		return errors.New("explicit operation plan ID must not be empty")
	}
	if !validOperationIntent(plan.Intent) {
		return fmt.Errorf("invalid operation plan intent %q", plan.Intent)
	}
	if operation != nil &&
		operation.PlanID == plan.ID &&
		!operationCarriesPlan(*operation, *plan) {
		return fmt.Errorf("operation plan ID %q was reused with a different payload", plan.ID)
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
	if err := validateOperationIdentityAndCapability(operation); err != nil {
		return err
	}
	if err := validateOperationServingVerification(operation); err != nil {
		return err
	}
	if err := validateOperationReplicaReferences(operation); err != nil {
		return err
	}
	if err := validateOperationCapacityTargetState(operation); err != nil {
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

func validateOperationIdentityAndCapability(operation Operation) error {
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
	if err := validateCapabilityForOperation(operation.Capability, operation); err != nil {
		return err
	}
	return nil
}

func validateOperationServingVerification(operation Operation) error {
	if err := validateOperationServingVerificationHeader(operation); err != nil {
		return err
	}
	if operation.ServingVerificationAttempt == 0 {
		if operation.ServingVerificationTarget != nil || operation.ServingVerificationProof != nil {
			return errors.New("serving verification state requires a positive attempt")
		}
		return nil
	}
	if operation.ServingVerificationTarget == nil {
		return errors.New("serving verification attempt requires an exact topology target")
	}
	if err := validateServingVerificationTopology(*operation.ServingVerificationTarget); err != nil {
		return fmt.Errorf("validate durable serving verification target: %w", err)
	}
	if err := validateOperationServingVerificationTarget(operation); err != nil {
		return err
	}
	return validateOperationServingVerificationProof(operation)
}

func validateOperationServingVerificationHeader(operation Operation) error {
	if operation.ServingVerificationAttempt < 0 {
		return fmt.Errorf(
			"serving verification attempt must not be negative: %d",
			operation.ServingVerificationAttempt,
		)
	}
	if operation.Capability.VerificationRequirement == ServingVerificationNotRequired &&
		(operation.ServingVerificationAttempt != 0 ||
			operation.ServingVerificationTarget != nil ||
			operation.ServingVerificationProof != nil ||
			operation.CapacityRecoveryPhase != CapacityRecoveryPhaseNone) {
		return errors.New("operation without serving verification must not carry verification state")
	}
	return validateOperationCapacityRecoveryVerification(operation)
}

func validateOperationCapacityRecoveryVerification(operation Operation) error {
	if operation.CapacityRecoveryPhase != CapacityRecoveryPhaseNone && operation.ServingVerificationAttempt <= 0 {
		return errors.New("capacity recovery verification requires a positive verification attempt")
	}
	if operation.CapacityRecoveryPhase != CapacityRecoveryPhaseNone &&
		operation.Phase != OperationPhaseCommitted &&
		operation.Phase != OperationPhaseUnknown &&
		operation.Phase != OperationPhaseAborting &&
		operation.Phase != OperationPhaseAborted {
		return fmt.Errorf(
			"operation phase %q cannot carry pending capacity-recovery verification",
			operation.Phase,
		)
	}
	switch operation.CapacityRecoveryPhase {
	case CapacityRecoveryPhaseNone, CapacityRecoveryPhaseVerifying:
	case CapacityRecoveryPhaseRepairing:
		if operation.ServingVerificationProof != nil {
			return errors.New("capacity repair cannot retain a serving verification proof")
		}
	default:
		return fmt.Errorf("invalid capacity recovery phase %q", operation.CapacityRecoveryPhase)
	}
	return nil
}

func validateOperationServingVerificationTarget(operation Operation) error {
	// Verification state exists only after membership commit or while exact base-topology compensation is restoring.
	switch operation.Phase {
	case OperationPhaseCommitted, OperationPhaseFailed, OperationPhaseUnknown:
		if operation.CommittedTopology == nil {
			return fmt.Errorf("operation phase %q requires committed topology for serving verification", operation.Phase)
		}
		if !servingVerificationTopologiesEqual(
			*operation.ServingVerificationTarget,
			*operation.CommittedTopology,
		) {
			return errors.New("serving verification target does not match committed topology")
		}
	case OperationPhaseAborting, OperationPhaseAborted:
		if operation.CompensationTopology == nil {
			return fmt.Errorf("operation phase %q requires compensation topology for serving verification", operation.Phase)
		}
		if !servingVerificationTopologiesEqual(
			*operation.ServingVerificationTarget,
			*operation.CompensationTopology,
		) {
			return errors.New("serving verification target does not match compensation topology")
		}
	default:
		return fmt.Errorf("operation phase %q cannot carry serving verification state", operation.Phase)
	}
	return nil
}

func validateOperationServingVerificationProof(operation Operation) error {
	proof := operation.ServingVerificationProof
	if proof == nil {
		return nil
	}
	if err := validateServingVerificationProof(*proof); err != nil {
		return fmt.Errorf("validate durable serving verification proof: %w", err)
	}
	if proof.Phase != ServingVerificationPhasePassed {
		return fmt.Errorf(
			"durable serving verification proof must be Passed, got %q",
			proof.Phase,
		)
	}
	request := ServingVerificationRequest{
		OperationID:         operation.ID,
		Attempt:             operation.Attempt,
		VerificationAttempt: operation.ServingVerificationAttempt,
		Topology:            *operation.ServingVerificationTarget,
	}
	if !servingVerificationPassedForRequest(*proof, request) {
		return errors.New("durable serving verification proof does not match the exact verification request")
	}
	if !operationPhaseAllowsServingVerificationProof(operation.Phase) {
		return fmt.Errorf("operation phase %q cannot carry serving verification proof", operation.Phase)
	}
	if (operation.Phase == OperationPhaseCommitted || operation.Phase == OperationPhaseUnknown) &&
		operation.CommittedTopology != nil &&
		!servingVerificationProofMatchesOperation(proof, operation) {
		return errors.New("durable serving verification proof does not match committed membership")
	}
	return nil
}

func operationPhaseAllowsServingVerificationProof(phase OperationPhase) bool {
	switch phase {
	case OperationPhaseCommitted,
		OperationPhaseUnknown,
		OperationPhaseAborting,
		OperationPhaseAborted:
		return true
	default:
		return false
	}
}

func validateOperationReplicaReferences(operation Operation) error {
	if operation.SpecGeneration < 0 {
		return fmt.Errorf("operation spec generation must not be negative: %d", operation.SpecGeneration)
	}
	if err := validateTopology(operation.BaseTopology); err != nil {
		return fmt.Errorf("validate durable base topology: %w", err)
	}
	if operation.TargetReplicas < 0 {
		return fmt.Errorf("operation target replicas must not be negative: %d", operation.TargetReplicas)
	}
	if err := validateReplicaNativeMemberships("restored", operation.RestoredMembership); err != nil {
		return err
	}
	if err := validateReplicaIncarnations("joining", operation.JoiningReplicas, true); err != nil {
		return err
	}
	if err := validateReplicaIDs("nominated", operation.NominatedReplicas); err != nil {
		return err
	}
	if err := validateReplicaSlotBindings("cleanup", operation.CleanupReplicaSlots); err != nil {
		return err
	}
	if len(operation.CleanupReplicaSlots) != 0 &&
		operation.Phase != OperationPhaseAborting &&
		operation.Phase != OperationPhaseAborted {
		return fmt.Errorf("operation phase %q cannot carry abort cleanup replica slots", operation.Phase)
	}
	return nil
}

func validateOperationCapacityTargetState(operation Operation) error {
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
		operation.Phase != OperationPhaseFailed &&
		operation.Phase != OperationPhaseUnknown &&
		operation.Phase != OperationPhaseAborting &&
		operation.Phase != OperationPhaseAborted {
		return fmt.Errorf("operation phase %q cannot carry applied capacity target proof", operation.Phase)
	}
	if operation.CapacityTargetApplied &&
		operation.Phase == OperationPhaseFailed &&
		operation.CommittedTopology == nil {
		return errors.New("failed operation requires committed topology for an applied capacity target proof")
	}
	if operation.CapacityTargetApplied &&
		operation.Phase == OperationPhaseFailed &&
		operation.TargetReplicas >= operation.BaseTopology.ReplicaCount() {
		return errors.New("only a failed committed reduction may carry an applied capacity target proof")
	}
	if !operation.CapacityTargetApplied &&
		(operation.CapacityTargetReplicas != 0 || operation.CapacityTopologyGeneration != 0) {
		return errors.New("capacity target proof values require an applied capacity target")
	}
	if err := validateCapacityTargetProof(operation); err != nil {
		return err
	}
	return nil
}

func validateCapacityTargetProof(operation Operation) error {
	if !operation.CapacityTargetApplied ||
		operation.Phase == OperationPhaseAborting ||
		operation.Phase == OperationPhaseAborted {
		return nil
	}
	committedGeneration := committedTopologyGeneration(operation)
	if operation.CapacityTopologyGeneration < committedGeneration {
		return fmt.Errorf(
			"capacity target topology generation %d precedes committed generation %d",
			operation.CapacityTopologyGeneration,
			committedGeneration,
		)
	}
	if operation.CapacityTopologyGeneration == committedGeneration &&
		operation.CapacityTargetReplicas != operation.TargetReplicas {
		return fmt.Errorf(
			"capacity target %d does not match committed target %d",
			operation.CapacityTargetReplicas,
			operation.TargetReplicas,
		)
	}
	if operation.CapacityTopologyGeneration > committedGeneration &&
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
	if err := validateOperationTimestampsAndPhase(operation); err != nil {
		return err
	}
	if err := validateOperationFailureState(operation); err != nil {
		return err
	}
	if err := validateTerminalAdmissionFailure(operation); err != nil {
		return err
	}
	if err := validateOperationCommitState(operation); err != nil {
		return err
	}
	if err := validateOperationCompensationState(operation); err != nil {
		return err
	}
	return validateAdoptedOperationState(operation)
}

func validateTerminalAdmissionFailure(operation Operation) error {
	failure := operation.TerminalAdmissionFailure
	if failure == nil {
		return nil
	}
	if err := validateTrafficCommand(failure.Command); err != nil {
		return fmt.Errorf("validate terminal admission command: %w", err)
	}
	if failure.Command.Action != TrafficActionAdmit {
		return fmt.Errorf(
			"terminal admission failure must reference an Admit command, got %q",
			failure.Command.Action,
		)
	}
	if failure.Command.Request.OperationID != operation.ID {
		return fmt.Errorf(
			"terminal admission command belongs to operation %q, expected %q",
			failure.Command.Request.OperationID,
			operation.ID,
		)
	}
	if err := validateFailure(&failure.Failure); err != nil {
		return fmt.Errorf("validate terminal admission failure: %w", err)
	}
	if failure.Failure.Classification != FailureClassificationTerminal {
		return fmt.Errorf(
			"terminal admission failure requires terminal classification, got %q",
			failure.Failure.Classification,
		)
	}
	if operation.PostCommitComplete {
		return errors.New("operation with a terminal admission failure cannot be post-commit complete")
	}

	var authoritativeTopology *MembershipTopology
	switch operation.Phase {
	case OperationPhaseFailed:
		authoritativeTopology = operation.CommittedTopology
	case OperationPhaseAborting, OperationPhaseAborted:
		authoritativeTopology = operation.CompensationTopology
	case OperationPhaseUnknown:
		authoritativeTopology = operation.CompensationTopology
	default:
		return fmt.Errorf(
			"operation phase %q cannot carry a terminal admission failure",
			operation.Phase,
		)
	}
	if authoritativeTopology == nil {
		return fmt.Errorf(
			"operation phase %q requires an authoritative topology for a terminal admission failure",
			operation.Phase,
		)
	}
	if failure.Command.Request.TopologyGeneration != authoritativeTopology.Generation {
		return fmt.Errorf(
			"terminal admission topology generation %d does not match authoritative generation %d",
			failure.Command.Request.TopologyGeneration,
			authoritativeTopology.Generation,
		)
	}
	if !containsAllReplicaIncarnations(
		topologyReplicaIncarnations(*authoritativeTopology),
		failure.Command.Request.Replicas,
	) {
		return errors.New("terminal admission command contains replicas outside authoritative membership")
	}
	return nil
}

func validateOperationTimestampsAndPhase(operation Operation) error {
	if operation.StartedAt.IsZero() || operation.LastTransitionTime.IsZero() {
		return errors.New("operation timestamps must not be zero")
	}
	if operation.LastTransitionTime.Before(operation.StartedAt) {
		return errors.New("operation last transition time must not precede start time")
	}
	if !validOperationPhase(operation.Phase) {
		return fmt.Errorf("invalid operation phase %q", operation.Phase)
	}
	return nil
}

func validateOperationFailureState(operation Operation) error {
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
	return nil
}

func validateOperationCommitState(operation Operation) error {
	if operation.CommittedTopology != nil {
		if err := validateTopology(*operation.CommittedTopology); err != nil {
			return fmt.Errorf("validate durable committed topology: %w", err)
		}
		if operation.CommittedTopology.Generation <= operation.BaseTopology.Generation {
			return fmt.Errorf(
				"committed topology generation %d must exceed base generation %d",
				operation.CommittedTopology.Generation,
				operation.BaseTopology.Generation,
			)
		}
		if operation.Phase != OperationPhaseCommitted &&
			operation.Phase != OperationPhaseUnknown &&
			operation.Phase != OperationPhaseFailed {
			return fmt.Errorf(
				"operation phase %q cannot carry committed topology generation %d",
				operation.Phase,
				operation.CommittedTopology.Generation,
			)
		}
		if !operationMatchesCommittedTopology(operation, *operation.CommittedTopology) {
			return errors.New("durable committed topology does not match the operation target")
		}
	}
	if operation.Phase == OperationPhaseCommitted && operation.CommittedTopology == nil {
		return errors.New("committed operation must carry an exact committed topology")
	}
	if operation.Phase == OperationPhaseFailed && operation.CommittedTopology != nil &&
		operation.TerminalAdmissionFailure == nil &&
		(operation.Capability.VerificationRequirement != ServingVerificationRequired ||
			operation.ServingVerificationAttempt <= 0) {
		return errors.New("post-commit failure requires a serving-verification attempt or terminal admission failure")
	}
	if operation.Phase == OperationPhaseFailed && operation.CommittedTopology != nil &&
		operation.Failure.Classification != FailureClassificationTerminal {
		return errors.New("post-commit failure must be terminal")
	}
	if operation.PostCommitComplete && operation.CommittedTopology == nil {
		return errors.New("post-commit completion requires an exact committed topology")
	}
	if operation.PostCommitComplete &&
		operation.TargetReplicas > 0 &&
		operation.Capability.VerificationRequirement == ServingVerificationRequired &&
		!servingVerificationProofMatchesOperation(operation.ServingVerificationProof, operation) {
		return errors.New("post-commit completion requires serving verification for the committed topology")
	}
	return nil
}

func validateOperationCompensationState(operation Operation) error {
	if operation.CompensationTopology != nil {
		if err := validateTopology(*operation.CompensationTopology); err != nil {
			return fmt.Errorf("validate durable compensation topology: %w", err)
		}
		if !servingVerificationTopologiesEqual(*operation.CompensationTopology, operation.BaseTopology) {
			return errors.New("compensation topology must match the operation's exact unmodified base topology")
		}
		if operation.Phase != OperationPhaseAborting &&
			operation.Phase != OperationPhaseAborted &&
			operation.Phase != OperationPhaseUnknown {
			return fmt.Errorf(
				"operation phase %q cannot carry a compensation topology",
				operation.Phase,
			)
		}
	}
	if (operation.Phase == OperationPhaseAborting || operation.Phase == OperationPhaseAborted) &&
		operation.CompensationTopology == nil {
		return fmt.Errorf("operation phase %q requires an exact compensation topology", operation.Phase)
	}
	return nil
}

func validateAdoptedOperationState(operation Operation) error {
	if !operation.Adopted {
		return nil
	}
	if operation.Intent != OperationIntentRecover {
		return fmt.Errorf("adopted operation must have Recover intent, got %q", operation.Intent)
	}
	if operation.Capability.TrafficRequirement != ReconfigurationTrafficKeepServing {
		return errors.New("adopted operation requires a KeepServing traffic capability")
	}
	if operation.Capability.VerificationRequirement != ServingVerificationRequired {
		return errors.New("adopted operation requires serving verification")
	}
	if operation.CommittedTopology == nil {
		return errors.New("adopted operation must carry an exact committed topology")
	}
	if operation.BackendOperationID != "" {
		return errors.New("adopted operation must not claim a backend operation ID")
	}
	if operation.Phase != OperationPhaseCommitted &&
		operation.Phase != OperationPhaseFailed &&
		operation.Phase != OperationPhaseUnknown {
		return fmt.Errorf("adopted operation cannot have phase %q", operation.Phase)
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
	if err := validateReplicaNativeMemberships("restored", plan.RestoredMembership); err != nil {
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
		if plan.TargetReplicas <= 0 || plan.TargetReplicas >= activeReplicas {
			return fmt.Errorf(
				"shrink target %d must be positive and below active replicas %d",
				plan.TargetReplicas,
				activeReplicas,
			)
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

	// Recovery expansion alone may authorize exact stable logical identities to be repaired or unfenced.
	isReplacementRestoration := plan.Intent == OperationIntentRecover && plan.TargetReplicas > activeReplicas
	if !isReplacementRestoration {
		if len(plan.RestoredMembership) != 0 {
			return errors.New("restored replicas are only valid for a recovery expansion")
		}
	} else {
		expectedRestorations := plan.TargetReplicas - activeReplicas
		if int32(len(plan.RestoredMembership)) != expectedRestorations {
			return fmt.Errorf(
				"recovery target %d from %d active replicas requires %d restored replicas, got %d",
				plan.TargetReplicas,
				activeReplicas,
				expectedRestorations,
				len(plan.RestoredMembership),
			)
		}
		if overlap := intersectReplicaIDs(
			replicaNativeMembershipIDs(plan.RestoredMembership),
			topologyReplicaIDs(topology),
		); len(overlap) != 0 {
			return fmt.Errorf("restored replicas are already active in the base topology: %v", overlap)
		}
		if overlap := intersectNativeMemberIDs(
			replicaNativeMembershipNativeIDs(plan.RestoredMembership),
			replicaMembershipNativeIDs(topology.Replicas),
		); len(overlap) != 0 {
			return fmt.Errorf("restored native members are already active in the base topology: %v", overlap)
		}
		if overlap := intersectCapacitySlotIDs(
			replicaNativeMembershipSlotIDs(plan.RestoredMembership),
			replicaMembershipSlotIDs(topology.Replicas),
		); len(overlap) != 0 {
			return fmt.Errorf("restored capacity slots are already active in the base topology: %v", overlap)
		}
	}
	return validatePlanTargetMembership(plan, topology)
}

func validatePlanTargetMembership(plan OperationPlan, topology MembershipTopology) error {
	isNativeMemberRemap := plan.Intent == OperationIntentRecover &&
		plan.TargetReplicas == topology.ReplicaCount()
	if !isNativeMemberRemap {
		if len(plan.TargetMembership) != 0 {
			return errors.New("target membership is only valid for a cardinally stable recovery")
		}
		return nil
	}

	if len(plan.TargetMembership) == 0 {
		return errors.New("cardinally stable recovery requires an exact target membership")
	}
	if err := validateReplicaMemberships("target", plan.TargetMembership); err != nil {
		return err
	}
	if int32(len(plan.TargetMembership)) != plan.TargetReplicas {
		return fmt.Errorf(
			"target membership has %d replicas, expected %d",
			len(plan.TargetMembership),
			plan.TargetReplicas,
		)
	}
	if !sameReplicaIDs(replicaMembershipIDs(plan.TargetMembership), topologyReplicaIDs(topology)) {
		return errors.New("target membership must preserve every logical replica identity")
	}
	if !replicaSlotMappingsEqual(plan.TargetMembership, topology.Replicas) {
		return errors.New("cardinally stable recovery must preserve every logical replica capacity slot")
	}
	if err := validateRuntimeIdentityContinuity(topology.Replicas, plan.TargetMembership); err != nil {
		return err
	}
	if sameReplicaMemberships(plan.TargetMembership, topology.Replicas) {
		return errors.New("target membership does not change any replica incarnation or native member mapping")
	}
	return nil
}

func cardinalRecoveryShape(previous, target []ReplicaMembership) OperationShape {
	if replicaNativeMappingsEqual(previous, target) {
		return OperationShapeFixedSlotReplacement
	}
	return OperationShapeNativeMemberRemapping
}

func replicaNativeMappingsEqual(left, right []ReplicaMembership) bool {
	if !sameReplicaIDs(replicaMembershipIDs(left), replicaMembershipIDs(right)) {
		return false
	}
	rightByReplica := make(map[ReplicaID][]NativeMemberID, len(right))
	for _, membership := range right {
		rightByReplica[membership.Incarnation.ReplicaID] = membership.NativeMembers
	}
	for _, membership := range left {
		if !sameNativeMemberIDs(
			membership.NativeMembers,
			rightByReplica[membership.Incarnation.ReplicaID],
		) {
			return false
		}
	}
	return true
}

func replicaSlotMappingsEqual(left, right []ReplicaMembership) bool {
	if !sameReplicaIDs(replicaMembershipIDs(left), replicaMembershipIDs(right)) {
		return false
	}
	rightByReplica := make(map[ReplicaID]CapacitySlotID, len(right))
	for _, membership := range right {
		rightByReplica[membership.Incarnation.ReplicaID] = membership.Incarnation.SlotID
	}
	for _, membership := range left {
		if membership.Incarnation.SlotID != rightByReplica[membership.Incarnation.ReplicaID] {
			return false
		}
	}
	return true
}

func validateTopology(topology MembershipTopology) error {
	if topology.Generation < 0 {
		return fmt.Errorf("topology generation must not be negative: %d", topology.Generation)
	}
	if topology.Generation == 0 && len(topology.Replicas) != 0 {
		return errors.New("non-empty topology requires a positive generation")
	}
	return validateReplicaMemberships("topology", topology.Replicas)
}

func validateReplicaMemberships(label string, memberships []ReplicaMembership) error {
	replicas := make(map[ReplicaID]struct{}, len(memberships))
	slotIDs := make(map[CapacitySlotID]struct{}, len(memberships))
	runtimeIDs := make(map[RuntimeIncarnationID]struct{}, len(memberships))
	podNames := make(map[string]struct{})
	podUIDs := make(map[PodUID]struct{})
	nativeMembers := make(map[NativeMemberID]struct{})
	for _, replica := range memberships {
		if err := validateReplicaIncarnation(replica.Incarnation, true); err != nil {
			return fmt.Errorf("validate %s replica incarnation: %w", label, err)
		}
		replicaID := replica.Incarnation.ReplicaID
		if _, exists := replicas[replicaID]; exists {
			return fmt.Errorf("duplicate %s replica ID %q", label, replicaID)
		}
		replicas[replicaID] = struct{}{}
		if _, exists := slotIDs[replica.Incarnation.SlotID]; exists {
			return fmt.Errorf("duplicate %s capacity slot ID %q", label, replica.Incarnation.SlotID)
		}
		slotIDs[replica.Incarnation.SlotID] = struct{}{}
		if _, exists := runtimeIDs[replica.Incarnation.RuntimeID]; exists {
			return fmt.Errorf("duplicate %s runtime incarnation ID %q", label, replica.Incarnation.RuntimeID)
		}
		runtimeIDs[replica.Incarnation.RuntimeID] = struct{}{}
		for _, capacityRef := range replica.Incarnation.CapacityRefs {
			podName := capacityRef.Namespace + "/" + capacityRef.Name
			if _, exists := podNames[podName]; exists {
				return fmt.Errorf("duplicate %s capacity Pod name %q", label, podName)
			}
			podNames[podName] = struct{}{}
			if _, exists := podUIDs[capacityRef.UID]; exists {
				return fmt.Errorf("duplicate %s capacity Pod UID %q", label, capacityRef.UID)
			}
			podUIDs[capacityRef.UID] = struct{}{}
		}

		if len(replica.NativeMembers) == 0 {
			return fmt.Errorf("%s replica %q has no native members", label, replicaID)
		}
		for _, member := range replica.NativeMembers {
			if member == "" {
				return fmt.Errorf("%s replica %q has an empty native member ID", label, replicaID)
			}
			if _, exists := nativeMembers[member]; exists {
				return fmt.Errorf("duplicate %s native member ID %q", label, member)
			}
			nativeMembers[member] = struct{}{}
		}
	}
	return nil
}

func validateReplicaNativeMemberships(label string, memberships []ReplicaNativeMembership) error {
	replicaIDs := make(map[ReplicaID]struct{}, len(memberships))
	slotIDs := make(map[CapacitySlotID]struct{}, len(memberships))
	nativeMembers := make(map[NativeMemberID]struct{})
	for _, membership := range memberships {
		if membership.ReplicaID == "" {
			return fmt.Errorf("%s logical replica ID must not be empty", label)
		}
		if _, exists := replicaIDs[membership.ReplicaID]; exists {
			return fmt.Errorf("duplicate %s logical replica ID %q", label, membership.ReplicaID)
		}
		replicaIDs[membership.ReplicaID] = struct{}{}
		if membership.SlotID == "" {
			return fmt.Errorf("%s replica %q has an empty capacity slot ID", label, membership.ReplicaID)
		}
		if _, exists := slotIDs[membership.SlotID]; exists {
			return fmt.Errorf("duplicate %s capacity slot ID %q", label, membership.SlotID)
		}
		slotIDs[membership.SlotID] = struct{}{}
		if len(membership.NativeMembers) == 0 {
			return fmt.Errorf("%s replica %q has no native members", label, membership.ReplicaID)
		}
		for _, memberID := range membership.NativeMembers {
			if memberID == "" {
				return fmt.Errorf("%s replica %q has an empty native member ID", label, membership.ReplicaID)
			}
			if _, exists := nativeMembers[memberID]; exists {
				return fmt.Errorf("duplicate %s native member ID %q", label, memberID)
			}
			nativeMembers[memberID] = struct{}{}
		}
	}
	return nil
}

func validateMembershipCapabilities(capabilities MembershipCapabilities) error {
	seen := make(map[OperationShape]struct{}, len(capabilities.OperationShapes))
	for _, shape := range capabilities.OperationShapes {
		if !validOperationShape(shape) {
			return fmt.Errorf("invalid supported membership operation shape %q", shape)
		}
		if _, exists := seen[shape]; exists {
			return fmt.Errorf("duplicate supported membership operation shape %q", shape)
		}
		seen[shape] = struct{}{}
	}
	return nil
}

func validateCapabilityForPlan(
	capability ResolvedOperationCapability,
	plan OperationPlan,
	topology MembershipTopology,
) error {
	if err := validateOperationCapability(capability); err != nil {
		return err
	}
	if operationShapeIntent(capability.Shape) != plan.Intent {
		return fmt.Errorf(
			"operation shape %q does not implement intent %q",
			capability.Shape,
			plan.Intent,
		)
	}
	if plan.Intent == OperationIntentRecover && plan.TargetReplicas == topology.ReplicaCount() {
		expectedShape := cardinalRecoveryShape(topology.Replicas, plan.TargetMembership)
		if capability.Shape != expectedShape {
			return fmt.Errorf(
				"cardinally stable recovery requires operation shape %q, got %q",
				expectedShape,
				capability.Shape,
			)
		}
	}
	return validateCapabilityGeometry(capability.Shape, plan.TargetReplicas, topology.ReplicaCount())
}

func validateCapabilityForObservedTransition(
	capability ResolvedOperationCapability,
	transition ObservedMembershipTransition,
) error {
	if err := validateOperationCapability(capability); err != nil {
		return err
	}
	if operationShapeIntent(capability.Shape) != OperationIntentRecover {
		return fmt.Errorf(
			"observed membership operation shape %q does not implement recovery",
			capability.Shape,
		)
	}

	baseReplicas := transition.PreviousTopology.ReplicaCount()
	if err := validateCapabilityGeometry(capability.Shape, transition.Plan.TargetReplicas, baseReplicas); err != nil {
		return err
	}
	expectedShape := OperationShapeSurvivorReduction
	if transition.Plan.TargetReplicas > baseReplicas {
		expectedShape = OperationShapeReplacementRestoration
	} else if transition.Plan.TargetReplicas == baseReplicas {
		expectedShape = cardinalRecoveryShape(
			transition.PreviousTopology.Replicas,
			transition.ObservedTopology.Replicas,
		)
	}
	if capability.Shape != expectedShape {
		return fmt.Errorf(
			"observed transition from %d to %d replicas requires operation shape %q, got %q",
			baseReplicas,
			transition.Plan.TargetReplicas,
			expectedShape,
			capability.Shape,
		)
	}
	return nil
}

func validateCapabilityForOperation(capability ResolvedOperationCapability, operation Operation) error {
	if err := validateOperationCapability(capability); err != nil {
		return err
	}
	if operationShapeIntent(capability.Shape) != operation.Intent {
		return fmt.Errorf(
			"operation shape %q does not implement intent %q",
			capability.Shape,
			operation.Intent,
		)
	}
	return validateCapabilityGeometry(capability.Shape, operation.TargetReplicas, operation.BaseTopology.ReplicaCount())
}

func validateOperationCapability(capability ResolvedOperationCapability) error {
	if !validOperationShape(capability.Shape) {
		return fmt.Errorf("invalid membership operation shape %q", capability.Shape)
	}
	if capability.TrafficRequirement != ReconfigurationTrafficKeepServing &&
		capability.TrafficRequirement != ReconfigurationTrafficQuiesceGroup {
		return fmt.Errorf(
			"invalid reconfiguration traffic requirement %q for operation shape %q",
			capability.TrafficRequirement,
			capability.Shape,
		)
	}
	if capability.VerificationRequirement != ServingVerificationNotRequired &&
		capability.VerificationRequirement != ServingVerificationRequired {
		return fmt.Errorf(
			"invalid serving verification requirement %q for operation shape %q",
			capability.VerificationRequirement,
			capability.Shape,
		)
	}
	return nil
}

func validateCapabilityGeometry(shape OperationShape, targetReplicas, baseReplicas int32) error {
	switch shape {
	case OperationShapeFreshGrowth, OperationShapeReplacementRestoration:
		if targetReplicas <= baseReplicas {
			return fmt.Errorf(
				"operation shape %q requires growth from %d replicas, got target %d",
				shape,
				baseReplicas,
				targetReplicas,
			)
		}
	case OperationShapePlannedHighRankSuffixShrink, OperationShapePlannedSelectedRetirement:
		if targetReplicas <= 0 || targetReplicas >= baseReplicas {
			return fmt.Errorf(
				"operation shape %q requires a positive reduction target below %d replicas, got %d",
				shape,
				baseReplicas,
				targetReplicas,
			)
		}
	case OperationShapeSurvivorReduction:
		if targetReplicas >= baseReplicas {
			return fmt.Errorf(
				"operation shape %q requires reduction from %d replicas, got target %d",
				shape,
				baseReplicas,
				targetReplicas,
			)
		}
	case OperationShapeFixedSlotReplacement, OperationShapeNativeMemberRemapping:
		if targetReplicas != baseReplicas {
			return fmt.Errorf(
				"operation shape %q requires cardinality %d, got target %d",
				shape,
				baseReplicas,
				targetReplicas,
			)
		}
	case OperationShapeFullRetirement:
		if baseReplicas == 0 || targetReplicas != 0 {
			return fmt.Errorf(
				"operation shape %q requires a non-empty base and target zero, got base %d and target %d",
				shape,
				baseReplicas,
				targetReplicas,
			)
		}
	default:
		return fmt.Errorf("invalid membership operation shape %q", shape)
	}
	return nil
}

func validateOperationReplicaGeometry(operation Operation) error {
	baseReplicaIDs := topologyReplicaIDs(operation.BaseTopology)
	baseReplicas := operation.BaseTopology.ReplicaCount()
	joiningReplicaIDs := replicaIncarnationIDs(operation.JoiningReplicas)
	joiningReplicas := int32(len(operation.JoiningReplicas))
	nominatedReplicas := int32(len(operation.NominatedReplicas))
	if err := validateOperationTargetMembership(operation); err != nil {
		return err
	}
	if err := validateOperationRestorationGeometry(operation, baseReplicaIDs, baseReplicas); err != nil {
		return err
	}
	if err := validateOperationIntentGeometry(operation, baseReplicas); err != nil {
		return err
	}

	// Joining and retiring identities always describe disjoint logical replicas.
	if overlap := intersectReplicaIDs(joiningReplicaIDs, operation.NominatedReplicas); len(overlap) != 0 {
		return fmt.Errorf("replicas cannot be both joining and nominated: %v", overlap)
	}

	switch {
	case operation.TargetReplicas < baseReplicas:
		return validateOperationReductionGeometry(
			operation,
			baseReplicaIDs,
			baseReplicas,
			joiningReplicas,
			nominatedReplicas,
		)
	case operation.TargetReplicas > baseReplicas:
		return validateOperationExpansionGeometry(
			operation,
			baseReplicaIDs,
			baseReplicas,
			joiningReplicaIDs,
			joiningReplicas,
			nominatedReplicas,
		)
	default:
		return validateOperationStableCardinalityGeometry(joiningReplicas, nominatedReplicas)
	}
}

func validateOperationRestorationGeometry(
	operation Operation,
	baseReplicaIDs []ReplicaID,
	baseReplicas int32,
) error {
	if operation.Capability.Shape == OperationShapeReplacementRestoration {
		expectedRestorations := operation.TargetReplicas - baseReplicas
		if int32(len(operation.RestoredMembership)) != expectedRestorations {
			return fmt.Errorf(
				"replacement restoration from %d to %d replicas requires %d restored identities, got %d",
				baseReplicas,
				operation.TargetReplicas,
				expectedRestorations,
				len(operation.RestoredMembership),
			)
		}
		if overlap := intersectReplicaIDs(
			replicaNativeMembershipIDs(operation.RestoredMembership),
			baseReplicaIDs,
		); len(overlap) != 0 {
			return fmt.Errorf("restored replicas are already active in the base topology: %v", overlap)
		}
		if overlap := intersectNativeMemberIDs(
			replicaNativeMembershipNativeIDs(operation.RestoredMembership),
			replicaMembershipNativeIDs(operation.BaseTopology.Replicas),
		); len(overlap) != 0 {
			return fmt.Errorf("restored native members are already active in the base topology: %v", overlap)
		}
		if overlap := intersectCapacitySlotIDs(
			replicaNativeMembershipSlotIDs(operation.RestoredMembership),
			replicaMembershipSlotIDs(operation.BaseTopology.Replicas),
		); len(overlap) != 0 {
			return fmt.Errorf("restored capacity slots are already active in the base topology: %v", overlap)
		}
	} else if len(operation.RestoredMembership) != 0 {
		return fmt.Errorf("operation shape %q must not carry restored replica identities", operation.Capability.Shape)
	}
	return nil
}

func validateOperationIntentGeometry(operation Operation, baseReplicas int32) error {
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
		if operation.TargetReplicas <= 0 || operation.TargetReplicas >= baseReplicas {
			return fmt.Errorf(
				"shrink target %d must be positive and below %d base replicas",
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
	return nil
}

func validateOperationReductionGeometry(
	operation Operation,
	baseReplicaIDs []ReplicaID,
	baseReplicas int32,
	joiningReplicas int32,
	nominatedReplicas int32,
) error {
	// A reduction must name every removed base replica and cannot introduce a joiner.
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
	return requireReplicaSubset(operation.NominatedReplicas, baseReplicaIDs, "nominated")
}

func validateOperationExpansionGeometry(
	operation Operation,
	baseReplicaIDs []ReplicaID,
	baseReplicas int32,
	joiningReplicaIDs []ReplicaID,
	joiningReplicas int32,
	nominatedReplicas int32,
) error {
	// An expansion freezes every added identity before submission and cannot retire a base replica.
	if nominatedReplicas != 0 {
		return errors.New("expanding operation must not nominate retiring replicas")
	}
	if operation.Phase == OperationPhasePending &&
		operation.Capability.Shape == OperationShapeFreshGrowth &&
		joiningReplicas != 0 {
		return errors.New("pending fresh growth must not preselect joining replica identities")
	}
	expectedJoiners := operation.TargetReplicas - baseReplicas
	joinersMayBeUnfrozen := operationJoinersMayBeUnfrozen(operation, joiningReplicas)
	if !joinersMayBeUnfrozen && joiningReplicas != expectedJoiners {
		return fmt.Errorf(
			"target %d from %d base replicas requires %d joining replicas, got %d",
			operation.TargetReplicas,
			baseReplicas,
			expectedJoiners,
			joiningReplicas,
		)
	}
	if overlap := intersectReplicaIDs(joiningReplicaIDs, baseReplicaIDs); len(overlap) != 0 {
		return fmt.Errorf("joining replicas are already active in the base topology: %v", overlap)
	}
	if operation.Capability.Shape == OperationShapeReplacementRestoration &&
		!sameReplicaIDs(replicaNativeMembershipIDs(operation.RestoredMembership), joiningReplicaIDs) &&
		!joinersMayBeUnfrozen {
		return errors.New("replacement restoration joining incarnations do not match planned replica identities")
	}
	if operation.Capability.Shape == OperationShapeReplacementRestoration &&
		!restoredReplicaSlotsMatchJoiningIncarnations(operation.RestoredMembership, operation.JoiningReplicas) &&
		!joinersMayBeUnfrozen {
		return errors.New("replacement restoration joining incarnations do not match planned capacity slots")
	}
	return nil
}

func operationJoinersMayBeUnfrozen(operation Operation, joiningReplicas int32) bool {
	if joiningReplicas != 0 {
		return false
	}
	if operation.Phase == OperationPhasePending {
		return true
	}
	return operation.CommittedTopology == nil &&
		(operation.Phase == OperationPhaseUnknown ||
			operation.Phase == OperationPhaseAborting ||
			operation.Phase == OperationPhaseAborted)
}

func validateOperationStableCardinalityGeometry(joiningReplicas, nominatedReplicas int32) error {
	// A cardinally stable recovery may remap native members but not logical replica identities.
	if joiningReplicas != 0 || nominatedReplicas != 0 {
		return errors.New("cardinally stable operation must not add or nominate logical replicas")
	}
	return nil
}

func validateOperationTargetMembership(operation Operation) error {
	isCardinalRecovery := operation.Capability.Shape == OperationShapeFixedSlotReplacement ||
		operation.Capability.Shape == OperationShapeNativeMemberRemapping
	if !isCardinalRecovery {
		if len(operation.TargetMembership) != 0 {
			return fmt.Errorf(
				"operation shape %q must not carry target membership",
				operation.Capability.Shape,
			)
		}
		return nil
	}

	if len(operation.TargetMembership) == 0 {
		return errors.New("cardinally stable recovery requires an exact target membership")
	}
	if err := validateReplicaMemberships("target", operation.TargetMembership); err != nil {
		return err
	}
	if int32(len(operation.TargetMembership)) != operation.TargetReplicas {
		return fmt.Errorf(
			"target membership has %d replicas, expected %d",
			len(operation.TargetMembership),
			operation.TargetReplicas,
		)
	}
	if !sameReplicaIDs(replicaMembershipIDs(operation.TargetMembership), topologyReplicaIDs(operation.BaseTopology)) {
		return errors.New("target membership must preserve every base logical replica identity")
	}
	if !replicaSlotMappingsEqual(operation.TargetMembership, operation.BaseTopology.Replicas) {
		return errors.New("cardinally stable recovery must preserve every base logical replica capacity slot")
	}
	if err := validateRuntimeIdentityContinuity(
		operation.BaseTopology.Replicas,
		operation.TargetMembership,
	); err != nil {
		return err
	}
	if sameReplicaMemberships(operation.TargetMembership, operation.BaseTopology.Replicas) {
		return errors.New("target membership does not change any replica incarnation or native member mapping")
	}
	preservesNativeMembers := replicaNativeMappingsEqual(
		operation.TargetMembership,
		operation.BaseTopology.Replicas,
	)
	if operation.Capability.Shape == OperationShapeFixedSlotReplacement && !preservesNativeMembers {
		return errors.New("fixed-slot replacement must preserve every native member mapping")
	}
	if operation.Capability.Shape == OperationShapeNativeMemberRemapping && preservesNativeMembers {
		return errors.New("native-member remapping must change at least one native member mapping")
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

func validateBackendOperation(operation BackendOperation) error {
	switch operation.Phase {
	case BackendOperationPhaseAbsent:
		if operation.ID != "" ||
			operation.Attempt != 0 ||
			operation.BackendID != "" ||
			operation.TargetReplicas != 0 ||
			operation.CommittedTopology != nil ||
			operation.Failure != nil {
			return errors.New("absent backend operation must otherwise be zero-valued")
		}
		return nil
	case BackendOperationPhaseAccepted,
		BackendOperationPhaseCommitting,
		BackendOperationPhaseUnknown:
		if operation.CommittedTopology != nil {
			return fmt.Errorf("backend operation phase %q must not carry a committed topology", operation.Phase)
		}
		if operation.Failure != nil {
			return fmt.Errorf("backend operation phase %q must not carry a failure", operation.Phase)
		}
		return nil
	case BackendOperationPhaseCommitted:
		if operation.CommittedTopology == nil {
			return errors.New("committed backend operation must carry an exact committed topology")
		}
		if operation.Failure != nil {
			return errors.New("committed backend operation must not carry a failure")
		}
		if err := validateTopology(*operation.CommittedTopology); err != nil {
			return fmt.Errorf("validate backend committed topology: %w", err)
		}
		return nil
	case BackendOperationPhaseFailed:
		if operation.CommittedTopology != nil {
			return errors.New("failed backend operation must not carry a committed topology")
		}
		return validateFailure(operation.Failure)
	default:
		return fmt.Errorf("invalid backend operation phase %q", operation.Phase)
	}
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

func validOperationShape(shape OperationShape) bool {
	switch shape {
	case OperationShapeFreshGrowth,
		OperationShapePlannedHighRankSuffixShrink,
		OperationShapePlannedSelectedRetirement,
		OperationShapeSurvivorReduction,
		OperationShapeReplacementRestoration,
		OperationShapeFixedSlotReplacement,
		OperationShapeNativeMemberRemapping,
		OperationShapeFullRetirement:
		return true
	default:
		return false
	}
}

func operationShapeIntent(shape OperationShape) OperationIntent {
	switch shape {
	case OperationShapeFreshGrowth:
		return OperationIntentGrow
	case OperationShapePlannedHighRankSuffixShrink, OperationShapePlannedSelectedRetirement:
		return OperationIntentShrink
	case OperationShapeSurvivorReduction,
		OperationShapeReplacementRestoration,
		OperationShapeFixedSlotReplacement,
		OperationShapeNativeMemberRemapping:
		return OperationIntentRecover
	case OperationShapeFullRetirement:
		return OperationIntentRetire
	default:
		return ""
	}
}

func normalizePlan(plan OperationPlan) OperationPlan {
	plan.NominatedReplicas = normalizeReplicaIDs(plan.NominatedReplicas)
	plan.RestoredMembership = normalizeReplicaNativeMemberships(plan.RestoredMembership)
	plan.TargetMembership = normalizeReplicaMemberships(plan.TargetMembership)
	return plan
}

func observedMembershipTransition(
	previous Operation,
	observedTopology MembershipTopology,
	plan OperationPlan,
) (ObservedMembershipTransition, error) {
	previousTopology := recoveryBaselineTopology(previous)
	if previousTopology == nil {
		return ObservedMembershipTransition{}, errors.New(
			"observed membership transition requires an exact previous serving topology",
		)
	}
	return newObservedMembershipTransition(*previousTopology, observedTopology, plan)
}

func newObservedMembershipTransition(
	previousTopology MembershipTopology,
	observedTopology MembershipTopology,
	plan OperationPlan,
) (ObservedMembershipTransition, error) {
	transition := ObservedMembershipTransition{
		PreviousTopology: cloneTopology(previousTopology),
		ObservedTopology: cloneTopology(observedTopology),
		Plan:             normalizePlan(plan),
	}
	if err := validateObservedMembershipTransition(transition); err != nil {
		return ObservedMembershipTransition{}, err
	}
	return transition, nil
}

func validateObservedMembershipTransition(transition ObservedMembershipTransition) error {
	if transition.PreviousTopology.Generation <= 0 {
		return fmt.Errorf(
			"previous committed topology generation must be positive: %d",
			transition.PreviousTopology.Generation,
		)
	}
	if err := validateTopology(transition.PreviousTopology); err != nil {
		return fmt.Errorf("validate previous committed topology: %w", err)
	}
	previousReplicas := topologyReplicaIDs(transition.PreviousTopology)
	if len(previousReplicas) == 0 {
		return errors.New("observed membership transition requires previous committed replicas")
	}
	if err := validateTopology(transition.ObservedTopology); err != nil {
		return fmt.Errorf("validate observed membership transition topology: %w", err)
	}
	if transition.ObservedTopology.Generation <= transition.PreviousTopology.Generation {
		return fmt.Errorf(
			"observed topology generation %d must exceed previous committed generation %d",
			transition.ObservedTopology.Generation,
			transition.PreviousTopology.Generation,
		)
	}
	if err := validateRuntimeIdentityContinuity(
		transition.PreviousTopology.Replicas,
		transition.ObservedTopology.Replicas,
	); err != nil {
		return err
	}

	plan := transition.Plan
	if plan.ID == "" {
		return errors.New("observed membership transition requires an explicit plan ID")
	}
	if plan.Intent != OperationIntentRecover {
		return fmt.Errorf("only a recovery plan may adopt observed membership, got %q", plan.Intent)
	}
	if err := validatePlan(plan, transition.PreviousTopology); err != nil {
		return fmt.Errorf("validate observed recovery plan: %w", err)
	}
	if plan.TargetReplicas != transition.ObservedTopology.ReplicaCount() {
		return fmt.Errorf(
			"recovery plan target %d does not match observed topology size %d",
			plan.TargetReplicas,
			transition.ObservedTopology.ReplicaCount(),
		)
	}

	observedReplicas := topologyReplicaIDs(transition.ObservedTopology)
	if plan.TargetReplicas < transition.PreviousTopology.ReplicaCount() {
		if err := requireReplicaSubset(observedReplicas, previousReplicas, "observed survivor"); err != nil {
			return err
		}
		missingReplicas := differenceReplicaIDs(previousReplicas, observedReplicas)
		if !sameReplicaIDs(plan.NominatedReplicas, missingReplicas) {
			return fmt.Errorf(
				"recovery plan nominations %v do not match missing replicas %v",
				plan.NominatedReplicas,
				missingReplicas,
			)
		}
		if !retainedReplicaMembershipsEqual(
			transition.PreviousTopology.Replicas,
			transition.ObservedTopology.Replicas,
		) {
			return errors.New("survivor reduction must preserve retained native-member mappings")
		}
		return nil
	}

	if plan.TargetReplicas > transition.PreviousTopology.ReplicaCount() {
		if err := requireReplicaSubset(previousReplicas, observedReplicas, "previous survivor"); err != nil {
			return err
		}
		addedReplicas := differenceReplicaIDs(observedReplicas, previousReplicas)
		if !sameReplicaIDs(replicaNativeMembershipIDs(plan.RestoredMembership), addedReplicas) {
			return fmt.Errorf(
				"recovery plan restorations %v do not match added replicas %v",
				replicaNativeMembershipIDs(plan.RestoredMembership),
				addedReplicas,
			)
		}
		if !retainedReplicaMembershipsEqual(
			transition.PreviousTopology.Replicas,
			transition.ObservedTopology.Replicas,
		) {
			return errors.New("replacement restoration must preserve retained replica incarnations and native members")
		}
		if !restoredMembershipsEqual(plan.RestoredMembership, transition.ObservedTopology.Replicas) {
			return errors.New("restored slot and native membership do not match the exact observed topology")
		}
		return nil
	}

	if !replicaSlotMappingsEqual(
		transition.ObservedTopology.Replicas,
		transition.PreviousTopology.Replicas,
	) {
		return errors.New("cardinally stable observed recovery must preserve every logical replica capacity slot")
	}
	if err := validateReplicaMemberships("target", plan.TargetMembership); err != nil {
		return err
	}
	if !sameReplicaMemberships(plan.TargetMembership, transition.ObservedTopology.Replicas) {
		return errors.New("recovery target membership does not match the exact observed topology")
	}
	return nil
}

func validateRuntimeIdentityContinuity(previous, current []ReplicaMembership) error {
	incarnations := append(
		replicaMembershipIncarnations(previous),
		replicaMembershipIncarnations(current)...,
	)
	if err := validateReplicaRuntimeIdentities("membership transition", incarnations); err != nil {
		return fmt.Errorf("validate runtime identity continuity: %w", err)
	}
	return nil
}

func cloneObservedMembershipTransition(transition ObservedMembershipTransition) ObservedMembershipTransition {
	cloned := transition
	cloned.PreviousTopology = cloneTopology(transition.PreviousTopology)
	cloned.ObservedTopology = cloneTopology(transition.ObservedTopology)
	cloned.Plan = normalizePlan(transition.Plan)
	return cloned
}

func operationCarriesPlan(operation Operation, plan OperationPlan) bool {
	normalized := normalizePlan(plan)
	restoredReplicasMatch := len(normalized.RestoredMembership) == 0
	if operation.Capability.Shape == OperationShapeReplacementRestoration {
		restoredReplicasMatch = sameReplicaNativeMemberships(
			operation.RestoredMembership,
			normalized.RestoredMembership,
		)
	}

	return operation.PlanID == normalized.ID &&
		operation.Intent == normalized.Intent &&
		operation.TargetReplicas == normalized.TargetReplicas &&
		sameReplicaIDs(operation.NominatedReplicas, normalized.NominatedReplicas) &&
		restoredReplicasMatch &&
		sameReplicaMemberships(operation.TargetMembership, normalized.TargetMembership)
}

func operationPlan(operation Operation) OperationPlan {
	return OperationPlan{
		ID:                 operation.PlanID,
		Intent:             operation.Intent,
		TargetReplicas:     operation.TargetReplicas,
		NominatedReplicas:  slices.Clone(operation.NominatedReplicas),
		RestoredMembership: cloneReplicaNativeMemberships(operation.RestoredMembership),
		TargetMembership:   cloneReplicaMemberships(operation.TargetMembership),
	}
}

func operationMatchesBaseTopology(operation Operation, topology MembershipTopology) bool {
	return servingVerificationTopologiesEqual(operation.BaseTopology, topology)
}

func operationMatchesCommittedTopology(operation Operation, topology MembershipTopology) bool {
	if topology.ReplicaCount() != operation.TargetReplicas ||
		!sameReplicaIDs(expectedCommittedReplicaIDs(operation), topologyReplicaIDs(topology)) {
		return false
	}
	if operation.Capability.Shape == OperationShapeFixedSlotReplacement ||
		operation.Capability.Shape == OperationShapeNativeMemberRemapping {
		return sameReplicaMemberships(operation.TargetMembership, topology.Replicas)
	}
	if !retainedReplicaMembershipsEqual(operation.BaseTopology.Replicas, topology.Replicas) ||
		!joiningReplicaIncarnationsEqual(operation.JoiningReplicas, topology.Replicas) {
		return false
	}
	if operation.Capability.Shape == OperationShapeReplacementRestoration {
		return restoredMembershipsEqual(operation.RestoredMembership, topology.Replicas)
	}
	return true
}

func retainedReplicaMembershipsEqual(previous, current []ReplicaMembership) bool {
	currentByReplica := make(map[ReplicaID]ReplicaMembership, len(current))
	for _, membership := range current {
		currentByReplica[membership.Incarnation.ReplicaID] = membership
	}
	for _, membership := range previous {
		currentMembership, retained := currentByReplica[membership.Incarnation.ReplicaID]
		if retained && (!sameReplicaIncarnation(membership.Incarnation, currentMembership.Incarnation) ||
			!sameNativeMemberIDs(membership.NativeMembers, currentMembership.NativeMembers)) {
			return false
		}
	}
	return true
}

func joiningReplicaIncarnationsEqual(joining []ReplicaIncarnation, current []ReplicaMembership) bool {
	currentByReplica := make(map[ReplicaID]ReplicaIncarnation, len(current))
	for _, membership := range current {
		currentByReplica[membership.Incarnation.ReplicaID] = membership.Incarnation
	}
	for _, incarnation := range joining {
		currentIncarnation, exists := currentByReplica[incarnation.ReplicaID]
		if !exists || !sameReplicaIncarnation(incarnation, currentIncarnation) {
			return false
		}
	}
	return true
}

func restoredMembershipsEqual(restored []ReplicaNativeMembership, current []ReplicaMembership) bool {
	currentByReplica := make(map[ReplicaID]ReplicaMembership, len(current))
	for _, membership := range current {
		currentByReplica[membership.Incarnation.ReplicaID] = membership
	}
	for _, membership := range restored {
		currentMembership, exists := currentByReplica[membership.ReplicaID]
		if !exists ||
			membership.SlotID != currentMembership.Incarnation.SlotID ||
			!sameNativeMemberIDs(membership.NativeMembers, currentMembership.NativeMembers) {
			return false
		}
	}
	return true
}

func restoredReplicaSlotsMatchJoiningIncarnations(
	restored []ReplicaNativeMembership,
	joining []ReplicaIncarnation,
) bool {
	joiningSlots := make(map[ReplicaID]CapacitySlotID, len(joining))
	for _, incarnation := range joining {
		joiningSlots[incarnation.ReplicaID] = incarnation.SlotID
	}
	for _, membership := range restored {
		if membership.SlotID != joiningSlots[membership.ReplicaID] {
			return false
		}
	}
	return true
}

func sameNativeMemberIDs(left, right []NativeMemberID) bool {
	return slices.Equal(normalizeNativeMemberIDs(left), normalizeNativeMemberIDs(right))
}

func normalizeNativeMemberIDs(memberIDs []NativeMemberID) []NativeMemberID {
	normalized := slices.Clone(memberIDs)
	slices.Sort(normalized)
	return normalized
}

func replicaNativeMembershipIDs(memberships []ReplicaNativeMembership) []ReplicaID {
	replicaIDs := make([]ReplicaID, len(memberships))
	for i, membership := range memberships {
		replicaIDs[i] = membership.ReplicaID
	}
	return normalizeReplicaIDs(replicaIDs)
}

func replicaNativeMembershipNativeIDs(memberships []ReplicaNativeMembership) []NativeMemberID {
	memberIDs := make([]NativeMemberID, 0)
	for _, membership := range memberships {
		memberIDs = append(memberIDs, membership.NativeMembers...)
	}
	return normalizeNativeMemberIDs(memberIDs)
}

func replicaNativeMembershipSlotIDs(memberships []ReplicaNativeMembership) []CapacitySlotID {
	slotIDs := make([]CapacitySlotID, len(memberships))
	for i, membership := range memberships {
		slotIDs[i] = membership.SlotID
	}
	return normalizeCapacitySlotIDs(slotIDs)
}

func replicaMembershipSlotIDs(memberships []ReplicaMembership) []CapacitySlotID {
	slotIDs := make([]CapacitySlotID, len(memberships))
	for i, membership := range memberships {
		slotIDs[i] = membership.Incarnation.SlotID
	}
	return normalizeCapacitySlotIDs(slotIDs)
}

func normalizeCapacitySlotIDs(slotIDs []CapacitySlotID) []CapacitySlotID {
	normalized := slices.Clone(slotIDs)
	slices.Sort(normalized)
	return slices.Compact(normalized)
}

func intersectCapacitySlotIDs(left, right []CapacitySlotID) []CapacitySlotID {
	rightSet := make(map[CapacitySlotID]struct{}, len(right))
	for _, slotID := range right {
		rightSet[slotID] = struct{}{}
	}
	intersection := make([]CapacitySlotID, 0)
	for _, slotID := range left {
		if _, exists := rightSet[slotID]; exists {
			intersection = append(intersection, slotID)
		}
	}
	return normalizeCapacitySlotIDs(intersection)
}

func replicaMembershipNativeIDs(memberships []ReplicaMembership) []NativeMemberID {
	memberIDs := make([]NativeMemberID, 0)
	for _, membership := range memberships {
		memberIDs = append(memberIDs, membership.NativeMembers...)
	}
	return normalizeNativeMemberIDs(memberIDs)
}

func intersectNativeMemberIDs(left, right []NativeMemberID) []NativeMemberID {
	rightSet := make(map[NativeMemberID]struct{}, len(right))
	for _, memberID := range right {
		rightSet[memberID] = struct{}{}
	}
	intersection := make([]NativeMemberID, 0)
	for _, memberID := range left {
		if _, exists := rightSet[memberID]; exists {
			intersection = append(intersection, memberID)
		}
	}
	return normalizeNativeMemberIDs(intersection)
}

func normalizeReplicaNativeMemberships(memberships []ReplicaNativeMembership) []ReplicaNativeMembership {
	normalized := cloneReplicaNativeMemberships(memberships)
	for i := range normalized {
		normalized[i].NativeMembers = normalizeNativeMemberIDs(normalized[i].NativeMembers)
	}
	slices.SortFunc(normalized, func(left, right ReplicaNativeMembership) int {
		if left.ReplicaID < right.ReplicaID {
			return -1
		}
		if left.ReplicaID > right.ReplicaID {
			return 1
		}
		if left.SlotID < right.SlotID {
			return -1
		}
		if left.SlotID > right.SlotID {
			return 1
		}
		return slices.Compare(left.NativeMembers, right.NativeMembers)
	})
	return normalized
}

func cloneReplicaNativeMemberships(memberships []ReplicaNativeMembership) []ReplicaNativeMembership {
	if memberships == nil {
		return nil
	}
	cloned := make([]ReplicaNativeMembership, len(memberships))
	for i, membership := range memberships {
		cloned[i] = ReplicaNativeMembership{
			ReplicaID:     membership.ReplicaID,
			SlotID:        membership.SlotID,
			NativeMembers: slices.Clone(membership.NativeMembers),
		}
	}
	return cloned
}

func sameReplicaNativeMemberships(left, right []ReplicaNativeMembership) bool {
	return slices.EqualFunc(
		normalizeReplicaNativeMemberships(left),
		normalizeReplicaNativeMemberships(right),
		func(left, right ReplicaNativeMembership) bool {
			return left.ReplicaID == right.ReplicaID &&
				left.SlotID == right.SlotID &&
				sameNativeMemberIDs(left.NativeMembers, right.NativeMembers)
		},
	)
}

func expectedCommittedReplicaIDs(operation Operation) []ReplicaID {
	baseReplicaIDs := topologyReplicaIDs(operation.BaseTopology)
	expected := make(map[ReplicaID]struct{}, len(baseReplicaIDs)+len(operation.JoiningReplicas))
	for _, replicaID := range baseReplicaIDs {
		expected[replicaID] = struct{}{}
	}
	for _, replicaID := range operation.NominatedReplicas {
		delete(expected, replicaID)
	}
	for _, incarnation := range operation.JoiningReplicas {
		expected[incarnation.ReplicaID] = struct{}{}
	}

	replicaIDs := make([]ReplicaID, 0, len(expected))
	for replicaID := range expected {
		replicaIDs = append(replicaIDs, replicaID)
	}
	slices.Sort(replicaIDs)
	return replicaIDs
}

func topologyReplicaIDs(topology MembershipTopology) []ReplicaID {
	return replicaMembershipIDs(topology.Replicas)
}

func topologyReplicaIncarnations(topology MembershipTopology) []ReplicaIncarnation {
	return replicaMembershipIncarnations(topology.Replicas)
}

func replicaMembershipIncarnations(memberships []ReplicaMembership) []ReplicaIncarnation {
	incarnations := make([]ReplicaIncarnation, len(memberships))
	for i, replica := range memberships {
		incarnations[i] = cloneReplicaIncarnation(replica.Incarnation)
	}
	return normalizeReplicaIncarnations(incarnations)
}

func topologyReplicaIncarnationsForIDs(
	topology MembershipTopology,
	replicaIDs []ReplicaID,
) []ReplicaIncarnation {
	required := make(map[ReplicaID]struct{}, len(replicaIDs))
	for _, replicaID := range replicaIDs {
		required[replicaID] = struct{}{}
	}
	incarnations := make([]ReplicaIncarnation, 0, len(required))
	for _, replica := range topology.Replicas {
		if _, exists := required[replica.Incarnation.ReplicaID]; exists {
			incarnations = append(incarnations, cloneReplicaIncarnation(replica.Incarnation))
		}
	}
	return normalizeReplicaIncarnations(incarnations)
}

func replicaMembershipIDs(memberships []ReplicaMembership) []ReplicaID {
	replicaIDs := make([]ReplicaID, len(memberships))
	for i, replica := range memberships {
		replicaIDs[i] = replica.Incarnation.ReplicaID
	}
	return normalizeReplicaIDs(replicaIDs)
}

func normalizeReplicaMemberships(memberships []ReplicaMembership) []ReplicaMembership {
	normalized := cloneReplicaMemberships(memberships)
	for i := range normalized {
		slices.Sort(normalized[i].NativeMembers)
	}
	slices.SortFunc(normalized, func(left, right ReplicaMembership) int {
		return compareReplicaIncarnations(left.Incarnation, right.Incarnation)
	})
	return normalized
}

func sameReplicaMemberships(left, right []ReplicaMembership) bool {
	return slices.EqualFunc(
		normalizeReplicaMemberships(left),
		normalizeReplicaMemberships(right),
		func(left, right ReplicaMembership) bool {
			return sameReplicaIncarnation(left.Incarnation, right.Incarnation) &&
				slices.Equal(left.NativeMembers, right.NativeMembers)
		},
	)
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
		ID:                 operation.ID,
		Attempt:            operation.Attempt,
		Intent:             operation.Intent,
		Capability:         operation.Capability,
		BaseTopology:       cloneTopology(operation.BaseTopology),
		TargetReplicas:     operation.TargetReplicas,
		JoiningReplicas:    cloneReplicaIncarnations(operation.JoiningReplicas),
		RestoredMembership: cloneReplicaNativeMemberships(operation.RestoredMembership),
		NominatedReplicas:  slices.Clone(operation.NominatedReplicas),
		TargetMembership:   cloneReplicaMemberships(operation.TargetMembership),
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
	cloned.BaseTopology = cloneTopology(operation.BaseTopology)
	cloned.RestoredMembership = cloneReplicaNativeMemberships(operation.RestoredMembership)
	cloned.JoiningReplicas = cloneReplicaIncarnations(operation.JoiningReplicas)
	cloned.NominatedReplicas = slices.Clone(operation.NominatedReplicas)
	cloned.TargetMembership = cloneReplicaMemberships(operation.TargetMembership)
	cloned.CleanupReplicaSlots = slices.Clone(operation.CleanupReplicaSlots)
	cloned.CommittedTopology = cloneTopologyPointer(operation.CommittedTopology)
	cloned.CompensationTopology = cloneTopologyPointer(operation.CompensationTopology)
	cloned.ServingVerificationTarget = cloneTopologyPointer(operation.ServingVerificationTarget)
	cloned.ServingVerificationProof = cloneServingVerificationProof(operation.ServingVerificationProof)
	cloned.TerminalAdmissionFailure = cloneTrafficAdmissionFailure(operation.TerminalAdmissionFailure)
	cloned.Failure = cloneFailure(operation.Failure)
	if operation.QueuedTargetReplicas != nil {
		queuedTarget := *operation.QueuedTargetReplicas
		cloned.QueuedTargetReplicas = &queuedTarget
	}
	return &cloned
}

func cloneTrafficAdmissionFailure(failure *TrafficAdmissionFailure) *TrafficAdmissionFailure {
	if failure == nil {
		return nil
	}

	return &TrafficAdmissionFailure{
		Command: *cloneTrafficCommand(&failure.Command),
		Failure: *cloneFailure(&failure.Failure),
	}
}

func cloneFailure(failure *OperationFailure) *OperationFailure {
	if failure == nil {
		return nil
	}

	cloned := *failure
	return &cloned
}

func cloneTopology(topology MembershipTopology) MembershipTopology {
	return MembershipTopology{
		Generation: topology.Generation,
		Replicas:   cloneReplicaMemberships(topology.Replicas),
	}
}

func topologyPointer(topology MembershipTopology) *MembershipTopology {
	cloned := cloneTopology(topology)
	return &cloned
}

func cloneTopologyPointer(topology *MembershipTopology) *MembershipTopology {
	if topology == nil {
		return nil
	}
	return topologyPointer(*topology)
}

func committedTopologyGeneration(operation Operation) int64 {
	if operation.CommittedTopology == nil {
		return 0
	}
	return operation.CommittedTopology.Generation
}

func recoveryBaselineTopology(operation Operation) *MembershipTopology {
	switch operation.Phase {
	case OperationPhaseUnknown, OperationPhaseFailed:
		return operation.CommittedTopology
	case OperationPhaseAborted:
		return operation.CompensationTopology
	default:
		return nil
	}
}

func operationAllowsObservedAdoption(operation Operation) bool {
	return (operation.Phase == OperationPhaseUnknown && operation.CommittedTopology != nil) ||
		operation.Phase == OperationPhaseAborted
}

func cloneReplicaMemberships(memberships []ReplicaMembership) []ReplicaMembership {
	if memberships == nil {
		return nil
	}
	cloned := make([]ReplicaMembership, len(memberships))
	for i, membership := range memberships {
		cloned[i] = ReplicaMembership{
			Incarnation:   cloneReplicaIncarnation(membership.Incarnation),
			NativeMembers: slices.Clone(membership.NativeMembers),
		}
	}
	return cloned
}
