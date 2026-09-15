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

	"k8s.io/apimachinery/pkg/util/uuid"
)

// ReconcileInput is the desired and durable state consumed by one Engine Group workflow step.
type ReconcileInput struct {
	GroupID              GroupID
	SpecGeneration       int64
	DesiredReplicas      int32
	Plan                 *OperationPlan
	Operation            *Operation
	ReleaseAuthorization *ReleaseAuthorization
	TrafficRevision      int64
	TrafficCommand       *TrafficCommand
}

// ReconcileResult contains fresh observations and complete durable workflow state. A caller must persist every field
// whose corresponding Changed flag is true before requeueing, even when Reconcile also returns an error. The error
// describes the external or invariant failure; it does not invalidate safety state returned alongside it.
type ReconcileResult struct {
	Operation                   *Operation
	ReleaseAuthorization        *ReleaseAuthorization
	Capacity                    CapacitySnapshot
	Topology                    MembershipTopology
	Traffic                     TrafficSnapshot
	ServingVerification         *ServingVerificationProof
	OperationChanged            bool
	ReleaseAuthorizationChanged bool
	TrafficRevision             int64
	TrafficCommand              *TrafficCommand
	TrafficStateChanged         bool
}

// WorkflowCoordinator reconciles physical capacity, engine membership, traffic, and safe release.
type WorkflowCoordinator struct {
	capacity     CapacityAdapter
	membership   MembershipAdapter
	traffic      TrafficAdapter
	verifier     ServingVerifier
	operations   *OperationCoordinator
	newReleaseID func() string
}

// NewWorkflowCoordinator constructs an Engine Group workflow. All adapters must be non-nil.
func NewWorkflowCoordinator(
	capacity CapacityAdapter,
	membership MembershipAdapter,
	traffic TrafficAdapter,
	verifier ServingVerifier,
) *WorkflowCoordinator {
	return &WorkflowCoordinator{
		capacity:     capacity,
		membership:   membership,
		traffic:      traffic,
		verifier:     verifier,
		operations:   NewOperationCoordinator(membership),
		newReleaseID: func() string { return string(uuid.NewUUID()) },
	}
}

// Reconcile advances at most one durable transition or one idempotent external side effect. Callers must honor the
// ReconcileResult persistence contract before handling a returned error.
func (c *WorkflowCoordinator) Reconcile(
	ctx context.Context,
	input ReconcileInput,
) (ReconcileResult, error) {
	result := ReconcileResult{
		Operation:            cloneOperation(input.Operation),
		ReleaseAuthorization: cloneReleaseAuthorization(input.ReleaseAuthorization),
		TrafficRevision:      input.TrafficRevision,
		TrafficCommand:       cloneTrafficCommand(input.TrafficCommand),
	}

	// Reject malformed desired or durable state before observing or mutating an external system.
	if err := validateReconcileInput(input); err != nil {
		return result, err
	}

	// Observe and validate the two non-membership state dimensions on every invocation.
	capacity, traffic, err := c.observeWorkflowState(ctx, input.GroupID)
	if err != nil {
		return result, err
	}
	result.Capacity = capacity
	result.Traffic = traffic

	// A durable traffic command is the only mutation that may run before normal membership reconciliation. Withdrawals
	// replay immediately; admission replay first revalidates fresh topology, capacity, and serving proof. Both fail
	// closed on unknown or conflicting revision history.
	result, handled, err := c.reconcileTrafficCommand(ctx, input.GroupID, result)
	if err != nil || handled {
		return result, err
	}

	result, handled, err = c.recoverPostCommitCompletion(ctx, input, result)
	if err != nil || handled {
		return result, err
	}
	result, handled, err = c.reconcileExistingPostCommitState(ctx, input, result)
	if err != nil || handled {
		return result, err
	}

	// A plan is consumed here only when no durable operation exists. Plans queued behind a current operation must not
	// block its safety or repair work; their validation is deferred until replacement or observed adoption.
	if input.Operation == nil {
		if err := validateOperationCandidate(nil, input.Plan); err != nil {
			return result, err
		}
		if err := validateNewRestorationPlan(nil, input.Plan, result.Capacity); err != nil {
			return result, err
		}
	}
	operationResult, err := c.operations.Step(ctx, OperationInput{
		GroupID:         input.GroupID,
		SpecGeneration:  input.SpecGeneration,
		DesiredReplicas: input.DesiredReplicas,
		Plan:            input.Plan,
		Operation:       input.Operation,
	})

	// Any independently changed topology must first be fenced under the old durable operation. A newly derived state,
	// including observed recovery adoption, is discarded until that fence is observable after restart.
	preTransitionResult, fenced, fenceErr := c.ensureObservedOperationTransitionFenced(
		result,
		input.Operation,
		operationResult,
	)
	if fenceErr != nil || !fenced {
		return preTransitionResult, errors.Join(err, fenceErr)
	}
	if newlyAdoptedOperation(input.Operation, operationResult.Operation) {
		if validationErr := validateAdoptedRestoration(
			input.Operation,
			*operationResult.Operation,
			result.Capacity,
		); validationErr != nil {
			return preTransitionResult, errors.Join(err, validationErr)
		}
	}
	result.Operation = operationResult.Operation
	result.Topology = operationResult.Topology
	result.OperationChanged = operationResult.OperationChanged
	if err != nil {
		// A rejected recovery plan cannot leave an independently changed topology serving indefinitely. When the
		// authoritative observation succeeded, establish the same deterministic fence used while awaiting a plan.
		if operationResult.TopologyObserved &&
			result.Operation != nil &&
			operationTopologyDiffersFromServingBaseline(*result.Operation, result.Topology) {
			fencedResult, _, fenceErr := c.ensureUnverifiedTopologyFenced(result)
			if fenceErr != nil {
				return fencedResult, errors.Join(err, fenceErr)
			}
			return fencedResult, err
		}
		return result, err
	}

	if err := validateTrafficAgainstMembership(result.Traffic, result.Topology, result.Operation); err != nil {
		return result, err
	}

	// Persist every newly created or changed operation before another external side effect.
	if result.OperationChanged || result.Operation == nil {
		return result, nil
	}
	return c.reconcileOperationPhase(ctx, input, result, operationResult.SubmissionNeeded)
}

func (c *WorkflowCoordinator) recoverPostCommitCompletion(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	operation := input.Operation
	if operation == nil ||
		operation.Phase != OperationPhaseUnknown ||
		operation.PostCommitComplete ||
		!postCommitWorkComplete(
			*operation,
			input.ReleaseAuthorization,
			result.Capacity,
			result.Traffic,
		) {
		return result, false, nil
	}

	topology, err := c.membership.ObserveTopology(ctx, input.GroupID)
	if err != nil {
		return result, true, fmt.Errorf("observe topology for recovered post-commit completion: %w", err)
	}
	if err := validateTopology(topology); err != nil {
		return result, true, fmt.Errorf("validate topology for recovered post-commit completion: %w", err)
	}
	result.Topology = cloneTopology(topology)

	// A later topology or unavailable exact incarnation invalidates the stale completion proof and fails closed.
	committedTopology := *operation.CommittedTopology
	if !servingVerificationTopologiesEqual(topology, committedTopology) ||
		!replicaAllocationsAvailable(topologyReplicaIncarnations(committedTopology), result.Capacity) {
		result, fenced, err := c.ensureUnverifiedTopologyFenced(result)
		return result, !fenced || result.OperationChanged || result.TrafficStateChanged, err
	}

	result.Operation.PostCommitComplete = true
	result.OperationChanged = true
	return result, true, nil
}

func (c *WorkflowCoordinator) reconcileExistingPostCommitState(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	operation := input.Operation
	if operation == nil {
		return result, false, nil
	}

	// Membership uncertainty never regresses while exact physical release may still be in flight.
	if operation.Phase == OperationPhaseUnknown &&
		operation.CommittedTopology != nil &&
		input.ReleaseAuthorization != nil {
		result, err := c.reconcileUncertainRelease(ctx, input, result)
		return result, true, err
	}

	// A later authoritative survivor topology may still prove old reducing-operation cleanup safe.
	if operation.Phase == OperationPhaseUnknown &&
		operation.CommittedTopology != nil &&
		!operation.PostCommitComplete &&
		operation.TargetReplicas < operation.BaseTopology.ReplicaCount() &&
		!reductionReleaseOwnershipComplete(*operation, input.ReleaseAuthorization, result.Capacity) {
		result, err := c.reconcileUnknownReductionPostwork(ctx, input, result)
		return result, true, err
	}

	// A durably committed membership operation must finish traffic or release work before replacement.
	if operation.Phase == OperationPhaseCommitted {
		result, err := c.reconcileCommitted(ctx, input, result)
		return result, true, err
	}
	return result, false, nil
}

func (c *WorkflowCoordinator) reconcileOperationPhase(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
	submissionNeeded bool,
) (ReconcileResult, error) {
	// Keep preparation and submission ordering explicit for each durable membership phase.
	switch result.Operation.Phase {
	case OperationPhasePending:
		return c.reconcilePending(ctx, input.GroupID, result)
	case OperationPhaseSubmitting:
		if !submissionNeeded {
			return result, nil
		}
		return c.reconcileSubmission(ctx, input.GroupID, result)
	case OperationPhaseFailed:
		return c.reconcileFailed(ctx, input, result)
	case OperationPhaseAborting:
		return c.reconcileAborting(ctx, input, result)
	case OperationPhaseAborted:
		return c.reconcileAborted(ctx, input, result)
	case OperationPhaseUnknown:
		return c.reconcileUnknown(ctx, input, result)
	case OperationPhaseAccepted, OperationPhaseCommitting:
		return c.reconcileInFlight(result)
	default:
		return result, fmt.Errorf("unsupported workflow operation phase %q", result.Operation.Phase)
	}
}

func (c *WorkflowCoordinator) reconcileInFlight(
	result ReconcileResult,
) (ReconcileResult, error) {
	// Progress that still reports an uncommitted phase cannot safely coexist with authoritative base drift.
	if !operationMatchesBaseTopology(*result.Operation, result.Topology) {
		result, fenced, err := c.ensureUnverifiedTopologyFenced(result)
		if err != nil || !fenced {
			return result, err
		}
		result.Operation = transitionOperation(result.Operation, OperationPhaseUnknown, c.operations.now())
		result.OperationChanged = true
		return result, nil
	}

	// Never reselect or repair capacity under an in-flight immutable request; only fence and keep observing it.
	activeReplicas := topologyReplicaIncarnations(result.Topology)
	if !replicaAllocationsAvailable(activeReplicas, result.Capacity) {
		result, _, err := c.ensureUnverifiedTopologyFenced(result)
		return result, err
	}
	return c.reconcileInFlightTraffic(result)
}

func (c *WorkflowCoordinator) reconcileInFlightTraffic(
	result ReconcileResult,
) (ReconcileResult, error) {
	// The traffic adapter promises that an observed drain remains durable until explicit admission. Reassert the
	// operation's exact withdrawal if that invariant nevertheless regresses while membership is still committing.
	if preCommitTrafficReady(*result.Operation, result.Traffic) {
		return result, nil
	}
	result, err := scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
		OperationID:        result.Operation.ID,
		TopologyGeneration: result.Topology.Generation,
		Replicas:           preCommitWithdrawalReplicas(*result.Operation),
	})
	if err != nil {
		return result, fmt.Errorf("restore in-flight Engine Group traffic safety: %w", err)
	}
	return result, nil
}

func (c *WorkflowCoordinator) abortForCapabilityLoss(
	result ReconcileResult,
	cause error,
) ReconcileResult {
	result.Operation.CompensationTopology = topologyPointer(result.Topology)
	result.Operation.Failure = &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "CapabilityLost",
		Message:        cause.Error(),
	}
	result.Operation = transitionOperation(
		result.Operation,
		OperationPhaseAborting,
		c.operations.now(),
	)
	result.OperationChanged = true
	return result
}

func (c *WorkflowCoordinator) abortForTerminalPreconditionFailure(
	result ReconcileResult,
	reason string,
	message string,
) ReconcileResult {
	result.Operation.CompensationTopology = topologyPointer(result.Topology)
	result.Operation.Failure = &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         reason,
		Message:        message,
	}
	result.Operation = transitionOperation(
		result.Operation,
		OperationPhaseAborting,
		c.operations.now(),
	)
	result.OperationChanged = true
	return result
}

func (c *WorkflowCoordinator) handleSubmissionPreparationError(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
	action string,
	cause error,
) (ReconcileResult, error) {
	if !errors.Is(cause, ErrMembershipOperationUnsupported) {
		return result, fmt.Errorf("%s: %w", action, cause)
	}

	durableOperation := cloneOperation(result.Operation)
	resolved, resolveErr := c.operations.ResolvePreparationRejection(ctx, groupID, result.Operation)
	preTransitionResult, fenced, fenceErr := c.ensureObservedOperationTransitionFenced(
		result,
		durableOperation,
		resolved,
	)
	if fenceErr != nil || !fenced {
		return preTransitionResult, errors.Join(resolveErr, fenceErr)
	}
	if resolveErr != nil {
		return preTransitionResult, fmt.Errorf("%s after membership changed: %w", action, resolveErr)
	}
	if newlyAdoptedOperation(durableOperation, resolved.Operation) {
		if err := validateAdoptedRestoration(durableOperation, *resolved.Operation, result.Capacity); err != nil {
			return preTransitionResult, err
		}
	}
	if resolved.OperationChanged {
		result.Operation = resolved.Operation
		result.Topology = resolved.Topology
		result.OperationChanged = true
		return result, nil
	}
	return c.abortForCapabilityLoss(result, cause), nil
}

func (c *WorkflowCoordinator) observeWorkflowState(
	ctx context.Context,
	groupID GroupID,
) (CapacitySnapshot, TrafficSnapshot, error) {
	// Capacity observation must expose complete identities and durable release fences.
	capacity, err := c.capacity.ObserveCapacity(ctx, groupID)
	if err != nil {
		return CapacitySnapshot{}, TrafficSnapshot{}, fmt.Errorf("observe Engine Group capacity: %w", err)
	}
	if err := validateCapacitySnapshot(capacity); err != nil {
		return CapacitySnapshot{}, TrafficSnapshot{}, fmt.Errorf("validate Engine Group capacity: %w", err)
	}

	// Traffic observation remains independent from Pod readiness and engine membership.
	traffic, err := c.traffic.ObserveTraffic(ctx, groupID)
	if err != nil {
		return CapacitySnapshot{}, TrafficSnapshot{}, fmt.Errorf("observe Engine Group traffic: %w", err)
	}
	if err := validateTrafficSnapshot(traffic); err != nil {
		return CapacitySnapshot{}, TrafficSnapshot{}, fmt.Errorf("validate Engine Group traffic: %w", err)
	}
	return cloneCapacitySnapshot(capacity), cloneTrafficSnapshot(traffic), nil
}

func (c *WorkflowCoordinator) ensureCapacity(
	ctx context.Context,
	groupID GroupID,
	request CapacityRequest,
) error {
	if err := validateCapacityRequest(request); err != nil {
		return fmt.Errorf("validate Engine Group capacity request: %w", err)
	}
	return c.capacity.EnsureCapacity(ctx, groupID, request)
}

type trafficObservationDisposition uint8

const (
	trafficObservationReplayDurable trafficObservationDisposition = iota
	trafficObservationConverged
	trafficObservationStateChanged
)

func (c *WorkflowCoordinator) reconcileTrafficCommand(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	result, disposition, err := c.reconcileObservedTrafficCommand(ctx, groupID, result)
	if err != nil {
		return result, true, err
	}
	switch disposition {
	case trafficObservationConverged:
		return result, false, nil
	case trafficObservationStateChanged:
		return result, true, nil
	case trafficObservationReplayDurable:
		return c.replayDurableTrafficCommand(ctx, groupID, result)
	default:
		return result, true, fmt.Errorf("unsupported traffic observation disposition %d", disposition)
	}
}

func (c *WorkflowCoordinator) reconcileObservedTrafficCommand(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, trafficObservationDisposition, error) {
	observed := result.Traffic.LatestCommand
	if observed == nil {
		return result, trafficObservationReplayDurable, nil
	}

	result, terminalAdmissionRecorded, err := c.recordObservedTerminalAdmissionFailure(result, *observed)
	if err != nil {
		return result, trafficObservationStateChanged, err
	}
	if terminalAdmissionRecorded && observed.Phase == TrafficCommandPhaseRefused {
		// Persist the no-mutation terminal outcome before post-commit failure handling establishes the recovery fence.
		return result, trafficObservationStateChanged, nil
	}

	result, handled, err := c.reconcileOlderTerminalTrafficFailure(ctx, groupID, result, *observed)
	if err != nil || handled {
		return result, trafficObservationStateChanged, err
	}
	if terminalAdmissionRecorded && observed.Command.Request.Revision < result.TrafficRevision {
		// Persist the terminal failure before dispatching an already-durable covering withdrawal. Otherwise a crash
		// could retain only the withdrawal and later retry the admission whose terminal outcome it superseded.
		return result, trafficObservationStateChanged, nil
	}

	switch {
	case observed.Command.Request.Revision > result.TrafficRevision:
		return result, trafficObservationStateChanged, fmt.Errorf(
			"traffic adapter revision %d is ahead of durable revision %d",
			observed.Command.Request.Revision,
			result.TrafficRevision,
		)
	case observed.Command.Request.Revision == result.TrafficRevision:
		return c.reconcileCurrentTrafficObservation(ctx, groupID, result, *observed)
	default:
		return result, trafficObservationReplayDurable, nil
	}
}

func (c *WorkflowCoordinator) recordObservedTerminalAdmissionFailure(
	result ReconcileResult,
	observed TrafficCommandObservation,
) (ReconcileResult, bool, error) {
	if observed.Command.Action != TrafficActionAdmit ||
		(observed.Phase != TrafficCommandPhaseRefused && observed.Phase != TrafficCommandPhaseFailed) ||
		observed.Failure.Classification != FailureClassificationTerminal ||
		observed.Command.Request.Revision > result.TrafficRevision {
		return result, false, nil
	}
	return c.recordTerminalAdmissionFailure(result, observed)
}

func (c *WorkflowCoordinator) reconcileOlderTerminalTrafficFailure(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
	observed TrafficCommandObservation,
) (ReconcileResult, bool, error) {
	if observed.Phase != TrafficCommandPhaseFailed ||
		observed.Failure.Classification != FailureClassificationTerminal ||
		observed.Command.Request.Revision >= result.TrafficRevision ||
		(result.TrafficCommand != nil && trafficCommandSafelySupersedes(observed.Command, *result.TrafficCommand)) {
		return result, false, nil
	}
	if observed.Command.Action != TrafficActionAdmit {
		return result, true, fmt.Errorf(
			"terminally failed traffic command revision %d cannot be superseded by durable revision %d",
			observed.Command.Request.Revision,
			result.TrafficRevision,
		)
	}

	withdrawn, _, err := c.revalidateAdmission(ctx, groupID, result, observed.Command)
	if err != nil {
		return result, true, fmt.Errorf("fence older terminally failed admission: %w", err)
	}
	return withdrawn, true, nil
}

func (c *WorkflowCoordinator) reconcileCurrentTrafficObservation(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
	observed TrafficCommandObservation,
) (ReconcileResult, trafficObservationDisposition, error) {
	if result.TrafficCommand == nil || !sameTrafficCommand(observed.Command, *result.TrafficCommand) {
		return result, trafficObservationStateChanged, fmt.Errorf(
			"traffic adapter revision %d does not match the durable command",
			observed.Command.Request.Revision,
		)
	}
	if observed.Phase != TrafficCommandPhaseFailed {
		// An exact Refused observation proves that no delayed effect can occur. Continue to fresh membership
		// observation so a topology-dependent replacement command can be durably issued at the next revision.
		return result, trafficObservationConverged, nil
	}
	if observed.Failure.Classification != FailureClassificationTerminal {
		// Retry the complete original absolute request rather than a downstream caller's recomputed remainder.
		// Failed guarantees revision N has stopped before N+1 is made durable.
		retried, err := scheduleTrafficCommand(result, observed.Command.Action, observed.Command.Request)
		if err != nil {
			return result, trafficObservationStateChanged, fmt.Errorf("retry failed traffic command: %w", err)
		}
		return retried, trafficObservationStateChanged, nil
	}
	if observed.Command.Action != TrafficActionAdmit {
		return result, trafficObservationStateChanged, fmt.Errorf(
			"traffic command revision %d failed terminally: %s",
			result.TrafficRevision,
			observed.Failure.Message,
		)
	}

	// Failed may expose a partial admission. Persist a higher-revision covering withdrawal before surfacing the
	// terminal failure so later safety reconciliation cannot be trapped behind it.
	withdrawn, _, err := c.revalidateAdmission(ctx, groupID, result, observed.Command)
	if err != nil {
		return result, trafficObservationStateChanged, fmt.Errorf("fence terminally failed admission: %w", err)
	}
	return withdrawn, trafficObservationStateChanged, nil
}

func (c *WorkflowCoordinator) replayDurableTrafficCommand(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	if result.TrafficRevision == 0 {
		return result, false, nil
	}
	if result.TrafficCommand == nil || result.TrafficCommand.Request.Revision != result.TrafficRevision {
		return result, true, errors.New("durable traffic revision has no exact command to replay")
	}

	// Replay only the latest durable command. The adapter's monotonic compare-and-set contract makes a delayed older
	// request unable to supersede this revision after timeout, restart, or concurrent controller handoff.
	command := *result.TrafficCommand
	if command.Action == TrafficActionAdmit {
		var safe bool
		var err error
		result, safe, err = c.revalidateAdmission(ctx, groupID, result, command)
		if err != nil || !safe {
			return result, true, err
		}
	}

	var err error
	switch command.Action {
	case TrafficActionAdmit:
		err = c.traffic.Admit(ctx, groupID, command.Request)
	case TrafficActionWithdraw:
		err = c.traffic.Withdraw(ctx, groupID, command.Request)
	default:
		return result, true, fmt.Errorf("unsupported durable traffic command action %q", command.Action)
	}
	if err != nil {
		return result, true, fmt.Errorf(
			"dispatch traffic command revision %d: %w",
			command.Request.Revision,
			err,
		)
	}
	return result, true, nil
}

func (c *WorkflowCoordinator) recordTerminalAdmissionFailure(
	result ReconcileResult,
	observation TrafficCommandObservation,
) (ReconcileResult, bool, error) {
	operation := result.Operation
	if operation == nil || observation.Command.Request.OperationID != operation.ID {
		return result, false, nil
	}

	recorded := &TrafficAdmissionFailure{
		Command: *cloneTrafficCommand(&observation.Command),
		Failure: *cloneFailure(observation.Failure),
	}
	if operation.TerminalAdmissionFailure != nil {
		if sameTrafficAdmissionFailure(*operation.TerminalAdmissionFailure, *recorded) {
			return result, false, nil
		}
		return result, false, fmt.Errorf(
			"operation %q already records a different terminal traffic-admission failure",
			operation.ID,
		)
	}

	switch operation.Phase {
	case OperationPhaseCommitted:
		if operation.CommittedTopology == nil {
			return result, false, errors.New("committed operation has no topology for terminal admission failure")
		}
		operation.Failure = cloneFailure(observation.Failure)
		result.Operation = transitionOperation(operation, OperationPhaseFailed, c.operations.now())
	case OperationPhaseUnknown:
		if operation.CommittedTopology != nil {
			operation.Failure = cloneFailure(observation.Failure)
			result.Operation = transitionOperation(operation, OperationPhaseFailed, c.operations.now())
		} else if operation.CompensationTopology != nil {
			operation.LastTransitionTime = c.operations.now()
		} else {
			return result, false, errors.New("unknown operation has no topology for terminal admission failure")
		}
	case OperationPhaseFailed:
		if operation.CommittedTopology == nil {
			return result, false, errors.New("pre-commit failed operation cannot own an admission failure")
		}
		operation.LastTransitionTime = c.operations.now()
	case OperationPhaseAborting, OperationPhaseAborted:
		if operation.CompensationTopology == nil {
			return result, false, errors.New("compensating operation has no topology for terminal admission failure")
		}
		operation.LastTransitionTime = c.operations.now()
	default:
		return result, false, fmt.Errorf(
			"operation phase %q cannot own a terminal traffic-admission failure",
			operation.Phase,
		)
	}

	result.Operation.TerminalAdmissionFailure = recorded
	result.Operation.ServingVerificationProof = nil
	result.Operation.PostCommitComplete = false
	result.OperationChanged = true
	return result, true, nil
}

func sameTrafficAdmissionFailure(left, right TrafficAdmissionFailure) bool {
	return sameTrafficCommand(left.Command, right.Command) &&
		left.Failure == right.Failure
}

func (c *WorkflowCoordinator) revalidateAdmission(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
	command TrafficCommand,
) (ReconcileResult, bool, error) {
	topology, err := c.membership.ObserveTopology(ctx, groupID)
	if err != nil {
		return result, false, fmt.Errorf("observe topology before traffic admission: %w", err)
	}
	if err := validateTopology(topology); err != nil {
		return result, false, fmt.Errorf("validate topology before traffic admission: %w", err)
	}
	result.Topology = cloneTopology(topology)

	if trafficAdmissionStillAuthorized(command, result.Operation, topology, result.Capacity) {
		return result, true, nil
	}

	// The persisted admission lost authority before dispatch. Fence every incarnation on both sides of the freshly
	// observed topology, not only the stale Admit payload: retained members may already be serving an independently
	// changed collective. A covering higher revision also makes delayed delivery of the old command harmless.
	request := command.Request
	request.TopologyGeneration = topology.Generation
	if result.Operation != nil {
		request.OperationID = unverifiedTopologyTrafficOperationID(*result.Operation, topology)
		request.Replicas = unverifiedTopologyDrainReplicas(*result.Operation, topology)
	}
	withdrawn, err := scheduleTrafficCommand(result, TrafficActionWithdraw, request)
	if err != nil {
		return result, false, fmt.Errorf("supersede stale traffic admission: %w", err)
	}
	return withdrawn, false, nil
}

func trafficAdmissionStillAuthorized(
	command TrafficCommand,
	operation *Operation,
	topology MembershipTopology,
	capacity CapacitySnapshot,
) bool {
	if operation == nil ||
		operation.TerminalAdmissionFailure != nil ||
		command.Action != TrafficActionAdmit ||
		command.Request.OperationID != operation.ID ||
		command.Request.TopologyGeneration != topology.Generation ||
		!containsAllReplicaIncarnations(topologyReplicaIncarnations(topology), command.Request.Replicas) ||
		!replicaAllocationsAvailable(topologyReplicaIncarnations(topology), capacity) {
		return false
	}

	var authoritativeTopology *MembershipTopology
	switch operation.Phase {
	case OperationPhaseCommitted, OperationPhaseFailed, OperationPhaseUnknown:
		authoritativeTopology = operation.CommittedTopology
	case OperationPhaseAborting, OperationPhaseAborted:
		authoritativeTopology = operation.CompensationTopology
	default:
		return false
	}
	if authoritativeTopology == nil || !servingVerificationTopologiesEqual(topology, *authoritativeTopology) {
		return false
	}
	if operation.Capability.VerificationRequirement != ServingVerificationRequired {
		return true
	}
	request := ServingVerificationRequest{
		OperationID:         operation.ID,
		Attempt:             operation.Attempt,
		VerificationAttempt: operation.ServingVerificationAttempt,
		Topology:            topology,
	}
	return operation.ServingVerificationProof != nil &&
		servingVerificationPassedForRequest(*operation.ServingVerificationProof, request)
}

func scheduleTrafficCommand(
	result ReconcileResult,
	action TrafficAction,
	request TrafficRequest,
) (ReconcileResult, error) {
	request.Revision = 0
	desired := TrafficCommand{Action: action, Request: request}
	if result.Traffic.LatestCommand != nil &&
		result.Traffic.LatestCommand.Phase == TrafficCommandPhaseAccepted &&
		result.TrafficCommand != nil &&
		sameTrafficCommand(result.Traffic.LatestCommand.Command, *result.TrafficCommand) &&
		!trafficCommandEffectComplete(*result.TrafficCommand, result.Traffic) &&
		!trafficCommandSafelySupersedes(*result.TrafficCommand, desired) {
		// Do not replace an in-flight absolute request with a recomputed remainder. Its terminal result must remain
		// observable at the controller's current durable revision before any non-safety-increasing command is issued.
		return result, nil
	}
	if result.TrafficCommand != nil && sameTrafficCommandPayload(*result.TrafficCommand, desired) {
		if result.Traffic.LatestCommand != nil &&
			sameTrafficCommand(result.Traffic.LatestCommand.Command, *result.TrafficCommand) {
			switch result.Traffic.LatestCommand.Phase {
			case TrafficCommandPhaseRefused:
				return result, fmt.Errorf(
					"traffic command revision %d was definitively refused: %s",
					result.TrafficRevision,
					result.Traffic.LatestCommand.Failure.Message,
				)
			case TrafficCommandPhaseFailed:
				if result.Traffic.LatestCommand.Failure.Classification == FailureClassificationTerminal {
					return result, fmt.Errorf(
						"traffic command revision %d failed terminally: %s",
						result.TrafficRevision,
						result.Traffic.LatestCommand.Failure.Message,
					)
				}
				// Retry the same absolute request under a new revision. Failed guarantees the prior revision has
				// stopped, and the adapter's exact snapshot makes any partial effect safe to converge idempotently.
			default:
				return result, nil
			}
		} else {
			return result, nil
		}
	}
	if result.TrafficRevision == math.MaxInt64 {
		return result, errors.New("traffic command revision exhausted")
	}

	// Reserve the next group-global revision and exact payload durably before an adapter mutation can be dispatched.
	desired.Request.Revision = result.TrafficRevision + 1
	if err := validateTrafficCommand(desired); err != nil {
		return result, err
	}
	result.TrafficRevision = desired.Request.Revision
	result.TrafficCommand = cloneTrafficCommand(&desired)
	result.TrafficStateChanged = true
	return result, nil
}

func trafficCommandEffectComplete(command TrafficCommand, traffic TrafficSnapshot) bool {
	switch command.Action {
	case TrafficActionAdmit:
		return containsAllReplicaIncarnations(traffic.Admitted, command.Request.Replicas) &&
			len(intersectReplicaIncarnations(traffic.Drained, command.Request.Replicas)) == 0
	case TrafficActionWithdraw:
		return containsAllReplicaIncarnations(traffic.Drained, command.Request.Replicas) &&
			len(intersectReplicaIncarnations(traffic.Admitted, command.Request.Replicas)) == 0
	default:
		return false
	}
}

func trafficCommandSafelySupersedes(previous, next TrafficCommand) bool {
	return next.Action == TrafficActionWithdraw &&
		containsAllReplicaIncarnations(next.Request.Replicas, previous.Request.Replicas)
}

func (c *WorkflowCoordinator) reconcileUnknownReductionPostwork(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	operation := *result.Operation
	if operation.TargetReplicas >= operation.BaseTopology.ReplicaCount() {
		return result, nil
	}

	// Continue old cleanup only when a committed topology at least as new contains no unexpected identity.
	topology, err := c.membership.ObserveTopology(ctx, input.GroupID)
	if err != nil {
		return result, fmt.Errorf("observe topology for uncertain reduction cleanup: %w", err)
	}
	if err := validateTopology(topology); err != nil {
		return result, fmt.Errorf("validate topology for uncertain reduction cleanup: %w", err)
	}
	result.Topology = cloneTopology(topology)
	safe, err := cleanupTopologySafe(operation, topology)
	if err != nil {
		return result, err
	}
	if !safe {
		return result, nil
	}

	// A missing, replaced, or unavailable survivor must be fully fenced even while old-victim cleanup is pending.
	if !replicaAllocationsAvailable(topologyReplicaIncarnations(topology), result.Capacity) {
		result, fenced, err := c.ensureUnverifiedTopologyFenced(result)
		if err != nil || !fenced {
			return result, err
		}
	}

	// A later unverified survivor topology may release only the exact old victims while every current and retiring
	// identity is fenced. Admission remains blocked until a distinct recovery adopts and verifies the new topology.
	trafficReady := retirementTrafficReady(operation, topology, result.Traffic)
	if !servingVerificationTopologiesEqual(topology, *operation.CommittedTopology) {
		result, trafficReady, err = c.ensureUnverifiedTopologyFenced(result)
		if err != nil || !trafficReady {
			return result, err
		}
	}
	// The later topology attests quiescence; old exact traffic drain and release obligations remain in force.
	allowedAdmissions := append(
		topologyReplicaIncarnations(topology),
		topologyReplicaIncarnationsForIDs(operation.BaseTopology, operation.NominatedReplicas)...,
	)
	if err := validateTrafficAgainstReplicaSet(result.Traffic, allowedAdmissions); err != nil {
		return result, err
	}
	if !trafficReady {
		result, err := scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
			OperationID:        operation.ID,
			TopologyGeneration: topology.Generation,
			Replicas: topologyReplicaIncarnationsForIDs(
				operation.BaseTopology,
				operation.NominatedReplicas,
			),
		})
		if err != nil {
			return result, fmt.Errorf("resume uncertain reduction drain: %w", err)
		}
		return result, nil
	}
	return c.reconcileRelease(ctx, input, result)
}

func (c *WorkflowCoordinator) reconcileUncertainRelease(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	// Revalidate survivor topology and traffic even while an already-issued release remains in progress.
	topology, err := c.membership.ObserveTopology(ctx, input.GroupID)
	if err != nil {
		return result, fmt.Errorf("observe topology during uncertain capacity release: %w", err)
	}
	if err := validateTopology(topology); err != nil {
		return result, fmt.Errorf("validate topology during uncertain capacity release: %w", err)
	}
	result.Topology = cloneTopology(topology)
	safe, err := cleanupTopologySafe(*input.Operation, topology)
	if err != nil {
		return result, err
	}
	if !safe {
		return result, nil
	}

	// A missing, replaced, or unavailable survivor must be fully fenced throughout an asynchronous victim release.
	if !replicaAllocationsAvailable(topologyReplicaIncarnations(topology), result.Capacity) {
		result, fenced, err := c.ensureUnverifiedTopologyFenced(result)
		if err != nil || !fenced {
			return result, err
		}
	}

	// A later survivor generation invalidates the old serving proof. Fence every current and retiring identity before
	// observing or replaying a release that may already be in flight; a later explicit recovery must re-verify it.
	trafficReady := retirementTrafficReady(*input.Operation, topology, result.Traffic)
	if !servingVerificationTopologiesEqual(topology, *input.Operation.CommittedTopology) {
		result, trafficReady, err = c.ensureUnverifiedTopologyFenced(result)
		if err != nil || !trafficReady {
			return result, err
		}
	}
	if err := validateAuthorizationAgainstTopology(
		*input.ReleaseAuthorization,
		*input.Operation,
		topology,
	); err != nil {
		return result, err
	}
	allowedAdmissions := append(
		topologyReplicaIncarnations(topology),
		topologyReplicaIncarnationsForIDs(
			input.Operation.BaseTopology,
			input.Operation.NominatedReplicas,
		)...,
	)
	if err := validateTrafficAgainstReplicaSet(result.Traffic, allowedAdmissions); err != nil {
		return result, err
	}
	if !trafficReady {
		return result, fmt.Errorf(
			"traffic drain for operation %q regressed while capacity release %q is outstanding",
			input.Operation.ID,
			input.ReleaseAuthorization.ID,
		)
	}

	// Observe first; an absent release is issued only after fresh topology, traffic, and UID proof below.
	observation, err := c.capacity.ObserveRelease(
		ctx,
		input.GroupID,
		result.ReleaseAuthorization.ID,
	)
	if err != nil {
		return result, fmt.Errorf(
			"observe capacity release %q while membership is unknown: %w",
			result.ReleaseAuthorization.ID,
			err,
		)
	}
	if err := validateCapacityReleaseObservation(
		observation,
		result.ReleaseAuthorization.ID,
	); err != nil {
		return result, err
	}

	// Only authoritative completion with every old UID absent can retire uncertain release state.
	switch observation.Phase {
	case CapacityReleasePhaseAbsent:
		return c.issueRevalidatedUnknownRelease(ctx, input, result)
	case CapacityReleasePhaseApplying:
		return result, nil
	case CapacityReleasePhaseApplied:
		if authorizedCapacityPresent(*result.ReleaseAuthorization, result.Capacity) {
			return result, nil
		}
		if !authorizationFencesObserved(*result.ReleaseAuthorization, result.Capacity) {
			return result, nil
		}
		if result.ReleaseAuthorization.TopologyGeneration != result.Topology.Generation ||
			result.ReleaseAuthorization.TargetReplicas != result.Topology.ReplicaCount() {
			authorization, err := c.buildReleaseAuthorization(
				*result.Operation,
				result.Capacity,
				result.Topology,
			)
			if err != nil {
				return result, err
			}
			clearCapacityTargetProof(result.Operation)
			result.ReleaseAuthorization = authorization
			result.ReleaseAuthorizationChanged = true
			result.OperationChanged = true
			return result, nil
		}
		result.OperationChanged = setCapacityTargetProof(
			result.Operation,
			*result.ReleaseAuthorization,
		) || result.OperationChanged
		result.ReleaseAuthorization = nil
		result.ReleaseAuthorizationChanged = true
		return result, nil
	case CapacityReleasePhaseRefused:
		if observation.Failure.Classification == FailureClassificationRetryable {
			result.ReleaseAuthorization = nil
			result.ReleaseAuthorizationChanged = true
		}
		return result, fmt.Errorf(
			"capacity release %q was refused while membership is unknown: %s: %s",
			observation.ReleaseID,
			observation.Failure.Reason,
			observation.Failure.Message,
		)
	case CapacityReleasePhaseFailed:
		if observation.Failure.Classification == FailureClassificationRetryable {
			// Failed guarantees the old release has stopped. Replan from the exact remaining allocations and fences.
			result.OperationChanged = clearCapacityTargetProof(result.Operation) || result.OperationChanged
			result.ReleaseAuthorization = nil
			result.ReleaseAuthorizationChanged = true
		}
		return result, fmt.Errorf(
			"capacity release %q failed while membership is unknown: %s: %s",
			observation.ReleaseID,
			observation.Failure.Reason,
			observation.Failure.Message,
		)
	default:
		return result, fmt.Errorf("unsupported capacity release phase %q", observation.Phase)
	}
}

func (c *WorkflowCoordinator) reconcileUnknown(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	activeReplicas := topologyReplicaIncarnations(result.Topology)
	if !replicaAllocationsAvailable(activeReplicas, result.Capacity) ||
		operationTopologyDiffersFromServingBaseline(*result.Operation, result.Topology) {
		// Membership or capacity uncertainty never inherits a prior admission or serving proof.
		result, fenced, err := c.ensureUnverifiedTopologyFenced(result)
		if err != nil || !fenced {
			return result, err
		}
		if result.Operation.CommittedTopology != nil &&
			servingVerificationTopologiesEqual(result.Topology, *result.Operation.CommittedTopology) &&
			!replicaIncarnationsPresent(activeReplicas, result.Capacity) {
			result, _, err := c.reconcileActiveCapacityRegression(ctx, input, result)
			return result, err
		}
		return result, nil
	}
	if result.Operation.CommittedTopology != nil &&
		servingVerificationTopologiesEqual(result.Topology, *result.Operation.CommittedTopology) {
		return c.reconcileCommitted(ctx, input, result)
	}
	return result, nil
}

func (c *WorkflowCoordinator) ensureUnverifiedTopologyFenced(
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	replicas := unverifiedTopologyDrainReplicas(*result.Operation, result.Topology)
	if len(replicas) == 0 {
		return result, true, nil
	}

	operationID := unverifiedTopologyTrafficOperationID(*result.Operation, result.Topology)
	ready := unverifiedTopologyTrafficReady(*result.Operation, result.Topology, result.Traffic)
	if ready {
		return result, true, nil
	}
	result, err := scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
		OperationID:        operationID,
		TopologyGeneration: result.Topology.Generation,
		Replicas:           replicas,
	})
	if err != nil {
		return result, false, fmt.Errorf("fence unverified Engine Group topology: %w", err)
	}
	return result, false, nil
}

func (c *WorkflowCoordinator) ensureObservedOperationTransitionFenced(
	result ReconcileResult,
	durableOperation *Operation,
	observed OperationResult,
) (ReconcileResult, bool, error) {
	if durableOperation == nil ||
		!observed.TopologyObserved ||
		!operationTopologyDiffersFromServingBaseline(*durableOperation, observed.Topology) {
		return result, true, nil
	}
	if correlatedCommittedTransition(*durableOperation, observed) {
		return c.ensureCorrelatedCommitJoinersFenced(result, *durableOperation, observed.Topology)
	}

	// Keep the old operation as the durable owner of traffic safety until the independently observed topology is
	// fully fenced. The caller may persist a replacement operation only after that exact fence is observable.
	result.Operation = cloneOperation(durableOperation)
	result.Topology = cloneTopology(observed.Topology)
	result.OperationChanged = false
	return c.ensureUnverifiedTopologyFenced(result)
}

func correlatedCommittedTransition(durableOperation Operation, observed OperationResult) bool {
	return observed.Operation != nil &&
		observed.Operation.ID == durableOperation.ID &&
		observed.Operation.Attempt == durableOperation.Attempt &&
		observed.Operation.Phase == OperationPhaseCommitted &&
		observed.Operation.CommittedTopology != nil &&
		servingVerificationTopologiesEqual(*observed.Operation.CommittedTopology, observed.Topology) &&
		operationMatchesCommittedTopology(*observed.Operation, observed.Topology)
}

func (c *WorkflowCoordinator) ensureCorrelatedCommitJoinersFenced(
	result ReconcileResult,
	durableOperation Operation,
	topology MembershipTopology,
) (ReconcileResult, bool, error) {
	joiningReplicas := verificationJoiningReplicas(durableOperation, topology)
	if len(joiningReplicas) == 0 ||
		(trafficStateStableFor(result.Traffic, TrafficActionWithdraw, joiningReplicas) &&
			containsAllReplicaIncarnations(result.Traffic.Drained, joiningReplicas) &&
			len(intersectReplicaIncarnations(result.Traffic.Admitted, joiningReplicas)) == 0) {
		return result, true, nil
	}

	// Preserve the old operation until every new incarnation is explicitly non-routable. Existing members continue
	// serving when the resolved operation allows it; unexplained or adopted transitions still use the full fence.
	result.Operation = cloneOperation(&durableOperation)
	result.Topology = cloneTopology(topology)
	result.OperationChanged = false
	result, err := scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
		OperationID:        durableOperation.ID,
		TopologyGeneration: topology.Generation,
		Replicas:           joiningReplicas,
	})
	if err != nil {
		return result, false, fmt.Errorf("fence correlated Engine Group commit joiners: %w", err)
	}
	return result, false, nil
}

func (c *WorkflowCoordinator) issueRevalidatedUnknownRelease(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	// Re-observe the exact topology generation that supplied quiescence proof before first issuing deletion.
	topology, err := c.membership.ObserveTopology(ctx, input.GroupID)
	if err != nil {
		return result, fmt.Errorf("observe topology for uncertain capacity release: %w", err)
	}
	if err := validateTopology(topology); err != nil {
		return result, fmt.Errorf("validate topology for uncertain capacity release: %w", err)
	}
	result.Topology = cloneTopology(topology)
	if topology.Generation != input.ReleaseAuthorization.TopologyGeneration {
		result.OperationChanged = clearCapacityTargetProof(result.Operation) || result.OperationChanged
		result.ReleaseAuthorization = nil
		result.ReleaseAuthorizationChanged = true
		return result, nil
	}
	safe, err := cleanupTopologySafe(*input.Operation, topology)
	if err != nil {
		return result, err
	}
	if !safe {
		return result, nil
	}
	if err := validateAuthorizationAgainstTopology(
		*input.ReleaseAuthorization,
		*input.Operation,
		topology,
	); err != nil {
		return result, err
	}

	// Revalidate traffic and every capacity binding against fresh observations before replaying the exact request.
	allowedAdmissions := append(
		topologyReplicaIncarnations(*input.Operation.CommittedTopology),
		topologyReplicaIncarnationsForIDs(
			input.Operation.BaseTopology,
			input.Operation.NominatedReplicas,
		)...,
	)
	if err := validateTrafficAgainstReplicaSet(result.Traffic, allowedAdmissions); err != nil {
		return result, err
	}
	trafficReady := retirementTrafficReady(*input.Operation, topology, result.Traffic)
	if !servingVerificationTopologiesEqual(topology, *input.Operation.CommittedTopology) {
		trafficReady = unverifiedTopologyTrafficReady(*input.Operation, topology, result.Traffic)
	}
	if !trafficReady {
		result.ReleaseAuthorization = nil
		result.ReleaseAuthorizationChanged = true
		return result, nil
	}
	if err := validateAuthorizationCompleteness(
		*input.ReleaseAuthorization,
		input.Operation.NominatedReplicas,
	); err != nil {
		result.ReleaseAuthorization = nil
		result.ReleaseAuthorizationChanged = true
		return result, fmt.Errorf("discard uncertain capacity release %q: %w", input.ReleaseAuthorization.ID, err)
	}
	if err := validateAuthorizationFresh(*input.ReleaseAuthorization, result.Capacity); err != nil {
		result.ReleaseAuthorization = nil
		result.ReleaseAuthorizationChanged = true
		return result, fmt.Errorf("discard uncertain capacity release %q: %w", input.ReleaseAuthorization.ID, err)
	}
	if err := c.capacity.Release(ctx, input.GroupID, *cloneReleaseAuthorization(input.ReleaseAuthorization)); err != nil {
		return result, fmt.Errorf("release revalidated uncertain Engine Group capacity: %w", err)
	}
	return result, nil
}

func (c *WorkflowCoordinator) reconcilePending(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, error) {
	operation := result.Operation
	delta := operation.TargetReplicas - operation.BaseTopology.ReplicaCount()
	var joiningReplicas []ReplicaIncarnation

	// Cardinally stable recovery names concrete target incarnations up front; never submit a stale physical target.
	if operation.Capability.Shape == OperationShapeFixedSlotReplacement ||
		operation.Capability.Shape == OperationShapeNativeMemberRemapping {
		targetIncarnations := replicaMembershipIncarnations(operation.TargetMembership)
		if !replicaIncarnationsPresent(targetIncarnations, result.Capacity) {
			return c.abortForTerminalPreconditionFailure(
				result,
				"TargetIncarnationLost",
				"the exact cardinal-recovery target incarnation is no longer allocated",
			), nil
		}
		if !replicaAllocationsAvailable(targetIncarnations, result.Capacity) {
			if err := c.ensureCapacity(ctx, groupID, CapacityRequest{
				OperationID:        operation.ID,
				TopologyGeneration: result.Topology.Generation,
				TargetReplicas:     operation.TargetReplicas,
				RequiredReplicas:   requiredReplicaAllocations(targetIncarnations),
			}); err != nil {
				return result, fmt.Errorf("ensure cardinal-recovery target capacity: %w", err)
			}
			return result, nil
		}
	}

	// Growth first allocates capacity and resolves the exact available joining identities.
	if delta > 0 {
		baseIncarnations := topologyReplicaIncarnations(operation.BaseTopology)

		// A replaced or missing retained incarnation makes this pending request stale before joiner selection begins.
		if !replicaIncarnationsPresent(baseIncarnations, result.Capacity) {
			return c.abortForTerminalPreconditionFailure(
				result,
				"BaseIncarnationLost",
				"an exact base incarnation required by the growth plan is no longer allocated",
			), nil
		}

		// Only transient unavailability of the same exact base incarnation may be repaired under this operation.
		if !replicaAllocationsAvailable(baseIncarnations, result.Capacity) {
			if err := c.ensureCapacity(ctx, groupID, CapacityRequest{
				OperationID:        operation.ID,
				TopologyGeneration: result.Topology.Generation,
				TargetReplicas:     operation.TargetReplicas,
				RequiredReplicas: requiredReplicaAllocations(
					baseIncarnations,
				),
			}); err != nil {
				return result, fmt.Errorf("restore growth base capacity: %w", err)
			}
			return result, nil
		}
		if operation.Capability.Shape == OperationShapeReplacementRestoration {
			requiredReplicas := normalizeRequiredReplicaAllocations(append(
				requiredReplicaAllocations(baseIncarnations),
				restoredReplicaAllocations(operation.RestoredMembership)...,
			))
			if surplus := capacityReplicaIDsOutside(result.Capacity, requiredReplicas); len(surplus) != 0 {
				return c.abortForTerminalPreconditionFailure(
					result,
					"ConflictingSurplusCapacity",
					fmt.Sprintf(
						"replacement restoration requires exact-release cleanup of unrelated replicas %v before retry",
						surplus,
					),
				), nil
			}
		}

		var ready bool
		var err error
		if operation.Capability.Shape == OperationShapeReplacementRestoration {
			joiningReplicas, ready, err = resolveRestoredReplicas(*operation, result.Capacity, result.Traffic)
		} else {
			joiningReplicas, ready, err = resolveJoiningReplicas(*operation, result.Capacity, result.Traffic)
		}
		if err != nil {
			return result, err
		}
		if !ready {
			requiredReplicas := requiredReplicaAllocations(baseIncarnations)
			if operation.Capability.Shape == OperationShapeReplacementRestoration {
				requiredReplicas = normalizeRequiredReplicaAllocations(append(
					requiredReplicas,
					restoredReplicaAllocations(operation.RestoredMembership)...,
				))
			}
			request := CapacityRequest{
				OperationID:        operation.ID,
				TopologyGeneration: result.Topology.Generation,
				TargetReplicas:     operation.TargetReplicas,
				RequiredReplicas:   requiredReplicas,
			}
			if err := c.ensureCapacity(ctx, groupID, request); err != nil {
				return result, fmt.Errorf("ensure %d Engine Group replicas: %w", operation.TargetReplicas, err)
			}
			return result, nil
		}
	}
	preparedCapacity := cloneOperation(operation)
	preparedCapacity.JoiningReplicas = cloneReplicaIncarnations(joiningReplicas)
	requiredIncarnations := frozenSubmissionIncarnations(*preparedCapacity)
	if !replicaIncarnationsPresent(requiredIncarnations, result.Capacity) {
		return c.abortForTerminalPreconditionFailure(
			result,
			"SubmissionIncarnationLost",
			"an exact replica incarnation required by the membership plan is no longer allocated",
		), nil
	}
	if !replicaAllocationsAvailable(requiredIncarnations, result.Capacity) {
		if err := c.ensureCapacity(ctx, groupID, CapacityRequest{
			OperationID:        operation.ID,
			TopologyGeneration: result.Topology.Generation,
			TargetReplicas:     max(operation.TargetReplicas, operation.BaseTopology.ReplicaCount()),
			RequiredReplicas:   requiredReplicaAllocations(requiredIncarnations),
		}); err != nil {
			return result, fmt.Errorf("ensure membership-plan capacity: %w", err)
		}
		return result, nil
	}

	// Traffic handling is resolved per operation shape. KeepServing drains only retirees;
	// QuiesceGroup drains every current member before the membership request may be submitted.
	if !preCommitTrafficReady(*operation, result.Traffic) {
		withdrawals := preCommitWithdrawalReplicas(*operation)
		result, err := scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
			OperationID:        operation.ID,
			TopologyGeneration: result.Topology.Generation,
			Replicas:           withdrawals,
		})
		if err != nil {
			return result, fmt.Errorf("prepare Engine Group traffic for membership change: %w", err)
		}
		return result, nil
	}

	// Freeze the complete exact request and validate it before persisting the submission marker.
	prepared, err := c.operations.PrepareSubmission(ctx, groupID, operation, joiningReplicas)
	if err != nil {
		return c.handleSubmissionPreparationError(ctx, groupID, result, "prepare membership submission", err)
	}
	result.Operation = prepared
	result.OperationChanged = true
	return result, nil
}

func (c *WorkflowCoordinator) reconcileSubmission(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, error) {
	// Replay is allowed only while the exact persisted request still satisfies fresh prerequisites.
	if !submissionPrerequisitesReady(*result.Operation, result.Capacity, result.Traffic) {
		return c.resumePreparation(ctx, groupID, result)
	}

	durableOperation := cloneOperation(result.Operation)
	operationResult, err := c.operations.Submit(ctx, groupID, result.Operation)
	preTransitionResult, fenced, fenceErr := c.ensureObservedOperationTransitionFenced(
		result,
		durableOperation,
		operationResult,
	)
	if fenceErr != nil || !fenced {
		return preTransitionResult, errors.Join(err, fenceErr)
	}
	result.Operation = operationResult.Operation
	result.Topology = operationResult.Topology
	result.OperationChanged = operationResult.OperationChanged
	if err != nil {
		var preflightError *submissionPreflightError
		if errors.As(err, &preflightError) && errors.Is(err, ErrMembershipOperationUnsupported) {
			return c.abortForCapabilityLoss(result, err), nil
		}
		return result, err
	}
	return result, nil
}

func (c *WorkflowCoordinator) reconcileFailed(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	if result.Operation.CommittedTopology != nil {
		return c.reconcilePostCommitFailure(ctx, input, result)
	}

	// A definitive failed attempt cannot later commit. Preserve its exact old base as the cleanup boundary and let the
	// drifted-abort path fence the current topology before exact surplus cleanup or explicit observed recovery.
	if !operationMatchesBaseTopology(*result.Operation, result.Topology) {
		failureMessage := fmt.Sprintf(
			"membership attempt failed before authoritative topology changed: %s: %s",
			result.Operation.Failure.Reason,
			result.Operation.Failure.Message,
		)
		result.Operation.CompensationTopology = topologyPointer(result.Operation.BaseTopology)
		result.Operation.Failure = &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "TopologyChangedAfterOperationFailure",
			Message:        failureMessage,
		}
		result.Operation = transitionOperation(
			result.Operation,
			OperationPhaseAborting,
			c.operations.now(),
		)
		result.OperationChanged = true
		return result, nil
	}

	// Persist terminal failure compensation before restoring any capacity or traffic side effect.
	if result.Operation.Failure.Classification == FailureClassificationTerminal {
		result.Operation.CompensationTopology = topologyPointer(result.Topology)
		result.Operation = transitionOperation(
			result.Operation,
			OperationPhaseAborting,
			c.operations.now(),
		)
		result.OperationChanged = true
		return result, nil
	}

	// Retryable failures reacquire capacity or drain evidence before reusing the same operation identity.
	if !submissionPrerequisitesReady(*result.Operation, result.Capacity, result.Traffic) {
		return c.resumePreparation(ctx, input.GroupID, result)
	}
	prepared, err := c.operations.PrepareSubmission(
		ctx,
		input.GroupID,
		result.Operation,
		result.Operation.JoiningReplicas,
	)
	if err != nil {
		return c.handleSubmissionPreparationError(ctx, input.GroupID, result, "prepare membership retry", err)
	}
	result.Operation = prepared
	result.OperationChanged = true
	return result, nil
}

func (c *WorkflowCoordinator) reconcilePostCommitFailure(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	// Preserve the exact failed-topology fence across restart before accepting any recovery plan.
	result, fenced, err := c.ensureServingFailureTopologyFenced(result)
	if err != nil || !fenced {
		return result, err
	}

	// Finish the failed reduction's exact old-victim release before replacing its only durable ownership record.
	if result.Operation.TargetReplicas < result.Operation.BaseTopology.ReplicaCount() {
		safe, err := cleanupTopologySafe(*result.Operation, result.Topology)
		if err != nil {
			return result, err
		}
		if !safe {
			return result, nil
		}
		if !failedReductionReleaseComplete(
			*result.Operation,
			result.ReleaseAuthorization,
			result.Capacity,
			result.Topology,
		) {
			return c.reconcileRelease(ctx, input, result)
		}
	}

	// Keep the failed record as the durable boundary until a distinct, explicit recovery plan is supplied.
	if input.Plan == nil || input.Plan.ID == result.Operation.PlanID {
		result, handled, err := c.reconcileActiveCapacityRegression(ctx, input, result)
		if err != nil || handled {
			return result, err
		}
		return result, nil
	}
	if input.Plan.Intent != OperationIntentRecover {
		return result, errors.New("post-commit failure requires a distinct explicit Recover plan")
	}
	if !servingVerificationTopologiesEqual(result.Topology, *result.Operation.CommittedTopology) {
		transition, err := observedMembershipTransition(*result.Operation, result.Topology, *input.Plan)
		if err != nil {
			return result, err
		}
		capability, err := c.operations.validateObservedTransition(ctx, input.GroupID, transition)
		if err != nil {
			return result, err
		}
		if capability.VerificationRequirement != ServingVerificationRequired {
			return result, errors.New("recovery after a post-commit failure must require serving verification")
		}
		adopted, err := c.operations.adoptObservedRecovery(
			OperationInput{
				GroupID:         input.GroupID,
				SpecGeneration:  input.SpecGeneration,
				DesiredReplicas: input.DesiredReplicas,
				Plan:            input.Plan,
			},
			transition,
			capability,
		)
		if err != nil {
			return result, err
		}
		if err := validateAdoptedRestoration(input.Operation, *adopted, result.Capacity); err != nil {
			return result, err
		}
		result.Operation = adopted
		result.OperationChanged = true
		return result, nil
	}
	return c.beginRequiredRecovery(ctx, input, result)
}

func (c *WorkflowCoordinator) reconcileAborting(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	result, exactActiveCapacityPresent, handled, err := c.prepareAbortCleanup(ctx, input, result)
	if err != nil || handled {
		return result, err
	}
	result, handled, err = c.reconcileAbortCleanupRelease(ctx, input, result)
	if err != nil || handled {
		return result, err
	}
	if !exactActiveCapacityPresent {
		// Cleanup is complete, but the compensation topology needs a separate recovery operation before serving.
		result.Operation = transitionOperation(
			result.Operation,
			OperationPhaseAborted,
			c.operations.now(),
		)
		result.OperationChanged = true
		return result, nil
	}
	if result.Operation.TerminalAdmissionFailure != nil {
		result, fenced, err := ensureTerminalAdmissionFailureFenced(result)
		if err != nil || !fenced {
			return result, err
		}
		result.Operation = transitionOperation(
			result.Operation,
			OperationPhaseAborted,
			c.operations.now(),
		)
		result.OperationChanged = true
		return result, nil
	}

	// The closed target now permits exact survivor repair without surplus capacity occupying its slots.
	result, restored, err := c.restoreAuthoritativeServingState(ctx, input.GroupID, result)
	if err != nil || !restored {
		return result, err
	}

	// Persist completion only after serving restoration and exact surplus cleanup are observable after restart.
	result.Operation = transitionOperation(
		result.Operation,
		OperationPhaseAborted,
		c.operations.now(),
	)
	result.OperationChanged = true
	return result, nil
}

func (c *WorkflowCoordinator) prepareAbortCleanup(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, bool, bool, error) {
	if !servingVerificationTopologiesEqual(*result.Operation.CompensationTopology, result.Topology) {
		// Fence an independently changed topology before persisting drift or continuing old-operation cleanup.
		result, fenced, err := c.ensureUnverifiedTopologyFenced(result)
		if err != nil || !fenced {
			return result, false, true, err
		}

		// Persist the drift outcome before deriving a new cleanup authorization from that topology.
		if markCompensationTopologyChanged(result.Operation, result.Topology, c.operations.now()) {
			result.OperationChanged = true
			return result, false, true, nil
		}
		result, err = c.reconcileDriftedAbortCleanup(ctx, input.GroupID, result)
		return result, false, true, err
	}
	if err := validateAbortCleanupTopology(*result.Operation, result.Topology); err != nil {
		return result, false, true, err
	}

	activeReplicas := topologyReplicaIncarnations(result.Topology)
	exactActiveCapacityPresent := replicaIncarnationsPresent(activeReplicas, result.Capacity)
	if !replicaAllocationsAvailable(activeReplicas, result.Capacity) {
		result, fenced, err := c.ensureUnverifiedTopologyFenced(result)
		if err != nil || !fenced {
			return result, exactActiveCapacityPresent, true, err
		}
		result, invalidated, err := invalidateServingVerificationAfterCapacityRegression(result)
		if err != nil || invalidated {
			return result, exactActiveCapacityPresent, true, err
		}
	}

	// Freeze every currently surplus allocation before authorizing any physical removal.
	if input.ReleaseAuthorization == nil {
		cleanupReplicaSlots, err := discoverAbortCleanupReplicaSlots(
			*result.Operation,
			result.Capacity,
			result.Topology,
		)
		if err != nil {
			return result, exactActiveCapacityPresent, true, err
		}
		if !sameReplicaSlotBindings(cleanupReplicaSlots, result.Operation.CleanupReplicaSlots) {
			result.Operation.CleanupReplicaSlots = cleanupReplicaSlots
			clearCapacityTargetProof(result.Operation)
			result.OperationChanged = true
			return result, exactActiveCapacityPresent, true, nil
		}
	}
	if err := validateAbortReleaseBarrier(input.ReleaseAuthorization, result.Topology); err != nil {
		return result, exactActiveCapacityPresent, true, err
	}
	return result, exactActiveCapacityPresent, false, nil
}

func validateAbortReleaseBarrier(
	authorization *ReleaseAuthorization,
	topology MembershipTopology,
) error {
	if authorization == nil {
		return nil
	}
	if authorization.TopologyGeneration > topology.Generation {
		return fmt.Errorf(
			"abort cleanup release topology generation %d exceeds current generation %d",
			authorization.TopologyGeneration,
			topology.Generation,
		)
	}
	if authorization.TargetReplicas < topology.ReplicaCount() {
		return fmt.Errorf(
			"abort cleanup release target %d is below current authoritative replicas %d",
			authorization.TargetReplicas,
			topology.ReplicaCount(),
		)
	}
	return nil
}

func (c *WorkflowCoordinator) reconcileAbortCleanupRelease(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	// Formerly serving cleanup replicas must drain before any current or stale authorization can remove them.
	if !abortCleanupTrafficReady(*result.Operation, result.Topology, result.Traffic) {
		result, err := scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			Replicas:           abortCleanupDrainReplicas(*result.Operation, result.Topology),
		})
		if err != nil {
			return result, true, fmt.Errorf("drain abort cleanup replicas: %w", err)
		}
		return result, true, nil
	}

	authorization := input.ReleaseAuthorization
	if authorization != nil &&
		(authorization.TopologyGeneration != result.Topology.Generation ||
			authorization.TargetReplicas != result.Topology.ReplicaCount()) {
		result, err := c.reconcileStaleAbortRelease(ctx, input.GroupID, result)
		return result, true, err
	}
	if authorization != nil {
		if err := validateAuthorizationForOperation(*authorization, *result.Operation); err != nil {
			return result, true, err
		}
		result, _, err := c.reconcileExactRelease(
			ctx,
			input.GroupID,
			result,
			abortCleanupReplicaIDs(*result.Operation),
		)
		return result, true, err
	}

	// Close older capacity work and remove surplus before a same-count repair can allocate a missing survivor.
	if abortCleanupComplete(*result.Operation, result.Capacity, result.Topology) {
		return result, false, nil
	}
	result.OperationChanged = clearCapacityTargetProof(result.Operation) || result.OperationChanged
	authorization, err := c.buildExactReleaseAuthorization(
		*result.Operation,
		result.Topology.ReplicaCount(),
		abortCleanupReplicaIDs(*result.Operation),
		result.Capacity,
		result.Topology.Generation,
	)
	if err != nil {
		return result, true, err
	}
	result.ReleaseAuthorization = authorization
	result.ReleaseAuthorizationChanged = true
	return result, true, nil
}

func (c *WorkflowCoordinator) reconcileDriftedAbortCleanup(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, error) {
	// Recompute cleanup ownership against authoritative membership without ever selecting a current logical member.
	cleanupReplicaSlots, err := discoverAbortCleanupReplicaSlots(
		*result.Operation,
		result.Capacity,
		result.Topology,
	)
	if err != nil {
		return result, err
	}
	if !sameReplicaSlotBindings(cleanupReplicaSlots, result.Operation.CleanupReplicaSlots) {
		result.Operation.CleanupReplicaSlots = cleanupReplicaSlots
		clearCapacityTargetProof(result.Operation)
		result.OperationChanged = true
		return result, nil
	}

	// Resolve an authorization from an older topology before deriving a replacement from the fenced current topology.
	if result.ReleaseAuthorization != nil &&
		(result.ReleaseAuthorization.TopologyGeneration != result.Topology.Generation ||
			result.ReleaseAuthorization.TargetReplicas != result.Topology.ReplicaCount()) {
		return c.reconcileStaleAbortRelease(ctx, groupID, result)
	}

	// Finish only an exact, current-generation authorization that cannot select authoritative membership.
	if result.ReleaseAuthorization != nil {
		if err := validateAuthorizationForOperation(*result.ReleaseAuthorization, *result.Operation); err != nil {
			return result, err
		}
		if err := validateAuthorizationAgainstTopology(
			*result.ReleaseAuthorization,
			*result.Operation,
			result.Topology,
		); err != nil {
			return result, err
		}
		result, applied, err := c.reconcileExactRelease(
			ctx,
			groupID,
			result,
			abortCleanupReplicaIDs(*result.Operation),
		)
		if err != nil || !applied {
			return result, err
		}
		return result, nil
	}

	// Authorize any remaining surplus by exact UID against the already-fenced authoritative topology.
	if !abortCleanupComplete(*result.Operation, result.Capacity, result.Topology) {
		result.OperationChanged = clearCapacityTargetProof(result.Operation) || result.OperationChanged
		authorization, err := c.buildExactReleaseAuthorization(
			*result.Operation,
			result.Topology.ReplicaCount(),
			abortCleanupReplicaIDs(*result.Operation),
			result.Capacity,
			result.Topology.Generation,
		)
		if err != nil {
			return result, err
		}
		result.ReleaseAuthorization = authorization
		result.ReleaseAuthorizationChanged = true
		return result, nil
	}

	// Old-operation cleanup is complete; remain fenced and expose an adoption-eligible recovery boundary.
	result.Operation = transitionOperation(result.Operation, OperationPhaseAborted, c.operations.now())
	result.OperationChanged = true
	return result, nil
}

func (c *WorkflowCoordinator) reconcileStaleAbortRelease(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, error) {
	authorization := result.ReleaseAuthorization
	if authorization.TopologyGeneration > result.Topology.Generation {
		return result, fmt.Errorf(
			"abort cleanup release topology generation %d exceeds current generation %d",
			authorization.TopologyGeneration,
			result.Topology.Generation,
		)
	}
	if authorization.TargetReplicas < result.Topology.ReplicaCount() {
		return result, fmt.Errorf(
			"abort cleanup release target %d is below current authoritative replicas %d",
			authorization.TargetReplicas,
			result.Topology.ReplicaCount(),
		)
	}

	observation, err := c.capacity.ObserveRelease(ctx, groupID, authorization.ID)
	if err != nil {
		return result, fmt.Errorf("observe stale abort cleanup release %q: %w", authorization.ID, err)
	}
	if err := validateCapacityReleaseObservation(observation, authorization.ID); err != nil {
		return result, err
	}
	switch observation.Phase {
	case CapacityReleasePhaseAbsent, CapacityReleasePhaseRefused, CapacityReleasePhaseFailed:
		result.OperationChanged = clearCapacityTargetProof(result.Operation) || result.OperationChanged
		result.ReleaseAuthorization = nil
		result.ReleaseAuthorizationChanged = true
		return result, nil
	case CapacityReleasePhaseApplying, CapacityReleasePhaseApplied:
		result, _, err := c.reconcileObservedExactRelease(
			ctx,
			groupID,
			result,
			abortCleanupReplicaIDs(*result.Operation),
			observation,
		)
		return result, err
	default:
		return result, fmt.Errorf("unsupported capacity release phase %q", observation.Phase)
	}
}

func (c *WorkflowCoordinator) reconcileAborted(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	if !servingVerificationTopologiesEqual(*result.Operation.CompensationTopology, result.Topology) {
		// A later topology cannot inherit the compensation proof. Keep it fenced and leave Aborted eligible for an
		// explicit observed-recovery plan instead of reopening cleanup already completed by the old operation.
		result, _, err := c.ensureUnverifiedTopologyFenced(result)
		return result, err
	}
	if err := validateAbortCleanupTopology(*result.Operation, result.Topology); err != nil {
		return result, err
	}
	cleanupReplicaSlots, err := discoverAbortCleanupReplicaSlots(
		*result.Operation,
		result.Capacity,
		result.Topology,
	)
	if err != nil {
		return result, err
	}
	if !sameReplicaSlotBindings(cleanupReplicaSlots, result.Operation.CleanupReplicaSlots) ||
		!abortCleanupComplete(*result.Operation, result.Capacity, result.Topology) ||
		!abortCleanupTrafficReady(*result.Operation, result.Topology, result.Traffic) {
		result.Operation.CleanupReplicaSlots = cleanupReplicaSlots
		clearCapacityTargetProof(result.Operation)
		result.Operation = transitionOperation(
			result.Operation,
			OperationPhaseAborting,
			c.operations.now(),
		)
		result.OperationChanged = true
		return result, nil
	}
	result, handled, err := c.reconcileActiveCapacityRegression(ctx, input, result)
	if err != nil || handled {
		return result, err
	}
	if result.Operation.TerminalAdmissionFailure != nil {
		result, fenced, err := ensureTerminalAdmissionFailureFenced(result)
		if err != nil || !fenced {
			return result, err
		}
		if input.Plan == nil || input.Plan.ID == result.Operation.PlanID {
			return result, nil
		}
		if input.Plan.Intent != OperationIntentRecover {
			return result, errors.New("terminal traffic-admission failure requires a distinct explicit Recover plan")
		}
		return c.replaceAbortedOperation(ctx, input, result)
	}

	// Aborted remains a live serving state and continuously repairs capacity or admission regressions.
	result, restored, err := c.restoreAuthoritativeServingState(ctx, input.GroupID, result)
	if err != nil || !restored {
		return result, err
	}

	// Keep an already-satisfied or still-acknowledged aborted plan as the durable audit record.
	if input.DesiredReplicas == result.Topology.ReplicaCount() &&
		(input.Plan == nil || operationCarriesPlan(*input.Operation, *input.Plan)) {
		return result, nil
	}
	return c.replaceAbortedOperation(ctx, input, result)
}

func (c *WorkflowCoordinator) replaceAbortedOperation(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	// The stale plan cannot be replayed against a different base; cardinal growth may be replanned automatically.
	plan := input.Plan
	if plan != nil && operationCarriesPlan(*input.Operation, *plan) {
		plan = nil
	}
	durableOperation := cloneOperation(result.Operation)
	if err := validateOperationCandidate(input.Operation, plan); err != nil {
		return result, err
	}
	if err := validateNewRestorationPlan(input.Operation, plan, result.Capacity); err != nil {
		return result, err
	}
	operationResult, err := c.operations.Step(ctx, OperationInput{
		GroupID:         input.GroupID,
		SpecGeneration:  input.SpecGeneration,
		DesiredReplicas: input.DesiredReplicas,
		Plan:            plan,
	})
	preTransitionResult, fenced, fenceErr := c.ensureObservedOperationTransitionFenced(
		result,
		durableOperation,
		operationResult,
	)
	if fenceErr != nil || !fenced {
		return preTransitionResult, errors.Join(err, fenceErr)
	}
	if err != nil {
		return result, err
	}
	result.Operation = operationResult.Operation
	result.Topology = operationResult.Topology
	result.OperationChanged = operationResult.OperationChanged
	return result, nil
}

func ensureTerminalAdmissionFailureFenced(
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	failure := result.Operation.TerminalAdmissionFailure
	if failure == nil {
		return result, true, nil
	}
	replicas := failure.Command.Request.Replicas
	if trafficStateStableFor(result.Traffic, TrafficActionWithdraw, replicas) &&
		containsAllReplicaIncarnations(result.Traffic.Drained, replicas) &&
		len(intersectReplicaIncarnations(result.Traffic.Admitted, replicas)) == 0 {
		return result, true, nil
	}

	result, err := scheduleTrafficCommand(
		result,
		TrafficActionWithdraw,
		failure.Command.Request,
	)
	if err != nil {
		return result, false, fmt.Errorf("fence terminal traffic-admission failure: %w", err)
	}
	return result, false, nil
}

func (c *WorkflowCoordinator) restoreAuthoritativeServingState(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	if result.Operation.TerminalAdmissionFailure != nil {
		return result, false, errors.New(
			"cannot restore traffic after a terminal admission failure without a distinct recovery operation",
		)
	}
	activeReplicas := topologyReplicaIncarnations(result.Topology)

	// Establish every required post-transition traffic fence before capacity repair can publish a replacement behind
	// a previously admitted logical identity. Verification itself waits for complete capacity below.
	if servingVerificationPending(*result.Operation, result.Topology) &&
		!verificationTrafficReady(*result.Operation, result.Topology, result.Traffic) {
		result, _, err := c.reconcileServingVerification(ctx, groupID, result)
		return result, false, err
	}

	// Restore complete physical capacity before making every current authoritative member routable again.
	if !replicaAllocationsAvailable(activeReplicas, result.Capacity) {
		result, fenced, err := c.ensureUnverifiedTopologyFenced(result)
		if err != nil || !fenced {
			return result, false, err
		}
		if !replicaIncarnationsPresent(activeReplicas, result.Capacity) {
			return result, false, nil
		}
		request := CapacityRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			TargetReplicas:     int32(len(activeReplicas)),
			RequiredReplicas:   requiredReplicaAllocations(activeReplicas),
		}
		if err := c.ensureCapacity(ctx, groupID, request); err != nil {
			return result, false, fmt.Errorf("restore aborted Engine Group capacity: %w", err)
		}
		return result, false, nil
	}

	result, verified, err := c.reconcileServingVerification(ctx, groupID, result)
	if err != nil || !verified {
		return result, false, err
	}

	// A traffic observation correlated to this operation proves compensation survived timeout or restart.
	if trafficCompensationComplete(result.Topology, result.Traffic) {
		return result, true, nil
	}
	if len(activeReplicas) == 0 {
		return result, false, errors.New(
			"empty compensation topology cannot complete while a traffic admission may still take effect",
		)
	}
	result, err = scheduleTrafficCommand(result, TrafficActionAdmit, TrafficRequest{
		OperationID:        result.Operation.ID,
		TopologyGeneration: result.Topology.Generation,
		Replicas:           activeReplicas,
	})
	if err != nil {
		return result, false, fmt.Errorf("restore traffic after aborted membership operation: %w", err)
	}
	return result, false, nil
}

func discoverAbortCleanupReplicaSlots(
	operation Operation,
	capacity CapacitySnapshot,
	topology MembershipTopology,
) ([]ReplicaSlotBinding, error) {
	activeReplicas := make(map[ReplicaID]struct{}, len(topology.Replicas))
	for _, replicaID := range topologyReplicaIDs(topology) {
		activeReplicas[replicaID] = struct{}{}
	}
	cleanupReplicaSlots := make([]ReplicaSlotBinding, 0, len(operation.CleanupReplicaSlots))
	add := func(replicaID ReplicaID, slotID CapacitySlotID) {
		if _, active := activeReplicas[replicaID]; active {
			return
		}
		cleanupReplicaSlots = append(cleanupReplicaSlots, ReplicaSlotBinding{
			ReplicaID: replicaID,
			SlotID:    slotID,
		})
	}

	// Preserve previously discovered slots even after their concrete allocations disappear.
	for _, binding := range operation.CleanupReplicaSlots {
		add(binding.ReplicaID, binding.SlotID)
	}

	// Every operation-owned identity excluded from authoritative membership needs a logical recreation fence. This
	// includes capacity that was planned or launched but disappeared before cleanup authorization was persisted.
	for _, membership := range operation.BaseTopology.Replicas {
		add(membership.Incarnation.ReplicaID, membership.Incarnation.SlotID)
	}
	for _, incarnation := range operation.JoiningReplicas {
		add(incarnation.ReplicaID, incarnation.SlotID)
	}
	for _, membership := range operation.TargetMembership {
		add(membership.Incarnation.ReplicaID, membership.Incarnation.SlotID)
	}
	for _, membership := range operation.RestoredMembership {
		add(membership.ReplicaID, membership.SlotID)
	}

	// Every other allocation outside authoritative membership is surplus capacity owned by the aborted operation.
	for _, allocation := range capacity.Allocations {
		add(allocation.Incarnation.ReplicaID, allocation.Incarnation.SlotID)
	}
	return normalizeReplicaSlotBindings(cleanupReplicaSlots)
}

func abortCleanupReplicaIDs(operation Operation) []ReplicaID {
	return replicaIDsForSlotBindings(operation.CleanupReplicaSlots)
}

func validateAbortCleanupTopology(operation Operation, topology MembershipTopology) error {
	if operation.CompensationTopology == nil {
		return errors.New("abort cleanup requires an exact compensation topology")
	}
	if !servingVerificationTopologiesEqual(*operation.CompensationTopology, topology) {
		return errors.New("authoritative topology changed during membership compensation")
	}
	if overlap := intersectReplicaIDs(abortCleanupReplicaIDs(operation), topologyReplicaIDs(topology)); len(overlap) != 0 {
		return fmt.Errorf("abort cleanup replicas became active in authoritative topology: %v", overlap)
	}
	return nil
}

func abortCleanupDrainReplicas(operation Operation, topology MembershipTopology) []ReplicaIncarnation {
	removedReplicaIDs := differenceReplicaIDs(
		topologyReplicaIDs(operation.BaseTopology),
		topologyReplicaIDs(topology),
	)
	return topologyReplicaIncarnationsForIDs(operation.BaseTopology, removedReplicaIDs)
}

func abortCleanupTrafficReady(
	operation Operation,
	topology MembershipTopology,
	traffic TrafficSnapshot,
) bool {
	drainReplicas := abortCleanupDrainReplicas(operation, topology)
	if len(drainReplicas) == 0 {
		return true
	}
	return trafficStateStableFor(traffic, TrafficActionWithdraw, drainReplicas) &&
		containsAllReplicaIncarnations(traffic.Drained, drainReplicas) &&
		len(intersectReplicaIncarnations(traffic.Admitted, drainReplicas)) == 0
}

func abortCleanupComplete(
	operation Operation,
	capacity CapacitySnapshot,
	topology MembershipTopology,
) bool {
	if !capacityTargetProofCurrent(operation, topology) {
		return false
	}

	activeReplicas := topologyReplicaIDs(topology)
	for _, allocation := range capacity.Allocations {
		if !slices.Contains(activeReplicas, allocation.Incarnation.ReplicaID) {
			return false
		}
	}

	allocations := replicaAllocationByID(capacity)
	fencedSlots := fencedReplicaSlots(capacity)
	for _, binding := range operation.CleanupReplicaSlots {
		if _, exists := allocations[binding.ReplicaID]; exists {
			return false
		}
		if slotID, fenced := fencedSlots[binding.ReplicaID]; !fenced || slotID != binding.SlotID {
			return false
		}
	}
	return true
}

func capacityTargetProofCurrent(operation Operation, topology MembershipTopology) bool {
	return operation.CapacityTargetApplied &&
		operation.CapacityTargetReplicas == topology.ReplicaCount() &&
		operation.CapacityTopologyGeneration == topology.Generation
}

func setCapacityTargetProof(operation *Operation, authorization ReleaseAuthorization) bool {
	changed := !operation.CapacityTargetApplied ||
		operation.CapacityTargetReplicas != authorization.TargetReplicas ||
		operation.CapacityTopologyGeneration != authorization.TopologyGeneration

	operation.CapacityTargetReplicas = authorization.TargetReplicas
	operation.CapacityTopologyGeneration = authorization.TopologyGeneration
	operation.CapacityTargetApplied = true
	return changed
}

func clearCapacityTargetProof(operation *Operation) bool {
	if !operation.CapacityTargetApplied &&
		operation.CapacityTargetReplicas == 0 &&
		operation.CapacityTopologyGeneration == 0 {
		return false
	}

	operation.CapacityTargetReplicas = 0
	operation.CapacityTopologyGeneration = 0
	operation.CapacityTargetApplied = false
	return true
}

func (c *WorkflowCoordinator) resumePreparation(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, error) {
	delta := result.Operation.TargetReplicas - result.Operation.BaseTopology.ReplicaCount()
	requiredIncarnations := frozenSubmissionIncarnations(*result.Operation)

	// An exact physical incarnation cannot be recreated under the frozen request after the backend proves no commit.
	if len(requiredIncarnations) > 0 && !replicaIncarnationsPresent(requiredIncarnations, result.Capacity) {
		return c.abortForTerminalPreconditionFailure(
			result,
			"JoiningIncarnationLost",
			"an exact replica incarnation disappeared before its membership request committed",
		), nil
	}
	if len(requiredIncarnations) > 0 && !replicaAllocationsAvailable(requiredIncarnations, result.Capacity) {
		if err := c.ensureCapacity(ctx, groupID, CapacityRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			TargetReplicas: max(
				result.Operation.TargetReplicas,
				result.Operation.BaseTopology.ReplicaCount(),
			),
			RequiredReplicas: requiredReplicaAllocations(requiredIncarnations),
		}); err != nil {
			return result, fmt.Errorf("restore frozen Engine Group capacity: %w", err)
		}
		return result, nil
	}

	// Expansion first restores its original exact request by recreating missing or unavailable capacity.
	if delta > 0 && !expansionCapacityReady(*result.Operation, result.Capacity, result.Traffic) {
		request := CapacityRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			TargetReplicas:     result.Operation.TargetReplicas,
			RequiredReplicas: normalizeRequiredReplicaAllocations(append(
				requiredReplicaAllocations(topologyReplicaIncarnations(result.Operation.BaseTopology)),
				requiredReplicaAllocations(result.Operation.JoiningReplicas)...,
			)),
		}
		if err := c.ensureCapacity(ctx, groupID, request); err != nil {
			return result, fmt.Errorf("restore Engine Group growth capacity: %w", err)
		}
		return result, nil
	}

	// Then restore the operation-specific traffic prerequisite without changing the frozen request.
	if !preCommitTrafficReady(*result.Operation, result.Traffic) {
		var err error
		result, err = scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			Replicas:           preCommitWithdrawalReplicas(*result.Operation),
		})
		if err != nil {
			return result, fmt.Errorf("resume Engine Group traffic preparation: %w", err)
		}
	}
	return result, nil
}

func (c *WorkflowCoordinator) reconcileActiveCapacityRegression(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	activeReplicas := topologyReplicaIncarnations(result.Topology)
	if len(activeReplicas) == 0 || replicaAllocationsAvailable(activeReplicas, result.Capacity) {
		return result, false, nil
	}

	// Withdraw the exact authoritative incarnation before attempting repair or accepting a replacement plan.
	result, fenced, err := c.ensureUnverifiedTopologyFenced(result)
	if err != nil || !fenced {
		return result, true, err
	}

	// Physical availability is weaker than serving progress. Invalidate any attempt that could have started before
	// this regression, then persist the new attempt identity before repairing capacity. Repairing prevents repeated
	// reconciles of the same outage from continually advancing the attempt. A regression after the durable transition
	// to Verifying advances again, so a check that predates or overlaps the later outage cannot authorize admission.
	result, invalidated, err := invalidateServingVerificationAfterCapacityRegression(result)
	if err != nil || invalidated {
		return result, true, err
	}

	// Finish current-operation release ownership before allowing a later recovery plan to replace its durable record.
	if result.Operation.Phase == OperationPhaseCommitted &&
		result.Operation.TargetReplicas < result.Operation.BaseTopology.ReplicaCount() &&
		!failedReductionReleaseComplete(
			*result.Operation,
			result.ReleaseAuthorization,
			result.Capacity,
			result.Topology,
		) {
		result, releaseErr := c.reconcileRelease(ctx, input, result)
		return result, true, releaseErr
	}

	// A distinct explicit recovery may remove or replace any unavailable member, even while its old physical
	// incarnation remains observable. Without one, preserve automatic same-incarnation repair.
	if input.Plan != nil && input.Plan.ID != result.Operation.PlanID {
		result, recoveryErr := c.beginRequiredRecovery(ctx, input, result)
		return result, true, recoveryErr
	}

	// A missing or different incarnation cannot inherit the old topology generation, serving proof, or admission.
	if !replicaIncarnationsPresent(activeReplicas, result.Capacity) {

		// Provision replacement capacity by stable logical identity, but never reinterpret it as the old incarnation.
		if err := c.ensureCapacity(ctx, input.GroupID, CapacityRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			TargetReplicas:     int32(len(activeReplicas)),
			RequiredReplicas:   requiredReplicaAllocations(activeReplicas),
		}); err != nil {
			return result, true, fmt.Errorf("provision replacement Engine Group capacity: %w", err)
		}
		return result, true, nil
	}

	// Only transient readiness of the same exact incarnation may be repaired under the current membership record.
	request := CapacityRequest{
		OperationID:        result.Operation.ID,
		TopologyGeneration: result.Topology.Generation,
		TargetReplicas:     int32(len(activeReplicas)),
		RequiredReplicas:   requiredReplicaAllocations(activeReplicas),
	}
	if err := c.ensureCapacity(ctx, input.GroupID, request); err != nil {
		return result, true, fmt.Errorf("restore exact Engine Group capacity: %w", err)
	}
	return result, true, nil
}

func invalidateServingVerificationAfterCapacityRegression(
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	if !operationCanRestoreServingAfterCapacityRecovery(*result.Operation) ||
		result.Operation.CapacityRecoveryPhase == CapacityRecoveryPhaseRepairing {
		return result, false, nil
	}
	if result.Operation.ServingVerificationAttempt == math.MaxInt32 {
		return result, false, errors.New("serving verification retry attempt exhausted during capacity recovery")
	}
	if result.Operation.ServingVerificationAttempt == 0 {
		result.Operation.ServingVerificationAttempt = 1
	} else {
		result.Operation.ServingVerificationAttempt++
	}
	result.Operation.ServingVerificationTarget = topologyPointer(result.Topology)
	result.Operation.ServingVerificationProof = nil
	result.Operation.CapacityRecoveryPhase = CapacityRecoveryPhaseRepairing
	result.Operation.PostCommitComplete = false
	result.OperationChanged = true
	return result, true, nil
}

func operationCanRestoreServingAfterCapacityRecovery(operation Operation) bool {
	if operation.Capability.VerificationRequirement != ServingVerificationRequired {
		return false
	}
	return operation.Phase == OperationPhaseCommitted ||
		operation.Phase == OperationPhaseUnknown ||
		operation.Phase == OperationPhaseAborting ||
		operation.Phase == OperationPhaseAborted
}

func (c *WorkflowCoordinator) beginRequiredRecovery(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	if input.Plan == nil || input.Plan.ID == result.Operation.PlanID {
		return result, nil
	}
	if err := validateOperationCandidate(input.Operation, input.Plan); err != nil {
		return result, err
	}
	if input.Plan.Intent != OperationIntentRecover {
		return result, errors.New("unavailable membership requires a distinct explicit Recover plan")
	}
	if err := validateRecoveryCoversUnavailableMembership(*input.Plan, result.Topology, result.Capacity); err != nil {
		return result, err
	}

	// Plan against fresh authoritative topology without discarding the fail-closed traffic state on rejection.
	durableOperation := cloneOperation(result.Operation)
	if err := validateNewRestorationPlan(input.Operation, input.Plan, result.Capacity); err != nil {
		return result, err
	}
	operationResult, err := c.operations.Step(ctx, OperationInput{
		GroupID:         input.GroupID,
		SpecGeneration:  input.SpecGeneration,
		DesiredReplicas: input.DesiredReplicas,
		Plan:            input.Plan,
	})
	preTransitionResult, fenced, fenceErr := c.ensureObservedOperationTransitionFenced(
		result,
		durableOperation,
		operationResult,
	)
	if fenceErr != nil || !fenced {
		return preTransitionResult, errors.Join(err, fenceErr)
	}
	if err != nil {
		return result, err
	}
	if operationResult.Operation == nil ||
		operationResult.Operation.Capability.VerificationRequirement != ServingVerificationRequired {
		return result, errors.New("recovery after incarnation loss must require serving verification")
	}
	result.Operation = operationResult.Operation
	result.Topology = operationResult.Topology
	result.OperationChanged = operationResult.OperationChanged
	return result, nil
}

func validateRecoveryCoversUnavailableMembership(
	plan OperationPlan,
	topology MembershipTopology,
	capacity CapacitySnapshot,
) error {
	unavailableReplicaIDs := unavailableActiveReplicaIDs(topology, capacity)
	if len(unavailableReplicaIDs) == 0 {
		return nil
	}
	if plan.TargetReplicas < topology.ReplicaCount() {
		if containsAllReplicaIDs(plan.NominatedReplicas, unavailableReplicaIDs) {
			return nil
		}
		return fmt.Errorf(
			"survivor recovery must nominate every unavailable replica: %v",
			unavailableReplicaIDs,
		)
	}
	if plan.TargetReplicas != topology.ReplicaCount() {
		return errors.New("unavailable membership requires fixed-cardinality replacement or survivor reduction")
	}

	targetByReplica := make(map[ReplicaID]ReplicaIncarnation, len(plan.TargetMembership))
	for _, membership := range plan.TargetMembership {
		targetByReplica[membership.Incarnation.ReplicaID] = membership.Incarnation
	}
	allocations := replicaAllocationByID(capacity)
	for _, replicaID := range unavailableReplicaIDs {
		target, targeted := targetByReplica[replicaID]
		allocation, allocated := allocations[replicaID]
		if !targeted || !allocated || !sameReplicaIncarnation(target, allocation.Incarnation) {
			return fmt.Errorf(
				"cardinal recovery target does not bind lost replica %q to its current replacement incarnation",
				replicaID,
			)
		}
	}
	return nil
}

func unavailableActiveReplicaIDs(topology MembershipTopology, capacity CapacitySnapshot) []ReplicaID {
	allocations := replicaAllocationByID(capacity)
	fencedReplicas := fencedReplicaIDs(capacity)
	mismatched := make([]ReplicaID, 0)
	for _, incarnation := range topologyReplicaIncarnations(topology) {
		allocation, exists := allocations[incarnation.ReplicaID]
		_, fenced := fencedReplicas[incarnation.ReplicaID]
		if !exists ||
			fenced ||
			allocation.Availability != ReplicaAvailabilityAvailable ||
			!sameReplicaIncarnation(allocation.Incarnation, incarnation) {
			mismatched = append(mismatched, incarnation.ReplicaID)
		}
	}
	return normalizeUniqueReplicaIDs(mismatched)
}

func (c *WorkflowCoordinator) reconcileCommitted(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	// Re-observe authoritative membership before any post-commit traffic or capacity mutation.
	topology, err := c.membership.ObserveTopology(ctx, input.GroupID)
	if err != nil {
		return result, fmt.Errorf("observe committed Engine Group topology: %w", err)
	}
	if err := validateTopology(topology); err != nil {
		return result, fmt.Errorf("validate committed Engine Group topology: %w", err)
	}
	result.Topology = cloneTopology(topology)

	// A changed generation or identity set invalidates all operation-scoped post-commit authority.
	if input.Operation.CommittedTopology == nil ||
		!servingVerificationTopologiesEqual(topology, *input.Operation.CommittedTopology) {
		// Fence the changed topology under the still-durable committed record before persisting Unknown.
		result, fenced, err := c.ensureUnverifiedTopologyFenced(result)
		if err != nil || !fenced {
			return result, err
		}
		result.Operation = transitionOperation(result.Operation, OperationPhaseUnknown, c.operations.now())
		result.OperationChanged = true
		return result, nil
	}
	if err := validateTrafficAgainstMembership(result.Traffic, result.Topology, input.Operation); err != nil {
		return result, err
	}
	result, handled, err := c.reconcileActiveCapacityRegression(ctx, input, result)
	if err != nil || handled {
		return result, err
	}
	if input.Operation.PostCommitComplete {
		return c.finishCommitted(ctx, input, result)
	}

	delta := input.Operation.TargetReplicas - input.Operation.BaseTopology.ReplicaCount()

	// Fence an adopted or otherwise unverified topology before capacity repair can publish replacement processes
	// behind logical identities that were admitted under an earlier incarnation.
	if servingVerificationPending(*result.Operation, result.Topology) &&
		!verificationTrafficReady(*result.Operation, result.Topology, result.Traffic) {
		result, _, err := c.reconcileServingVerification(ctx, input.GroupID, result)
		return result, err
	}

	result, verified, err := c.reconcileServingVerification(ctx, input.GroupID, result)
	if err != nil || !verified {
		return result, err
	}

	// Admit only exact members that the operation actually withdrew or replaced. Adopted and group-quiescing
	// transitions require current-operation ownership of the complete topology before any physical release.
	result, admitted, err := c.ensureCommittedTopologyAdmitted(result)
	if err != nil || !admitted {
		return result, err
	}

	if delta > 0 {
		return c.finishCommitted(ctx, input, result)
	}

	// Retired replicas remain allocated until drain evidence and exact release authorization are durable.
	if delta < 0 {
		if !retirementTrafficReady(*input.Operation, result.Topology, result.Traffic) {
			result, err = scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
				OperationID:        input.Operation.ID,
				TopologyGeneration: result.Topology.Generation,
				Replicas: topologyReplicaIncarnationsForIDs(
					input.Operation.BaseTopology,
					input.Operation.NominatedReplicas,
				),
			})
			if err != nil {
				return result, fmt.Errorf("restore committed retirement drain state: %w", err)
			}
			return result, nil
		}
		return c.reconcileRelease(ctx, input, result)
	}

	return c.finishCommitted(ctx, input, result)
}

func (c *WorkflowCoordinator) reconcileServingVerification(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	if result.Operation.Capability.VerificationRequirement == ServingVerificationNotRequired ||
		len(result.Topology.Replicas) == 0 {
		return result, true, nil
	}

	result, ready, err := prepareServingVerification(result)
	if err != nil || !ready {
		return result, false, err
	}

	request, err := servingVerificationRequestFor(*result.Operation, result.Topology)
	if err != nil {
		return result, false, err
	}
	if result.Operation.ServingVerificationProof != nil &&
		servingVerificationPassedForRequest(*result.Operation.ServingVerificationProof, request) {
		if result.Operation.CapacityRecoveryPhase == CapacityRecoveryPhaseVerifying {
			result.Operation.CapacityRecoveryPhase = CapacityRecoveryPhaseNone
			result.OperationChanged = true
			return result, false, nil
		}
		return result, true, nil
	}

	observed, err := c.verifier.ObserveVerification(
		ctx,
		groupID,
		request.OperationID,
		request.Attempt,
		request.VerificationAttempt,
	)
	if err != nil {
		return result, false, fmt.Errorf("observe Engine Group serving verification: %w", err)
	}
	if err := validateServingVerificationProof(observed); err != nil {
		return result, false, fmt.Errorf("validate Engine Group serving verification: %w", err)
	}
	result.ServingVerification = cloneServingVerificationProof(&observed)

	return c.reconcileObservedServingVerification(ctx, groupID, result, request, observed)
}

func prepareServingVerification(result ReconcileResult) (ReconcileResult, bool, error) {
	operation := result.Operation
	if operation.CapacityRecoveryPhase == CapacityRecoveryPhaseRepairing {
		if !replicaAllocationsAvailable(topologyReplicaIncarnations(result.Topology), result.Capacity) {
			return result, false, nil
		}
		// Persist that repair completed before the verifier is allowed to observe or start this attempt. A later
		// availability regression from Verifying will mint another attempt rather than reuse this one.
		result.Operation.CapacityRecoveryPhase = CapacityRecoveryPhaseVerifying
		result.OperationChanged = true
		return result, false, nil
	}

	// Verification cannot make premature routing safe after the fact. Until an exact proof is durable, preserve the
	// operation's pre-commit drain and keep every joining identity non-routable. Regression fails closed: replaying an
	// earlier traffic mutation against a different topology would not be an idempotent retry.
	if servingVerificationPending(*operation, result.Topology) &&
		!verificationTrafficReady(*operation, result.Topology, result.Traffic) {
		result, err := restoreServingVerificationTrafficFence(result)
		return result, false, err
	}
	if operation.ServingVerificationAttempt == 0 {
		result.Operation.ServingVerificationAttempt = 1
		result.Operation.ServingVerificationTarget = topologyPointer(result.Topology)
		result.OperationChanged = true
		return result, false, nil
	}
	return result, true, nil
}

func restoreServingVerificationTrafficFence(result ReconcileResult) (ReconcileResult, error) {
	operation := result.Operation
	request := TrafficRequest{
		OperationID:        operation.ID,
		TopologyGeneration: result.Topology.Generation,
	}

	// An adopted transition has no controller-issued pre-commit proof, while a QuiesceGroup regression invalidates
	// its earlier proof. Fence both sides of either transition before verification, admission, or physical release.
	if operation.Adopted ||
		operation.Capability.TrafficRequirement == ReconfigurationTrafficQuiesceGroup {
		request.Replicas = unverifiedTopologyDrainReplicas(*operation, result.Topology)
		result, err := scheduleTrafficCommand(result, TrafficActionWithdraw, request)
		if err != nil {
			return result, fmt.Errorf("fence adopted Engine Group topology: %w", err)
		}
		return result, nil
	}

	if operation.Capability.TrafficRequirement != ReconfigurationTrafficKeepServing {
		return result, errors.New("unsupported serving-verification traffic state")
	}

	// KeepServing preserves unaffected traffic but still restores the operation-specific fence: joining replicas
	// for growth, or the exact retiring replicas for reduction. The later Admit is explicit compensation for joiners.
	request.Replicas = verificationJoiningReplicas(*operation, result.Topology)
	if len(request.Replicas) == 0 {
		request.Replicas = preCommitWithdrawalReplicas(*operation)
	}
	if len(request.Replicas) == 0 {
		return result, errors.New("KeepServing verification has no traffic fence to restore")
	}
	result, err := scheduleTrafficCommand(result, TrafficActionWithdraw, request)
	if err != nil {
		return result, fmt.Errorf("restore serving-verification traffic fence: %w", err)
	}
	return result, nil
}

func servingVerificationRequestFor(
	operation Operation,
	topology MembershipTopology,
) (ServingVerificationRequest, error) {
	if operation.ServingVerificationTarget == nil {
		return ServingVerificationRequest{}, errors.New("serving verification attempt has no durable topology target")
	}
	if !servingVerificationTopologiesEqual(*operation.ServingVerificationTarget, topology) {
		return ServingVerificationRequest{}, errors.New("authoritative topology changed during serving verification")
	}

	return ServingVerificationRequest{
		OperationID:         operation.ID,
		Attempt:             operation.Attempt,
		VerificationAttempt: operation.ServingVerificationAttempt,
		Topology:            cloneTopology(*operation.ServingVerificationTarget),
	}, nil
}

func (c *WorkflowCoordinator) reconcileObservedServingVerification(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
	request ServingVerificationRequest,
	observed ServingVerificationProof,
) (ReconcileResult, bool, error) {
	switch observed.Phase {
	case ServingVerificationPhaseAbsent:
		if observed.OperationID != request.OperationID ||
			observed.Attempt != request.Attempt ||
			observed.VerificationAttempt != request.VerificationAttempt {
			return result, false, errors.New("absent serving verification does not match the requested attempt")
		}
		if err := c.verifier.EnsureVerification(ctx, groupID, request); err != nil {
			return result, false, fmt.Errorf("ensure Engine Group serving verification: %w", err)
		}
		return result, false, nil
	case ServingVerificationPhaseRunning:
		if !servingVerificationProofMatchesRequest(observed, request) {
			return result, false, errors.New("running serving verification does not match committed topology")
		}
		return result, false, nil
	case ServingVerificationPhasePassed:
		if !servingVerificationPassedForRequest(observed, request) {
			return result, false, errors.New("passed serving verification does not match committed topology")
		}
		result.Operation.ServingVerificationProof = cloneServingVerificationProof(&observed)
		result.OperationChanged = true
		return result, false, nil
	case ServingVerificationPhaseFailed:
		if !servingVerificationProofMatchesRequest(observed, request) {
			return result, false, errors.New("failed serving verification does not match committed topology")
		}
		return c.reconcileFailedServingVerification(result, observed.Failure)
	case ServingVerificationPhaseUnknown:
		if !servingVerificationProofMatchesRequest(observed, request) {
			return result, false, errors.New("unknown serving verification does not match committed topology")
		}
		return c.reconcileFailedServingVerification(result, &OperationFailure{
			Classification: FailureClassificationTerminal,
			Reason:         "ServingVerificationUnknown",
			Message:        "the verifier cannot recover the exact serving-verification outcome",
		})
	default:
		return result, false, fmt.Errorf("unsupported serving verification phase %q", observed.Phase)
	}
}

func (c *WorkflowCoordinator) reconcileFailedServingVerification(
	result ReconcileResult,
	failure *OperationFailure,
) (ReconcileResult, bool, error) {
	// A failed progress check fences the complete topology before either retrying or exposing a terminal record.
	result, fenced, err := c.ensureServingFailureTopologyFenced(result)
	if err != nil || !fenced {
		return result, false, err
	}
	if failure.Classification != FailureClassificationRetryable {
		return c.recordTerminalServingVerificationFailure(result, failure)
	}

	// Persist a distinct retry identity before starting another external verification attempt.
	if result.Operation.ServingVerificationAttempt == math.MaxInt32 {
		return result, false, errors.New("serving verification retry attempt exhausted")
	}
	result.Operation.ServingVerificationAttempt++
	result.Operation.ServingVerificationProof = nil
	result.OperationChanged = true
	return result, false, nil
}

func (c *WorkflowCoordinator) recordTerminalServingVerificationFailure(
	result ReconcileResult,
	failure *OperationFailure,
) (ReconcileResult, bool, error) {
	// A terminal verification result ends same-topology capacity recovery. Keeping Repairing or Verifying on a
	// terminal record would imply that reconciliation may still admit this attempt and would make Failed state invalid.
	if result.Operation.CapacityRecoveryPhase != CapacityRecoveryPhaseNone {
		result.Operation.CapacityRecoveryPhase = CapacityRecoveryPhaseNone
		result.OperationChanged = true
	}

	// Failed compensation cannot discard its exact restoration and cleanup state. Keep it durably fail-closed in
	// Aborting so higher-level recovery can recreate the group without admitting an unverified topology.
	if result.Operation.Phase == OperationPhaseAborting || result.Operation.Phase == OperationPhaseAborted {
		if result.Operation.Phase == OperationPhaseAborting &&
			result.Operation.Failure.Classification == failure.Classification &&
			result.Operation.Failure.Reason == failure.Reason &&
			result.Operation.Failure.Message == failure.Message {
			return result, false, nil
		}
		result.Operation.Failure = cloneFailure(failure)
		result.Operation = transitionOperation(
			result.Operation,
			OperationPhaseAborting,
			c.operations.now(),
		)
		result.OperationChanged = true
		return result, false, nil
	}

	result.Operation.Failure = cloneFailure(failure)
	result.Operation = transitionOperation(
		result.Operation,
		OperationPhaseFailed,
		c.operations.now(),
	)
	result.OperationChanged = true
	return result, false, nil
}

func (c *WorkflowCoordinator) reconcileRelease(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	// Victim fences and a current group-global target barrier jointly prove release completion.
	if input.ReleaseAuthorization == nil &&
		capacityTargetProofCurrent(*input.Operation, result.Topology) &&
		releaseFencesComplete(*input.Operation, result.Capacity) {
		if input.Operation.Phase == OperationPhaseFailed {
			return result, nil
		}
		// A later survivor topology needs its own adopted recovery before current identities can be repaired safely.
		if input.Operation.Phase == OperationPhaseUnknown &&
			result.Topology.Generation > committedTopologyGeneration(*input.Operation) {
			result.Operation.PostCommitComplete = true
			result.OperationChanged = true
			return result, nil
		}
		return c.finishCommitted(ctx, input, result)
	}

	// First persist the absolute target barrier and exact UID-bound authorization without external mutation.
	if input.ReleaseAuthorization == nil {
		result.OperationChanged = clearCapacityTargetProof(result.Operation) || result.OperationChanged
		authorization, err := c.buildReleaseAuthorization(
			*input.Operation,
			result.Capacity,
			result.Topology,
		)
		if err != nil {
			return result, err
		}
		result.ReleaseAuthorization = authorization
		result.ReleaseAuthorizationChanged = true
		return result, nil
	}

	// Refuse an authorization that is not bound to the current committed membership operation.
	if err := validateAuthorizationForOperation(*input.ReleaseAuthorization, *input.Operation); err != nil {
		return result, err
	}
	if err := validateAuthorizationAgainstTopology(
		*input.ReleaseAuthorization,
		*input.Operation,
		result.Topology,
	); err != nil {
		return result, err
	}
	result, _, err := c.reconcileExactRelease(
		ctx,
		input.GroupID,
		result,
		input.Operation.NominatedReplicas,
	)
	return result, err
}

func (c *WorkflowCoordinator) reconcileExactRelease(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
	requiredReplicas []ReplicaID,
) (ReconcileResult, bool, error) {
	// Recover an asynchronous release outcome before replaying the exact durable authorization.
	releaseID := result.ReleaseAuthorization.ID
	observation, err := c.capacity.ObserveRelease(ctx, groupID, releaseID)
	if err != nil {
		return result, false, fmt.Errorf("observe capacity release %q: %w", releaseID, err)
	}
	if err := validateCapacityReleaseObservation(observation, releaseID); err != nil {
		return result, false, err
	}
	return c.reconcileObservedExactRelease(ctx, groupID, result, requiredReplicas, observation)
}

func (c *WorkflowCoordinator) reconcileObservedExactRelease(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
	requiredReplicas []ReplicaID,
	observation CapacityReleaseObservation,
) (ReconcileResult, bool, error) {
	releaseID := result.ReleaseAuthorization.ID
	switch observation.Phase {
	case CapacityReleasePhaseAbsent:
		if err := validateAuthorizationCompleteness(*result.ReleaseAuthorization, requiredReplicas); err != nil {
			result.ReleaseAuthorization = nil
			result.ReleaseAuthorizationChanged = true
			return result, false, fmt.Errorf(
				"discard unapplied capacity release %q: %w",
				releaseID,
				err,
			)
		}
		if err := validateAuthorizationFresh(*result.ReleaseAuthorization, result.Capacity); err != nil {
			result.ReleaseAuthorization = nil
			result.ReleaseAuthorizationChanged = true
			return result, false, fmt.Errorf(
				"discard unapplied capacity release %q: %w",
				releaseID,
				err,
			)
		}
		if err := c.capacity.Release(ctx, groupID, *cloneReleaseAuthorization(result.ReleaseAuthorization)); err != nil {
			return result, false, fmt.Errorf("release authorized Engine Group capacity: %w", err)
		}
		return result, false, nil
	case CapacityReleasePhaseApplying:
		return result, false, nil
	case CapacityReleasePhaseApplied:
		if authorizedCapacityPresent(*result.ReleaseAuthorization, result.Capacity) {
			return result, false, nil
		}
		if !authorizationFencesObserved(*result.ReleaseAuthorization, result.Capacity) {
			return result, false, nil
		}
		result.OperationChanged = setCapacityTargetProof(
			result.Operation,
			*result.ReleaseAuthorization,
		) || result.OperationChanged
		result.ReleaseAuthorization = nil
		result.ReleaseAuthorizationChanged = true
		return result, true, nil
	case CapacityReleasePhaseRefused:
		if observation.Failure.Classification == FailureClassificationRetryable {
			result.ReleaseAuthorization = nil
			result.ReleaseAuthorizationChanged = true
		}
		return result, false, fmt.Errorf(
			"capacity release %q was refused: %s: %s",
			observation.ReleaseID,
			observation.Failure.Reason,
			observation.Failure.Message,
		)
	case CapacityReleasePhaseFailed:
		if observation.Failure.Classification == FailureClassificationRetryable {
			// Rebuild a fresh authorization from the exact capacity and fences left by the stopped release.
			result.OperationChanged = clearCapacityTargetProof(result.Operation) || result.OperationChanged
			result.ReleaseAuthorization = nil
			result.ReleaseAuthorizationChanged = true
		}
		return result, false, fmt.Errorf(
			"capacity release %q failed: %s: %s",
			observation.ReleaseID,
			observation.Failure.Reason,
			observation.Failure.Message,
		)
	default:
		return result, false, fmt.Errorf("unsupported capacity release phase %q", observation.Phase)
	}
}

func (c *WorkflowCoordinator) finishCommitted(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	// Reconcile serving traffic from authoritative topology without admitting unavailable physical capacity.
	activeReplicas := topologyReplicaIncarnations(result.Topology)
	if !replicaAllocationsAvailable(activeReplicas, result.Capacity) {
		return result, errors.New("committed Engine Group capacity regressed before completion")
	}
	if result.Operation.Capability.VerificationRequirement == ServingVerificationRequired &&
		len(activeReplicas) > 0 {
		request := ServingVerificationRequest{
			OperationID:         result.Operation.ID,
			Attempt:             result.Operation.Attempt,
			VerificationAttempt: result.Operation.ServingVerificationAttempt,
			Topology:            result.Topology,
		}
		if result.Operation.ServingVerificationProof == nil ||
			!servingVerificationPassedForRequest(*result.Operation.ServingVerificationProof, request) {
			return result, errors.New("cannot admit Engine Group topology without matching serving verification")
		}
	}
	result, admitted, err := c.ensureCommittedTopologyAdmitted(result)
	if err != nil || !admitted {
		return result, err
	}

	// Persist completion of all operation-scoped postwork before planning or replacing the record.
	if !result.Operation.PostCommitComplete {
		result.Operation.PostCommitComplete = true
		result.OperationChanged = true
		return result, nil
	}

	// Preserve the last record only when cardinal intent is satisfied and no distinct explicit plan is pending.
	if input.DesiredReplicas == result.Topology.ReplicaCount() &&
		(input.Plan == nil || operationCarriesPlan(*input.Operation, *input.Plan)) {
		return result, nil
	}

	// A completed identity-aware plan is consumed; a queued cardinal restoration is planned from fresh topology.
	plan := input.Plan
	if plan != nil && operationCarriesPlan(*input.Operation, *plan) {
		plan = nil
	}

	// Replace the completed record with a newly durable Pending operation for the latest desired target.
	durableOperation := cloneOperation(result.Operation)
	if err := validateOperationCandidate(input.Operation, plan); err != nil {
		return result, err
	}
	if err := validateNewRestorationPlan(input.Operation, plan, result.Capacity); err != nil {
		return result, err
	}
	operationResult, err := c.operations.Step(ctx, OperationInput{
		GroupID:         input.GroupID,
		SpecGeneration:  input.SpecGeneration,
		DesiredReplicas: input.DesiredReplicas,
		Plan:            plan,
	})
	preTransitionResult, fenced, fenceErr := c.ensureObservedOperationTransitionFenced(
		result,
		durableOperation,
		operationResult,
	)
	if fenceErr != nil || !fenced {
		return preTransitionResult, errors.Join(err, fenceErr)
	}
	if err != nil {
		return result, err
	}
	result.Operation = operationResult.Operation
	result.Topology = operationResult.Topology
	result.OperationChanged = operationResult.OperationChanged
	return result, nil
}

func (c *WorkflowCoordinator) ensureCommittedTopologyAdmitted(
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	if result.Operation.TerminalAdmissionFailure != nil {
		return result, false, errors.New(
			"cannot admit committed membership after a terminal admission failure without a distinct recovery operation",
		)
	}
	activeReplicas := topologyReplicaIncarnations(result.Topology)
	if len(activeReplicas) == 0 {
		return result, true, nil
	}

	// Only transitions that fenced the complete topology need current-operation ownership of every admission.
	fullAdmissionRequired := result.Operation.Adopted ||
		result.Operation.Capability.TrafficRequirement == ReconfigurationTrafficQuiesceGroup
	admissionComplete := trafficStateStableFor(result.Traffic, TrafficActionAdmit, activeReplicas) &&
		containsAllReplicaIncarnations(result.Traffic.Admitted, activeReplicas)
	if fullAdmissionRequired {
		admissionComplete = fullTopologyAdmissionComplete(activeReplicas, result.Traffic)
	}
	if admissionComplete {
		return result, true, nil
	}

	// Preserve unaffected KeepServing members; otherwise explicitly compensate the complete topology fence.
	replicas := missingReplicaIncarnations(activeReplicas, result.Traffic.Admitted)
	if fullAdmissionRequired && len(replicas) == 0 {
		replicas = activeReplicas
	}
	if len(replicas) == 0 {
		return result, false, errors.New("committed Engine Group admission is incomplete without missing replicas")
	}
	result, err := scheduleTrafficCommand(result, TrafficActionAdmit, TrafficRequest{
		OperationID:        result.Operation.ID,
		TopologyGeneration: result.Topology.Generation,
		Replicas:           replicas,
	})
	if err != nil {
		return result, false, fmt.Errorf("admit committed Engine Group topology: %w", err)
	}
	return result, false, nil
}

func resolveJoiningReplicas(
	operation Operation,
	capacity CapacitySnapshot,
	traffic TrafficSnapshot,
) ([]ReplicaIncarnation, bool, error) {
	// Index the immutable base before classifying capacity as retained or joining.
	baseReplicaIDs := topologyReplicaIDs(operation.BaseTopology)
	baseReplicas := make(map[ReplicaID]struct{}, len(baseReplicaIDs))
	for _, replicaID := range baseReplicaIDs {
		baseReplicas[replicaID] = struct{}{}
	}
	allocations := replicaAllocationByID(capacity)
	fencedReplicas := fencedReplicaIDs(capacity)

	// Every retained replica must still have a complete available allocation before membership can grow.
	for _, baseIncarnation := range topologyReplicaIncarnations(operation.BaseTopology) {
		allocation, exists := allocations[baseIncarnation.ReplicaID]
		_, fenced := fencedReplicas[baseIncarnation.ReplicaID]
		if !exists || fenced ||
			allocation.Availability != ReplicaAvailabilityAvailable ||
			!sameReplicaIncarnation(allocation.Incarnation, baseIncarnation) {
			return nil, false, nil
		}
	}

	// Premature traffic admission is an invariant violation, not a signal to choose different capacity.
	admittedReplicaIDs := replicaIncarnationIDs(traffic.Admitted)
	candidates := make([]ReplicaIncarnation, 0)
	for _, allocation := range capacity.Allocations {
		replicaID := allocation.Incarnation.ReplicaID
		if _, active := baseReplicas[replicaID]; active {
			continue
		}
		if _, fenced := fencedReplicas[replicaID]; fenced {
			continue
		}
		if slices.Contains(admittedReplicaIDs, replicaID) {
			return nil, false, fmt.Errorf("uncommitted replica %q is already admitted to traffic", replicaID)
		}
		if allocation.Availability == ReplicaAvailabilityAvailable {
			candidates = append(candidates, cloneReplicaIncarnation(allocation.Incarnation))
		}
	}
	candidates = normalizeReplicaIncarnations(candidates)

	needed := int(operation.TargetReplicas) - len(baseReplicaIDs)
	if needed < 0 {
		return nil, false, errors.New("joining replicas requested for a reducing operation")
	}
	if len(capacity.Allocations) < int(operation.TargetReplicas) || len(candidates) < needed {
		return nil, false, nil
	}
	return cloneReplicaIncarnations(candidates[:needed]), true, nil
}

func resolveRestoredReplicas(
	operation Operation,
	capacity CapacitySnapshot,
	traffic TrafficSnapshot,
) ([]ReplicaIncarnation, bool, error) {
	allocations := replicaAllocationByID(capacity)
	fencedReplicas := fencedReplicaIDs(capacity)
	admittedReplicaIDs := replicaIncarnationIDs(traffic.Admitted)
	restored := make([]ReplicaIncarnation, 0, len(operation.RestoredMembership))
	for _, membership := range operation.RestoredMembership {
		replicaID := membership.ReplicaID
		if slices.Contains(admittedReplicaIDs, replicaID) {
			return nil, false, fmt.Errorf("uncommitted restored replica %q is already admitted to traffic", replicaID)
		}
		allocation, exists := allocations[replicaID]
		_, fenced := fencedReplicas[replicaID]
		if exists && allocation.Incarnation.SlotID != membership.SlotID {
			return nil, false, fmt.Errorf(
				"restored replica %q was allocated in slot %q instead of stable slot %q",
				replicaID,
				allocation.Incarnation.SlotID,
				membership.SlotID,
			)
		}
		if !exists || fenced || allocation.Availability != ReplicaAvailabilityAvailable {
			return nil, false, nil
		}
		restored = append(restored, cloneReplicaIncarnation(allocation.Incarnation))
	}
	if !replicaAllocationsAvailable(topologyReplicaIncarnations(operation.BaseTopology), capacity) {
		return nil, false, nil
	}
	return normalizeReplicaIncarnations(restored), true, nil
}

func capacityReplicaIDsOutside(
	capacity CapacitySnapshot,
	required []RequiredReplicaAllocation,
) []ReplicaID {
	requiredIDs := make(map[ReplicaID]struct{}, len(required))
	for _, replica := range required {
		requiredIDs[replica.ReplicaID] = struct{}{}
	}
	surplus := make([]ReplicaID, 0)
	for _, allocation := range capacity.Allocations {
		if _, required := requiredIDs[allocation.Incarnation.ReplicaID]; !required {
			surplus = append(surplus, allocation.Incarnation.ReplicaID)
		}
	}
	return normalizeUniqueReplicaIDs(surplus)
}

func submissionPrerequisitesReady(
	operation Operation,
	capacity CapacitySnapshot,
	traffic TrafficSnapshot,
) bool {
	return frozenSubmissionCapacityReady(operation, capacity) &&
		expansionCapacityReady(operation, capacity, traffic) &&
		preCommitTrafficReady(operation, traffic)
}

func frozenSubmissionCapacityReady(operation Operation, capacity CapacitySnapshot) bool {
	requiredIncarnations := frozenSubmissionIncarnations(operation)
	return len(requiredIncarnations) == 0 || replicaAllocationsAvailable(requiredIncarnations, capacity)
}

func expansionCapacityReady(
	operation Operation,
	capacity CapacitySnapshot,
	traffic TrafficSnapshot,
) bool {
	if operation.TargetReplicas <= operation.BaseTopology.ReplicaCount() {
		return true
	}

	allocations := replicaAllocationByID(capacity)
	fencedReplicas := fencedReplicaIDs(capacity)
	requiredIncarnations := append(
		topologyReplicaIncarnations(operation.BaseTopology),
		operation.JoiningReplicas...,
	)
	joiningReplicaIDs := replicaIncarnationIDs(operation.JoiningReplicas)
	admittedReplicaIDs := replicaIncarnationIDs(traffic.Admitted)
	for _, incarnation := range requiredIncarnations {
		allocation, exists := allocations[incarnation.ReplicaID]
		_, fenced := fencedReplicas[incarnation.ReplicaID]
		if !exists || fenced ||
			allocation.Availability != ReplicaAvailabilityAvailable ||
			!sameReplicaIncarnation(allocation.Incarnation, incarnation) {
			return false
		}
		if slices.Contains(joiningReplicaIDs, incarnation.ReplicaID) &&
			slices.Contains(admittedReplicaIDs, incarnation.ReplicaID) {
			return false
		}
	}
	return true
}

func preCommitWithdrawalReplicas(operation Operation) []ReplicaIncarnation {
	if operation.Capability.TrafficRequirement == ReconfigurationTrafficQuiesceGroup {
		return topologyReplicaIncarnations(operation.BaseTopology)
	}
	if operation.TargetReplicas == operation.BaseTopology.ReplicaCount() {
		return changedBaseReplicaIncarnations(operation)
	}
	if operation.TargetReplicas < operation.BaseTopology.ReplicaCount() {
		return topologyReplicaIncarnationsForIDs(operation.BaseTopology, operation.NominatedReplicas)
	}
	return nil
}

func preCommitTrafficReady(operation Operation, traffic TrafficSnapshot) bool {
	requiredDrains := preCommitWithdrawalReplicas(operation)
	if len(requiredDrains) == 0 {
		return true
	}
	return trafficStateStableFor(traffic, TrafficActionWithdraw, requiredDrains) &&
		containsAllReplicaIncarnations(traffic.Drained, requiredDrains) &&
		len(intersectReplicaIncarnations(traffic.Admitted, requiredDrains)) == 0
}

func servingVerificationPending(operation Operation, topology MembershipTopology) bool {
	if operation.Capability.VerificationRequirement != ServingVerificationRequired || topology.ReplicaCount() == 0 {
		return false
	}
	if operation.ServingVerificationProof == nil || operation.ServingVerificationAttempt <= 0 {
		return true
	}
	return !servingVerificationPassedForRequest(
		*operation.ServingVerificationProof,
		ServingVerificationRequest{
			OperationID:         operation.ID,
			Attempt:             operation.Attempt,
			VerificationAttempt: operation.ServingVerificationAttempt,
			Topology:            topology,
		},
	)
}

func verificationTrafficReady(
	operation Operation,
	topology MembershipTopology,
	traffic TrafficSnapshot,
) bool {
	// A whole-topology fence established after an earlier failed check is stronger than every operation-specific
	// pre-commit drain and remains valid across verification retries for the same exact topology.
	if servingFailureTrafficReady(operation, topology, traffic) ||
		unverifiedTopologyTrafficReady(operation, topology, traffic) {
		return true
	}
	if operation.Adopted {
		// An observed transition has no controller-issued pre-commit proof. Fence its complete current topology and
		// every missing identity before verification; QuiesceGroup transitions are never eligible for adoption.
		requiredDrains := unverifiedTopologyDrainReplicas(operation, topology)
		if !trafficStateStableFor(traffic, TrafficActionWithdraw, requiredDrains) ||
			!containsAllReplicaIncarnations(traffic.Drained, requiredDrains) ||
			len(intersectReplicaIncarnations(traffic.Admitted, requiredDrains)) != 0 {
			return false
		}
	} else if !preCommitTrafficReady(operation, traffic) {
		return false
	}
	joiningReplicas := verificationJoiningReplicas(operation, topology)
	if len(joiningReplicas) > 0 {
		if !trafficStateStableFor(traffic, TrafficActionWithdraw, joiningReplicas) {
			return false
		}
		// KeepServing growth establishes a post-commit fence because it had no pre-commit traffic mutation. A
		// QuiesceGroup transition relies on the adapter's invariant that only Admit can make new members routable.
		if operation.Capability.TrafficRequirement == ReconfigurationTrafficKeepServing {
			return trafficStateStableFor(traffic, TrafficActionWithdraw, joiningReplicas) &&
				containsAllReplicaIncarnations(traffic.Drained, joiningReplicas) &&
				len(intersectReplicaIncarnations(traffic.Admitted, joiningReplicas)) == 0
		}
		if len(intersectReplicaIncarnations(traffic.Admitted, joiningReplicas)) != 0 {
			return false
		}
	}
	return operation.Capability.TrafficRequirement != ReconfigurationTrafficQuiesceGroup ||
		len(intersectReplicaIncarnations(traffic.Admitted, topologyReplicaIncarnations(topology))) == 0
}

func verificationJoiningReplicas(operation Operation, topology MembershipTopology) []ReplicaIncarnation {
	return missingReplicaIncarnations(
		topologyReplicaIncarnations(topology),
		topologyReplicaIncarnations(operation.BaseTopology),
	)
}

func changedBaseReplicaIncarnations(operation Operation) []ReplicaIncarnation {
	targetByReplica := make(map[ReplicaID]ReplicaMembership, len(operation.TargetMembership))
	for _, membership := range operation.TargetMembership {
		targetByReplica[membership.Incarnation.ReplicaID] = membership
	}

	changed := make([]ReplicaIncarnation, 0, len(operation.BaseTopology.Replicas))
	for _, baseMembership := range operation.BaseTopology.Replicas {
		targetMembership, exists := targetByReplica[baseMembership.Incarnation.ReplicaID]
		if !exists ||
			!sameReplicaIncarnation(baseMembership.Incarnation, targetMembership.Incarnation) ||
			!sameNativeMemberIDs(baseMembership.NativeMembers, targetMembership.NativeMembers) {
			changed = append(changed, cloneReplicaIncarnation(baseMembership.Incarnation))
		}
	}
	return normalizeReplicaIncarnations(changed)
}

func verificationFailureTrafficOperationID(operation Operation, topology MembershipTopology) string {
	return fmt.Sprintf(
		"%s/attempt-%d-topology-%d-serving-verification-failed",
		operation.ID,
		operation.Attempt,
		topology.Generation,
	)
}

func servingFailureTrafficReady(
	operation Operation,
	topology MembershipTopology,
	traffic TrafficSnapshot,
) bool {
	requiredDrains := unverifiedTopologyDrainReplicas(operation, topology)
	return trafficStateStableFor(traffic, TrafficActionWithdraw, requiredDrains) &&
		containsAllReplicaIncarnations(traffic.Drained, requiredDrains) &&
		len(intersectReplicaIncarnations(traffic.Admitted, requiredDrains)) == 0
}

func (c *WorkflowCoordinator) ensureServingFailureTopologyFenced(
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	replicas := unverifiedTopologyDrainReplicas(*result.Operation, result.Topology)
	if len(replicas) == 0 {
		return result, true, nil
	}

	operationID := verificationFailureTrafficOperationID(*result.Operation, result.Topology)
	ready := servingFailureTrafficReady(*result.Operation, result.Topology, result.Traffic)
	if ready {
		return result, true, nil
	}
	result, err := scheduleTrafficCommand(result, TrafficActionWithdraw, TrafficRequest{
		OperationID:        operationID,
		TopologyGeneration: result.Topology.Generation,
		Replicas:           replicas,
	})
	if err != nil {
		return result, false, fmt.Errorf("fence topology after serving verification failure: %w", err)
	}
	return result, false, nil
}

func unverifiedTopologyTrafficOperationID(operation Operation, topology MembershipTopology) string {
	return fmt.Sprintf(
		"%s/attempt-%d-topology-%d-unverified",
		operation.ID,
		operation.Attempt,
		topology.Generation,
	)
}

func unverifiedTopologyTrafficReady(
	operation Operation,
	topology MembershipTopology,
	traffic TrafficSnapshot,
) bool {
	requiredDrains := unverifiedTopologyDrainReplicas(operation, topology)
	return trafficStateStableFor(traffic, TrafficActionWithdraw, requiredDrains) &&
		containsAllReplicaIncarnations(traffic.Drained, requiredDrains) &&
		len(intersectReplicaIncarnations(traffic.Admitted, requiredDrains)) == 0
}

func operationTopologyDiffersFromServingBaseline(
	operation Operation,
	topology MembershipTopology,
) bool {
	return !servingVerificationTopologiesEqual(operationServingBaseline(operation), topology)
}

func operationServingBaseline(operation Operation) MembershipTopology {
	if operation.CompensationTopology != nil {
		return *operation.CompensationTopology
	}
	if operation.CommittedTopology != nil {
		return *operation.CommittedTopology
	}
	return operation.BaseTopology
}

func unverifiedTopologyDrainReplicas(
	operation Operation,
	topology MembershipTopology,
) []ReplicaIncarnation {
	// Fence both sides of an unverified transition so replaced incarnations cannot retain old traffic admission.
	replicas := append(
		topologyReplicaIncarnations(topology),
		topologyReplicaIncarnations(operationServingBaseline(operation))...,
	)
	return normalizeReplicaIncarnations(append(replicas, topologyReplicaIncarnations(operation.BaseTopology)...))
}

func markCompensationTopologyChanged(
	operation *Operation,
	topology MembershipTopology,
	now time.Time,
) bool {
	message := fmt.Sprintf(
		"authoritative topology generation %d no longer matches compensation generation %d",
		topology.Generation,
		operation.CompensationTopology.Generation,
	)
	if operation.Phase == OperationPhaseAborting &&
		operation.Failure.Reason == "CompensationTopologyChanged" &&
		operation.Failure.Message == message {
		return false
	}

	operation.Failure = &OperationFailure{
		Classification: FailureClassificationTerminal,
		Reason:         "CompensationTopologyChanged",
		Message:        message,
	}
	operation.Phase = OperationPhaseAborting
	operation.LastTransitionTime = now
	return true
}

func retirementTrafficReady(
	operation Operation,
	topology MembershipTopology,
	traffic TrafficSnapshot,
) bool {
	retiringReplicas := topologyReplicaIncarnationsForIDs(operation.BaseTopology, operation.NominatedReplicas)
	if trafficStateStableFor(traffic, TrafficActionWithdraw, retiringReplicas) &&
		containsAllReplicaIncarnations(traffic.Drained, retiringReplicas) &&
		len(intersectReplicaIncarnations(traffic.Admitted, retiringReplicas)) == 0 {
		return true
	}
	return unverifiedTopologyTrafficReady(operation, topology, traffic) ||
		servingFailureTrafficReady(operation, topology, traffic)
}

func trafficAdmissionComplete(operation Operation, traffic TrafficSnapshot) bool {
	return trafficStateStableFor(traffic, TrafficActionAdmit, operation.JoiningReplicas) &&
		containsAllReplicaIncarnations(traffic.Admitted, operation.JoiningReplicas) &&
		len(intersectReplicaIncarnations(traffic.Drained, operation.JoiningReplicas)) == 0
}

func fullTopologyAdmissionComplete(
	activeReplicas []ReplicaIncarnation,
	traffic TrafficSnapshot,
) bool {
	return trafficStateStableFor(traffic, TrafficActionAdmit, activeReplicas) &&
		containsAllReplicaIncarnations(traffic.Admitted, activeReplicas) &&
		len(intersectReplicaIncarnations(traffic.Drained, activeReplicas)) == 0
}

func trafficCompensationComplete(
	topology MembershipTopology,
	traffic TrafficSnapshot,
) bool {
	activeReplicas := topologyReplicaIncarnations(topology)
	// An empty authoritative topology has no survivor to admit, but it cannot complete while the latest accepted
	// command may still make any incarnation routable.
	if len(activeReplicas) == 0 {
		return len(traffic.Admitted) == 0 &&
			(traffic.LatestCommand == nil ||
				traffic.LatestCommand.Phase == TrafficCommandPhaseRefused ||
				traffic.LatestCommand.Phase == TrafficCommandPhaseFailed ||
				traffic.LatestCommand.Command.Action != TrafficActionAdmit ||
				traffic.LatestCommand.Command.Request.TopologyGeneration != topology.Generation)
	}
	return trafficStateStableFor(traffic, TrafficActionAdmit, activeReplicas) &&
		containsAllReplicaIncarnations(traffic.Admitted, activeReplicas) &&
		len(intersectReplicaIncarnations(traffic.Drained, activeReplicas)) == 0
}

func trafficStateStableFor(
	traffic TrafficSnapshot,
	expectedAction TrafficAction,
	replicas []ReplicaIncarnation,
) bool {
	if traffic.LatestCommand == nil ||
		traffic.LatestCommand.Phase == TrafficCommandPhaseRefused ||
		traffic.LatestCommand.Phase == TrafficCommandPhaseFailed {
		return true
	}

	var conflictingAction TrafficAction
	switch expectedAction {
	case TrafficActionAdmit:
		conflictingAction = TrafficActionWithdraw
	case TrafficActionWithdraw:
		conflictingAction = TrafficActionAdmit
	default:
		return false
	}
	return traffic.LatestCommand.Command.Action != conflictingAction ||
		len(intersectReplicaIncarnations(traffic.LatestCommand.Command.Request.Replicas, replicas)) == 0
}

func validateTrafficAgainstMembership(
	traffic TrafficSnapshot,
	topology MembershipTopology,
	operation *Operation,
) error {
	// A committed reduction may need to withdraw a retired identity that was admitted or recreated unexpectedly.
	allowedAdmissions := topologyReplicaIncarnations(topology)
	if operation != nil &&
		(operation.Phase == OperationPhaseCommitted ||
			operation.Phase == OperationPhaseFailed ||
			operation.Phase == OperationPhaseUnknown) &&
		operation.CommittedTopology != nil {
		allowedAdmissions = append(
			allowedAdmissions,
			preCommitWithdrawalReplicas(*operation)...,
		)
	}
	if operation != nil &&
		(operation.Phase == OperationPhaseAborting || operation.Phase == OperationPhaseAborted) {
		allowedAdmissions = append(allowedAdmissions, abortCleanupDrainReplicas(*operation, topology)...)
	}

	return validateTrafficAgainstReplicaSet(traffic, allowedAdmissions)
}

func validateTrafficAgainstReplicaSet(
	traffic TrafficSnapshot,
	allowedAdmissions []ReplicaIncarnation,
) error {
	// Runtime admission is the unsafe direction and rejects every identity outside authoritative membership. Extra
	// exact drain tombstones are fail-closed historical evidence and may safely survive across later operations.
	return requireReplicaIncarnationSubset(traffic.Admitted, allowedAdmissions, "traffic-admitted")
}

func requireReplicaIncarnationSubset(
	subset []ReplicaIncarnation,
	superset []ReplicaIncarnation,
	label string,
) error {
	for _, incarnation := range subset {
		if !slices.ContainsFunc(superset, func(candidate ReplicaIncarnation) bool {
			return sameReplicaIncarnation(incarnation, candidate)
		}) {
			return fmt.Errorf(
				"%s incarnation %q is outside the allowed exact membership",
				label,
				incarnation.ReplicaID,
			)
		}
	}
	return nil
}

func postCommitWorkComplete(
	operation Operation,
	authorization *ReleaseAuthorization,
	capacity CapacitySnapshot,
	traffic TrafficSnapshot,
) bool {
	// Only a previously committed topology can carry durable post-commit completion evidence.
	if operation.CommittedTopology == nil ||
		(operation.Phase != OperationPhaseCommitted && operation.Phase != OperationPhaseUnknown) {
		return false
	}
	if operation.Adopted {
		return false
	}
	if operation.TargetReplicas > 0 &&
		operation.Capability.VerificationRequirement == ServingVerificationRequired &&
		!servingVerificationProofMatchesOperation(operation.ServingVerificationProof, operation) {
		return false
	}
	if (operation.Capability.TrafficRequirement == ReconfigurationTrafficQuiesceGroup ||
		operation.Capability.VerificationRequirement == ServingVerificationRequired) &&
		!fullTopologyAdmissionComplete(topologyReplicaIncarnations(*operation.CommittedTopology), traffic) {
		return false
	}

	delta := operation.TargetReplicas - operation.BaseTopology.ReplicaCount()
	if delta > 0 {
		return trafficAdmissionComplete(operation, traffic)
	}
	if delta == 0 {
		return true
	}

	// A reduction is complete only after no authorization remains, every victim is fenced, and none is admitted.
	retiringReplicas := topologyReplicaIncarnationsForIDs(operation.BaseTopology, operation.NominatedReplicas)
	if authorization != nil || len(intersectReplicaIncarnations(traffic.Admitted, retiringReplicas)) != 0 {
		return false
	}
	if !operation.CapacityTargetApplied ||
		operation.CapacityTargetReplicas != operation.TargetReplicas ||
		operation.CapacityTopologyGeneration != committedTopologyGeneration(operation) {
		return false
	}
	if !releaseFencesComplete(operation, capacity) {
		return false
	}
	return true
}

func failedReductionReleaseComplete(
	operation Operation,
	authorization *ReleaseAuthorization,
	capacity CapacitySnapshot,
	topology MembershipTopology,
) bool {
	return authorization == nil &&
		capacityTargetProofCurrent(operation, topology) &&
		releaseFencesComplete(operation, capacity)
}

func reductionReleaseOwnershipComplete(
	operation Operation,
	authorization *ReleaseAuthorization,
	capacity CapacitySnapshot,
) bool {
	return authorization == nil && operation.CapacityTargetApplied && releaseFencesComplete(operation, capacity)
}

func releaseFencesComplete(operation Operation, capacity CapacitySnapshot) bool {
	allocations := replicaAllocationByID(capacity)
	fencedSlots := fencedReplicaSlots(capacity)
	baseSlots := make(map[ReplicaID]CapacitySlotID, len(operation.BaseTopology.Replicas))
	for _, membership := range operation.BaseTopology.Replicas {
		baseSlots[membership.Incarnation.ReplicaID] = membership.Incarnation.SlotID
	}

	// Every nominated identity must be absent and durably fenced against autonomous recreation.
	for _, replicaID := range operation.NominatedReplicas {
		if _, exists := allocations[replicaID]; exists {
			return false
		}
		if slotID, fenced := fencedSlots[replicaID]; !fenced || slotID != baseSlots[replicaID] {
			return false
		}
	}
	return true
}

func replicaAllocationsAvailable(incarnations []ReplicaIncarnation, capacity CapacitySnapshot) bool {
	allocations := replicaAllocationByID(capacity)
	fencedReplicas := fencedReplicaIDs(capacity)
	for _, incarnation := range incarnations {
		allocation, exists := allocations[incarnation.ReplicaID]
		_, fenced := fencedReplicas[incarnation.ReplicaID]
		if !exists || fenced ||
			allocation.Availability != ReplicaAvailabilityAvailable ||
			!sameReplicaIncarnation(allocation.Incarnation, incarnation) {
			return false
		}
	}
	return true
}

func replicaIncarnationsPresent(incarnations []ReplicaIncarnation, capacity CapacitySnapshot) bool {
	allocations := replicaAllocationByID(capacity)
	fencedReplicas := fencedReplicaIDs(capacity)
	for _, incarnation := range incarnations {
		allocation, exists := allocations[incarnation.ReplicaID]
		_, fenced := fencedReplicas[incarnation.ReplicaID]
		if !exists || fenced || !sameReplicaIncarnation(allocation.Incarnation, incarnation) {
			return false
		}
	}
	return true
}

func frozenSubmissionIncarnations(operation Operation) []ReplicaIncarnation {
	switch operation.Capability.Shape {
	case OperationShapeFixedSlotReplacement, OperationShapeNativeMemberRemapping:
		return replicaMembershipIncarnations(operation.TargetMembership)
	case OperationShapeFreshGrowth, OperationShapeReplacementRestoration:
		return normalizeReplicaIncarnations(append(
			topologyReplicaIncarnations(operation.BaseTopology),
			operation.JoiningReplicas...,
		))
	case OperationShapeSurvivorReduction:
		retainedReplicaIDs := differenceReplicaIDs(
			topologyReplicaIDs(operation.BaseTopology),
			operation.NominatedReplicas,
		)
		return topologyReplicaIncarnationsForIDs(operation.BaseTopology, retainedReplicaIDs)
	case OperationShapePlannedHighRankSuffixShrink,
		OperationShapePlannedSelectedRetirement,
		OperationShapeFullRetirement:
		return topologyReplicaIncarnations(operation.BaseTopology)
	default:
		return nil
	}
}

func fencedReplicaIDs(capacity CapacitySnapshot) map[ReplicaID]struct{} {
	fencedReplicas := make(map[ReplicaID]struct{}, len(capacity.FencedReplicaSlots))
	for _, binding := range capacity.FencedReplicaSlots {
		fencedReplicas[binding.ReplicaID] = struct{}{}
	}
	return fencedReplicas
}

func fencedReplicaSlots(capacity CapacitySnapshot) map[ReplicaID]CapacitySlotID {
	fencedSlots := make(map[ReplicaID]CapacitySlotID, len(capacity.FencedReplicaSlots))
	for _, binding := range capacity.FencedReplicaSlots {
		fencedSlots[binding.ReplicaID] = binding.SlotID
	}
	return fencedSlots
}

func shouldValidateRestorationPlan(operation *Operation, plan *OperationPlan) bool {
	return plan != nil &&
		len(plan.RestoredMembership) > 0 &&
		(operation == nil || operation.PlanID != plan.ID)
}

func newlyAdoptedOperation(previous, observed *Operation) bool {
	return observed != nil &&
		observed.Adopted &&
		(previous == nil || previous.ID != observed.ID)
}

func validateNewRestorationPlan(
	operation *Operation,
	plan *OperationPlan,
	capacity CapacitySnapshot,
) error {
	if !shouldValidateRestorationPlan(operation, plan) {
		return nil
	}
	return validateRestorationPlanAgainstCapacity(*plan, capacity)
}

func validateRestorationPlanAgainstCapacity(plan OperationPlan, capacity CapacitySnapshot) error {
	fencedSlots := fencedReplicaSlots(capacity)
	for _, restored := range plan.RestoredMembership {
		slotID, exists := fencedSlots[restored.ReplicaID]
		if !exists {
			return fmt.Errorf("restored replica %q has no durable capacity-slot fence", restored.ReplicaID)
		}
		if restored.SlotID != slotID {
			return fmt.Errorf(
				"restored replica %q must reuse fenced slot %q, got %q",
				restored.ReplicaID,
				slotID,
				restored.SlotID,
			)
		}
	}
	return nil
}

func validateAdoptedRestoration(
	previous *Operation,
	adopted Operation,
	capacity CapacitySnapshot,
) error {
	if len(adopted.RestoredMembership) == 0 {
		return nil
	}
	if previous == nil {
		return errors.New("adopted restoration requires durable historical membership")
	}

	// EnsureCapacity normally consumes a logical fence when it recreates the required slot. Adoption therefore uses
	// the old durable operation (plus any still-observable fence) as historical proof, and the current allocation as
	// corroboration. It must not require the consumed fence to remain visible.
	historicalCapacity := CapacitySnapshot{FencedReplicaSlots: slices.Clone(capacity.FencedReplicaSlots)}
	allocations := replicaAllocationByID(capacity)
	for _, restored := range adopted.RestoredMembership {
		slotID, found, err := knownReplicaSlot(*previous, historicalCapacity, restored.ReplicaID)
		if err != nil {
			return err
		}
		if !found {
			return fmt.Errorf(
				"adopted restored replica %q has no durable historical capacity slot",
				restored.ReplicaID,
			)
		}
		if slotID != restored.SlotID {
			return fmt.Errorf(
				"adopted restored replica %q must reuse historical slot %q, got %q",
				restored.ReplicaID,
				slotID,
				restored.SlotID,
			)
		}
		if allocation, exists := allocations[restored.ReplicaID]; exists &&
			allocation.Incarnation.SlotID != restored.SlotID {
			return fmt.Errorf(
				"adopted restored replica %q is allocated in slot %q instead of historical slot %q",
				restored.ReplicaID,
				allocation.Incarnation.SlotID,
				restored.SlotID,
			)
		}
	}
	return nil
}

func cleanupTopologySafe(operation Operation, topology MembershipTopology) (bool, error) {
	if operation.CommittedTopology == nil {
		return false, errors.New("cleanup requires an exact committed topology")
	}
	committedTopology := *operation.CommittedTopology
	if topology.Generation < committedTopology.Generation {
		return false, nil
	}
	expectedReplicas := topologyReplicaIDs(committedTopology)
	observedReplicas := topologyReplicaIDs(topology)
	if topology.Generation == committedTopology.Generation {
		if !servingVerificationTopologiesEqual(topology, committedTopology) {
			return false, fmt.Errorf(
				"topology generation %d no longer matches operation %q committed membership",
				topology.Generation,
				operation.ID,
			)
		}
		return true, nil
	}
	if err := requireReplicaSubset(observedReplicas, expectedReplicas, "later survivor"); err != nil {
		return false, err
	}
	// A later generation may replace or remap survivors. Callers must fully fence that unverified topology; cleanup
	// remains limited to old victims that are still absent and exact UID authorization is revalidated separately.
	return true, nil
}

func (c *WorkflowCoordinator) buildReleaseAuthorization(
	operation Operation,
	capacity CapacitySnapshot,
	proofTopology MembershipTopology,
) (*ReleaseAuthorization, error) {
	if operation.CommittedTopology == nil {
		return nil, errors.New("capacity release requires an exact committed topology")
	}
	committedGeneration := operation.CommittedTopology.Generation
	if proofTopology.Generation < committedGeneration {
		return nil, fmt.Errorf(
			"release proof topology generation %d precedes committed generation %d",
			proofTopology.Generation,
			committedGeneration,
		)
	}
	return c.buildExactReleaseAuthorization(
		operation,
		proofTopology.ReplicaCount(),
		operation.NominatedReplicas,
		capacity,
		proofTopology.Generation,
	)
}

func (c *WorkflowCoordinator) buildExactReleaseAuthorization(
	operation Operation,
	targetReplicas int32,
	replicaIDs []ReplicaID,
	capacity CapacitySnapshot,
	proofTopologyGeneration int64,
) (*ReleaseAuthorization, error) {
	allocations := replicaAllocationByID(capacity)
	replicaIDs = normalizeReplicaIDs(replicaIDs)
	authorizedReplicas := make([]AuthorizedReplica, 0, len(replicaIDs))

	// Authorize every identity; already-absent capacity still needs a durable logical fence.
	for _, replicaID := range replicaIDs {
		allocation, exists := allocations[replicaID]
		if !exists {
			slotID, found, err := knownReplicaSlot(operation, capacity, replicaID)
			if err != nil {
				return nil, err
			}
			if !found {
				return nil, fmt.Errorf("cannot authorize fence for replica %q without a stable capacity slot", replicaID)
			}
			authorizedReplicas = append(authorizedReplicas, AuthorizedReplica{
				ReplicaID: replicaID,
				SlotID:    slotID,
			})
			continue
		}
		capacityRefs := slices.Clone(allocation.Incarnation.CapacityRefs)
		slices.SortFunc(capacityRefs, compareCapacityRefs)
		authorizedReplicas = append(authorizedReplicas, AuthorizedReplica{
			ReplicaID:    replicaID,
			SlotID:       allocation.Incarnation.SlotID,
			CapacityRefs: capacityRefs,
		})
	}
	releaseID := c.newReleaseID()
	if releaseID == "" {
		return nil, errors.New("generate capacity release ID: empty value")
	}
	return &ReleaseAuthorization{
		ID:                 releaseID,
		OperationID:        operation.ID,
		TopologyGeneration: proofTopologyGeneration,
		TargetReplicas:     targetReplicas,
		Replicas:           authorizedReplicas,
	}, nil
}

func knownReplicaSlot(
	operation Operation,
	capacity CapacitySnapshot,
	replicaID ReplicaID,
) (CapacitySlotID, bool, error) {
	var resolved CapacitySlotID
	add := func(slotID CapacitySlotID) error {
		if slotID == "" {
			return nil
		}
		if resolved != "" && resolved != slotID {
			return fmt.Errorf(
				"replica %q has conflicting stable capacity slots %q and %q",
				replicaID,
				resolved,
				slotID,
			)
		}
		resolved = slotID
		return nil
	}

	for _, binding := range capacity.FencedReplicaSlots {
		if binding.ReplicaID == replicaID {
			if err := add(binding.SlotID); err != nil {
				return "", false, err
			}
		}
	}
	for _, allocation := range capacity.Allocations {
		if allocation.Incarnation.ReplicaID == replicaID {
			if err := add(allocation.Incarnation.SlotID); err != nil {
				return "", false, err
			}
		}
	}
	for _, binding := range operation.CleanupReplicaSlots {
		if binding.ReplicaID == replicaID {
			if err := add(binding.SlotID); err != nil {
				return "", false, err
			}
		}
	}
	for _, membership := range append(
		cloneReplicaMemberships(operation.BaseTopology.Replicas),
		operation.TargetMembership...,
	) {
		if membership.Incarnation.ReplicaID == replicaID {
			if err := add(membership.Incarnation.SlotID); err != nil {
				return "", false, err
			}
		}
	}
	for _, topology := range []*MembershipTopology{operation.CommittedTopology, operation.CompensationTopology} {
		if topology == nil {
			continue
		}
		for _, membership := range topology.Replicas {
			if membership.Incarnation.ReplicaID == replicaID {
				if err := add(membership.Incarnation.SlotID); err != nil {
					return "", false, err
				}
			}
		}
	}
	for _, incarnation := range operation.JoiningReplicas {
		if incarnation.ReplicaID == replicaID {
			if err := add(incarnation.SlotID); err != nil {
				return "", false, err
			}
		}
	}
	for _, membership := range operation.RestoredMembership {
		if membership.ReplicaID == replicaID {
			if err := add(membership.SlotID); err != nil {
				return "", false, err
			}
		}
	}
	return resolved, resolved != "", nil
}

func validateAuthorizationForOperation(
	authorization ReleaseAuthorization,
	operation Operation,
) error {
	if authorization.OperationID != operation.ID {
		return fmt.Errorf(
			"release authorization operation %q does not match current operation %q",
			authorization.OperationID,
			operation.ID,
		)
	}
	if operation.Phase == OperationPhaseAborting {
		return validateAbortAuthorizationReplicas(authorization, operation)
	}

	switch operation.Phase {
	case OperationPhaseCommitted:
		committedGeneration := committedTopologyGeneration(operation)
		if authorization.TopologyGeneration != committedGeneration {
			return fmt.Errorf(
				"release authorization topology generation %d does not match committed generation %d",
				authorization.TopologyGeneration,
				committedGeneration,
			)
		}
		if authorization.TargetReplicas != operation.TargetReplicas {
			return fmt.Errorf(
				"release authorization target %d does not match committed target %d",
				authorization.TargetReplicas,
				operation.TargetReplicas,
			)
		}
	case OperationPhaseFailed, OperationPhaseUnknown:
		committedGeneration := committedTopologyGeneration(operation)
		if authorization.TopologyGeneration < committedGeneration {
			return fmt.Errorf(
				"release authorization topology generation %d precedes committed generation %d",
				authorization.TopologyGeneration,
				committedGeneration,
			)
		}
		if authorization.TopologyGeneration == committedGeneration &&
			authorization.TargetReplicas != operation.TargetReplicas {
			return fmt.Errorf(
				"release authorization target %d does not match committed target %d",
				authorization.TargetReplicas,
				operation.TargetReplicas,
			)
		}
		if authorization.TopologyGeneration > committedGeneration &&
			authorization.TargetReplicas > operation.TargetReplicas {
			return fmt.Errorf(
				"later survivor release target %d exceeds committed target %d",
				authorization.TargetReplicas,
				operation.TargetReplicas,
			)
		}
	default:
		return fmt.Errorf("operation phase %q cannot authorize capacity release", operation.Phase)
	}

	authorizedIDs := make([]ReplicaID, len(authorization.Replicas))
	for i, replica := range authorization.Replicas {
		authorizedIDs[i] = replica.ReplicaID
	}
	return requireReplicaSubset(authorizedIDs, operation.NominatedReplicas, "authorized")
}

func validateAbortAuthorizationReplicas(
	authorization ReleaseAuthorization,
	operation Operation,
) error {
	authorizedIDs := make([]ReplicaID, len(authorization.Replicas))
	for i, replica := range authorization.Replicas {
		authorizedIDs[i] = replica.ReplicaID
	}
	return requireReplicaSubset(authorizedIDs, abortCleanupReplicaIDs(operation), "abort-cleanup authorized")
}

func validateAuthorizationAgainstTopology(
	authorization ReleaseAuthorization,
	operation Operation,
	topology MembershipTopology,
) error {
	activeReplicaIDs := topologyReplicaIDs(topology)
	activePodNames := make(map[string]struct{})
	activePodUIDs := make(map[PodUID]struct{})
	for _, incarnation := range topologyReplicaIncarnations(topology) {
		for _, capacityRef := range incarnation.CapacityRefs {
			activePodNames[capacityRef.Namespace+"/"+capacityRef.Name] = struct{}{}
			activePodUIDs[capacityRef.UID] = struct{}{}
		}
	}
	for _, replica := range authorization.Replicas {
		if slices.Contains(activeReplicaIDs, replica.ReplicaID) {
			return fmt.Errorf("release authorization targets active replica %q", replica.ReplicaID)
		}
		for _, capacityRef := range replica.CapacityRefs {
			_, nameActive := activePodNames[capacityRef.Namespace+"/"+capacityRef.Name]
			_, uidActive := activePodUIDs[capacityRef.UID]
			if nameActive || uidActive {
				return fmt.Errorf(
					"release authorization targets capacity still present in authoritative topology: %s/%s (%s)",
					capacityRef.Namespace,
					capacityRef.Name,
					capacityRef.UID,
				)
			}
		}
	}
	if authorization.TopologyGeneration > topology.Generation {
		return fmt.Errorf(
			"release authorization topology generation %d exceeds current generation %d",
			authorization.TopologyGeneration,
			topology.Generation,
		)
	}
	if authorization.TargetReplicas < topology.ReplicaCount() {
		return fmt.Errorf(
			"release authorization target %d is below current authoritative replicas %d",
			authorization.TargetReplicas,
			topology.ReplicaCount(),
		)
	}
	if authorization.TopologyGeneration < topology.Generation {
		return nil
	}
	if authorization.TargetReplicas != topology.ReplicaCount() {
		return fmt.Errorf(
			"release authorization target %d does not match topology replicas %d",
			authorization.TargetReplicas,
			topology.ReplicaCount(),
		)
	}

	authorizedIDs := make([]ReplicaID, len(authorization.Replicas))
	for i, replica := range authorization.Replicas {
		authorizedIDs[i] = replica.ReplicaID
	}
	expectedIDs := normalizeReplicaIDs(operation.NominatedReplicas)
	if operation.Phase == OperationPhaseAborting || operation.Phase == OperationPhaseAborted {
		expectedIDs = abortCleanupReplicaIDs(operation)
	}
	if !sameReplicaIDs(authorizedIDs, expectedIDs) {
		return fmt.Errorf(
			"release authorization replicas %v do not match topology-excluded replicas %v",
			normalizeReplicaIDs(authorizedIDs),
			expectedIDs,
		)
	}
	return nil
}

func validateAuthorizationFresh(
	authorization ReleaseAuthorization,
	capacity CapacitySnapshot,
) error {
	allocations := replicaAllocationByID(capacity)
	authorizedBindings := make(map[string]capacityBinding)

	// Build the immutable logical-replica, slot, Pod-name, and UID bindings carried by authorization.
	for _, replica := range authorization.Replicas {
		for _, capacityRef := range replica.CapacityRefs {
			authorizedBindings[capacityRef.Namespace+"/"+capacityRef.Name] = capacityBinding{
				replicaID: replica.ReplicaID,
				slotID:    replica.SlotID,
				uid:       capacityRef.UID,
			}
		}
	}

	// A replacement or remapped slot must never inherit deletion authority from an older allocation.
	for _, authorizedReplica := range authorization.Replicas {
		allocation, exists := allocations[authorizedReplica.ReplicaID]
		if !exists {
			continue
		}
		if allocation.Incarnation.SlotID != authorizedReplica.SlotID {
			return fmt.Errorf(
				"release authorization for replica %q is stale: capacity slot changed from %q to %q",
				authorizedReplica.ReplicaID,
				authorizedReplica.SlotID,
				allocation.Incarnation.SlotID,
			)
		}
		for _, capacityRef := range allocation.Incarnation.CapacityRefs {
			binding, authorized := authorizedBindings[capacityRef.Namespace+"/"+capacityRef.Name]
			if !authorized ||
				binding.replicaID != allocation.Incarnation.ReplicaID ||
				binding.slotID != allocation.Incarnation.SlotID ||
				binding.uid != capacityRef.UID {
				return fmt.Errorf(
					"release authorization for replica %q is stale at Pod %s/%s",
					authorizedReplica.ReplicaID,
					capacityRef.Namespace,
					capacityRef.Name,
				)
			}
		}
	}

	// Moving an authorized Pod identity under another logical allocation is equally stale.
	for _, allocation := range capacity.Allocations {
		for _, capacityRef := range allocation.Incarnation.CapacityRefs {
			binding, authorized := authorizedBindings[capacityRef.Namespace+"/"+capacityRef.Name]
			if authorized &&
				(binding.uid != capacityRef.UID ||
					binding.replicaID != allocation.Incarnation.ReplicaID ||
					binding.slotID != allocation.Incarnation.SlotID) {
				return fmt.Errorf(
					"release authorization is stale: Pod %s/%s no longer matches replica %q, slot %q, UID %q",
					capacityRef.Namespace,
					capacityRef.Name,
					binding.replicaID,
					binding.slotID,
					binding.uid,
				)
			}
		}
	}
	return nil
}

func validateAuthorizationCompleteness(
	authorization ReleaseAuthorization,
	requiredReplicas []ReplicaID,
) error {
	// Index both identity sets before checking whether durable authority covers current capacity.
	authorizedReplicas := make(map[ReplicaID]struct{}, len(authorization.Replicas))
	for _, replica := range authorization.Replicas {
		authorizedReplicas[replica.ReplicaID] = struct{}{}
	}
	// Every required replica must be included, even when it currently needs only a durable logical fence.
	for _, replicaID := range requiredReplicas {
		if _, authorized := authorizedReplicas[replicaID]; !authorized {
			return fmt.Errorf(
				"release authorization omits required replica %q",
				replicaID,
			)
		}
	}
	return nil
}

type capacityBinding struct {
	replicaID ReplicaID
	slotID    CapacitySlotID
	uid       PodUID
}

func authorizedCapacityPresent(
	authorization ReleaseAuthorization,
	capacity CapacitySnapshot,
) bool {
	currentUIDs := make(map[PodUID]struct{})
	for _, allocation := range capacity.Allocations {
		for _, capacityRef := range allocation.Incarnation.CapacityRefs {
			currentUIDs[capacityRef.UID] = struct{}{}
		}
	}

	for _, replica := range authorization.Replicas {
		for _, capacityRef := range replica.CapacityRefs {
			if _, exists := currentUIDs[capacityRef.UID]; exists {
				return true
			}
		}
	}
	return false
}

func authorizationFencesObserved(
	authorization ReleaseAuthorization,
	capacity CapacitySnapshot,
) bool {
	fencedSlots := fencedReplicaSlots(capacity)

	// Applied is complete only when every authorized logical identity has an observable durable fence.
	for _, replica := range authorization.Replicas {
		if slotID, fenced := fencedSlots[replica.ReplicaID]; !fenced || slotID != replica.SlotID {
			return false
		}
	}
	return true
}

func validateReconcileInput(input ReconcileInput) error {
	if err := validateOperationStateInput(OperationInput{
		GroupID:         input.GroupID,
		SpecGeneration:  input.SpecGeneration,
		DesiredReplicas: input.DesiredReplicas,
		Plan:            input.Plan,
		Operation:       input.Operation,
	}); err != nil {
		return err
	}
	if input.TrafficRevision < 0 {
		return fmt.Errorf("traffic revision must not be negative: %d", input.TrafficRevision)
	}
	if input.TrafficRevision == 0 && input.TrafficCommand != nil {
		return errors.New("traffic command requires a positive durable revision")
	}
	if input.TrafficRevision > 0 && input.TrafficCommand == nil {
		return errors.New("positive traffic revision requires its exact durable command")
	}
	if input.TrafficCommand != nil {
		if err := validateTrafficCommand(*input.TrafficCommand); err != nil {
			return fmt.Errorf("validate durable traffic command: %w", err)
		}
		if input.TrafficCommand.Request.Revision != input.TrafficRevision {
			return fmt.Errorf(
				"durable traffic command revision %d does not match status revision %d",
				input.TrafficCommand.Request.Revision,
				input.TrafficRevision,
			)
		}
	}
	if input.ReleaseAuthorization == nil {
		return nil
	}
	if err := validateReleaseAuthorization(*input.ReleaseAuthorization); err != nil {
		return err
	}
	if input.Operation == nil {
		return errors.New("release authorization requires a membership operation")
	}
	if input.Operation.Phase == OperationPhaseAborting {
		return validateAuthorizationForOperation(*input.ReleaseAuthorization, *input.Operation)
	}
	if input.Operation.Phase == OperationPhaseUnknown && input.Operation.CompensationTopology != nil {
		if input.ReleaseAuthorization.OperationID != input.Operation.ID {
			return errors.New("compensation release authorization belongs to a different operation")
		}
		return validateAbortAuthorizationReplicas(*input.ReleaseAuthorization, *input.Operation)
	}
	if input.Operation.Phase != OperationPhaseCommitted &&
		((input.Operation.Phase != OperationPhaseFailed && input.Operation.Phase != OperationPhaseUnknown) ||
			input.Operation.CommittedTopology == nil ||
			input.Operation.CommittedTopology.Generation <= input.Operation.BaseTopology.Generation) {
		return errors.New("release authorization requires committed membership or its failed or unknown postwork")
	}
	if input.Operation.TargetReplicas >= input.Operation.BaseTopology.ReplicaCount() {
		return errors.New("release authorization requires a committed reduction")
	}
	if input.Operation.PostCommitComplete {
		return errors.New("post-commit-complete operation must not carry a release authorization")
	}
	return validateAuthorizationForOperation(*input.ReleaseAuthorization, *input.Operation)
}
