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
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"

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
}

// ReconcileResult contains fresh observations and complete durable workflow state.
type ReconcileResult struct {
	Operation                   *Operation
	ReleaseAuthorization        *ReleaseAuthorization
	Capacity                    CapacitySnapshot
	Topology                    MembershipTopology
	Traffic                     TrafficSnapshot
	OperationChanged            bool
	ReleaseAuthorizationChanged bool
}

// WorkflowCoordinator reconciles physical capacity, engine membership, traffic, and safe release.
type WorkflowCoordinator struct {
	capacity     CapacityAdapter
	membership   MembershipAdapter
	traffic      TrafficAdapter
	operations   *OperationCoordinator
	newReleaseID func() string
}

// NewWorkflowCoordinator constructs an Engine Group workflow. All adapters must be non-nil.
func NewWorkflowCoordinator(
	capacity CapacityAdapter,
	membership MembershipAdapter,
	traffic TrafficAdapter,
) *WorkflowCoordinator {
	return &WorkflowCoordinator{
		capacity:     capacity,
		membership:   membership,
		traffic:      traffic,
		operations:   NewOperationCoordinator(membership),
		newReleaseID: func() string { return string(uuid.NewUUID()) },
	}
}

// Reconcile advances at most one durable transition or one idempotent external side effect.
func (c *WorkflowCoordinator) Reconcile(
	ctx context.Context,
	input ReconcileInput,
) (ReconcileResult, error) {
	result := ReconcileResult{
		Operation:            cloneOperation(input.Operation),
		ReleaseAuthorization: cloneReleaseAuthorization(input.ReleaseAuthorization),
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

	// Persist independently provable post-commit completion before replacing or recovering membership state.
	if input.Operation != nil &&
		!input.Operation.PostCommitComplete &&
		postCommitWorkComplete(
			*input.Operation,
			input.ReleaseAuthorization,
			result.Capacity,
			result.Traffic,
		) {
		result.Operation.PostCommitComplete = true
		result.OperationChanged = true
		return result, nil
	}

	// Membership uncertainty never regresses while exact physical release may still be in flight.
	if input.Operation != nil &&
		input.Operation.Phase == OperationPhaseUnknown &&
		input.ReleaseAuthorization != nil {
		return c.reconcileUncertainRelease(ctx, input, result)
	}

	// A later authoritative survivor topology may still prove old reducing-operation cleanup safe.
	if input.Operation != nil &&
		input.Operation.Phase == OperationPhaseUnknown &&
		input.Operation.CommittedTopologyGeneration != 0 &&
		!input.Operation.PostCommitComplete &&
		input.Operation.TargetReplicas < int32(len(input.Operation.BaseReplicas)) {
		return c.reconcileUnknownReductionPostwork(ctx, input, result)
	}

	// A durably committed membership operation must finish traffic or release work before replacement.
	if input.Operation != nil && input.Operation.Phase == OperationPhaseCommitted {
		return c.reconcileCommitted(ctx, input, result)
	}

	// Advance membership observation without hiding capacity or traffic prerequisites in that coordinator.
	operationResult, err := c.operations.Step(ctx, OperationInput{
		GroupID:         input.GroupID,
		SpecGeneration:  input.SpecGeneration,
		DesiredReplicas: input.DesiredReplicas,
		Plan:            input.Plan,
		Operation:       input.Operation,
	})
	result.Operation = operationResult.Operation
	result.Topology = operationResult.Topology
	result.OperationChanged = operationResult.OperationChanged
	if err != nil {
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

func (c *WorkflowCoordinator) reconcileOperationPhase(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
	submissionNeeded bool,
) (ReconcileResult, error) {
	// Keep preparation and submission ordering explicit for each durable membership phase.
	switch result.Operation.Phase {
	case OperationPhasePending:
		if err := c.operations.requireCapability(ctx, input.GroupID, result.Operation.Intent); err != nil {
			if errors.Is(err, errMembershipOperationUnsupported) {
				return c.abortForCapabilityLoss(result, err), nil
			}
			return result, err
		}
		return c.reconcilePending(ctx, input.GroupID, result)
	case OperationPhaseSubmitting:
		if !submissionNeeded {
			return result, nil
		}
		return c.reconcileSubmission(ctx, input.GroupID, result)
	case OperationPhaseFailed:
		if result.Operation.Failure.Classification == FailureClassificationRetryable {
			if err := c.operations.requireCapability(ctx, input.GroupID, result.Operation.Intent); err != nil {
				if errors.Is(err, errMembershipOperationUnsupported) {
					return c.abortForCapabilityLoss(result, err), nil
				}
				return result, err
			}
		}
		return c.reconcileFailed(ctx, input.GroupID, result)
	case OperationPhaseAborting:
		return c.reconcileAborting(ctx, input, result)
	case OperationPhaseAborted:
		return c.reconcileAborted(ctx, input, result)
	case OperationPhaseUnknown:
		return result, nil
	case OperationPhaseAccepted, OperationPhaseCommitting:
		return result, nil
	default:
		return result, fmt.Errorf("unsupported workflow operation phase %q", result.Operation.Phase)
	}
}

func (c *WorkflowCoordinator) abortForCapabilityLoss(
	result ReconcileResult,
	cause error,
) ReconcileResult {
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

func (c *WorkflowCoordinator) reconcileUnknownReductionPostwork(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	operation := *result.Operation
	if operation.TargetReplicas >= int32(len(operation.BaseReplicas)) {
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
	// The later topology attests quiescence; old exact traffic drain and release obligations remain in force.
	allowedAdmissions := append(topologyReplicaIDs(topology), operation.NominatedReplicas...)
	if err := validateTrafficAgainstReplicaSet(result.Traffic, allowedAdmissions, result.Operation); err != nil {
		return result, err
	}
	if !retirementTrafficReady(operation, result.Traffic) {
		if err := c.traffic.Withdraw(
			ctx,
			input.GroupID,
			TrafficRequest{
				OperationID:        operation.ID,
				TopologyGeneration: topology.Generation,
				Replicas:           slices.Clone(operation.NominatedReplicas),
			},
		); err != nil {
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
	if err := validateAuthorizationAgainstTopology(
		*input.ReleaseAuthorization,
		*input.Operation,
		topology,
	); err != nil {
		return result, err
	}
	allowedAdmissions := append(topologyReplicaIDs(topology), input.Operation.NominatedReplicas...)
	if err := validateTrafficAgainstReplicaSet(result.Traffic, allowedAdmissions, input.Operation); err != nil {
		return result, err
	}
	if !retirementTrafficReady(*input.Operation, result.Traffic) {
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
	default:
		return result, fmt.Errorf("unsupported capacity release phase %q", observation.Phase)
	}
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
		expectedCommittedReplicaIDs(*input.Operation),
		input.Operation.NominatedReplicas...,
	)
	if err := validateTrafficAgainstReplicaSet(result.Traffic, allowedAdmissions, input.Operation); err != nil {
		return result, err
	}
	if !retirementTrafficReady(*input.Operation, result.Traffic) {
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
	delta := operation.TargetReplicas - int32(len(operation.BaseReplicas))

	// Growth first allocates capacity, then freezes exact available joiner identities into Submitting.
	if delta > 0 {
		joiningReplicas, ready, err := resolveJoiningReplicas(*operation, result.Capacity, result.Traffic)
		if err != nil {
			return result, err
		}
		if !ready {
			request := CapacityRequest{
				OperationID:        operation.ID,
				TopologyGeneration: result.Topology.Generation,
				TargetReplicas:     operation.TargetReplicas,
				RequiredReplicas:   normalizeReplicaIDs(operation.BaseReplicas),
			}
			if err := c.capacity.EnsureCapacity(ctx, groupID, request); err != nil {
				return result, fmt.Errorf("ensure %d Engine Group replicas: %w", operation.TargetReplicas, err)
			}
			return result, nil
		}

		prepared, err := c.operations.PrepareSubmission(operation, joiningReplicas)
		if err != nil {
			return result, fmt.Errorf("prepare growth submission: %w", err)
		}
		result.Operation = prepared
		result.OperationChanged = true
		return result, nil
	}

	// Reduction withdraws exact nominees and waits for operation-scoped drain before Submitting.
	if delta < 0 && !retirementTrafficReady(*operation, result.Traffic) {
		if err := c.traffic.Withdraw(
			ctx,
			groupID,
			TrafficRequest{
				OperationID:        operation.ID,
				TopologyGeneration: result.Topology.Generation,
				Replicas:           slices.Clone(operation.NominatedReplicas),
			},
		); err != nil {
			return result, fmt.Errorf("withdraw nominated replicas from traffic: %w", err)
		}
		return result, nil
	}

	// Cardinally stable recovery and fully drained reduction need only a durable submission marker.
	prepared, err := c.operations.PrepareSubmission(operation, nil)
	if err != nil {
		return result, fmt.Errorf("prepare membership submission: %w", err)
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
		if err := c.operations.requireCapability(ctx, groupID, result.Operation.Intent); err != nil {
			return result, err
		}
		return c.resumePreparation(ctx, groupID, result)
	}

	operationResult, err := c.operations.Submit(ctx, groupID, result.Operation)
	result.Operation = operationResult.Operation
	result.Topology = operationResult.Topology
	result.OperationChanged = operationResult.OperationChanged
	if err != nil {
		return result, err
	}
	return result, nil
}

func (c *WorkflowCoordinator) reconcileFailed(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, error) {
	// Persist terminal failure compensation before restoring any capacity or traffic side effect.
	if result.Operation.Failure.Classification == FailureClassificationTerminal {
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
		return c.resumePreparation(ctx, groupID, result)
	}
	prepared, err := c.operations.PrepareSubmission(result.Operation, result.Operation.JoiningReplicas)
	if err != nil {
		return result, fmt.Errorf("prepare membership retry: %w", err)
	}
	result.Operation = prepared
	result.OperationChanged = true
	return result, nil
}

func (c *WorkflowCoordinator) reconcileAborting(
	ctx context.Context,
	input ReconcileInput,
	result ReconcileResult,
) (ReconcileResult, error) {
	// Freeze every currently surplus allocation before authorizing any physical removal.
	if input.ReleaseAuthorization == nil {
		cleanupReplicas := discoverAbortCleanupReplicas(
			*result.Operation,
			result.Capacity,
			result.Topology,
		)
		if !sameReplicaIDs(cleanupReplicas, result.Operation.CleanupReplicas) {
			result.Operation.CleanupReplicas = cleanupReplicas
			clearCapacityTargetProof(result.Operation)
			result.OperationChanged = true
			return result, nil
		}
	}
	if err := validateAbortCleanupTopology(*result.Operation, result.Topology); err != nil {
		return result, err
	}

	// Reject a release barrier that could remove authoritative capacity before any compensation side effect.
	if input.ReleaseAuthorization != nil {
		if input.ReleaseAuthorization.TopologyGeneration > result.Topology.Generation {
			return result, fmt.Errorf(
				"abort cleanup release topology generation %d exceeds current generation %d",
				input.ReleaseAuthorization.TopologyGeneration,
				result.Topology.Generation,
			)
		}
		if input.ReleaseAuthorization.TargetReplicas < result.Topology.ReplicaCount() {
			return result, fmt.Errorf(
				"abort cleanup release target %d is below current authoritative replicas %d",
				input.ReleaseAuthorization.TargetReplicas,
				result.Topology.ReplicaCount(),
			)
		}
	}

	// Formerly serving cleanup replicas must drain before any current or stale authorization can remove them.
	if !abortCleanupTrafficReady(*result.Operation, result.Topology, result.Traffic) {
		if err := c.traffic.Withdraw(
			ctx,
			input.GroupID,
			TrafficRequest{
				OperationID:        result.Operation.ID,
				TopologyGeneration: result.Topology.Generation,
				Replicas:           abortCleanupDrainReplicas(*result.Operation, result.Topology),
			},
		); err != nil {
			return result, fmt.Errorf("drain abort cleanup replicas: %w", err)
		}
		return result, nil
	}

	// Resolve an older barrier before issuing capacity work against the authoritative topology generation.
	if input.ReleaseAuthorization != nil &&
		(input.ReleaseAuthorization.TopologyGeneration != result.Topology.Generation ||
			input.ReleaseAuthorization.TargetReplicas != result.Topology.ReplicaCount()) {
		return c.reconcileStaleAbortRelease(ctx, input.GroupID, result)
	}

	// Finish an already-authorized exact cleanup before repairing missing authoritative identities.
	if input.ReleaseAuthorization != nil {
		if err := validateAuthorizationForOperation(
			*input.ReleaseAuthorization,
			*result.Operation,
		); err != nil {
			return result, err
		}
		result, applied, err := c.reconcileExactRelease(
			ctx,
			input.GroupID,
			result,
			result.Operation.CleanupReplicas,
		)
		if err != nil || !applied {
			return result, err
		}
		return result, nil
	}

	// Close older capacity work and remove surplus before a same-count repair can allocate a missing survivor.
	if !abortCleanupComplete(*result.Operation, result.Capacity, result.Topology) {
		result.OperationChanged = clearCapacityTargetProof(result.Operation) || result.OperationChanged
		authorization, err := c.buildExactReleaseAuthorization(
			result.Operation.ID,
			result.Topology.ReplicaCount(),
			result.Operation.CleanupReplicas,
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
	case CapacityReleasePhaseAbsent, CapacityReleasePhaseRefused:
		result.OperationChanged = clearCapacityTargetProof(result.Operation) || result.OperationChanged
		result.ReleaseAuthorization = nil
		result.ReleaseAuthorizationChanged = true
		return result, nil
	case CapacityReleasePhaseApplying, CapacityReleasePhaseApplied:
		result, _, err := c.reconcileObservedExactRelease(
			ctx,
			groupID,
			result,
			result.Operation.CleanupReplicas,
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
	cleanupReplicas := discoverAbortCleanupReplicas(
		*result.Operation,
		result.Capacity,
		result.Topology,
	)
	if err := validateAbortCleanupTopology(*result.Operation, result.Topology); err != nil {
		return result, err
	}
	if !sameReplicaIDs(cleanupReplicas, result.Operation.CleanupReplicas) ||
		!abortCleanupComplete(*result.Operation, result.Capacity, result.Topology) ||
		!abortCleanupTrafficReady(*result.Operation, result.Topology, result.Traffic) {
		result.Operation.CleanupReplicas = cleanupReplicas
		clearCapacityTargetProof(result.Operation)
		result.Operation = transitionOperation(
			result.Operation,
			OperationPhaseAborting,
			c.operations.now(),
		)
		result.OperationChanged = true
		return result, nil
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

	// The stale plan cannot be replayed against a different base; cardinal growth may be replanned automatically.
	plan := input.Plan
	if plan != nil && operationCarriesPlan(*input.Operation, *plan) {
		plan = nil
	}
	operationResult, err := c.operations.Step(ctx, OperationInput{
		GroupID:         input.GroupID,
		SpecGeneration:  input.SpecGeneration,
		DesiredReplicas: input.DesiredReplicas,
		Plan:            plan,
	})
	if err != nil {
		return result, err
	}
	result.Operation = operationResult.Operation
	result.Topology = operationResult.Topology
	result.OperationChanged = operationResult.OperationChanged
	return result, nil
}

func (c *WorkflowCoordinator) restoreAuthoritativeServingState(
	ctx context.Context,
	groupID GroupID,
	result ReconcileResult,
) (ReconcileResult, bool, error) {
	activeReplicas := topologyReplicaIDs(result.Topology)

	// Restore complete physical capacity before making every current authoritative member routable again.
	if !replicaAllocationsAvailable(activeReplicas, result.Capacity) {
		request := CapacityRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			TargetReplicas:     int32(len(activeReplicas)),
			RequiredReplicas:   activeReplicas,
		}
		if err := c.capacity.EnsureCapacity(ctx, groupID, request); err != nil {
			return result, false, fmt.Errorf("restore aborted Engine Group capacity: %w", err)
		}
		return result, false, nil
	}

	// A traffic observation correlated to this operation proves compensation survived timeout or restart.
	if trafficCompensationComplete(*result.Operation, result.Topology, result.Traffic) {
		return result, true, nil
	}
	if err := c.traffic.Admit(
		ctx,
		groupID,
		TrafficRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			Replicas:           activeReplicas,
		},
	); err != nil {
		return result, false, fmt.Errorf("restore traffic after aborted membership operation: %w", err)
	}
	return result, false, nil
}

func discoverAbortCleanupReplicas(
	operation Operation,
	capacity CapacitySnapshot,
	topology MembershipTopology,
) []ReplicaID {
	activeReplicas := make(map[ReplicaID]struct{}, len(topology.Replicas))
	for _, replicaID := range topologyReplicaIDs(topology) {
		activeReplicas[replicaID] = struct{}{}
	}

	// Every base identity excluded from authoritative membership needs drain and a logical recreation fence.
	cleanupReplicas := append(
		slices.Clone(operation.CleanupReplicas),
		differenceReplicaIDs(operation.BaseReplicas, topologyReplicaIDs(topology))...,
	)

	// Every other allocation outside authoritative membership is surplus capacity owned by the aborted operation.
	for _, allocation := range capacity.Allocations {
		if _, active := activeReplicas[allocation.ID]; !active {
			cleanupReplicas = append(cleanupReplicas, allocation.ID)
		}
	}
	return normalizeUniqueReplicaIDs(cleanupReplicas)
}

func validateAbortCleanupTopology(operation Operation, topology MembershipTopology) error {
	if overlap := intersectReplicaIDs(operation.CleanupReplicas, topologyReplicaIDs(topology)); len(overlap) != 0 {
		return fmt.Errorf("abort cleanup replicas became active in authoritative topology: %v", overlap)
	}
	return nil
}

func abortCleanupDrainReplicas(operation Operation, topology MembershipTopology) []ReplicaID {
	return differenceReplicaIDs(operation.BaseReplicas, topologyReplicaIDs(topology))
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
	return traffic.OperationID == operation.ID &&
		containsAllReplicaIDs(traffic.Drained, drainReplicas) &&
		len(intersectReplicaIDs(traffic.Admitted, drainReplicas)) == 0
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
		if !slices.Contains(activeReplicas, allocation.ID) {
			return false
		}
	}

	allocations := replicaAllocationByID(capacity)
	fencedReplicas := fencedReplicaIDs(capacity)
	for _, replicaID := range operation.CleanupReplicas {
		if _, exists := allocations[replicaID]; exists {
			return false
		}
		if _, fenced := fencedReplicas[replicaID]; !fenced {
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
	delta := result.Operation.TargetReplicas - int32(len(result.Operation.BaseReplicas))

	// Expansion can only restore its original exact request by recreating missing or unavailable capacity.
	if delta > 0 {
		request := CapacityRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			TargetReplicas:     result.Operation.TargetReplicas,
			RequiredReplicas: normalizeReplicaIDs(append(
				slices.Clone(result.Operation.BaseReplicas),
				result.Operation.JoiningReplicas...,
			)),
		}
		if err := c.capacity.EnsureCapacity(ctx, groupID, request); err != nil {
			return result, fmt.Errorf("restore Engine Group growth capacity: %w", err)
		}
		return result, nil
	}

	// Reduction repeats the same operation-scoped withdrawal until every nominee is observably drained.
	if delta < 0 {
		if err := c.traffic.Withdraw(
			ctx,
			groupID,
			TrafficRequest{
				OperationID:        result.Operation.ID,
				TopologyGeneration: result.Topology.Generation,
				Replicas:           slices.Clone(result.Operation.NominatedReplicas),
			},
		); err != nil {
			return result, fmt.Errorf("resume nominated replica withdrawal: %w", err)
		}
	}
	return result, nil
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
	if topology.Generation != input.Operation.CommittedTopologyGeneration ||
		!operationMatchesCommittedTopology(*input.Operation, topology) {
		result.Operation = transitionOperation(result.Operation, OperationPhaseUnknown, c.operations.now())
		result.OperationChanged = true
		return result, nil
	}
	if err := validateTrafficAgainstMembership(result.Traffic, result.Topology, input.Operation); err != nil {
		return result, err
	}
	if input.Operation.PostCommitComplete {
		return c.finishCommitted(ctx, input, result)
	}

	delta := input.Operation.TargetReplicas - int32(len(input.Operation.BaseReplicas))

	// Repair every committed allocation before exposing newly active membership to traffic.
	if delta > 0 {
		activeReplicas := topologyReplicaIDs(result.Topology)
		if !replicaAllocationsAvailable(activeReplicas, result.Capacity) {
			request := CapacityRequest{
				OperationID:        input.Operation.ID,
				TopologyGeneration: result.Topology.Generation,
				TargetReplicas:     int32(len(activeReplicas)),
				RequiredReplicas:   activeReplicas,
			}
			if err := c.capacity.EnsureCapacity(ctx, input.GroupID, request); err != nil {
				return result, fmt.Errorf("restore committed growth capacity: %w", err)
			}
			return result, nil
		}

		// Joining replicas become routable only after Committed itself was durable on a prior invocation.
		if trafficAdmissionComplete(*input.Operation, result.Traffic) {
			return c.finishCommitted(ctx, input, result)
		}
		if err := c.traffic.Admit(
			ctx,
			input.GroupID,
			TrafficRequest{
				OperationID:        input.Operation.ID,
				TopologyGeneration: result.Topology.Generation,
				Replicas:           slices.Clone(input.Operation.JoiningReplicas),
			},
		); err != nil {
			return result, fmt.Errorf("admit committed joining replicas: %w", err)
		}
		return result, nil
	}

	// Retired replicas remain allocated until drain evidence and exact release authorization are durable.
	if delta < 0 {
		if !retirementTrafficReady(*input.Operation, result.Traffic) {
			if err := c.traffic.Withdraw(
				ctx,
				input.GroupID,
				TrafficRequest{
					OperationID:        input.Operation.ID,
					TopologyGeneration: result.Topology.Generation,
					Replicas:           slices.Clone(input.Operation.NominatedReplicas),
				},
			); err != nil {
				return result, fmt.Errorf("restore committed retirement drain state: %w", err)
			}
			return result, nil
		}
		return c.reconcileRelease(ctx, input, result)
	}

	return c.finishCommitted(ctx, input, result)
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
		// A later survivor topology needs its own adopted recovery before current identities can be repaired safely.
		if input.Operation.Phase == OperationPhaseUnknown &&
			result.Topology.Generation > input.Operation.CommittedTopologyGeneration {
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
	activeReplicas := topologyReplicaIDs(result.Topology)
	if !replicaAllocationsAvailable(activeReplicas, result.Capacity) {
		request := CapacityRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			TargetReplicas:     int32(len(activeReplicas)),
			RequiredReplicas:   normalizeReplicaIDs(activeReplicas),
		}
		if err := c.capacity.EnsureCapacity(ctx, input.GroupID, request); err != nil {
			return result, fmt.Errorf("restore committed Engine Group capacity: %w", err)
		}
		return result, nil
	}
	if !containsAllReplicaIDs(result.Traffic.Admitted, activeReplicas) {
		request := TrafficRequest{
			OperationID:        result.Operation.ID,
			TopologyGeneration: result.Topology.Generation,
			Replicas:           activeReplicas,
		}
		if err := c.traffic.Admit(ctx, input.GroupID, request); err != nil {
			return result, fmt.Errorf("admit committed Engine Group topology: %w", err)
		}
		return result, nil
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
	operationResult, err := c.operations.Step(ctx, OperationInput{
		GroupID:         input.GroupID,
		SpecGeneration:  input.SpecGeneration,
		DesiredReplicas: input.DesiredReplicas,
		Plan:            plan,
	})
	if err != nil {
		return result, err
	}
	result.Operation = operationResult.Operation
	result.Topology = operationResult.Topology
	result.OperationChanged = operationResult.OperationChanged
	return result, nil
}

func resolveJoiningReplicas(
	operation Operation,
	capacity CapacitySnapshot,
	traffic TrafficSnapshot,
) ([]ReplicaID, bool, error) {
	// Index the immutable base before classifying capacity as retained or joining.
	baseReplicas := make(map[ReplicaID]struct{}, len(operation.BaseReplicas))
	for _, replicaID := range operation.BaseReplicas {
		baseReplicas[replicaID] = struct{}{}
	}
	allocations := replicaAllocationByID(capacity)
	fencedReplicas := fencedReplicaIDs(capacity)

	// Every retained replica must still have a complete available allocation before membership can grow.
	for _, replicaID := range operation.BaseReplicas {
		allocation, exists := allocations[replicaID]
		_, fenced := fencedReplicas[replicaID]
		if !exists || fenced || allocation.Availability != ReplicaAvailabilityAvailable {
			return nil, false, nil
		}
	}

	// Premature traffic admission is an invariant violation, not a signal to choose different capacity.
	candidates := make([]ReplicaID, 0)
	for _, allocation := range capacity.Allocations {
		if _, active := baseReplicas[allocation.ID]; active {
			continue
		}
		if _, fenced := fencedReplicas[allocation.ID]; fenced {
			continue
		}
		if slices.Contains(traffic.Admitted, allocation.ID) {
			return nil, false, fmt.Errorf("uncommitted replica %q is already admitted to traffic", allocation.ID)
		}
		if allocation.Availability == ReplicaAvailabilityAvailable {
			candidates = append(candidates, allocation.ID)
		}
	}
	slices.Sort(candidates)

	needed := int(operation.TargetReplicas) - len(operation.BaseReplicas)
	if needed < 0 {
		return nil, false, errors.New("joining replicas requested for a reducing operation")
	}
	if len(capacity.Allocations) < int(operation.TargetReplicas) || len(candidates) < needed {
		return nil, false, nil
	}
	return slices.Clone(candidates[:needed]), true, nil
}

func submissionPrerequisitesReady(
	operation Operation,
	capacity CapacitySnapshot,
	traffic TrafficSnapshot,
) bool {
	delta := operation.TargetReplicas - int32(len(operation.BaseReplicas))
	if delta > 0 {
		allocations := replicaAllocationByID(capacity)
		fencedReplicas := fencedReplicaIDs(capacity)
		for _, replicaID := range append(
			slices.Clone(operation.BaseReplicas),
			operation.JoiningReplicas...,
		) {
			allocation, exists := allocations[replicaID]
			_, fenced := fencedReplicas[replicaID]
			if !exists || fenced || allocation.Availability != ReplicaAvailabilityAvailable {
				return false
			}
			if slices.Contains(traffic.Admitted, replicaID) &&
				slices.Contains(operation.JoiningReplicas, replicaID) {
				return false
			}
		}
		return true
	}
	if delta < 0 {
		return retirementTrafficReady(operation, traffic)
	}
	return true
}

func retirementTrafficReady(operation Operation, traffic TrafficSnapshot) bool {
	return traffic.OperationID == operation.ID &&
		containsAllReplicaIDs(traffic.Drained, operation.NominatedReplicas) &&
		len(intersectReplicaIDs(traffic.Admitted, operation.NominatedReplicas)) == 0
}

func trafficAdmissionComplete(operation Operation, traffic TrafficSnapshot) bool {
	return traffic.OperationID == operation.ID &&
		containsAllReplicaIDs(traffic.Admitted, operation.JoiningReplicas)
}

func trafficCompensationComplete(
	operation Operation,
	topology MembershipTopology,
	traffic TrafficSnapshot,
) bool {
	activeReplicas := topologyReplicaIDs(topology)
	// An empty authoritative topology has no survivor whose admission must prove an explicit compensation call.
	if len(activeReplicas) == 0 {
		return len(traffic.Admitted) == 0
	}
	return traffic.OperationID == operation.ID &&
		containsAllReplicaIDs(traffic.Admitted, activeReplicas) &&
		len(intersectReplicaIDs(traffic.Drained, activeReplicas)) == 0
}

func validateTrafficAgainstMembership(
	traffic TrafficSnapshot,
	topology MembershipTopology,
	operation *Operation,
) error {
	// A committed reduction may need to withdraw a retired identity that was admitted or recreated unexpectedly.
	allowedAdmissions := topologyReplicaIDs(topology)
	if operation != nil &&
		operation.Phase == OperationPhaseCommitted &&
		operation.CommittedTopologyGeneration == topology.Generation &&
		operationMatchesCommittedTopology(*operation, topology) &&
		operation.TargetReplicas < int32(len(operation.BaseReplicas)) {
		allowedAdmissions = append(allowedAdmissions, operation.NominatedReplicas...)
	}
	if operation != nil &&
		(operation.Phase == OperationPhaseAborting || operation.Phase == OperationPhaseAborted) {
		allowedAdmissions = append(allowedAdmissions, abortCleanupDrainReplicas(*operation, topology)...)
	}

	return validateTrafficAgainstReplicaSet(traffic, allowedAdmissions, operation)
}

func validateTrafficAgainstReplicaSet(
	traffic TrafficSnapshot,
	allowedAdmissions []ReplicaID,
	operation *Operation,
) error {
	// Runtime admission rejects every identity outside committed membership or an explicit retiring set.
	if err := requireReplicaSubset(traffic.Admitted, allowedAdmissions, "traffic-admitted"); err != nil {
		return err
	}
	if len(traffic.Drained) == 0 {
		return nil
	}

	// Historical operation-scoped drain evidence is not evidence for the current operation.
	if operation == nil || traffic.OperationID != operation.ID {
		return nil
	}

	// Current operation-scoped drain state may contain planned victims and abort-cleanup replicas that once served.
	allowedDrains := slices.Clone(operation.NominatedReplicas)
	if operation.Phase == OperationPhaseAborting || operation.Phase == OperationPhaseAborted {
		allowedDrains = append(allowedDrains, operation.BaseReplicas...)
	}
	return requireReplicaSubset(traffic.Drained, normalizeUniqueReplicaIDs(allowedDrains), "traffic-drained")
}

func postCommitWorkComplete(
	operation Operation,
	authorization *ReleaseAuthorization,
	capacity CapacitySnapshot,
	traffic TrafficSnapshot,
) bool {
	// Only a previously committed topology can carry durable post-commit completion evidence.
	if operation.CommittedTopologyGeneration == 0 ||
		(operation.Phase != OperationPhaseCommitted && operation.Phase != OperationPhaseUnknown) {
		return false
	}
	if operation.Adopted {
		return false
	}

	delta := operation.TargetReplicas - int32(len(operation.BaseReplicas))
	if delta > 0 {
		return trafficAdmissionComplete(operation, traffic)
	}
	if delta == 0 {
		return true
	}

	// A reduction is complete only after no authorization remains, every victim is fenced, and none is admitted.
	if authorization != nil || len(intersectReplicaIDs(traffic.Admitted, operation.NominatedReplicas)) != 0 {
		return false
	}
	if !operation.CapacityTargetApplied ||
		operation.CapacityTargetReplicas != operation.TargetReplicas ||
		operation.CapacityTopologyGeneration != operation.CommittedTopologyGeneration {
		return false
	}
	if !releaseFencesComplete(operation, capacity) {
		return false
	}
	return true
}

func releaseFencesComplete(operation Operation, capacity CapacitySnapshot) bool {
	allocations := replicaAllocationByID(capacity)
	fencedReplicas := fencedReplicaIDs(capacity)

	// Every nominated identity must be absent and durably fenced against autonomous recreation.
	for _, replicaID := range operation.NominatedReplicas {
		if _, exists := allocations[replicaID]; exists {
			return false
		}
		if _, fenced := fencedReplicas[replicaID]; !fenced {
			return false
		}
	}
	return true
}

func replicaAllocationsAvailable(replicaIDs []ReplicaID, capacity CapacitySnapshot) bool {
	allocations := replicaAllocationByID(capacity)
	fencedReplicas := fencedReplicaIDs(capacity)
	for _, replicaID := range replicaIDs {
		allocation, exists := allocations[replicaID]
		_, fenced := fencedReplicas[replicaID]
		if !exists || fenced || allocation.Availability != ReplicaAvailabilityAvailable {
			return false
		}
	}
	return true
}

func fencedReplicaIDs(capacity CapacitySnapshot) map[ReplicaID]struct{} {
	fencedReplicas := make(map[ReplicaID]struct{}, len(capacity.FencedReplicas))
	for _, replicaID := range capacity.FencedReplicas {
		fencedReplicas[replicaID] = struct{}{}
	}
	return fencedReplicas
}

func cleanupTopologySafe(operation Operation, topology MembershipTopology) (bool, error) {
	if topology.Generation < operation.CommittedTopologyGeneration {
		return false, nil
	}
	expectedReplicas := expectedCommittedReplicaIDs(operation)
	observedReplicas := topologyReplicaIDs(topology)
	if topology.Generation == operation.CommittedTopologyGeneration {
		if !sameReplicaIDs(observedReplicas, expectedReplicas) {
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
	return true, nil
}

func (c *WorkflowCoordinator) buildReleaseAuthorization(
	operation Operation,
	capacity CapacitySnapshot,
	proofTopology MembershipTopology,
) (*ReleaseAuthorization, error) {
	if proofTopology.Generation < operation.CommittedTopologyGeneration {
		return nil, fmt.Errorf(
			"release proof topology generation %d precedes committed generation %d",
			proofTopology.Generation,
			operation.CommittedTopologyGeneration,
		)
	}
	return c.buildExactReleaseAuthorization(
		operation.ID,
		proofTopology.ReplicaCount(),
		operation.NominatedReplicas,
		capacity,
		proofTopology.Generation,
	)
}

func (c *WorkflowCoordinator) buildExactReleaseAuthorization(
	operationID string,
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
			authorizedReplicas = append(authorizedReplicas, AuthorizedReplica{ReplicaID: replicaID})
			continue
		}
		capacityRefs := slices.Clone(allocation.CapacityRefs)
		slices.SortFunc(capacityRefs, compareCapacityRefs)
		authorizedReplicas = append(authorizedReplicas, AuthorizedReplica{
			ReplicaID:    replicaID,
			SlotID:       allocation.SlotID,
			CapacityRefs: capacityRefs,
		})
	}
	releaseID := c.newReleaseID()
	if releaseID == "" {
		return nil, errors.New("generate capacity release ID: empty value")
	}
	return &ReleaseAuthorization{
		ID:                 releaseID,
		OperationID:        operationID,
		TopologyGeneration: proofTopologyGeneration,
		TargetReplicas:     targetReplicas,
		Replicas:           authorizedReplicas,
	}, nil
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
		authorizedIDs := make([]ReplicaID, len(authorization.Replicas))
		for i, replica := range authorization.Replicas {
			authorizedIDs[i] = replica.ReplicaID
		}
		return requireReplicaSubset(authorizedIDs, operation.CleanupReplicas, "abort-cleanup authorized")
	}

	switch operation.Phase {
	case OperationPhaseCommitted:
		if authorization.TopologyGeneration != operation.CommittedTopologyGeneration {
			return fmt.Errorf(
				"release authorization topology generation %d does not match committed generation %d",
				authorization.TopologyGeneration,
				operation.CommittedTopologyGeneration,
			)
		}
		if authorization.TargetReplicas != operation.TargetReplicas {
			return fmt.Errorf(
				"release authorization target %d does not match committed target %d",
				authorization.TargetReplicas,
				operation.TargetReplicas,
			)
		}
	case OperationPhaseUnknown:
		if authorization.TopologyGeneration < operation.CommittedTopologyGeneration {
			return fmt.Errorf(
				"release authorization topology generation %d precedes committed generation %d",
				authorization.TopologyGeneration,
				operation.CommittedTopologyGeneration,
			)
		}
		if authorization.TopologyGeneration == operation.CommittedTopologyGeneration &&
			authorization.TargetReplicas != operation.TargetReplicas {
			return fmt.Errorf(
				"release authorization target %d does not match committed target %d",
				authorization.TargetReplicas,
				operation.TargetReplicas,
			)
		}
		if authorization.TopologyGeneration > operation.CommittedTopologyGeneration &&
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

func validateAuthorizationAgainstTopology(
	authorization ReleaseAuthorization,
	operation Operation,
	topology MembershipTopology,
) error {
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
		if allocation.SlotID != authorizedReplica.SlotID {
			return fmt.Errorf(
				"release authorization for replica %q is stale: capacity slot changed from %q to %q",
				authorizedReplica.ReplicaID,
				authorizedReplica.SlotID,
				allocation.SlotID,
			)
		}
		for _, capacityRef := range allocation.CapacityRefs {
			binding, authorized := authorizedBindings[capacityRef.Namespace+"/"+capacityRef.Name]
			if !authorized ||
				binding.replicaID != allocation.ID ||
				binding.slotID != allocation.SlotID ||
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
		for _, capacityRef := range allocation.CapacityRefs {
			binding, authorized := authorizedBindings[capacityRef.Namespace+"/"+capacityRef.Name]
			if authorized &&
				(binding.uid != capacityRef.UID ||
					binding.replicaID != allocation.ID ||
					binding.slotID != allocation.SlotID) {
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
		for _, capacityRef := range allocation.CapacityRefs {
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
	fencedReplicas := fencedReplicaIDs(capacity)

	// Applied is complete only when every authorized logical identity has an observable durable fence.
	for _, replica := range authorization.Replicas {
		if _, fenced := fencedReplicas[replica.ReplicaID]; !fenced {
			return false
		}
	}
	return true
}

func compareCapacityRefs(left CapacityRef, right CapacityRef) int {
	if byNamespace := cmp.Compare(left.Namespace, right.Namespace); byNamespace != 0 {
		return byNamespace
	}
	if byName := cmp.Compare(left.Name, right.Name); byName != 0 {
		return byName
	}
	return cmp.Compare(string(left.UID), string(right.UID))
}

func validateReconcileInput(input ReconcileInput) error {
	if err := validateOperationInput(OperationInput{
		GroupID:         input.GroupID,
		SpecGeneration:  input.SpecGeneration,
		DesiredReplicas: input.DesiredReplicas,
		Plan:            input.Plan,
		Operation:       input.Operation,
	}); err != nil {
		return err
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
	if input.Operation.Phase != OperationPhaseCommitted &&
		(input.Operation.Phase != OperationPhaseUnknown ||
			input.Operation.CommittedTopologyGeneration <= input.Operation.BaseTopologyGeneration) {
		return errors.New("release authorization requires a committed or subsequently unknown membership operation")
	}
	if input.Operation.TargetReplicas >= int32(len(input.Operation.BaseReplicas)) {
		return errors.New("release authorization requires a committed reduction")
	}
	if input.Operation.PostCommitComplete {
		return errors.New("post-commit-complete operation must not carry a release authorization")
	}
	return validateAuthorizationForOperation(*input.ReleaseAuthorization, *input.Operation)
}
