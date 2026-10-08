/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package enginegroup

import (
	"context"
	"errors"
	"fmt"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	domain "github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
)

func (r *Reconciler) reconcileEngineGroupRetirement(
	ctx context.Context,
	group *api.DynamoGraphDeploymentEngineGroup,
	runtime Runtime,
	checkpoint engineGroupCheckpoint,
	snapshot kubejournal.Snapshot,
) (ctrl.Result, error) {
	// Deletion changes the workflow target, not spec.replicas or its positive scaling bounds.
	before := group.Status.DeepCopy()
	checkpoint.restoreProjectionInputs(group)
	plan, rejection, planningErr := resolveEngineGroupRetirement(ctx, engineGroupID(group), runtime, checkpoint.State)

	// An accepted collective cannot be canceled by deletion. Finish or observe it before starting retirement.
	// A nil plan on inconclusive/unsupported resolution still maintains existing fail-closed accepted targets.
	coordinator := domain.NewCoordinator(runtime.Capacity, runtime.Membership, runtime.Traffic, runtime.Verifier)
	result, reconcileErr := coordinator.Reconcile(ctx, engineGroupID(group), plan, checkpoint.State)
	var observationErr *domain.ObservationError
	errors.As(reconcileErr, &observationErr)
	r.projectEngineGroupStatus(group, runtime.Profile, result.Status, observationErr, planningErr)
	if err := projectEngineGroupOperation(group, result.Status); err != nil {
		group.Status = *before
		return r.reconcileCheckpointFailure(ctx, group, err)
	}

	// Surface unsupported or inconclusive retirement without changing the scale target's validation.
	if planningErr != nil {
		setEngineGroupCondition(group, engineGroupConditionProgressing, metav1.ConditionFalse, "RetirementInconclusive", planningErr.Error())
	} else if rejection != nil {
		setEngineGroupCondition(group, engineGroupConditionProgressing, metav1.ConditionFalse, rejection.Reason, rejection.Message)
	} else if result.Status.Transition == nil || result.Status.Transition.Outcome != domain.TransitionOutcomeBlocked {
		setEngineGroupCondition(group, engineGroupConditionProgressing, metav1.ConditionTrue, "RetirementPending",
			"Retirement must converge to accepted empty capacity, traffic, and membership before finalization")
	}

	// Persist every next level before replaying effects, including after a restart during retirement.
	store := engineGroupCheckpointStore(r, group)
	if err := persistEngineGroupCheckpoint(ctx, store, snapshot, checkpoint, newEngineGroupCheckpoint(group, result.Status)); err != nil {
		group.Status = *before
		return r.reconcileCheckpointFailure(ctx, group, err)
	}
	if err := r.updateEngineGroupStatus(ctx, group, before); err != nil {
		return ctrl.Result{}, err
	}
	return ctrl.Result{RequeueAfter: engineGroupRequeueAfter}, errors.Join(planningErr, reconcileErr)
}

func resolveEngineGroupRetirement(
	ctx context.Context,
	groupID domain.GroupID,
	runtime Runtime,
	status domain.GroupStatus,
) (*domain.ResolvedPlan, *domain.Failure, error) {
	// A previously committed empty topology still needs capacity/traffic convergence, not a new membership mutation.
	base, found := status.Topologies.Current()
	if !found {
		return nil, nil, errors.New("retirement has no authoritative base topology")
	}
	if len(base.Replicas) == 0 {
		return nil, nil, nil
	}

	// Ask the backend to resolve terminal retirement rather than deriving an engine-specific shutdown here.
	resolution, err := runtime.Planner.ResolveScalePlan(ctx, groupID, 0, status)
	if err != nil {
		return nil, nil, fmt.Errorf("resolve Engine Group retirement: %w", err)
	}
	if resolution.Plan != nil && resolution.Rejection != nil {
		return nil, nil, errors.New("retirement resolution contains both a plan and a rejection")
	}
	if rejection := resolution.Rejection; rejection != nil {
		if rejection.Classification != domain.FailureClassificationTerminal || rejection.Reason == "" {
			return nil, nil, errors.New("retirement rejection must be terminal and contain a reason")
		}
		return nil, rejection, nil
	}

	// Only exact full retirement is allowed: never turn deletion into a resize, remap, or new bootstrap.
	plan := resolution.Plan
	if plan == nil || plan.ProfileFingerprint != runtime.Profile.Fingerprint ||
		plan.Change.Kind != domain.PlanKindRetire || plan.Change.Retire == nil ||
		plan.VerificationRequirement != domain.VerificationRequirementNone ||
		len(plan.Change.Retire.Replicas) != len(base.Replicas) {
		return nil, nil, errors.New("retirement requires a profile-matched Retire plan for every committed replica without serving verification")
	}
	retiring := make(map[domain.ReplicaID]bool, len(base.Replicas))
	for _, replicaID := range plan.Change.Retire.Replicas {
		if retiring[replicaID] {
			return nil, nil, errors.New("retirement plan contains duplicate replicas")
		}
		retiring[replicaID] = true
	}

	// Cardinality alone is insufficient: every exact committed logical identity must be selected.
	for _, replica := range base.Replicas {
		if !retiring[replica.ReplicaID] {
			return nil, nil, fmt.Errorf("retirement plan omits committed replica %q", replica.ReplicaID)
		}
	}
	return plan, nil, nil
}
