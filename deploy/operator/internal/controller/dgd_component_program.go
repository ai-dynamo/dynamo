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

package controller

import (
	"context"
	"errors"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type componentProgram struct {
	sharedResources *dgdSharedResourcesReconciler
	rollout         *dgdWorkerRolloutReconciler
	restart         *dgdRestartReconciler
	restartProgress *componentRestartProgressResolver
	workloads       *componentWorkloadsReconciler
	scalingAdapters *dgdScalingAdaptersReconciler
	lwsEnabled      bool
}

// newComponentProgram wires the DCD pathway at the DGD composition root.
func (r *DynamoGraphDeploymentReconciler) newComponentProgram() *componentProgram {
	rollout := newDGDWorkerRolloutReconciler(r.Client, r.Recorder)
	return &componentProgram{
		sharedResources: newDGDSharedResourcesReconciler(
			r.Client,
			r.Recorder,
			r.Config,
			r.RuntimeConfig,
			r.RestConfig,
			r.DockerSecretRetriever,
			r.SSHKeyManager,
			r.RBACManager,
		),
		rollout:         rollout,
		restart:         newDGDRestartReconciler(),
		restartProgress: newComponentRestartProgressResolver(r.Client),
		workloads:       newComponentWorkloadsReconciler(r.Client, r.Recorder, rollout),
		scalingAdapters: newDGDScalingAdaptersReconciler(r.Client, r.Recorder),
		lwsEnabled:      r.RuntimeConfig.Gate.Enabled(features.LWS),
	}
}

// Reconcile composes the complete component pathway. Each earlier operation
// returns a typed value consumed by later operations. Non-status DGD changes
// are persisted through req.DGD; status accumulates in the returned result.
func (p *componentProgram) Reconcile(
	ctx context.Context,
	req workloadProgramRequest,
) (programResult workloadProgramResult, retErr error) {
	programResult = newWorkloadProgramResult(req.DGD)
	clearComponentGPUShapes(programResult.Status.Components)
	defer func() {
		if retErr == nil {
			return
		}
		reason := reasonFailedToReconcileResources
		if classified, ok := workloadProgramFailureReason(retErr); ok {
			reason = classified
		}
		programResult.Fail(req.DGD.Generation, reason, retErr)
	}()
	log.FromContext(ctx).Info(
		"Reconciling Dynamo components deployments",
		"hasMultinode", req.DGD.HasAnyMultinodeComponent(),
		"lwsEnabled", p.lwsEnabled,
	)

	previousRolloutPhase := rollingUpdatePhase(programResult.Status.RollingUpdate)
	rollingUpdateCtx, err := p.reconcileWorkerRollout(ctx, req.DGD, &programResult.Status)
	if err != nil {
		return programResult, err
	}
	p.recordRollingUpdateTransition(previousRolloutPhase, rollingUpdateCtx, &programResult)
	checkpoints, err := p.sharedResources.Reconcile(ctx, req.DGD, rollingUpdateCtx.WorkerHashByComponent)
	if checkpoints.Statuses != nil {
		programResult.Status.Checkpoints = checkpoints.Statuses
	}
	if err != nil {
		return programResult, err
	}
	hasMultinode := req.DGD.HasAnyMultinodeComponent()
	if hasMultinode && !p.lwsEnabled {
		err := fmt.Errorf("no multinode orchestrator available")
		log.FromContext(ctx).Error(
			err,
			err.Error(),
			"hasMultinode", hasMultinode,
			"lwsEnabled", p.lwsEnabled,
		)
		return programResult, failWorkloadProgram(reasonNoMultinodeOrchestrator, err)
	}
	previousRestart := programResult.Status.Restart
	restart := p.restart.Resolve(
		ctx,
		req.DGD,
		&programResult.Status,
		func(restartCtx context.Context, restartDGD *nvidiacomv1beta1.DynamoGraphDeployment, inProgress []string) []string {
			return p.restartProgress.ResolveWithRollingUpdateContext(restartCtx, restartDGD, inProgress, rollingUpdateCtx)
		},
	)
	recordRestartTransition(previousRestart, restart.Status, &programResult)
	programResult.Status.Restart = restart.Status

	result, err := p.workloads.Reconcile(
		ctx,
		req.DGD,
		restart.State,
		checkpoints.Infos,
		rollingUpdateCtx,
	)
	if err != nil {
		// Preserve newly observed component status while leaving the generation unobserved.
		if result.ComponentStatus != nil {
			programResult.Status.Components = result.ComponentStatus
		}
		return programResult, fmt.Errorf("failed to reconcile Dynamo components deployments: %w", err)
	}
	result = applyCheckpointStartupReadiness(result, checkpoints.Infos)
	if result.State != nvidiacomv1beta1.DGDStatePending || result.Reason != reasonWaitingForCheckpoint {
		if err := p.scalingAdapters.Reconcile(ctx, req.DGD); err != nil {
			log.FromContext(ctx).Error(err, "Failed to reconcile scaling adapters")
			return programResult, fmt.Errorf("failed to reconcile scaling adapters: %w", err)
		}
	}

	programResult.applyReconcileResult(req.DGD.Generation, result)
	return programResult, nil
}

func (p *componentProgram) reconcileWorkerRollout(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
) (dynamo.RollingUpdateContext, error) {
	if supportsManagedRollingUpdate(dgd) {
		return p.reconcileManagedWorkerRollout(ctx, dgd, status)
	}
	if err := p.rollout.migrateCurrentWorkerHashIfNeeded(ctx, dgd); err != nil {
		return dynamo.RollingUpdateContext{}, failWorkloadProgram(reasonFailedToMigrateWorkerHash, err)
	}
	if err := p.rollout.ReconcileUnsupported(ctx, dgd, false); err != nil {
		return dynamo.RollingUpdateContext{}, err
	}
	return dynamo.RollingUpdateContext{}, nil
}

// supportsManagedRollingUpdate checks whether the component pathway can use
// operator-managed rolling updates. Multinode component workloads rely on
// external orchestration and retain the compatibility hash behavior instead.
func supportsManagedRollingUpdate(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) bool {
	return !dgd.HasAnyMultinodeComponent()
}

func (p *componentProgram) reconcileManagedWorkerRollout(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
) (dynamo.RollingUpdateContext, error) {
	plan, err := p.rollout.buildManagedWorkerRolloutPlan(ctx, dgd)
	if err != nil {
		var collision *workerDCDIdentityCollisionError
		if errors.As(err, &collision) {
			return dynamo.RollingUpdateContext{}, failWorkloadProgram(reasonDCDIdentityCollision, err)
		}
		return dynamo.RollingUpdateContext{}, failWorkloadProgram(reasonRollingUpdateFailed, err)
	}
	if err := p.rollout.reconcileManagedWorkerInventory(ctx, dgd, status, plan); err != nil {
		return plan.context, failWorkloadProgram(reasonRollingUpdateFailed, err)
	}
	return plan.context, nil
}

func (p *componentProgram) recordRollingUpdateTransition(
	previous nvidiacomv1beta1.RollingUpdatePhase,
	rollingUpdateCtx dynamo.RollingUpdateContext,
	result *workloadProgramResult,
) {
	current := rollingUpdatePhase(result.Status.RollingUpdate)
	switch {
	case current == nvidiacomv1beta1.RollingUpdatePhasePending && previous != current:
		result.Eventf(
			corev1.EventTypeNormal,
			"RollingUpdateStarted",
			"Starting rolling update to worker hash %s",
			rollingUpdateCtx.NewWorkerHash,
		)
	case current == nvidiacomv1beta1.RollingUpdatePhaseCompleted && previous != current:
		result.Eventf(
			corev1.EventTypeNormal,
			"RollingUpdateCompleted",
			"Rolling update completed, worker hash %s",
			rollingUpdateCtx.NewWorkerHash,
		)
	}
}
