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
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
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
	if err := p.reconcileWorkerRollout(ctx, req.DGD, &programResult.Status); err != nil {
		return programResult, err
	}
	p.recordRollingUpdateTransition(req.DGD, previousRolloutPhase, &programResult)
	checkpoints, err := p.sharedResources.Reconcile(ctx, req.DGD)
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
		p.restartProgress.Resolve,
	)
	recordRestartTransition(previousRestart, restart.Status, &programResult)
	programResult.Status.Restart = restart.Status

	result, err := p.workloads.Reconcile(
		ctx,
		req.DGD,
		restart.State,
		checkpoints.Infos,
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
) error {
	if err := p.rollout.migrateCurrentWorkerHashIfNeeded(ctx, dgd); err != nil {
		log.FromContext(ctx).Error(err, "Failed to migrate worker hash")
		return failWorkloadProgram(reasonFailedToMigrateWorkerHash, err)
	}

	if supportsManagedRollingUpdate(dgd) {
		return p.reconcileManagedWorkerRollout(ctx, dgd, status)
	}
	return p.rollout.ReconcileUnsupported(ctx, dgd, false)
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
) error {
	logger := log.FromContext(ctx)

	if err := p.rollout.initializeWorkerHashIfNeeded(ctx, dgd); err != nil {
		logger.Error(err, "Failed to initialize worker hash")
		return failWorkloadProgram(reasonFailedToInitializeWorkerHash, err)
	}

	rollingUpdateInProgress := isRollingUpdateInProgress(status)
	triggerRollingUpdate := false
	if !rollingUpdateInProgress {
		var err error
		triggerRollingUpdate, err = p.rollout.shouldTriggerRollingUpdate(dgd)
		if err != nil {
			logger.Error(err, "Failed to check rolling update trigger")
			return failWorkloadProgram(reasonRollingUpdateFailed, err)
		}
	}
	if rollingUpdateInProgress || triggerRollingUpdate {
		if err := p.rollout.reconcileRollingUpdate(ctx, dgd, status); err != nil {
			logger.Error(err, "Failed to reconcile rolling update")
			return failWorkloadProgram(reasonRollingUpdateFailed, err)
		}
	}
	return nil
}

func (p *componentProgram) recordRollingUpdateTransition(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	previous nvidiacomv1beta1.RollingUpdatePhase,
	result *workloadProgramResult,
) {
	current := rollingUpdatePhase(result.Status.RollingUpdate)
	switch {
	case current == nvidiacomv1beta1.RollingUpdatePhasePending && previous != current:
		desired, err := desiredWorkerHashes(dgd)
		if err != nil {
			return
		}
		result.Eventf(
			corev1.EventTypeNormal,
			"RollingUpdateStarted",
			"Starting rolling update to worker hash %s",
			activeWorkerHashForDCDGeneration(dgd, desired),
		)
	case current == nvidiacomv1beta1.RollingUpdatePhaseCompleted && previous != current:
		currentHashes := currentWorkerHashes(dgd)
		workerHash := currentHashes.v2
		if workerHash == "" {
			workerHash = currentHashes.v1
		}
		result.Eventf(
			corev1.EventTypeNormal,
			"RollingUpdateCompleted",
			"Rolling update completed, worker hash %s",
			workerHash,
		)
	}
}
