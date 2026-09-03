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
	rollout, err := p.reconcileWorkerRollout(ctx, req.DGD, &programResult.Status)
	if err != nil {
		return programResult, err
	}
	p.recordRollingUpdateTransition(previousRolloutPhase, rollout, &programResult)
	workerDCDSuffix := ""
	if rollout != nil {
		workerDCDSuffix = rollout.targetDCDSuffix
	}
	checkpoints, err := p.sharedResources.Reconcile(ctx, req.DGD, workerDCDSuffix)
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
		rollout,
		p.restartProgress.Resolve,
	)
	recordRestartTransition(previousRestart, restart.Status, &programResult)
	programResult.Status.Restart = restart.Status

	result, err := p.workloads.Reconcile(
		ctx,
		req.DGD,
		restart.State,
		checkpoints.Infos,
		rollout,
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
) (*managedWorkerRollout, error) {
	if supportsManagedRollingUpdate(dgd) {
		return p.reconcileManagedWorkerRollout(ctx, dgd, status)
	}

	// LWS has no controller-owned rollout contract. Its DCD and LWS writes are
	// receipts only, so this program does not project them into DGD generation state.
	return nil, nil
}

// supportsManagedRollingUpdate selects the component pathway with a controller-owned
// rollout contract. Multinode component workloads remain provider-owned.
func supportsManagedRollingUpdate(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) bool {
	return !dgd.HasAnyMultinodeComponent()
}

func (p *componentProgram) reconcileManagedWorkerRollout(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
) (*managedWorkerRollout, error) {
	rollout, err := p.rollout.buildManagedWorkerRollout(ctx, dgd)
	if err != nil {
		return nil, failWorkloadProgram(reasonRollingUpdateFailed, err)
	}
	if err := p.rollout.advanceManagedWorkerRollout(ctx, dgd, status, rollout); err != nil {
		return &rollout, failWorkloadProgram(reasonRollingUpdateFailed, err)
	}
	return &rollout, nil
}

func (p *componentProgram) recordRollingUpdateTransition(
	previous nvidiacomv1beta1.RollingUpdatePhase,
	rollout *managedWorkerRollout,
	result *workloadProgramResult,
) {
	if rollout == nil {
		return
	}
	current := rollingUpdatePhase(result.Status.RollingUpdate)
	switch {
	case current == nvidiacomv1beta1.RollingUpdatePhasePending && previous != current:
		result.Eventf(
			corev1.EventTypeNormal,
			"RollingUpdateStarted",
			"Starting rolling update to worker hash %s",
			rollout.desiredV2Hash,
		)
	case current == nvidiacomv1beta1.RollingUpdatePhaseCompleted && previous != current:
		result.Eventf(
			corev1.EventTypeNormal,
			"RollingUpdateCompleted",
			"Rolling update completed, worker hash %s",
			rollout.desiredV2Hash,
		)
	}
}
