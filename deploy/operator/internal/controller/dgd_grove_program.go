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
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

type groveProgram struct {
	sharedResources *dgdSharedResourcesReconciler
	rollout         *dgdWorkerRolloutReconciler
	restart         *dgdRestartReconciler
	restartProgress *groveRestartProgressResolver
	workloads       *groveWorkloadsReconciler
	scalingAdapters *dgdScalingAdaptersReconciler
	topology        *dgdGroveTopologyConditionReconciler
	gate            features.Gate
}

// newGroveProgram wires the Grove pathway at the DGD composition root.
func (r *DynamoGraphDeploymentReconciler) newGroveProgram() *groveProgram {
	rollout := newDGDWorkerRolloutReconciler(r.Client, r.Recorder)
	return &groveProgram{
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
		restartProgress: newGroveRestartProgressResolver(r.Client),
		workloads: newGroveWorkloadsReconciler(
			r.Client,
			r.Recorder,
			rollout,
			r.Config,
			r.RuntimeConfig,
			r.DockerSecretRetriever,
		),
		scalingAdapters: newDGDScalingAdaptersReconciler(r.Client, r.Recorder),
		topology:        newDGDGroveTopologyConditionReconciler(r.Client),
		gate:            r.RuntimeConfig.Gate,
	}
}

// Reconcile composes the complete Grove pathway. Each earlier operation
// returns a typed value consumed by later operations. Non-status DGD changes
// are persisted through req.DGD; status accumulates in the returned result.
func (p *groveProgram) Reconcile(
	ctx context.Context,
	req workloadProgramRequest,
) (programResult workloadProgramResult, retErr error) {
	programResult = newWorkloadProgramResult(req.DGD)

	// Fail a durable Grove selection when Grove is unavailable rather than falling back.
	if !p.gate.Enabled(features.Grove) {
		err := failWorkloadProgram(
			reasonSelectedWorkloadProviderUnavailable,
			fmt.Errorf("selected workload provider %q is unavailable because Grove is disabled", workloadProviderGrove),
		)
		programResult.Fail(req.DGD.Generation, reasonSelectedWorkloadProviderUnavailable, err)
		return programResult, reconcile.TerminalError(err)
	}

	defer func() {
		if retErr != nil {
			reason := reasonFailedToReconcileResources
			if classified, ok := workloadProgramFailureReason(retErr); ok {
				reason = classified
			}
			programResult.Fail(req.DGD.Generation, reason, retErr)
		}
		p.topology.Reconcile(ctx, req.DGD, &programResult)
	}()
	log.FromContext(ctx).Info(
		"Reconciling Grove resources",
		"hasMultinode", req.DGD.HasAnyMultinodeComponent(),
	)

	if err := p.rollout.migrateCurrentWorkerHashIfNeeded(ctx, req.DGD); err != nil {
		log.FromContext(ctx).Error(err, "Failed to migrate worker hash")
		return programResult, failWorkloadProgram(reasonFailedToMigrateWorkerHash, err)
	}
	checkpoints, err := p.sharedResources.Reconcile(ctx, req.DGD)
	if checkpoints.Statuses != nil {
		programResult.Status.Checkpoints = checkpoints.Statuses
	}
	if err != nil {
		return programResult, err
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
		return programResult, fmt.Errorf("failed to reconcile Grove workloads: %w", err)
	}
	result = applyCheckpointStartupReadiness(result, checkpoints.Infos)
	if result.State != nvidiacomv1beta1.DGDStatePending || result.Reason != reasonWaitingForCheckpoint {
		if err := p.scalingAdapters.Reconcile(ctx, req.DGD); err != nil {
			log.FromContext(ctx).Error(err, "Failed to reconcile scaling adapters")
			return programResult, fmt.Errorf("failed to reconcile scaling adapters: %w", err)
		}
	}
	if result.State == nvidiacomv1beta1.DGDStateSuccessful && p.legacyCleanup != nil {
		if err := p.legacyCleanup.Reconcile(ctx, req.DGD); err != nil {
			return programResult, err
		}
		apiMeta.RemoveStatusCondition(&programResult.Status.Conditions, disaggregatedSetEligibleConditionType)
	}

	programResult.applyReconcileResult(req.DGD.Generation, result)
	return programResult, nil
}
