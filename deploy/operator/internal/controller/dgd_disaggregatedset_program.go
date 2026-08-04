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
	corev1 "k8s.io/api/core/v1"
	apiMeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type disaggregatedSetProgram struct {
	owner           disaggregatedSetProgramOwner
	sharedResources *dgdSharedResourcesReconciler
	rollout         *dgdWorkerRolloutReconciler
	restart         *dgdRestartReconciler
	workloads       *disaggregatedSetWorkloadsReconciler
	scalingAdapters *dgdScalingAdaptersReconciler
}

type disaggregatedSetProgramOwner interface {
	shouldUseDisaggregatedSet(dgd *nvidiacomv1beta1.DynamoGraphDeployment) (bool, string)
	newComponentProgram() *componentProgram
	restoreDisaggregatedSetServiceOwnershipToDCDs(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) error
	deleteDisaggregatedSetIfExists(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) error
}

func (r *DynamoGraphDeploymentReconciler) newDisaggregatedSetProgram() *disaggregatedSetProgram {
	rollout := newDGDWorkerRolloutReconciler(r.Client, r.Recorder)
	return &disaggregatedSetProgram{
		owner: r,
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
		workloads:       newDisaggregatedSetWorkloadsReconciler(r, rollout),
		scalingAdapters: newDGDScalingAdaptersReconciler(r.Client, r.Recorder),
	}
}

func (p *disaggregatedSetProgram) Reconcile(
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

	useDS, fallbackReason := p.owner.shouldUseDisaggregatedSet(req.DGD)
	if !useDS {
		fallbackProgram := p.owner.newComponentProgram()
		programResult, retErr = fallbackProgram.Reconcile(ctx, req)
		if fallbackReason != "" {
			programResult.Eventf(
				corev1.EventTypeWarning,
				"DisaggregatedSetFallback",
				"DisaggregatedSet requested but falling back to DynamoComponentDeployments: %s",
				fallbackReason,
			)
		}
		if retErr != nil {
			return programResult, retErr
		}
		setDisaggregatedSetEligibilityCondition(&programResult, req.DGD.Generation, metav1.ConditionFalse, "FallbackToComponentProgram", fallbackReason)
		if err := p.owner.restoreDisaggregatedSetServiceOwnershipToDCDs(ctx, req.DGD); err != nil {
			retErr = fmt.Errorf("failed to restore DisaggregatedSet service ownership: %w", err)
			return programResult, retErr
		}
		if err := p.owner.deleteDisaggregatedSetIfExists(ctx, req.DGD); err != nil {
			retErr = fmt.Errorf("failed to delete stale DisaggregatedSet: %w", err)
			return programResult, retErr
		}
		return programResult, nil
	}
	setDisaggregatedSetEligibilityCondition(&programResult, req.DGD.Generation, metav1.ConditionTrue, "Selected", "DisaggregatedSet pathway selected")

	if err := p.reconcileWorkerRollout(ctx, req.DGD, &programResult.Status); err != nil {
		return programResult, err
	}

	checkpoints, err := p.sharedResources.Reconcile(ctx, req.DGD)
	if checkpoints.Statuses != nil {
		programResult.Status.Checkpoints = checkpoints.Statuses
	}
	if err != nil {
		return programResult, err
	}

	previousRestart := programResult.Status.Restart
	restart := p.workloads.ResolveRestart(ctx, req.DGD, &programResult.Status, p.restart)
	recordRestartTransition(previousRestart, restart.Status, &programResult)
	programResult.Status.Restart = restart.Status

	result, err := p.workloads.Reconcile(ctx, req.DGD, restart.State, checkpoints.Infos)
	if err != nil {
		return programResult, fmt.Errorf("failed to reconcile DisaggregatedSet pathway: %w", err)
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

func setDisaggregatedSetEligibilityCondition(
	result *workloadProgramResult,
	generation int64,
	status metav1.ConditionStatus,
	reason string,
	message string,
) {
	apiMeta.SetStatusCondition(&result.Status.Conditions, metav1.Condition{
		Type:               "DisaggregatedSetEligible",
		Status:             status,
		ObservedGeneration: generation,
		Reason:             reason,
		Message:            message,
	})
}

func (p *disaggregatedSetProgram) reconcileWorkerRollout(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
) error {
	if err := p.rollout.migrateCurrentWorkerHashIfNeeded(ctx, dgd); err != nil {
		log.FromContext(ctx).Error(err, "Failed to migrate worker hash")
		return failWorkloadProgram(reasonFailedToMigrateWorkerHash, err)
	}
	return p.rollout.ReconcileUnsupported(ctx, dgd, false)
}
