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
	"slices"
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

type groveProgram struct {
	sharedResources    *dgdSharedResourcesReconciler
	rollout            *dgdWorkerRolloutReconciler
	restart            *dgdRestartReconciler
	restartProgress    *groveRestartProgressResolver
	lpxRestartProgress *lpxRestartProgressResolver
	workloads          *groveWorkloadsReconciler
	scalingAdapters    *dgdScalingAdaptersReconciler
	topology           *dgdGroveTopologyConditionReconciler
	gate               features.Gate
	lpx                *dgdLPXHandoff
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
		rollout:            rollout,
		restart:            newDGDRestartReconciler(),
		restartProgress:    newGroveRestartProgressResolver(r.Client),
		lpxRestartProgress: newLPXRestartProgressResolver(r.Client),
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
		lpx:             &dgdLPXHandoff{client: r.Client},
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
	clearComponentGPUShapes(programResult.Status.Components)

	// Fail a durable Grove selection when Grove is unavailable rather than falling back.
	if !p.gate.Enabled(features.Grove) {
		err := failWorkloadProgram(
			reasonSelectedWorkloadProviderUnavailable,
			fmt.Errorf("selected workload provider %q is unavailable because Grove is disabled", workloadProviderGrove),
		)
		programResult.Fail(req.DGD.Generation, reasonSelectedWorkloadProviderUnavailable, err)
		return programResult, reconcile.TerminalError(err)
	}
	var ordinaryDGD *nvidiacomv1beta1.DynamoGraphDeployment

	defer func() {
		if retErr != nil {
			reason := reasonFailedToReconcileResources
			if classified, ok := workloadProgramFailureReason(retErr); ok {
				reason = classified
			}
			programResult.Fail(req.DGD.Generation, reason, retErr)
		}
		if ordinaryDGD == nil {
			ordinaryDGD = projectWithoutExternallyManagedComponents(req.DGD)
		}
		p.topology.Reconcile(ctx, ordinaryDGD, &programResult)
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
	ordinaryDGD = projectWithoutExternallyManagedComponents(req.DGD)

	previousRestart := programResult.Status.Restart
	restart := p.restart.Resolve(
		ctx,
		req.DGD,
		&programResult.Status,
		func(ctx context.Context, source *nvidiacomv1beta1.DynamoGraphDeployment, inProgress []string) []string {
			return resolveCompositeGroveRestartProgress(
				ctx,
				source,
				ordinaryDGD,
				inProgress,
				p.restartProgress,
				p.lpxRestartProgress,
			)
		},
	)
	recordRestartTransition(previousRestart, restart.Status, &programResult)
	programResult.Status.Restart = restart.Status

	result, err := p.workloads.Reconcile(
		ctx,
		req.DGD,
		ordinaryDGD,
		restart.State,
		checkpoints.Infos,
	)

	if err != nil {
		// Preserve newly observed component status while leaving the generation unobserved.
		if result.ComponentStatus != nil {
			programResult.Status.Components = result.ComponentStatus
		}
		return programResult, fmt.Errorf("failed to reconcile Grove workloads: %w", err)
	}

	if req.DGD.HasLPXComponent() && !apiequality.Semantic.DeepEqual(req.DGD.Status.Restart, restart.Status) {
		// Persist the selected restart before delivering its token to the child.
		programResult.RequeueAfter = time.Nanosecond
		return programResult, nil
	}

	// Keep LPX creation and updates after ordinary reconciliation and restart selection.
	var child *nvidiacomv1alpha1.LPXGraphDeployment
	if req.DGD.HasLPXComponent() {
		child, err = p.lpx.Reconcile(ctx, req.DGD)
		if err != nil {
			return programResult, fmt.Errorf("reconcile LPX child: %w", err)
		}
	}

	result = mergeLPXChildStatus(req.DGD, child, result)
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

// projectWithoutExternallyManagedComponents copies source without externally managed components.
// The source must be non-nil; shared nested data must not be mutated by callers.
func projectWithoutExternallyManagedComponents(source *nvidiacomv1beta1.DynamoGraphDeployment) *nvidiacomv1beta1.DynamoGraphDeployment {
	projected := *source
	projected.Spec.Components = slices.DeleteFunc(
		slices.Clone(projected.Spec.Components),
		func(component nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
			return component.ManagedByExternalController()
		},
	)
	return &projected
}
