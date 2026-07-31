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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type workloadProgramRequest struct {
	// DGD is the mutable primary object. Programs may mutate and directly
	// persist non-status fields; status is returned through workloadProgramResult.
	DGD *nvidiacomv1beta1.DynamoGraphDeployment
}

type workloadProgramEvent struct {
	Type    string
	Reason  string
	Message string
}

type workloadProgramResult struct {
	ctrl.Result
	Status nvidiacomv1beta1.DynamoGraphDeploymentStatus
	Events []workloadProgramEvent
}

type programInputs struct {
	HasMultinode       bool
	CheckpointInfos    map[string]*checkpoint.CheckpointInfo
	CheckpointStatuses map[string]nvidiacomv1beta1.ComponentCheckpointStatus
}

type programRestart struct {
	State  *dynamo.RestartState
	Status *nvidiacomv1beta1.RestartStatus
}

type workloadReconcileRequest struct {
	DGD             *nvidiacomv1beta1.DynamoGraphDeployment
	RestartState    *dynamo.RestartState
	CheckpointInfos map[string]*checkpoint.CheckpointInfo
}

// workloadProgram owns the complete graph-workload state machine for one
// pathway. The common DGD controller selects one program and invokes it once;
// it does not drive provider rendering, rollout, readiness, or cleanup through
// lifecycle callbacks.
type workloadProgram interface {
	Reconcile(context.Context, workloadProgramRequest) (workloadProgramResult, error)
}

func newWorkloadProgramResult(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) workloadProgramResult {
	status := dgd.DeepCopy().Status
	return workloadProgramResult{Status: status}
}

func (r *workloadProgramResult) Eventf(
	eventType string,
	reason string,
	format string,
	args ...any,
) {
	r.Events = append(r.Events, workloadProgramEvent{
		Type:    eventType,
		Reason:  reason,
		Message: fmt.Sprintf(format, args...),
	})
}

func (r *workloadProgramResult) Fail(generation int64, reason Reason, err error) {
	r.Status.State = nvidiacomv1beta1.DGDStateFailed
	meta.SetStatusCondition(&r.Status.Conditions, metav1.Condition{
		Type:               "Ready",
		Status:             metav1.ConditionFalse,
		ObservedGeneration: generation,
		Reason:             string(reason),
		Message:            err.Error(),
	})
}

func (r *workloadProgramResult) applyReconcileResult(
	generation int64,
	result ReconcileResult,
) {
	r.Status.State = result.State
	r.Status.Components = result.ComponentStatus
	r.Status.Restart = result.RestartStatus
	if rollingUpdateInProgress(r.Status.RollingUpdate) {
		r.Status.State = nvidiacomv1beta1.DGDStatePending
	}
	meta.SetStatusCondition(&r.Status.Conditions, readyCondition(generation, r.Status, result))
	r.Status.ObservedGeneration = generation
}

func readyCondition(
	generation int64,
	status nvidiacomv1beta1.DynamoGraphDeploymentStatus,
	workloads ReconcileResult,
) metav1.Condition {
	if rollingUpdateInProgress(status.RollingUpdate) {
		return metav1.Condition{
			Type:               "Ready",
			Status:             metav1.ConditionFalse,
			ObservedGeneration: generation,
			Reason:             "rolling_update_in_progress",
			Message:            "Rolling update in progress",
		}
	}

	conditionStatus := metav1.ConditionFalse
	if workloads.State == nvidiacomv1beta1.DGDStateSuccessful {
		conditionStatus = metav1.ConditionTrue
	}
	return metav1.Condition{
		Type:               "Ready",
		Status:             conditionStatus,
		ObservedGeneration: generation,
		Reason:             string(workloads.Reason),
		Message:            string(workloads.Message),
	}
}

func rollingUpdateInProgress(status *nvidiacomv1beta1.RollingUpdateStatus) bool {
	if status == nil {
		return false
	}
	return status.Phase == nvidiacomv1beta1.RollingUpdatePhasePending ||
		status.Phase == nvidiacomv1beta1.RollingUpdatePhaseInProgress
}

func rollingUpdatePhase(status *nvidiacomv1beta1.RollingUpdateStatus) nvidiacomv1beta1.RollingUpdatePhase {
	if status == nil {
		return nvidiacomv1beta1.RollingUpdatePhaseNone
	}
	return status.Phase
}

type workloadProgramFailure struct {
	reason Reason
	err    error
}

func (e *workloadProgramFailure) Error() string {
	return e.err.Error()
}

func (e *workloadProgramFailure) Unwrap() error {
	return e.err
}

func failWorkloadProgram(reason Reason, err error) error {
	return &workloadProgramFailure{reason: reason, err: err}
}

func workloadProgramFailureReason(err error) (Reason, bool) {
	var failure *workloadProgramFailure
	if !errors.As(err, &failure) {
		return "", false
	}
	return failure.reason, true
}

// groveReconcileFunc is the temporary strangler seam around the existing Grove
// pathway. It disappears when Grove reconciliation moves into groveProgram.
type groveReconcileFunc func(
	context.Context,
	*nvidiacomv1beta1.DynamoGraphDeployment,
	*dynamo.RestartState,
	map[string]*checkpoint.CheckpointInfo,
) (ReconcileResult, error)

type componentProgram struct {
	// The DGD reconciler temporarily supplies shared controller dependencies and
	// managed rolling-update helpers. Later extractions can narrow this without
	// moving component-path orchestration back into the common controller flow.
	reconciler *DynamoGraphDeploymentReconciler
	lwsEnabled bool
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
	inputs, err := p.reconciler.reconcileProgramInputs(ctx, req.DGD)
	if inputs.CheckpointStatuses != nil {
		programResult.Status.Checkpoints = inputs.CheckpointStatuses
	}
	if err != nil {
		return programResult, err
	}
	if inputs.HasMultinode && !p.lwsEnabled {
		err := fmt.Errorf("no multinode orchestrator available")
		log.FromContext(ctx).Error(
			err,
			err.Error(),
			"hasMultinode", inputs.HasMultinode,
			"lwsEnabled", p.lwsEnabled,
		)
		return programResult, failWorkloadProgram(reasonNoMultinodeOrchestrator, err)
	}
	restart := p.reconciler.resolveProgramRestartState(ctx, req.DGD, &programResult.Status, &programResult)
	programResult.Status.Restart = restart.Status

	result, err := p.reconcileWorkloads(ctx, workloadReconcileRequest{
		DGD:             req.DGD,
		RestartState:    restart.State,
		CheckpointInfos: inputs.CheckpointInfos,
	})
	if err != nil {
		return programResult, fmt.Errorf("failed to reconcile Dynamo components deployments: %w", err)
	}
	result, err = p.reconciler.reconcileProgramResult(ctx, req.DGD, inputs, restart, result)
	if err != nil {
		return programResult, err
	}

	programResult.applyReconcileResult(req.DGD.Generation, result)
	return programResult, nil
}

// reconcileWorkloads owns the component pathway's complete DCD graph
// reconciliation.
// Managed rolling-update helpers remain on the DGD reconciler until their
// dedicated extraction; this program owns when they participate in the flow.
func (p *componentProgram) reconcileWorkloads(
	ctx context.Context,
	req workloadReconcileRequest,
) (ReconcileResult, error) {
	r := p.reconciler
	dynamoDeployment := req.DGD
	resources := []Resource{}
	logger := log.FromContext(ctx)

	rollingUpdateCtx, err := r.buildRollingUpdateContext(ctx, dynamoDeployment)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to build rolling update context: %w", err)
	}

	existingRestartAnnotations, err := r.getExistingRestartAnnotationsDCD(ctx, dynamoDeployment)
	if err != nil {
		logger.Error(err, "failed to get existing restart annotations")
		return ReconcileResult{}, fmt.Errorf("failed to get existing restart annotations: %w", err)
	}
	if rollingUpdateCtx.InProgress() {
		logger.Info("Rolling update in progress",
			"newWorkerHash", rollingUpdateCtx.NewWorkerHash,
			"oldWorkerComponentReplicas", rollingUpdateCtx.OldWorkerReplicaTargetsByComponent)
	}

	// Generate all DCDs, including the desired generations during a managed
	// rolling update.
	dynamoComponentsDeployments, err := dynamo.GenerateDynamoComponentsDeployments(
		dynamoDeployment,
		req.RestartState,
		existingRestartAnnotations,
		rollingUpdateCtx,
	)
	if err != nil {
		logger.Error(err, "failed to generate the DynamoComponentsDeployments")
		return ReconcileResult{}, fmt.Errorf("failed to generate the DynamoComponentsDeployments: %w", err)
	}

	// Apply resolved checkpoint policy and synchronize every desired DCD.
	for key, dcd := range dynamoComponentsDeployments {
		if err := p.applyCheckpointStartupPolicy(dcd, req.CheckpointInfos[key]); err != nil {
			return ReconcileResult{}, fmt.Errorf("failed to apply checkpoint startup policy for %s: %w", key, err)
		}
		logger.Info("Reconciling DynamoComponentDeployment", "key", key, "name", dcd.Name)
		if err := p.preserveExistingBackendFramework(ctx, dcd); err != nil {
			logger.Error(err, "failed to preserve existing DynamoComponentDeployment backendFramework", "name", dcd.Name)
			return ReconcileResult{}, fmt.Errorf("failed to preserve existing DynamoComponentDeployment backendFramework: %w", err)
		}
		_, syncedDCD, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(context.Context) (*nvidiacomv1beta1.DynamoComponentDeployment, bool, error) {
			return dcd, false, nil
		})
		if err != nil {
			logger.Error(err, "failed to sync the DynamoComponentDeployment", "name", dcd.Name)
			return ReconcileResult{}, fmt.Errorf("failed to sync the DynamoComponentDeployment: %w", err)
		}
		resources = append(resources, syncedDCD)
	}

	// Old worker DCDs are scaled through direct patches so their stored specs
	// are not overwritten with the new generation's desired spec.
	if rollingUpdateCtx.InProgress() {
		if err := r.scaleOldWorkerDCDs(ctx, dynamoDeployment, rollingUpdateCtx); err != nil {
			logger.Error(err, "failed to scale old worker DCDs")
			return ReconcileResult{}, fmt.Errorf("failed to scale old worker DCDs: %w", err)
		}
	}

	result := r.checkResourcesReadiness(resources)

	// Include old worker generations in status while a managed rolling update
	// is active. A transient aggregation error remains non-fatal so readiness
	// can continue with the statuses already collected above.
	if rollingUpdateCtx.InProgress() {
		oldWorkerStatuses, err := r.aggregateOldWorkerComponentStatuses(ctx, dynamoDeployment, rollingUpdateCtx)
		if err != nil {
			logger.Error(err, "failed to aggregate old worker component statuses")
		} else if len(oldWorkerStatuses) > 0 {
			mergeWorkerComponentStatuses(result.ComponentStatus, oldWorkerStatuses)
		}
	}

	return result, nil
}

func (p *componentProgram) reconcileWorkerRollout(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
) error {
	if err := p.reconciler.migrateCurrentWorkerHashIfNeeded(ctx, dgd); err != nil {
		log.FromContext(ctx).Error(err, "Failed to migrate worker hash")
		return failWorkloadProgram(reasonFailedToMigrateWorkerHash, err)
	}

	if p.supportsManagedRollingUpdate(dgd) {
		return p.reconcileManagedWorkerRollout(ctx, dgd, status)
	}
	return reconcileUnsupportedWorkerRollout(ctx, p.reconciler, dgd, false)
}

func (p *componentProgram) reconcileManagedWorkerRollout(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
) error {
	r := p.reconciler
	logger := log.FromContext(ctx)

	if err := r.initializeWorkerHashIfNeeded(ctx, dgd); err != nil {
		logger.Error(err, "Failed to initialize worker hash")
		return failWorkloadProgram(reasonFailedToInitializeWorkerHash, err)
	}

	rollingUpdateInProgress := r.isRollingUpdateInProgress(status)
	triggerRollingUpdate := false
	if !rollingUpdateInProgress {
		var err error
		triggerRollingUpdate, err = r.shouldTriggerRollingUpdate(dgd)
		if err != nil {
			logger.Error(err, "Failed to check rolling update trigger")
			return failWorkloadProgram(reasonRollingUpdateFailed, err)
		}
	}
	if rollingUpdateInProgress || triggerRollingUpdate {
		if err := r.reconcileRollingUpdate(ctx, dgd, status); err != nil {
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
		desired, err := p.reconciler.desiredWorkerHashes(dgd)
		if err != nil {
			return
		}
		result.Eventf(
			corev1.EventTypeNormal,
			"RollingUpdateStarted",
			"Starting rolling update to worker hash %s",
			p.reconciler.activeWorkerHashForDCDGeneration(dgd, desired),
		)
	case current == nvidiacomv1beta1.RollingUpdatePhaseCompleted && previous != current:
		currentHashes := p.reconciler.currentWorkerHashes(dgd)
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

func reconcileUnsupportedWorkerRollout(
	ctx context.Context,
	r *DynamoGraphDeploymentReconciler,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	isGrove bool,
) error {
	logger := log.FromContext(ctx)

	if r.currentWorkerHashes(dgd).empty() {
		hashes, err := r.desiredWorkerHashes(dgd)
		if err != nil {
			logger.Error(err, "Failed to compute worker hash for unsupported pathway")
			return failWorkloadProgram(reasonFailedToInitializeWorkerHash, err)
		}
		r.setCurrentWorkerHashes(dgd, workerHashesForCompletedGeneration(hashes.v2, hashes))
		if err := r.Update(ctx, dgd); err != nil {
			logger.Error(err, "Failed to initialize worker hash for unsupported pathway")
			return failWorkloadProgram(reasonFailedToInitializeWorkerHash, err)
		}
	}

	// For unsupported pathways, log if a rolling update would have been triggered.
	triggerRollingUpdate, err := r.shouldTriggerRollingUpdate(dgd)
	if err != nil {
		logger.Error(err, "Failed to check rolling update trigger for unsupported pathway")
		return failWorkloadProgram(reasonRollingUpdateFailed, err)
	}
	if !triggerRollingUpdate {
		return nil
	}

	// Update the hash to prevent repeated warnings. If the unsupported path is
	// processing a v2-only worker change, preserve the migrated v2-only state
	// instead of resurrecting the downgrade-compatible v1 annotation for pod
	// contents it no longer represents.
	hashes, err := r.desiredWorkerHashes(dgd)
	if err != nil {
		logger.Error(err, "Failed to compute worker hash for unsupported pathway")
		return failWorkloadProgram(reasonFailedToInitializeWorkerHash, err)
	}
	r.setCurrentWorkerHashes(dgd, r.workerHashesForUnsupportedPathway(dgd, hashes))
	if err := r.Update(ctx, dgd); err != nil {
		// Preserve the existing best-effort behavior: the next reconciliation
		// retries the metadata update and may emit another warning.
		logger.Error(err, "Failed to update worker hash for unsupported pathway")
		return nil
	}

	logger.Info(
		"Worker spec change detected but rolling update not supported for this pathway",
		"isGrove", isGrove,
		"hasMultinode", dgd.HasAnyMultinodeComponent(),
	)
	if r.Recorder != nil {
		r.Recorder.Event(
			dgd,
			corev1.EventTypeWarning,
			"RollingUpdateNotSupported",
			"Worker spec changed but custom rolling updates are not supported for Grove/multinode deployments",
		)
	}
	return nil
}

func (p *componentProgram) applyCheckpointStartupPolicy(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	checkpointInfo *checkpoint.CheckpointInfo,
) error {
	if dcd == nil || checkpointInfo == nil || !checkpointInfo.Enabled {
		return nil
	}

	// DGD-managed automatic checkpoints deliberately bypass identity lookup and
	// are resolved by the exact DynamoCheckpoint CR created for this
	// DGD/component generation. Propagate that resolved reference into the child
	// DCD so the DCD controller does not independently fall back to legacy
	// identity-based reuse logic.
	if checkpointInfo.Exists && checkpointInfo.CheckpointName != "" {
		if dcd.Spec.Experimental == nil {
			dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{}
		}
		if dcd.Spec.Experimental.Checkpoint == nil {
			dcd.Spec.Experimental.Checkpoint = &nvidiacomv1beta1.ComponentCheckpointConfig{}
		}
		checkpointName := checkpointInfo.CheckpointName
		dcd.Spec.Experimental.Checkpoint.Enabled = true
		dcd.Spec.Experimental.Checkpoint.CheckpointRef = &checkpointName
		dcd.Spec.Experimental.Checkpoint.Identity = nil
		dcd.Spec.Experimental.Checkpoint.Job = nil
		startupPolicy := checkpointInfo.StartupPolicy
		if startupPolicy == "" {
			startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
		}
		dcd.Spec.Experimental.Checkpoint.StartupPolicy = nvidiacomv1beta1.CheckpointStartupPolicy(startupPolicy)
	}

	if checkpointInfo.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint && !checkpointInfo.Ready {
		dcd.Spec.Replicas = ptr.To(int32(0))
		return nil
	}
	if checkpointInfo.StartupPolicy == "" ||
		checkpointInfo.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyImmediate {
		labels := dynamo.GetPodTemplateLabels(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
		if labels == nil {
			if dcd.Spec.PodTemplate == nil {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{}
			}
			if dcd.Spec.PodTemplate.Labels == nil {
				dcd.Spec.PodTemplate.Labels = map[string]string{}
			}
			labels = dcd.Spec.PodTemplate.Labels
		}
		annotations := dynamo.GetPodTemplateAnnotations(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
		if annotations == nil {
			if dcd.Spec.PodTemplate == nil {
				dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{}
			}
			if dcd.Spec.PodTemplate.Annotations == nil {
				dcd.Spec.PodTemplate.Annotations = map[string]string{}
			}
			annotations = dcd.Spec.PodTemplate.Annotations
		}
		return checkpoint.ApplyRestoreCandidateMetadata(labels, annotations, checkpointInfo)
	}
	return nil
}

func (p *componentProgram) preserveExistingBackendFramework(
	ctx context.Context,
	desired *nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	r := p.reconciler
	existing := &nvidiacomv1beta1.DynamoComponentDeployment{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to get existing DynamoComponentDeployment %s/%s: %w", desired.Namespace, desired.Name, err)
	}

	// backendFramework is immutable on DCDs. Older generated children may have
	// an empty value, so preserve the stored value on update while allowing new
	// children to be created with the inferred backend.
	desired.Spec.BackendFramework = existing.Spec.BackendFramework
	return nil
}

type groveProgram struct {
	reconciler *DynamoGraphDeploymentReconciler
	reconcile  groveReconcileFunc
	lwsEnabled bool
}

// Reconcile composes the current Grove pathway around its temporary workload
// adapter while keeping shared resource ordering and result publication
// identical to the component program.
func (p *groveProgram) Reconcile(
	ctx context.Context,
	req workloadProgramRequest,
) (programResult workloadProgramResult, retErr error) {
	programResult = newWorkloadProgramResult(req.DGD)
	defer func() {
		if retErr != nil {
			reason := reasonFailedToReconcileResources
			if classified, ok := workloadProgramFailureReason(retErr); ok {
				reason = classified
			}
			programResult.Fail(req.DGD.Generation, reason, retErr)
		}
		p.reconciler.propagateTopologyCondition(ctx, req.DGD, &programResult)
	}()
	log.FromContext(ctx).Info(
		"Reconciling Grove resources",
		"hasMultinode", req.DGD.HasAnyMultinodeComponent(),
		"lwsEnabled", p.lwsEnabled,
	)

	if err := p.reconciler.migrateCurrentWorkerHashIfNeeded(ctx, req.DGD); err != nil {
		log.FromContext(ctx).Error(err, "Failed to migrate worker hash")
		return programResult, failWorkloadProgram(reasonFailedToMigrateWorkerHash, err)
	}
	if err := reconcileUnsupportedWorkerRollout(ctx, p.reconciler, req.DGD, true); err != nil {
		return programResult, err
	}
	inputs, err := p.reconciler.reconcileProgramInputs(ctx, req.DGD)
	if inputs.CheckpointStatuses != nil {
		programResult.Status.Checkpoints = inputs.CheckpointStatuses
	}
	if err != nil {
		return programResult, err
	}
	restart := p.reconciler.resolveProgramRestartState(ctx, req.DGD, &programResult.Status, &programResult)
	programResult.Status.Restart = restart.Status

	result, err := p.reconcileWorkloads(ctx, workloadReconcileRequest{
		DGD:             req.DGD,
		RestartState:    restart.State,
		CheckpointInfos: inputs.CheckpointInfos,
	})
	if err != nil {
		return programResult, fmt.Errorf("failed to reconcile Grove workloads: %w", err)
	}
	result, err = p.reconciler.reconcileProgramResult(ctx, req.DGD, inputs, restart, result)
	if err != nil {
		return programResult, err
	}

	programResult.applyReconcileResult(req.DGD.Generation, result)
	return programResult, nil
}

func (p *groveProgram) reconcileWorkloads(
	ctx context.Context,
	req workloadReconcileRequest,
) (ReconcileResult, error) {
	return p.reconcile(
		ctx,
		req.DGD,
		req.RestartState,
		req.CheckpointInfos,
	)
}

func (r *DynamoGraphDeploymentReconciler) selectWorkloadProgram(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) workloadProgram {
	if r.isGrovePathway(dgd) {
		return &groveProgram{
			reconciler: r,
			reconcile:  r.reconcileGroveResources,
			lwsEnabled: r.RuntimeConfig.Gate.Enabled(features.LWS),
		}
	}
	return &componentProgram{
		reconciler: r,
		lwsEnabled: r.RuntimeConfig.Gate.Enabled(features.LWS),
	}
}
