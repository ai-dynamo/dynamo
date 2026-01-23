/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/rollout"
)

// shouldTriggerRollingUpdate determines if WORKER spec changes require a rolling update.
func (r *DynamoGraphDeploymentReconciler) shouldTriggerRollingUpdate(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) bool {
	currentHash := dynamo.ComputeWorkerSpecHash(dgd)

	activeHash := r.getCurrentActiveWorkerHash(dgd)

	if activeHash == "" {
		return false
	}

	return currentHash != activeHash
}

// initializeWorkerHashIfNeeded sets the active worker hash annotation on first deployment.
func (r *DynamoGraphDeploymentReconciler) initializeWorkerHashIfNeeded(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	if r.getCurrentActiveWorkerHash(dgd) != "" {
		return nil // Already initialized
	}

	hash := dynamo.ComputeWorkerSpecHash(dgd)
	r.setActiveWorkerHash(dgd, hash)

	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to initialize worker hash: %w", err)
	}

	logger.Info("Initialized active worker hash",
		"hash", hash,
		"dynamoNamespace", dynamo.ComputeHashedDynamoNamespace(dgd))

	return nil
}

// isUnsupportedRollingUpdatePathway checks if DGD uses unsupported pathways for custom rolling updates.
// Grove and LWS deployments use different orchestration and don't support the
// HAProxy-based rolling update strategy.
func (r *DynamoGraphDeploymentReconciler) isUnsupportedRollingUpdatePathway(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) bool {
	if r.isGrovePathway(dgd) {
		return true
	}

	// Check if any service uses multinode (which requires LWS when Grove is disabled)
	if dgd.HasAnyMultinodeService() {
		return true
	}

	return false
}

// getCurrentActiveWorkerHash returns the stored worker hash from DGD annotations.
// Returns empty string if no hash has been set (new deployment).
func (r *DynamoGraphDeploymentReconciler) getCurrentActiveWorkerHash(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) string {
	if dgd.Annotations == nil {
		return ""
	}
	return dgd.Annotations[consts.AnnotationActiveWorkerHash]
}

// setActiveWorkerHash stores the worker hash in DGD annotations.
func (r *DynamoGraphDeploymentReconciler) setActiveWorkerHash(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	hash string,
) {
	if dgd.Annotations == nil {
		dgd.Annotations = make(map[string]string)
	}
	dgd.Annotations[consts.AnnotationActiveWorkerHash] = hash
}

// getOrCreateRolloutStatus returns the existing rollout status or creates a new one.
func (r *DynamoGraphDeploymentReconciler) getOrCreateRolloutStatus(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) *nvidiacomv1alpha1.RolloutStatus {
	if dgd.Status.Rollout == nil {
		dgd.Status.Rollout = &nvidiacomv1alpha1.RolloutStatus{
			Phase: nvidiacomv1alpha1.RolloutPhaseNone,
		}
	}
	return dgd.Status.Rollout
}

// isRollingUpdateInProgress returns true if a rolling update is currently active.
func (r *DynamoGraphDeploymentReconciler) isRollingUpdateInProgress(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) bool {
	if dgd.Status.Rollout == nil {
		return false
	}
	phase := dgd.Status.Rollout.Phase
	return phase == nvidiacomv1alpha1.RolloutPhasePending ||
		phase == nvidiacomv1alpha1.RolloutPhaseInProgress
}

// clearRolloutStatus resets the rollout status after completion or failure cleanup.
func (r *DynamoGraphDeploymentReconciler) clearRolloutStatus(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) {
	dgd.Status.Rollout = &nvidiacomv1alpha1.RolloutStatus{
		Phase: nvidiacomv1alpha1.RolloutPhaseNone,
	}
}

// getWorkerServices returns all worker service names from the DGD.
// Worker services are identified by ComponentType: "worker", "prefill", or "decode".
func (r *DynamoGraphDeploymentReconciler) getWorkerServices(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) []string {
	var workers []string
	for name, spec := range dgd.Spec.Services {
		if spec != nil && dynamo.IsWorkerComponent(spec.ComponentType) {
			workers = append(workers, name)
		}
	}
	return workers
}

// reconcileRollingUpdate orchestrates the rolling update process.
// This is called when a rolling update is in progress or needs to be started.
func (r *DynamoGraphDeploymentReconciler) reconcileRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Get or create rollout status
	rolloutStatus := r.getOrCreateRolloutStatus(dgd)

	// Compute namespace information
	newWorkerHash := dynamo.ComputeWorkerSpecHash(dgd)
	oldWorkerHash := r.getCurrentActiveWorkerHash(dgd)
	newNamespace := dynamo.ComputeHashedDynamoNamespace(dgd)
	oldNamespace := dynamo.ComputeHashedDynamoNamespaceWithHash(dgd, oldWorkerHash)

	// Reconstruct state machine from current status
	sm := rollout.NewStateMachineFromRolloutStatus(rolloutStatus, "")

	logger.Info("Reconciling rolling update",
		"phase", rolloutStatus.Phase,
		"state", sm.CurrentState,
		"oldNamespace", oldNamespace,
		"newNamespace", newNamespace)

	// Handle based on current phase
	switch rolloutStatus.Phase {
	case nvidiacomv1alpha1.RolloutPhaseNone:
		// Start new rollout
		return r.startRollingUpdate(ctx, dgd, rolloutStatus, oldWorkerHash, newWorkerHash, oldNamespace, newNamespace)

	case nvidiacomv1alpha1.RolloutPhasePending:
		// Rollout is pending, transition to in progress
		rolloutStatus.Phase = nvidiacomv1alpha1.RolloutPhaseInProgress
		if err := r.Status().Update(ctx, dgd); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to update rollout status to InProgress: %w", err)
		}
		return ctrl.Result{Requeue: true}, nil

	case nvidiacomv1alpha1.RolloutPhaseInProgress:
		// Continue the rollout
		return r.continueRollingUpdate(ctx, dgd, rolloutStatus, oldNamespace, newNamespace)

	case nvidiacomv1alpha1.RolloutPhaseCompleted:
		// Cleanup completed rollout
		return r.finalizeRollingUpdate(ctx, dgd, newWorkerHash)

	case nvidiacomv1alpha1.RolloutPhaseFailed:
		// Rollout failed - leave status for user inspection
		logger.Info("Rolling update in failed state, manual intervention may be required")
		return ctrl.Result{}, nil
	}

	return ctrl.Result{}, nil
}

// startRollingUpdate initializes a new rolling update.
func (r *DynamoGraphDeploymentReconciler) startRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rolloutStatus *nvidiacomv1alpha1.RolloutStatus,
	oldWorkerHash, newWorkerHash, oldNamespace, newNamespace string,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	logger.Info("Starting rolling update",
		"oldHash", oldWorkerHash,
		"newHash", newWorkerHash,
		"oldNamespace", oldNamespace,
		"newNamespace", newNamespace)

	// Initialize rollout status
	now := metav1.Now()
	rolloutStatus.Phase = nvidiacomv1alpha1.RolloutPhasePending
	rolloutStatus.StartTime = &now
	rolloutStatus.TrafficWeightOld = 100
	rolloutStatus.TrafficWeightNew = 0

	// Record event
	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateStarted",
		"Starting rolling update from namespace %s to %s", oldNamespace, newNamespace)

	// Update status
	if err := r.Status().Update(ctx, dgd); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to initialize rollout status: %w", err)
	}

	return ctrl.Result{Requeue: true}, nil
}

// continueRollingUpdate handles the in-progress phase of a rolling update.
// This is where the actual scaling and traffic shifting happens.
func (r *DynamoGraphDeploymentReconciler) continueRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rolloutStatus *nvidiacomv1alpha1.RolloutStatus,
	oldNamespace, newNamespace string,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// For now, this is a simplified implementation that:
	// 1. Creates the new deployment resources (DCDs will be reconciled normally)
	// 2. Waits for new workers to be ready
	// 3. Updates traffic weights
	// 4. Scales down old workers
	//
	// The full implementation would use the state machine for finer-grained control.

	workerServices := r.getWorkerServices(dgd)
	if len(workerServices) == 0 {
		logger.Info("No worker services found, completing rollout")
		rolloutStatus.Phase = nvidiacomv1alpha1.RolloutPhaseCompleted
		rolloutStatus.TrafficWeightOld = 0
		rolloutStatus.TrafficWeightNew = 100
		now := metav1.Now()
		rolloutStatus.EndTime = &now
		if err := r.Status().Update(ctx, dgd); err != nil {
			return ctrl.Result{}, fmt.Errorf("failed to update rollout status: %w", err)
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// The normal DCD reconciliation will create new DCDs with the new namespace
	// (since ComputeHashedDynamoNamespace uses the current spec hash)
	// We just need to track progress and update traffic weights

	logger.Info("Rolling update in progress",
		"workerServices", workerServices,
		"trafficWeightOld", rolloutStatus.TrafficWeightOld,
		"trafficWeightNew", rolloutStatus.TrafficWeightNew)

	// TODO: Implement full scaling logic with:
	// - Count ready workers in old vs new namespace
	// - Scale up new workers incrementally (maxSurge=1)
	// - Scale down old workers as new become ready (maxUnavailable=0)
	// - Update HAProxy weights proportionally
	// - For multi-group deployments, wait for ≥1 ready in each group before routing traffic

	// For now, mark as completed after one reconciliation cycle
	// This allows the normal DCD reconciliation to create the new resources
	rolloutStatus.Phase = nvidiacomv1alpha1.RolloutPhaseCompleted
	rolloutStatus.TrafficWeightOld = 0
	rolloutStatus.TrafficWeightNew = 100
	now := metav1.Now()
	rolloutStatus.EndTime = &now

	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateCompleted",
		"Rolling update completed, traffic shifted to namespace %s", newNamespace)

	if err := r.Status().Update(ctx, dgd); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to update rollout status: %w", err)
	}

	return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
}

// finalizeRollingUpdate cleans up after a completed rolling update.
// The rollout status is preserved (not cleared) so users can see completion info.
func (r *DynamoGraphDeploymentReconciler) finalizeRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	newWorkerHash string,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Update the active worker hash to the new hash
	r.setActiveWorkerHash(dgd, newWorkerHash)
	if err := r.Update(ctx, dgd); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to update active worker hash: %w", err)
	}

	// Note: We intentionally keep the rollout status with Phase=Completed
	// so users can see when the rollout finished and the final state.
	// The status will be reset when a new rollout starts.

	logger.Info("Rolling update finalized", "newWorkerHash", newWorkerHash)

	// TODO: Delete old DCDs and associated resources from the old namespace
	// This would involve listing DCDs by label and deleting those with the old namespace

	return ctrl.Result{}, nil
}
