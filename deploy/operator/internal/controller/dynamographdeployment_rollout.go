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

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

// shouldTriggerRollingUpdate determines if WORKER spec changes require a rolling update.
func (r *DynamoGraphDeploymentReconciler) shouldTriggerRollingUpdate(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) bool {
	currentHash := dynamo.ComputeWorkerSpecHash(dgd)

	activeHash := r.getCurrentActiveWorkerHash(dgd)

	// If no active hash exists (new deployment), no rolling update needed
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
		"workerSuffix", hash[:8])

	return nil
}

// isSupportedRollingUpdatePathway checks if DGD uses supported pathways for custom rolling updates.
// Grove and LWS deployments use different orchestration and don't support the custom rolling update strategy as of now.
func (r *DynamoGraphDeploymentReconciler) isSupportedRollingUpdatePathway(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) bool {
	return !r.isGrovePathway(dgd) && !dgd.HasAnyMultinodeService()
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

// getOrCreateRollingUpdateStatus returns the existing rolling update status or creates a new one.
func (r *DynamoGraphDeploymentReconciler) getOrCreateRollingUpdateStatus(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) *nvidiacomv1alpha1.RollingUpdateStatus {
	if dgd.Status.RollingUpdate == nil {
		dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
			Phase: nvidiacomv1alpha1.RollingUpdatePhaseNone,
		}
	}
	return dgd.Status.RollingUpdate
}

// isRollingUpdateInProgress returns true if a rolling update is currently active.
func (r *DynamoGraphDeploymentReconciler) isRollingUpdateInProgress(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) bool {
	if dgd.Status.RollingUpdate == nil {
		return false
	}
	phase := dgd.Status.RollingUpdate.Phase
	return phase == nvidiacomv1alpha1.RollingUpdatePhasePending ||
		phase == nvidiacomv1alpha1.RollingUpdatePhaseInProgress
}

// clearRollingUpdateStatus resets the rolling update status after completion or failure cleanup.
func (r *DynamoGraphDeploymentReconciler) clearRollingUpdateStatus(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) {
	dgd.Status.RollingUpdate = &nvidiacomv1alpha1.RollingUpdateStatus{
		Phase: nvidiacomv1alpha1.RollingUpdatePhaseNone,
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

// reconcileRollingUpdate orchestrates the rolling update state machine and proxy weights.
// It updates the rolling update status and proxy configuration but does NOT return early.
// The caller (main Reconcile) should always proceed to reconcileResources afterward.
//
// This function is responsible for:
// - Phase transitions (None -> Pending -> InProgress -> Completed)
// - Proxy weight updates based on worker readiness
// - Storing namespace info in status for reconcileResources to use
//
// reconcileResources is responsible for:
// - Creating/updating DCDs for both old and new namespaces
// - Calculating replica counts based on fresh DCD queries
func (r *DynamoGraphDeploymentReconciler) reconcileRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	// Get or create rollingUpdate status
	rollingUpdateStatus := r.getOrCreateRollingUpdateStatus(dgd)

	// Compute hash information
	newWorkerHash := dynamo.ComputeWorkerSpecHash(dgd)
	oldWorkerHash := r.getCurrentActiveWorkerHash(dgd)

	logger.Info("Reconciling rolling update",
		"phase", rollingUpdateStatus.Phase,
		"oldWorkerHash", oldWorkerHash,
		"newWorkerHash", newWorkerHash)

	if rollingUpdateStatus.Phase == nvidiacomv1alpha1.RollingUpdatePhaseCompleted {
		if oldWorkerHash != newWorkerHash {
			logger.Info("Rolling update completed but annotation stale, updating annotation",
				"oldHash", oldWorkerHash, "newHash", newWorkerHash)
			r.setActiveWorkerHash(dgd, newWorkerHash)
			return r.Update(ctx, dgd)
		}
		// Annotation matches, we're done
		logger.V(1).Info("Rolling update completed and annotation matches")
		return nil
	}

	if oldWorkerHash == newWorkerHash &&
		rollingUpdateStatus.Phase == nvidiacomv1alpha1.RollingUpdatePhaseInProgress {
		logger.Info("Detected stuck rolling update: hashes match but phase is InProgress",
			"hash", newWorkerHash,
			"phase", rollingUpdateStatus.Phase)
		return r.completeRollingUpdate(ctx, dgd, rollingUpdateStatus, oldWorkerHash, newWorkerHash)
	}

	switch rollingUpdateStatus.Phase {
	case nvidiacomv1alpha1.RollingUpdatePhaseNone:
		return r.startRollingUpdate(ctx, dgd, rollingUpdateStatus, oldWorkerHash, newWorkerHash)

	case nvidiacomv1alpha1.RollingUpdatePhasePending:
		rollingUpdateStatus.Phase = nvidiacomv1alpha1.RollingUpdatePhaseInProgress
		if err := r.Status().Update(ctx, dgd); err != nil {
			return fmt.Errorf("failed to update rolling update status to InProgress: %w", err)
		}
		return nil

	case nvidiacomv1alpha1.RollingUpdatePhaseInProgress:
		return r.continueRollingUpdate(ctx, dgd, rollingUpdateStatus, oldWorkerHash, newWorkerHash)

	case nvidiacomv1alpha1.RollingUpdatePhaseCompleted:
		// Cleanup is now done atomically in completeRollingUpdate, nothing to do here
		logger.Info("Rolling update already completed")
		return nil

	case nvidiacomv1alpha1.RollingUpdatePhaseFailed:
		logger.Info("Rolling update in failed state, manual intervention may be required")
		return nil
	}

	return nil
}

// startRollingUpdate initializes a new rolling update.
func (r *DynamoGraphDeploymentReconciler) startRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rollingUpdateStatus *nvidiacomv1alpha1.RollingUpdateStatus,
	oldWorkerHash, newWorkerHash string,
) error {
	logger := log.FromContext(ctx)

	logger.Info("Starting rolling update",
		"oldHash", oldWorkerHash,
		"newHash", newWorkerHash)

	// Initialize rolling update status
	// Note: Worker hashes are computed dynamically from worker hash annotation,
	// not stored in status, to avoid staleness issues
	now := metav1.Now()
	rollingUpdateStatus.Phase = nvidiacomv1alpha1.RollingUpdatePhasePending
	rollingUpdateStatus.StartTime = &now

	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateStarted",
		"Starting rolling update from worker hash %s to %s", oldWorkerHash, newWorkerHash)

	if err := r.Status().Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to initialize rolling update status: %w", err)
	}

	return nil
}

// continueRollingUpdate handles the in-progress phase of a rolling update.
// Traffic weighting is handled automatically by the frontend's multi-pool manager,
// which discovers workers via prefix-based namespace matching and load balances across them.
func (r *DynamoGraphDeploymentReconciler) continueRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rollingUpdateStatus *nvidiacomv1alpha1.RollingUpdateStatus,
	oldWorkerHash, newWorkerHash string,
) error {
	logger := log.FromContext(ctx)

	workerServices := r.getWorkerServices(dgd)
	if len(workerServices) == 0 {
		logger.Info("No worker services found, completing rolling update")
		return r.completeRollingUpdate(ctx, dgd, rollingUpdateStatus, oldWorkerHash, newWorkerHash)
	}

	oldInfo, err := r.getWorkerInfoForWorkerHash(ctx, dgd, oldWorkerHash)
	if err != nil {
		logger.Error(err, "Failed to get old worker hash status")
		// Continue with empty status - old DCDs may not exist yet
		oldInfo = &dynamoNamespaceWorkerInfo{}
	}

	newInfo, err := r.getWorkerInfoForWorkerHash(ctx, dgd, newWorkerHash)
	if err != nil {
		logger.Error(err, "Failed to get new worker hash status")
		newInfo = &dynamoNamespaceWorkerInfo{}
	}

	desiredReplicas := r.getDesiredWorkerReplicas(dgd)

	logger.Info("Rolling update progress",
		"oldReadyWorkers", oldInfo.TotalReadyWorkers(),
		"newReadyWorkers", newInfo.TotalReadyWorkers(),
		"desiredReplicas", desiredReplicas,
		"oldWorkerHash", oldWorkerHash,
		"newWorkerHash", newWorkerHash)

	// Check if rolling update is complete: all new workers ready and all old workers scaled down
	if newInfo.TotalReadyWorkers() >= desiredReplicas && oldInfo.TotalReadyWorkers() == 0 {
		return r.completeRollingUpdate(ctx, dgd, rollingUpdateStatus, oldWorkerHash, newWorkerHash)
	}

	// Update status
	if err := r.Status().Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to update rolling update status: %w", err)
	}

	return nil
}

// completeRollingUpdate marks the rolling update as completed, cleans up old resources, and updates status.
// This performs all cleanup atomically to avoid race conditions with subsequent reconciles.
func (r *DynamoGraphDeploymentReconciler) completeRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rollingUpdateStatus *nvidiacomv1alpha1.RollingUpdateStatus,
	oldWorkerHash, newWorkerHash string,
) error {
	logger := log.FromContext(ctx)

	// Delete old worker DCDs by their worker hash label
	if oldWorkerHash != "" && oldWorkerHash != newWorkerHash {
		if err := r.deleteOldDCDs(ctx, dgd, oldWorkerHash); err != nil {
			logger.Error(err, "Failed to delete old DCDs", "oldWorkerHash", oldWorkerHash)
			r.Recorder.Eventf(dgd, corev1.EventTypeWarning, "CleanupPartialFailure",
				"Failed to delete some old worker DCDs with hash %s: %v", oldWorkerHash, err)
			// Continue anyway - we don't want cleanup failures to block the rolling update completion
		} else {
			logger.Info("Old resources cleaned up", "oldWorkerHash", oldWorkerHash)
		}
	}

	// Update rolling update status to Completed
	rollingUpdateStatus.Phase = nvidiacomv1alpha1.RollingUpdatePhaseCompleted
	now := metav1.Now()
	rollingUpdateStatus.EndTime = &now

	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateCompleted",
		"Rolling update completed, worker hash %s", newWorkerHash)

	if err := r.Status().Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to update rolling update status: %w", err)
	}

	// Update the active worker hash to the new hash
	r.setActiveWorkerHash(dgd, newWorkerHash)
	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to update active worker hash: %w", err)
	}

	logger.Info("Rolling update finalized", "newWorkerHash", newWorkerHash)

	return nil
}

// workerServiceInfo holds ready replica count for a worker service.
type workerServiceInfo struct {
	readyReplicas int32
	desired       int32
}

// dynamoNamespaceWorkerInfo holds aggregated worker status for a single dynamo namespace.
type dynamoNamespaceWorkerInfo struct {
	// totalReadyWorkers is the sum of ready replicas across all worker services
	totalReadyWorkers int32
	// services contains per-component-type status (e.g., "prefill", "decode", "worker")
	services map[string]*workerServiceInfo
}

func (s *dynamoNamespaceWorkerInfo) TotalReadyWorkers() int32 {
	return s.totalReadyWorkers
}

// getWorkerInfoForWorkerHash queries DCDs for a specific worker hash and returns
// aggregated worker info.
func (r *DynamoGraphDeploymentReconciler) getWorkerInfoForWorkerHash(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	workerHash string,
) (*dynamoNamespaceWorkerInfo, error) {
	dcdList := &nvidiacomv1alpha1.DynamoComponentDeploymentList{}
	listOpts := []client.ListOption{
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			consts.KubeLabelDynamoWorkerHash:          workerHash,
		},
	}

	if err := r.List(ctx, dcdList, listOpts...); err != nil {
		return nil, fmt.Errorf("failed to list DCDs: %w", err)
	}

	status := &dynamoNamespaceWorkerInfo{
		services: make(map[string]*workerServiceInfo),
	}

	for _, dcd := range dcdList.Items {
		if !dynamo.IsWorkerComponent(dcd.Spec.ComponentType) {
			continue
		}

		// Add ready replicas
		readyReplicas := int32(0)
		if dcd.Status.Service != nil && dcd.Status.Service.ReadyReplicas != nil {
			readyReplicas = *dcd.Status.Service.ReadyReplicas
		}

		// Add desired replicas
		desiredReplicas := int32(0)
		if dcd.Spec.Replicas != nil {
			desiredReplicas = *dcd.Spec.Replicas
		}
		status.services[dcd.Spec.ServiceName] = &workerServiceInfo{
			readyReplicas: readyReplicas,
			desired:       desiredReplicas,
		}

		status.totalReadyWorkers += readyReplicas
	}

	return status, nil
}

// getDesiredWorkerReplicas returns the total desired replicas across all worker services.
func (r *DynamoGraphDeploymentReconciler) getDesiredWorkerReplicas(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) int32 {
	var total int32
	for _, spec := range dgd.Spec.Services {
		if spec != nil && dynamo.IsWorkerComponent(spec.ComponentType) {
			if spec.Replicas != nil {
				total += *spec.Replicas
			} else {
				total += 1 // Default to 1 if not specified
			}
		}
	}
	return total
}

// scaleOldWorkerDCDs patches the replicas field on old worker DCDs during a rolling update.
// This is done via direct patching rather than generating the full DCD spec to avoid
// overwriting the old spec with the new spec (which would trigger an unwanted rolling update).
func (r *DynamoGraphDeploymentReconciler) scaleOldWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rollingUpdateCtx *dynamo.RollingUpdateContext,
) error {
	logger := log.FromContext(ctx)

	if rollingUpdateCtx == nil || !rollingUpdateCtx.InProgress {
		return nil
	}

	for serviceName, desiredReplicas := range rollingUpdateCtx.OldWorkerReplicas {
		// Construct the old DCD name using the hash-based naming convention
		oldDCDName := dynamo.GetDynamoComponentName(dgd, serviceName) + "-" + rollingUpdateCtx.OldWorkerHash

		// Get the existing DCD
		existingDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{}
		err := r.Get(ctx, client.ObjectKey{Name: oldDCDName, Namespace: dgd.Namespace}, existingDCD)
		if err != nil {
			if apierrors.IsNotFound(err) {
				// Old DCD doesn't exist yet (first reconcile of rolling update)
				// This is expected - the old DCD was created before rolling update started
				logger.V(1).Info("Old worker DCD not found, skipping scale",
					"dcdName", oldDCDName,
					"service", serviceName)
				continue
			}
			return fmt.Errorf("failed to get old worker DCD %s: %w", oldDCDName, err)
		}

		// Check if replicas need to be updated
		currentReplicas := int32(1)
		if existingDCD.Spec.Replicas != nil {
			currentReplicas = *existingDCD.Spec.Replicas
		}

		if currentReplicas == desiredReplicas {
			logger.V(1).Info("Old worker DCD replicas already at desired value",
				"dcdName", oldDCDName,
				"replicas", desiredReplicas)
			continue
		}

		// Patch only the replicas field
		patch := client.MergeFrom(existingDCD.DeepCopy())
		existingDCD.Spec.Replicas = &desiredReplicas

		if err := r.Patch(ctx, existingDCD, patch); err != nil {
			return fmt.Errorf("failed to patch old worker DCD %s replicas: %w", oldDCDName, err)
		}

		logger.Info("Scaled old worker DCD",
			"dcdName", oldDCDName,
			"service", serviceName,
			"oldReplicas", currentReplicas,
			"newReplicas", desiredReplicas)
	}

	return nil
}

// deleteOldDCDs deletes all worker DCDs belonging to this DGD that have the old worker hash.
func (r *DynamoGraphDeploymentReconciler) deleteOldDCDs(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	oldWorkerHash string,
) error {
	logger := log.FromContext(ctx)

	// List all DCDs that belong to this DGD and have the old worker hash
	dcdList := &nvidiacomv1alpha1.DynamoComponentDeploymentList{}
	listOpts := []client.ListOption{
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			consts.KubeLabelDynamoWorkerHash:          oldWorkerHash,
		},
	}

	if err := r.List(ctx, dcdList, listOpts...); err != nil {
		return fmt.Errorf("failed to list old DCDs: %w", err)
	}

	if len(dcdList.Items) == 0 {
		logger.Info("No old DCDs found to delete", "oldWorkerHash", oldWorkerHash)
		return nil
	}

	logger.Info("Deleting old DCDs", "count", len(dcdList.Items), "oldWorkerHash", oldWorkerHash)

	var deleteErrors []error
	for i := range dcdList.Items {
		dcd := &dcdList.Items[i]
		logger.Info("Deleting old DCD", "name", dcd.Name, "oldWorkerHash", oldWorkerHash)

		if err := r.Delete(ctx, dcd); err != nil {
			if !apierrors.IsNotFound(err) {
				deleteErrors = append(deleteErrors, fmt.Errorf("failed to delete DCD %s: %w", dcd.Name, err))
			}
		}
	}

	if len(deleteErrors) > 0 {
		return fmt.Errorf("failed to delete %d DCDs: %v", len(deleteErrors), deleteErrors)
	}

	return nil
}
