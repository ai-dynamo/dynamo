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
	"slices"
	"sort"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

// shouldTriggerRollingUpdate determines if worker spec changes require a rolling update.
func (r *DynamoGraphDeploymentReconciler) shouldTriggerRollingUpdate(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) bool {
	computedHash := dynamo.ComputeWorkerSpecHash(dgd)

	currentHash := r.getCurrentWorkerHash(dgd)

	// If no current hash exists (new deployment), no rolling update needed
	if currentHash == "" {
		return false
	}

	return computedHash != currentHash
}

// initializeWorkerHashIfNeeded sets the current worker hash annotation on first deployment.
func (r *DynamoGraphDeploymentReconciler) initializeWorkerHashIfNeeded(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	if r.getCurrentWorkerHash(dgd) != "" {
		return nil // Already initialized
	}

	hash := dynamo.ComputeWorkerSpecHash(dgd)
	r.setCurrentWorkerHash(dgd, hash)

	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to initialize worker hash: %w", err)
	}

	logger.Info("Initialized current worker hash",
		"hash", hash,
		"workerSuffix", hash[:8])

	return nil
}

// supportsManagedRollingUpdate checks if DGD pathway supports operator managed rolling updates.
// Grove and LWS deployments currently do not support operator managed rolling updates.
// They fall back to the default rolling update mechanism.
func (r *DynamoGraphDeploymentReconciler) supportsManagedRollingUpdate(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) bool {
	return !r.isGrovePathway(dgd) && !dgd.HasAnyMultinodeService()
}

// getCurrentWorkerHash returns the stored worker hash from DGD annotations.
// during a rolling update, this is the old worker hash and is not updated until the rolling update is completed.
// Returns empty string if no hash has been set (new deployment).
func (r *DynamoGraphDeploymentReconciler) getCurrentWorkerHash(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) string {
	if dgd.Annotations == nil {
		return ""
	}
	return dgd.Annotations[consts.AnnotationCurrentWorkerHash]
}

// setCurrentWorkerHash stores the worker hash in DGD annotations.
func (r *DynamoGraphDeploymentReconciler) setCurrentWorkerHash(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	hash string,
) {
	if dgd.Annotations == nil {
		dgd.Annotations = make(map[string]string)
	}
	dgd.Annotations[consts.AnnotationCurrentWorkerHash] = hash
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

// reconcileRollingUpdate handles the rolling update lifecycle.
func (r *DynamoGraphDeploymentReconciler) reconcileRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	// Get or create rollingUpdate status
	rollingUpdateStatus := r.getOrCreateRollingUpdateStatus(dgd)

	// Compute hash information
	newWorkerHash := dynamo.ComputeWorkerSpecHash(dgd)
	oldWorkerHash := r.getCurrentWorkerHash(dgd)

	logger.Info("Reconciling rolling update",
		"phase", rollingUpdateStatus.Phase,
		"oldWorkerHash", oldWorkerHash,
		"newWorkerHash", newWorkerHash)

	if rollingUpdateStatus.Phase == nvidiacomv1alpha1.RollingUpdatePhaseCompleted && oldWorkerHash != newWorkerHash {
		logger.Info("Rolling update completed but annotation stale, updating annotation",
			"oldHash", oldWorkerHash, "newHash", newWorkerHash)
		return r.completeRollingUpdate(ctx, dgd, rollingUpdateStatus, oldWorkerHash, newWorkerHash)
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

	now := metav1.Now()
	rollingUpdateStatus.Phase = nvidiacomv1alpha1.RollingUpdatePhasePending
	rollingUpdateStatus.StartTime = &now
	rollingUpdateStatus.UpdatedServices = nil

	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateStarted",
		"Starting rolling update from worker hash %s to %s", oldWorkerHash, newWorkerHash)

	if err := r.Status().Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to initialize rolling update status: %w", err)
	}

	return nil
}

// continueRollingUpdate handles the in-progress phase of a rolling update.
func (r *DynamoGraphDeploymentReconciler) continueRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rollingUpdateStatus *nvidiacomv1alpha1.RollingUpdateStatus,
	oldWorkerHash, newWorkerHash string,
) error {
	logger := log.FromContext(ctx)

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

	// Compute per-service completion
	var updatedServices []string
	for serviceName, spec := range dgd.Spec.Services {
		if spec == nil || !dynamo.IsWorkerComponent(spec.ComponentType) {
			continue
		}

		desired := int32(1)
		if spec.Replicas != nil {
			desired = *spec.Replicas
		}

		newSvc := newInfo.services[serviceName]
		oldSvc := oldInfo.services[serviceName]

		newReady := newSvc != nil && newSvc.readyReplicas >= desired
		oldGone := oldSvc == nil || oldSvc.readyReplicas == 0

		if newReady && oldGone {
			updatedServices = append(updatedServices, serviceName)
		}
	}
	sort.Strings(updatedServices)
	rollingUpdateStatus.UpdatedServices = updatedServices

	// Check if rolling update is complete: all new workers ready and all old workers scaled down
	if newInfo.TotalReadyWorkers() >= desiredReplicas && oldInfo.TotalReadyWorkers() == 0 {
		return r.completeRollingUpdate(ctx, dgd, rollingUpdateStatus, oldWorkerHash, newWorkerHash)
	}

	// Persist updated services list mid-rollout
	if err := r.Status().Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to update rolling update status with updated services: %w", err)
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

	// Mark all worker services as updated
	var allWorkerServices []string
	for serviceName, spec := range dgd.Spec.Services {
		if spec != nil && dynamo.IsWorkerComponent(spec.ComponentType) {
			allWorkerServices = append(allWorkerServices, serviceName)
		}
	}
	sort.Strings(allWorkerServices)
	rollingUpdateStatus.UpdatedServices = allWorkerServices

	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateCompleted",
		"Rolling update completed, worker hash %s", newWorkerHash)

	if err := r.Status().Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to update rolling update status: %w", err)
	}

	// Update the current worker hash to the new hash
	r.setCurrentWorkerHash(dgd, newWorkerHash)
	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to update current worker hash: %w", err)
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
	rollingUpdateCtx dynamo.RollingUpdateContext,
) error {
	logger := log.FromContext(ctx)

	if !rollingUpdateCtx.InProgress() {
		return nil
	}

	for serviceName, desiredReplicas := range rollingUpdateCtx.OldWorkerReplicas {
		// Construct the old DCD name using the hash-based naming convention
		oldDCDName := dynamo.GetDCDResourceName(dgd, serviceName, rollingUpdateCtx.OldWorkerHash)

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

// aggregateOldWorkerServiceStatuses fetches old worker DCDs and returns their service statuses
// keyed by service name. This allows merging old worker replica counts into the aggregated
// service status during rolling updates.
func (r *DynamoGraphDeploymentReconciler) aggregateOldWorkerServiceStatuses(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rollingUpdateCtx dynamo.RollingUpdateContext,
) (map[string]nvidiacomv1alpha1.ServiceReplicaStatus, error) {
	logger := log.FromContext(ctx)
	oldStatuses := make(map[string]nvidiacomv1alpha1.ServiceReplicaStatus)

	for serviceName := range rollingUpdateCtx.OldWorkerReplicas {
		oldDCDName := dynamo.GetDCDResourceName(dgd, serviceName, rollingUpdateCtx.OldWorkerHash)

		existingDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{}
		err := r.Get(ctx, client.ObjectKey{Name: oldDCDName, Namespace: dgd.Namespace}, existingDCD)
		if err != nil {
			if apierrors.IsNotFound(err) {
				logger.V(1).Info("Old worker DCD not found, skipping status aggregation",
					"dcdName", oldDCDName,
					"service", serviceName)
				continue
			}
			return nil, fmt.Errorf("failed to get old worker DCD %s: %w", oldDCDName, err)
		}

		if existingDCD.Status.Service != nil {
			oldStatuses[serviceName] = *existingDCD.Status.Service
		}
	}

	return oldStatuses, nil
}

// mergeWorkerServiceStatuses merges old worker service statuses into the existing service statuses.
// For each worker service present in both maps, it aggregates replica counts so that the status
// reflects the total across old and new worker DCDs during a rolling update.
func mergeWorkerServiceStatuses(
	serviceStatuses map[string]nvidiacomv1alpha1.ServiceReplicaStatus,
	oldWorkerStatuses map[string]nvidiacomv1alpha1.ServiceReplicaStatus,
) {
	for serviceName, oldStatus := range oldWorkerStatuses {
		newStatus, exists := serviceStatuses[serviceName]
		if !exists {
			continue
		}

		// Build sorted ComponentNames from old and new DCD names
		componentNames := []string{oldStatus.ComponentName, newStatus.ComponentName}
		slices.Sort(componentNames)
		newStatus.ComponentNames = componentNames

		// Aggregate replica counts
		newStatus.Replicas += oldStatus.Replicas
		// UpdatedReplicas stays as-is (only new are "updated")
		newStatus.ReadyReplicas = addOptionalInt32(newStatus.ReadyReplicas, oldStatus.ReadyReplicas)
		newStatus.AvailableReplicas = addOptionalInt32(newStatus.AvailableReplicas, oldStatus.AvailableReplicas)

		serviceStatuses[serviceName] = newStatus
	}
}

// addOptionalInt32 adds two optional int32 pointers. Returns nil only if both are nil.
func addOptionalInt32(a, b *int32) *int32 {
	if a == nil && b == nil {
		return nil
	}
	var sum int32
	if a != nil {
		sum += *a
	}
	if b != nil {
		sum += *b
	}
	return &sum
}
