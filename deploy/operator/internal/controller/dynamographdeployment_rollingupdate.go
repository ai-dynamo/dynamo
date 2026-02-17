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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
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
	computedHash := dynamo.ComputeDGDWorkersSpecHash(dgd)

	currentHash := r.getCurrentWorkerHash(dgd)

	// If no current hash exists (new deployment), no rolling update needed
	if currentHash == "" {
		return false
	}

	return computedHash != currentHash
}

// initializeWorkerHashIfNeeded sets the current worker hash annotation on first deployment.
// For existing DGDs being upgraded from a pre-rolling-update operator version, this handles
// patching the legacy DCDs with the new worker hash label and then triggering a rolling update on the next reconcile.
func (r *DynamoGraphDeploymentReconciler) initializeWorkerHashIfNeeded(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	if r.getCurrentWorkerHash(dgd) != "" {
		return nil // Already initialized
	}

	// Check for legacy (pre-rolling-update) worker DCDs
	legacyDCDs, err := r.findLegacyWorkerDCDs(ctx, dgd)
	if err != nil {
		return fmt.Errorf("failed to check for legacy worker DCDs: %w", err)
	}

	if len(legacyDCDs) > 0 {
		logger.Info("Found legacy worker DCDs without hash label, initiating migration",
			"count", len(legacyDCDs))

		// Backfill hash label on legacy DCDs so they're manageable by the rolling update machinery
		for i := range legacyDCDs {
			dcd := &legacyDCDs[i]
			patch := client.MergeFrom(dcd.DeepCopy())
			if dcd.Labels == nil {
				dcd.Labels = make(map[string]string)
			}
			dcd.Labels[consts.KubeLabelDynamoWorkerHash] = consts.LegacyWorkerHash
			if err := r.Patch(ctx, dcd, patch); err != nil {
				return fmt.Errorf("failed to backfill hash label on legacy DCD %s: %w", dcd.Name, err)
			}
			logger.Info("Backfilled worker hash label on legacy DCD",
				"dcdName", dcd.Name, "hash", consts.LegacyWorkerHash)
		}

		// Set sentinel hash — next reconcile triggers a real rolling update from "legacy" -> computed hash
		r.setCurrentWorkerHash(dgd, consts.LegacyWorkerHash)
		if err := r.Update(ctx, dgd); err != nil {
			return fmt.Errorf("failed to set legacy worker hash: %w", err)
		}

		r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "LegacyMigrationStarted",
			"Detected %d legacy worker DCDs, initiating rolling update migration", len(legacyDCDs))
		return nil
	}

	// Normal first deploy — set the actual computed hash
	hash := dynamo.ComputeDGDWorkersSpecHash(dgd)
	r.setCurrentWorkerHash(dgd, hash)

	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to initialize worker hash: %w", err)
	}

	logger.Info("Initialized current worker hash", "hash", hash)

	return nil
}

// findLegacyWorkerDCDs returns worker DCDs owned by this DGD that lack the worker hash label.
// These are DCDs created by a pre-rolling-update operator version.
func (r *DynamoGraphDeploymentReconciler) findLegacyWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) ([]nvidiacomv1alpha1.DynamoComponentDeployment, error) {
	// List all DCDs for this DGD
	dcdList := &nvidiacomv1alpha1.DynamoComponentDeploymentList{}
	listOpts := []client.ListOption{
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		},
	}

	if err := r.List(ctx, dcdList, listOpts...); err != nil {
		return nil, fmt.Errorf("failed to list DCDs for DGD %s: %w", dgd.Name, err)
	}

	var legacyDCDs []nvidiacomv1alpha1.DynamoComponentDeployment
	for _, dcd := range dcdList.Items {
		if !dynamo.IsWorkerComponent(dcd.Spec.ComponentType) {
			continue
		}
		// Legacy DCDs lack the worker hash label
		if dcd.Labels[consts.KubeLabelDynamoWorkerHash] == "" {
			legacyDCDs = append(legacyDCDs, dcd)
		}
	}

	return legacyDCDs, nil
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
// during a rolling update, this is the previous worker hash and is not updated until the rolling update is completed.
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
	newWorkerHash := dynamo.ComputeDGDWorkersSpecHash(dgd)
	prevWorkerHash := r.getCurrentWorkerHash(dgd)

	logger.Info("Reconciling rolling update",
		"phase", rollingUpdateStatus.Phase,
		"prevWorkerHash", prevWorkerHash,
		"newWorkerHash", newWorkerHash)

	if (rollingUpdateStatus.Phase == nvidiacomv1alpha1.RollingUpdatePhaseCompleted) && prevWorkerHash != newWorkerHash {
		// Check if DCDs with the new hash already exist and are serving.
		// If so, this is just a stale annotation — update it without starting a new rollout.
		newInfo, err := r.getWorkerInfoForWorkerHash(ctx, dgd, newWorkerHash)
		if err == nil && newInfo.TotalReadyWorkers() > 0 {
			logger.Info("Updating stale worker hash annotation",
				"prevWorkerHash", prevWorkerHash, "newHash", newWorkerHash)
			r.setCurrentWorkerHash(dgd, newWorkerHash)
			return r.Update(ctx, dgd)
		}
		// New spec change: reset to start a proper rolling update cycle with surge/drain.
		logger.Info("New worker spec change detected, starting new rolling update cycle",
			"prevWorkerHash", prevWorkerHash, "newHash", newWorkerHash,
			"previousPhase", rollingUpdateStatus.Phase)
		rollingUpdateStatus.Phase = nvidiacomv1alpha1.RollingUpdatePhaseNone
		rollingUpdateStatus.StartTime = nil
		rollingUpdateStatus.EndTime = nil
		rollingUpdateStatus.UpdatedServices = nil
	}

	if prevWorkerHash == newWorkerHash &&
		rollingUpdateStatus.Phase == nvidiacomv1alpha1.RollingUpdatePhaseInProgress {
		logger.Info("Detected stuck rolling update: hashes match but phase is InProgress",
			"hash", newWorkerHash,
			"phase", rollingUpdateStatus.Phase)
		return r.completeRollingUpdate(ctx, dgd, rollingUpdateStatus, newWorkerHash)
	}

	switch rollingUpdateStatus.Phase {
	case nvidiacomv1alpha1.RollingUpdatePhaseNone:
		return r.startRollingUpdate(ctx, dgd, rollingUpdateStatus, newWorkerHash)

	case nvidiacomv1alpha1.RollingUpdatePhasePending:
		rollingUpdateStatus.Phase = nvidiacomv1alpha1.RollingUpdatePhaseInProgress
		if err := r.Status().Update(ctx, dgd); err != nil {
			return fmt.Errorf("failed to update rolling update status to InProgress: %w", err)
		}
		return nil

	case nvidiacomv1alpha1.RollingUpdatePhaseInProgress:
		return r.continueRollingUpdate(ctx, dgd, rollingUpdateStatus, newWorkerHash)

	case nvidiacomv1alpha1.RollingUpdatePhaseCompleted:
		// Cleanup is now done atomically in completeRollingUpdate, nothing to do here
		logger.Info("Rolling update already completed")
		return nil
	}

	return nil
}

// startRollingUpdate initializes a new rolling update.
func (r *DynamoGraphDeploymentReconciler) startRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rollingUpdateStatus *nvidiacomv1alpha1.RollingUpdateStatus,
	newWorkerHash string,
) error {
	logger := log.FromContext(ctx)

	prevWorkerHash := r.getCurrentWorkerHash(dgd)

	logger.Info("Starting rolling update",
		"prevHash", prevWorkerHash,
		"newHash", newWorkerHash)

	now := metav1.Now()
	rollingUpdateStatus.Phase = nvidiacomv1alpha1.RollingUpdatePhasePending
	rollingUpdateStatus.StartTime = &now
	rollingUpdateStatus.UpdatedServices = nil

	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateStarted",
		"Starting rolling update from worker hash %s to %s", prevWorkerHash, newWorkerHash)

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
	newWorkerHash string,
) error {
	logger := log.FromContext(ctx)

	oldInfo, err := r.getOldWorkerInfo(ctx, dgd, newWorkerHash)
	if err != nil {
		logger.Error(err, "Failed to get old worker info")
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

	// Count total worker services
	totalWorkerServices := 0
	for _, spec := range dgd.Spec.Services {
		if spec != nil && dynamo.IsWorkerComponent(spec.ComponentType) {
			totalWorkerServices++
		}
	}

	// Rolling update is complete when every worker service is individually updated
	if len(updatedServices) == totalWorkerServices && totalWorkerServices > 0 {
		return r.completeRollingUpdate(ctx, dgd, rollingUpdateStatus, newWorkerHash)
	}

	// Persist updated services list mid-rolling update
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
	newWorkerHash string,
) error {
	logger := log.FromContext(ctx)

	// Delete all non-current worker DCDs (any number of old generations)
	if err := r.deleteOldWorkerDCDs(ctx, dgd, newWorkerHash); err != nil {
		logger.Error(err, "Failed to delete non-current worker DCDs", "newWorkerHash", newWorkerHash)
		r.Recorder.Eventf(dgd, corev1.EventTypeWarning, "CleanupPartialFailure",
			"Failed to delete some old worker DCDs: %v", err)
		// Continue anyway - we don't want cleanup failures to block the rolling update completion
	} else {
		logger.Info("Old resources cleaned up", "newWorkerHash", newWorkerHash)
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

// getOldWorkerInfo aggregates ready replicas across ALL non-current worker DCDs.
func (r *DynamoGraphDeploymentReconciler) getOldWorkerInfo(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	newWorkerHash string,
) (*dynamoNamespaceWorkerInfo, error) {
	oldDCDs, err := r.listOldWorkerDCDs(ctx, dgd, newWorkerHash)
	if err != nil {
		return nil, fmt.Errorf("failed to list non-current worker DCDs: %w", err)
	}

	status := &dynamoNamespaceWorkerInfo{
		services: make(map[string]*workerServiceInfo),
	}

	for _, dcd := range oldDCDs {
		readyReplicas := int32(0)
		if dcd.Status.Service != nil && dcd.Status.Service.ReadyReplicas != nil {
			readyReplicas = *dcd.Status.Service.ReadyReplicas
		}

		if existing, ok := status.services[dcd.Spec.ServiceName]; ok {
			existing.readyReplicas += readyReplicas
		} else {
			status.services[dcd.Spec.ServiceName] = &workerServiceInfo{
				readyReplicas: readyReplicas,
			}
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
// When multiple old generations exist for the same service, replicas are distributed to the
// newest old DCD first, with older DCDs drained to 0 (matching K8s Deployment controller behavior).
func (r *DynamoGraphDeploymentReconciler) scaleOldWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rollingUpdateCtx dynamo.RollingUpdateContext,
) error {
	logger := log.FromContext(ctx)

	if !rollingUpdateCtx.InProgress() {
		return nil
	}

	oldDCDs, err := r.listOldWorkerDCDs(ctx, dgd, rollingUpdateCtx.NewWorkerHash)
	if err != nil {
		return fmt.Errorf("failed to list old worker DCDs: %w", err)
	}

	// Group old DCDs by service name
	dcdsByService := make(map[string][]*nvidiacomv1alpha1.DynamoComponentDeployment)
	for i := range oldDCDs {
		svc := oldDCDs[i].Spec.ServiceName
		dcdsByService[svc] = append(dcdsByService[svc], &oldDCDs[i])
	}

	for serviceName, dcds := range dcdsByService {
		oldNeeded, ok := rollingUpdateCtx.OldWorkerReplicas[serviceName]
		if !ok {
			continue
		}

		// Sort by creation time descending (newest first) so newest old DCDs get replicas first
		sort.Slice(dcds, func(i, j int) bool {
			return dcds[i].CreationTimestamp.After(dcds[j].CreationTimestamp.Time)
		})

		remaining := oldNeeded
		for _, dcd := range dcds {
			var desiredReplicas int32
			if remaining > 0 {
				currentSpec := int32(1)
				if dcd.Spec.Replicas != nil {
					currentSpec = *dcd.Spec.Replicas
				}
				// Give this DCD up to its current spec count, but no more than remaining
				desiredReplicas = min(remaining, currentSpec)
				remaining -= desiredReplicas
			}

			currentReplicas := int32(1)
			if dcd.Spec.Replicas != nil {
				currentReplicas = *dcd.Spec.Replicas
			}

			if currentReplicas == desiredReplicas {
				logger.V(1).Info("Old worker DCD replicas already at desired value",
					"dcdName", dcd.Name, "replicas", desiredReplicas)
				continue
			}

			patch := client.MergeFrom(dcd.DeepCopy())
			dcd.Spec.Replicas = &desiredReplicas

			if err := r.Patch(ctx, dcd, patch); err != nil {
				return fmt.Errorf("failed to patch old worker DCD %s replicas: %w", dcd.Name, err)
			}

			logger.Info("Scaled old worker DCD",
				"dcdName", dcd.Name,
				"service", serviceName,
				"oldReplicas", currentReplicas,
				"newReplicas", desiredReplicas)
		}
	}

	return nil
}

// listOldWorkerDCDs returns all worker DCDs for this DGD whose worker hash label
// does NOT match the given newWorkerHash. This captures all old generations (including legacy).
func (r *DynamoGraphDeploymentReconciler) listOldWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	newWorkerHash string,
) ([]nvidiacomv1alpha1.DynamoComponentDeployment, error) {
	dcdList := &nvidiacomv1alpha1.DynamoComponentDeploymentList{}
	listOpts := []client.ListOption{
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		},
	}

	if err := r.List(ctx, dcdList, listOpts...); err != nil {
		return nil, err
	}

	var workers []nvidiacomv1alpha1.DynamoComponentDeployment
	for _, dcd := range dcdList.Items {
		if !dynamo.IsWorkerComponent(dcd.Spec.ComponentType) {
			continue
		}
		if dcd.Labels[consts.KubeLabelDynamoWorkerHash] != newWorkerHash {
			workers = append(workers, dcd)
		}
	}
	return workers, nil
}

// deleteOldWorkerDCDs deletes all worker DCDs belonging to this DGD whose hash label
// does NOT match the given newWorkerHash. This cleans up all old generations at once.
func (r *DynamoGraphDeploymentReconciler) deleteOldWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	newWorkerHash string,
) error {
	logger := log.FromContext(ctx)

	oldDCDs, err := r.listOldWorkerDCDs(ctx, dgd, newWorkerHash)
	if err != nil {
		return fmt.Errorf("failed to list non-current worker DCDs: %w", err)
	}

	if len(oldDCDs) == 0 {
		logger.Info("No non-current worker DCDs found to delete", "newWorkerHash", newWorkerHash)
		return nil
	}

	logger.Info("Deleting non-current worker DCDs", "count", len(oldDCDs), "newWorkerHash", newWorkerHash)

	var deleteErrors []error
	for i := range oldDCDs {
		dcd := &oldDCDs[i]
		logger.Info("Deleting non-current worker DCD", "name", dcd.Name, "hash", dcd.Labels[consts.KubeLabelDynamoWorkerHash])

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

// aggregateOldWorkerServiceStatuses fetches all non-current worker DCDs and returns their
// aggregated service statuses keyed by service name. Accumulates across multiple old generations.
func (r *DynamoGraphDeploymentReconciler) aggregateOldWorkerServiceStatuses(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rollingUpdateCtx dynamo.RollingUpdateContext,
) (map[string]nvidiacomv1alpha1.ServiceReplicaStatus, error) {
	oldStatuses := make(map[string]nvidiacomv1alpha1.ServiceReplicaStatus)

	oldDCDs, err := r.listOldWorkerDCDs(ctx, dgd, rollingUpdateCtx.NewWorkerHash)
	if err != nil {
		return nil, fmt.Errorf("failed to list old worker DCDs for status aggregation: %w", err)
	}

	for _, dcd := range oldDCDs {
		if _, inRollout := rollingUpdateCtx.OldWorkerReplicas[dcd.Spec.ServiceName]; !inRollout {
			continue
		}
		if dcd.Status.Service == nil {
			continue
		}
		existing, found := oldStatuses[dcd.Spec.ServiceName]
		if !found {
			status := *dcd.Status.Service
			status.ComponentNames = []string{dcd.Status.Service.ComponentName}
			oldStatuses[dcd.Spec.ServiceName] = status
		} else {
			// Accumulate across multiple old DCDs
			existing.Replicas += dcd.Status.Service.Replicas
			existing.ReadyReplicas = addOptionalInt32(existing.ReadyReplicas, dcd.Status.Service.ReadyReplicas)
			existing.AvailableReplicas = addOptionalInt32(existing.AvailableReplicas, dcd.Status.Service.AvailableReplicas)
			existing.ComponentNames = append(existing.ComponentNames, dcd.Status.Service.ComponentName)
			oldStatuses[dcd.Spec.ServiceName] = existing
		}
	}

	return oldStatuses, nil
}

// resolveRollingUpdateParams reads the deployment strategy annotations from a service spec
// and resolves maxSurge and maxUnavailable to concrete replica counts.
// Defaults: maxSurge=25%, maxUnavailable=25% (matches Kubernetes Deployment defaults).
// TODO: support the recreate strategy
func resolveRollingUpdateParams(annotations map[string]string, desiredReplicas int32) (maxSurge int32, maxUnavailable int32) {
	surgeValue := intstr.FromString("25%")
	unavailValue := intstr.FromString("25%")

	if v := annotations[KubeAnnotationDeploymentRollingUpdateMaxSurge]; v != "" {
		surgeValue = intstr.Parse(v)
	}
	if v := annotations[KubeAnnotationDeploymentRollingUpdateMaxUnavailable]; v != "" {
		unavailValue = intstr.Parse(v)
	}

	// Resolve percentages against desiredReplicas. Round up for surge (more aggressive scale-up),
	// round down for unavailable (more conservative, matches Kubernetes deployment controller behavior).
	// https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#max-unavailable
	surge, _ := intstr.GetScaledValueFromIntOrPercent(&surgeValue, int(desiredReplicas), true)
	unavail, _ := intstr.GetScaledValueFromIntOrPercent(&unavailValue, int(desiredReplicas), false)

	// Ensure at least one of surge/unavailable is > 0 to guarantee progress
	if surge == 0 && unavail == 0 {
		surge = 1
	}

	return int32(surge), int32(unavail)
}

// buildRollingUpdateContext creates a RollingUpdateContext.
// It computes namespaces and pre-calculates old and new worker replica counts.
//
// Replica calculation:
//   - oldReplicas = max(0, desiredReplicas - newReadyReplicas - maxUnavailable)
//   - newReplicas = min(desiredReplicas, desiredReplicas + maxSurge - oldReplicas)
func (r *DynamoGraphDeploymentReconciler) buildRollingUpdateContext(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) dynamo.RollingUpdateContext {
	logger := log.FromContext(ctx)

	// Compute hashes
	newWorkerHash := dynamo.ComputeDGDWorkersSpecHash(dgd)
	prevWorkerHash := r.getCurrentWorkerHash(dgd)

	if prevWorkerHash == newWorkerHash {
		return dynamo.RollingUpdateContext{
			NewWorkerHash:     newWorkerHash,
			OldWorkerReplicas: make(map[string]int32),
			NewWorkerReplicas: make(map[string]int32),
		}
	}

	// Pre-calculate old and new worker replicas based on new worker readiness
	oldWorkerReplicas := make(map[string]int32)
	newWorkerReplicas := make(map[string]int32)

	for serviceName, spec := range dgd.Spec.Services {
		if spec == nil || !dynamo.IsWorkerComponent(spec.ComponentType) {
			continue
		}

		// Get desired replicas from spec
		desiredReplicas := int32(1)
		if spec.Replicas != nil {
			desiredReplicas = *spec.Replicas
		}

		maxSurge, maxUnavailable := resolveRollingUpdateParams(spec.Annotations, desiredReplicas)

		// Query new DCD to get ready replicas (using hash-based naming)
		newDCDName := dynamo.GetDCDResourceName(dgd, serviceName, newWorkerHash)
		newDCD := &nvidiacomv1alpha1.DynamoComponentDeployment{}
		err := r.Get(ctx, types.NamespacedName{Name: newDCDName, Namespace: dgd.Namespace}, newDCD)

		newReadyReplicas := int32(0)
		if err == nil && newDCD.Status.Service != nil && newDCD.Status.Service.ReadyReplicas != nil {
			newReadyReplicas = *newDCD.Status.Service.ReadyReplicas
		}

		// Calculate old replicas: allow scaling down by maxUnavailable
		// oldReplicas = max(0, desiredReplicas - newReadyReplicas - maxUnavailable)
		oldNeeded := desiredReplicas - newReadyReplicas - maxUnavailable
		if oldNeeded < 0 {
			oldNeeded = 0
		}

		// Calculate new replicas: stay within surge budget
		// newReplicas = min(desiredReplicas, desiredReplicas + maxSurge - oldNeeded)
		newNeeded := desiredReplicas + maxSurge - oldNeeded
		if newNeeded > desiredReplicas {
			newNeeded = desiredReplicas
		}
		if newNeeded < 0 {
			newNeeded = 0
		}

		newWorkerReplicas[serviceName] = newNeeded
		oldWorkerReplicas[serviceName] = oldNeeded

		logger.V(1).Info("Calculated worker replicas for rollingUpdate",
			"service", serviceName,
			"desired", desiredReplicas,
			"newReady", newReadyReplicas,
			"maxSurge", maxSurge,
			"maxUnavailable", maxUnavailable,
			"newNeeded", newNeeded,
			"oldNeeded", oldNeeded)
	}

	return dynamo.RollingUpdateContext{
		NewWorkerHash:     newWorkerHash,
		OldWorkerReplicas: oldWorkerReplicas,
		NewWorkerReplicas: newWorkerReplicas,
	}
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

		// Build sorted ComponentNames from old and new DCD names.
		componentNames := append(oldStatus.ComponentNames, newStatus.ComponentName)
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
