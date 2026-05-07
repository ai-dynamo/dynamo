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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

// shouldTriggerRollingUpdate determines if worker spec changes require a rolling update.
func (r *DynamoGraphDeploymentReconciler) shouldTriggerRollingUpdate(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (bool, error) {
	computedHash, err := nvidiacomv1beta1.ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		return false, fmt.Errorf("failed to compute worker hash: %w", err)
	}

	currentHash := r.getCurrentWorkerHash(dgd)

	// If no current hash exists (new deployment), no rolling update needed
	if currentHash == "" {
		return false, nil
	}

	if r.getCurrentWorkerHashVersion(dgd) != consts.CurrentWorkerHashVersionV2 {
		if currentHash == computedHash {
			return false, nil
		}
		legacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
		if err != nil {
			return false, fmt.Errorf("failed to compute legacy worker hash: %w", err)
		}
		if currentHash == legacyHash {
			return false, nil
		}
	}

	return computedHash != currentHash, nil
}

// initializeWorkerHashIfNeeded sets the current worker hash annotation on first deployment.
// For existing DGDs being upgraded from a pre-rolling-update operator version, this handles
// patching the legacy DCDs with the new worker hash label and then triggering a rolling update on the next reconcile.
func (r *DynamoGraphDeploymentReconciler) initializeWorkerHashIfNeeded(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	if r.getCurrentWorkerHash(dgd) != "" {
		return r.migrateCurrentWorkerHashIfNeeded(ctx, dgd)
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
		r.setLegacyWorkerHash(dgd)
		if err := r.Update(ctx, dgd); err != nil {
			return fmt.Errorf("failed to set legacy worker hash: %w", err)
		}

		r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "LegacyMigrationStarted",
			"Detected %d legacy worker DCDs, initiating rolling update migration", len(legacyDCDs))
		return nil
	}

	// Normal first deploy — set the actual computed hash
	hash, err := nvidiacomv1beta1.ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		return fmt.Errorf("failed to compute worker hash: %w", err)
	}
	r.setCurrentWorkerHash(dgd, hash)

	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to initialize worker hash: %w", err)
	}

	logger.Info("Initialized current worker hash", "hash", hash)

	return nil
}

func (r *DynamoGraphDeploymentReconciler) migrateCurrentWorkerHashIfNeeded(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	currentHash := r.getCurrentWorkerHash(dgd)
	if currentHash == "" || r.getCurrentWorkerHashVersion(dgd) == consts.CurrentWorkerHashVersionV2 {
		return nil
	}
	if currentHash == consts.LegacyWorkerHash {
		return nil
	}

	computedHash, err := nvidiacomv1beta1.ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		return fmt.Errorf("failed to compute worker hash: %w", err)
	}
	if currentHash != computedHash {
		legacyHash, err := dynamo.ComputeLegacyAlphaDGDWorkersSpecHash(dgd)
		if err != nil {
			return fmt.Errorf("failed to compute legacy worker hash: %w", err)
		}
		if currentHash != legacyHash {
			return nil
		}
	}

	r.setCurrentWorkerHash(dgd, computedHash)
	dynamo.ClearLegacyAlphaDGDWorkersSpecHash(dgd)
	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to migrate worker hash annotation: %w", err)
	}

	logger.Info("Migrated worker hash annotation to current version",
		"previousHash", currentHash,
		"currentHash", computedHash,
		"version", consts.CurrentWorkerHashVersionV2)
	r.Recorder.Event(dgd, corev1.EventTypeNormal, "WorkerHashMigrated",
		"Migrated worker hash annotation to the current hash version without rolling workers")

	return nil
}

// findLegacyWorkerDCDs returns worker DCDs owned by this DGD that lack the worker hash label.
// These are DCDs created by a pre-rolling-update operator version.
func (r *DynamoGraphDeploymentReconciler) findLegacyWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) ([]nvidiacomv1beta1.DynamoComponentDeployment, error) {
	// List all DCDs for this DGD
	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	listOpts := []client.ListOption{
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		},
	}

	if err := r.List(ctx, dcdList, listOpts...); err != nil {
		return nil, fmt.Errorf("failed to list DCDs for DGD %s: %w", dgd.Name, err)
	}

	var legacyDCDs []nvidiacomv1beta1.DynamoComponentDeployment
	for _, dcd := range dcdList.Items {
		if !dynamo.IsWorkerComponent(string(dcd.Spec.ComponentType)) {
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
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) bool {
	return !r.isGrovePathway(dgd) && !dgd.HasAnyMultinodeComponent()
}

// getCurrentWorkerHash returns the stored worker hash from DGD annotations.
// during a rolling update, this is the previous worker hash and is not updated until the rolling update is completed.
// Returns empty string if no hash has been set (new deployment).
func (r *DynamoGraphDeploymentReconciler) getCurrentWorkerHash(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) string {
	if dgd.Annotations == nil {
		return ""
	}
	return dgd.Annotations[consts.AnnotationCurrentWorkerHash]
}

func (r *DynamoGraphDeploymentReconciler) getCurrentWorkerHashVersion(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) string {
	if dgd.Annotations == nil {
		return ""
	}
	return dgd.Annotations[consts.AnnotationCurrentWorkerHashVersion]
}

// setCurrentWorkerHash stores the worker hash in DGD annotations.
func (r *DynamoGraphDeploymentReconciler) setCurrentWorkerHash(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	hash string,
) {
	if dgd.Annotations == nil {
		dgd.Annotations = make(map[string]string)
	}
	dgd.Annotations[consts.AnnotationCurrentWorkerHash] = hash
	dgd.Annotations[consts.AnnotationCurrentWorkerHashVersion] = consts.CurrentWorkerHashVersionV2
}

func (r *DynamoGraphDeploymentReconciler) setLegacyWorkerHash(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) {
	if dgd.Annotations == nil {
		dgd.Annotations = make(map[string]string)
	}
	dgd.Annotations[consts.AnnotationCurrentWorkerHash] = consts.LegacyWorkerHash
	delete(dgd.Annotations, consts.AnnotationCurrentWorkerHashVersion)
}

// getOrCreateRollingUpdateStatus returns the existing rolling update status or creates a new one.
func (r *DynamoGraphDeploymentReconciler) getOrCreateRollingUpdateStatus(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) *nvidiacomv1beta1.RollingUpdateStatus {
	if dgd.Status.RollingUpdate == nil {
		dgd.Status.RollingUpdate = &nvidiacomv1beta1.RollingUpdateStatus{
			Phase: nvidiacomv1beta1.RollingUpdatePhaseNone,
		}
	}
	return dgd.Status.RollingUpdate
}

// isRollingUpdateInProgress returns true if a rolling update is currently active.
func (r *DynamoGraphDeploymentReconciler) isRollingUpdateInProgress(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) bool {
	if dgd.Status.RollingUpdate == nil {
		return false
	}
	phase := dgd.Status.RollingUpdate.Phase
	return phase == nvidiacomv1beta1.RollingUpdatePhasePending ||
		phase == nvidiacomv1beta1.RollingUpdatePhaseInProgress
}

// reconcileRollingUpdate handles the rolling update lifecycle.
func (r *DynamoGraphDeploymentReconciler) reconcileRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	rollingUpdateStatus := r.getOrCreateRollingUpdateStatus(dgd)

	newWorkerHash, err := nvidiacomv1beta1.ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		return fmt.Errorf("failed to compute worker hash: %w", err)
	}
	prevWorkerHash := r.getCurrentWorkerHash(dgd)

	logger.Info("Reconciling rolling update",
		"phase", rollingUpdateStatus.Phase,
		"prevWorkerHash", prevWorkerHash,
		"newWorkerHash", newWorkerHash)

	if (rollingUpdateStatus.Phase == nvidiacomv1beta1.RollingUpdatePhaseCompleted) && prevWorkerHash != newWorkerHash {
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
		rollingUpdateStatus.Phase = nvidiacomv1beta1.RollingUpdatePhaseNone
		rollingUpdateStatus.StartTime = nil
		rollingUpdateStatus.EndTime = nil
		rollingUpdateStatus.UpdatedComponents = nil
	}

	if prevWorkerHash == newWorkerHash &&
		rollingUpdateStatus.Phase == nvidiacomv1beta1.RollingUpdatePhaseInProgress {
		logger.Info("Detected stuck rolling update: hashes match but phase is InProgress",
			"hash", newWorkerHash,
			"phase", rollingUpdateStatus.Phase)
		return r.completeRollingUpdate(ctx, dgd, newWorkerHash)
	}

	switch rollingUpdateStatus.Phase {
	case nvidiacomv1beta1.RollingUpdatePhaseNone:
		return r.startRollingUpdate(ctx, dgd, newWorkerHash)

	case nvidiacomv1beta1.RollingUpdatePhasePending:
		rollingUpdateStatus.Phase = nvidiacomv1beta1.RollingUpdatePhaseInProgress
		return nil // deferred function in Reconcile() persists status

	case nvidiacomv1beta1.RollingUpdatePhaseInProgress:
		return r.continueRollingUpdate(ctx, dgd, newWorkerHash)

	case nvidiacomv1beta1.RollingUpdatePhaseCompleted:
		logger.Info("Rolling update already completed")
		return nil
	}

	return nil
}

// startRollingUpdate initializes a new rolling update.
func (r *DynamoGraphDeploymentReconciler) startRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	newWorkerHash string,
) error {
	logger := log.FromContext(ctx)

	prevWorkerHash := r.getCurrentWorkerHash(dgd)

	logger.Info("Starting rolling update",
		"prevHash", prevWorkerHash,
		"newHash", newWorkerHash)

	now := metav1.Now()
	rollingUpdateStatus := r.getOrCreateRollingUpdateStatus(dgd)
	rollingUpdateStatus.Phase = nvidiacomv1beta1.RollingUpdatePhasePending
	rollingUpdateStatus.StartTime = &now
	rollingUpdateStatus.UpdatedComponents = nil

	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateStarted",
		"Starting rolling update from worker hash %s to %s", prevWorkerHash, newWorkerHash)

	return nil // deferred function in Reconcile() persists status
}

// continueRollingUpdate handles the in-progress phase of a rolling update.
func (r *DynamoGraphDeploymentReconciler) continueRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
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

	// Compute per-component completion.
	var updatedComponents []string
	for i := range dgd.Spec.Components {
		spec := &dgd.Spec.Components[i]
		componentName := spec.ComponentName
		if !dynamo.IsWorkerComponent(string(spec.ComponentType)) {
			continue
		}

		desired := int32(1)
		if spec.Replicas != nil {
			desired = *spec.Replicas
		}

		newComponent := newInfo.components[componentName]
		oldComponent := oldInfo.components[componentName]

		newReady := newComponent != nil && newComponent.readyReplicas >= desired
		oldGone := oldComponent == nil || oldComponent.readyReplicas == 0

		if newReady && oldGone {
			updatedComponents = append(updatedComponents, componentName)
		}
	}
	sort.Strings(updatedComponents)
	rollingUpdateStatus := r.getOrCreateRollingUpdateStatus(dgd)
	rollingUpdateStatus.UpdatedComponents = updatedComponents

	// Count total worker components.
	totalWorkerComponents := 0
	for i := range dgd.Spec.Components {
		spec := &dgd.Spec.Components[i]
		if dynamo.IsWorkerComponent(string(spec.ComponentType)) {
			totalWorkerComponents++
		}
	}

	// Rolling update is complete when every worker component is individually updated.
	if len(updatedComponents) == totalWorkerComponents && totalWorkerComponents > 0 {
		return r.completeRollingUpdate(ctx, dgd, newWorkerHash)
	}

	return nil // deferred function in Reconcile() persists UpdatedComponents
}

// completeRollingUpdate marks the rolling update as completed, cleans up old resources, and updates status.
// This performs all cleanup atomically to avoid race conditions with subsequent reconciles.
func (r *DynamoGraphDeploymentReconciler) completeRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	newWorkerHash string,
) error {
	logger := log.FromContext(ctx)

	// Delete all non-current worker DCDs (any number of old generations)
	if err := r.deleteOldWorkerDCDs(ctx, dgd, newWorkerHash); err != nil {
		return fmt.Errorf("failed to delete old worker DCDs: %w", err)
	}

	r.setCurrentWorkerHash(dgd, newWorkerHash)
	if err := r.Update(ctx, dgd); err != nil {
		return fmt.Errorf("failed to update current worker hash: %w", err)
	}

	rollingUpdateStatus := r.getOrCreateRollingUpdateStatus(dgd)
	rollingUpdateStatus.Phase = nvidiacomv1beta1.RollingUpdatePhaseCompleted
	now := metav1.Now()
	rollingUpdateStatus.EndTime = &now

	// Mark all worker components as updated.
	var allWorkerComponents []string
	for i := range dgd.Spec.Components {
		spec := &dgd.Spec.Components[i]
		componentName := spec.ComponentName
		if dynamo.IsWorkerComponent(string(spec.ComponentType)) {
			allWorkerComponents = append(allWorkerComponents, componentName)
		}
	}
	sort.Strings(allWorkerComponents)
	rollingUpdateStatus.UpdatedComponents = allWorkerComponents

	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateCompleted",
		"Rolling update completed, worker hash %s", newWorkerHash)

	logger.Info("Rolling update finalized", "newWorkerHash", newWorkerHash)

	return nil
}

// dcdComponentState holds replica signals extracted from a DCD's Spec and Status.
type dcdComponentState struct {
	Spec      int32 `json:"spec"`      // DCD Spec.Replicas (declared intent)
	Available int32 `json:"available"` // Status.Component.AvailableReplicas (serving traffic)
	Actual    int32 `json:"actual"`    // Status.Component.Replicas (non-terminated pods, excludes Terminating)
}

// dcdComponentStateFromDCD extracts replica signals from a single DCD.
func dcdComponentStateFromDCD(dcd *nvidiacomv1beta1.DynamoComponentDeployment) dcdComponentState {
	s := dcdComponentState{}
	if dcd.Spec.Replicas != nil {
		s.Spec = *dcd.Spec.Replicas
	}
	if dcd.Status.Component != nil {
		s.Actual = dcd.Status.Component.Replicas
		if dcd.Status.Component.AvailableReplicas != nil {
			s.Available = *dcd.Status.Component.AvailableReplicas
		}
	}
	return s
}

// workerComponentInfo holds ready replica count for a worker component.
type workerComponentInfo struct {
	readyReplicas int32
	desired       int32
}

// dynamoNamespaceWorkerInfo holds aggregated worker status for a single dynamo namespace.
type dynamoNamespaceWorkerInfo struct {
	// totalReadyWorkers is the sum of ready replicas across all worker components.
	totalReadyWorkers int32
	// components contains per-component status (e.g., "prefill", "decode", "worker").
	components map[string]*workerComponentInfo
}

func (s *dynamoNamespaceWorkerInfo) TotalReadyWorkers() int32 {
	return s.totalReadyWorkers
}

// getWorkerInfoForWorkerHash queries DCDs for a specific worker hash and returns
// aggregated worker info.
func (r *DynamoGraphDeploymentReconciler) getWorkerInfoForWorkerHash(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	workerHash string,
) (*dynamoNamespaceWorkerInfo, error) {
	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
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
		components: make(map[string]*workerComponentInfo),
	}

	for _, dcd := range dcdList.Items {
		if !dynamo.IsWorkerComponent(string(dcd.Spec.ComponentType)) {
			continue
		}
		componentName := dynamo.GetDCDComponentName(&dcd)

		// Add ready replicas
		readyReplicas := int32(0)
		if dcd.Status.Component != nil && dcd.Status.Component.ReadyReplicas != nil {
			readyReplicas = *dcd.Status.Component.ReadyReplicas
		}

		// Add desired replicas
		desiredReplicas := int32(0)
		if dcd.Spec.Replicas != nil {
			desiredReplicas = *dcd.Spec.Replicas
		}
		status.components[componentName] = &workerComponentInfo{
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
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	newWorkerHash string,
) (*dynamoNamespaceWorkerInfo, error) {
	oldDCDs, err := r.listOldWorkerDCDs(ctx, dgd, newWorkerHash)
	if err != nil {
		return nil, fmt.Errorf("failed to list non-current worker DCDs: %w", err)
	}

	status := &dynamoNamespaceWorkerInfo{
		components: make(map[string]*workerComponentInfo),
	}

	for _, dcd := range oldDCDs {
		componentName := dynamo.GetDCDComponentName(&dcd)
		readyReplicas := int32(0)
		if dcd.Status.Component != nil && dcd.Status.Component.ReadyReplicas != nil {
			readyReplicas = *dcd.Status.Component.ReadyReplicas
		}

		if existing, ok := status.components[componentName]; ok {
			existing.readyReplicas += readyReplicas
		} else {
			status.components[componentName] = &workerComponentInfo{
				readyReplicas: readyReplicas,
			}
		}

		status.totalReadyWorkers += readyReplicas
	}

	return status, nil
}

// getOldWorkerComponentStates returns per-component old DCD state aggregated across all old generations.
func (r *DynamoGraphDeploymentReconciler) getOldWorkerComponentStates(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	newWorkerHash string,
) (map[string]dcdComponentState, error) {
	oldDCDs, err := r.listOldWorkerDCDs(ctx, dgd, newWorkerHash)
	if err != nil {
		return nil, err
	}

	states := make(map[string]dcdComponentState)
	for i := range oldDCDs {
		componentName := dynamo.GetDCDComponentName(&oldDCDs[i])
		s := dcdComponentStateFromDCD(&oldDCDs[i])
		agg := states[componentName]
		agg.Spec += s.Spec
		agg.Available += s.Available
		agg.Actual += s.Actual
		states[componentName] = agg
	}

	return states, nil
}

// getDesiredWorkerReplicas returns the total desired replicas across all worker components.
func (r *DynamoGraphDeploymentReconciler) getDesiredWorkerReplicas(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) int32 {
	var total int32
	for i := range dgd.Spec.Components {
		spec := &dgd.Spec.Components[i]
		if dynamo.IsWorkerComponent(string(spec.ComponentType)) {
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
// When multiple old generations exist for the same component, replicas are distributed to the
// newest old DCD first, with older DCDs drained to 0 (matching K8s Deployment controller behavior).
func (r *DynamoGraphDeploymentReconciler) scaleOldWorkerDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
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

	// Group old DCDs by logical component name.
	dcdsByComponent := make(map[string][]*nvidiacomv1beta1.DynamoComponentDeployment)
	for i := range oldDCDs {
		componentName := dynamo.GetDCDComponentName(&oldDCDs[i])
		dcdsByComponent[componentName] = append(dcdsByComponent[componentName], &oldDCDs[i])
	}

	for componentName, dcds := range dcdsByComponent {
		oldNeeded, ok := rollingUpdateCtx.OldWorkerReplicas[componentName]
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
				"component", componentName,
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
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	newWorkerHash string,
) ([]nvidiacomv1beta1.DynamoComponentDeployment, error) {
	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	listOpts := []client.ListOption{
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		},
	}

	if err := r.List(ctx, dcdList, listOpts...); err != nil {
		return nil, err
	}

	var workers []nvidiacomv1beta1.DynamoComponentDeployment
	for _, dcd := range dcdList.Items {
		if !dynamo.IsWorkerComponent(string(dcd.Spec.ComponentType)) {
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
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
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

// aggregateOldWorkerComponentStatuses fetches all non-current worker DCDs and returns their
// aggregated component statuses keyed by component name. Accumulates across multiple old generations.
func (r *DynamoGraphDeploymentReconciler) aggregateOldWorkerComponentStatuses(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	rollingUpdateCtx dynamo.RollingUpdateContext,
) (map[string]nvidiacomv1beta1.ComponentReplicaStatus, error) {
	oldStatuses := make(map[string]nvidiacomv1beta1.ComponentReplicaStatus)

	oldDCDs, err := r.listOldWorkerDCDs(ctx, dgd, rollingUpdateCtx.NewWorkerHash)
	if err != nil {
		return nil, fmt.Errorf("failed to list old worker DCDs for status aggregation: %w", err)
	}

	for _, dcd := range oldDCDs {
		componentName := dynamo.GetDCDComponentName(&dcd)
		if _, inRollout := rollingUpdateCtx.OldWorkerReplicas[componentName]; !inRollout {
			continue
		}
		if dcd.Status.Component == nil {
			continue
		}
		existing, found := oldStatuses[componentName]
		if !found {
			status := *dcd.Status.Component
			status.ComponentNames = componentReplicaResourceNames(dcd.Status.Component, dcd.Name)
			oldStatuses[componentName] = status
		} else {
			// Accumulate across multiple old DCDs
			existing.Replicas += dcd.Status.Component.Replicas
			existing.ReadyReplicas = addOptionalInt32(existing.ReadyReplicas, dcd.Status.Component.ReadyReplicas)
			existing.AvailableReplicas = addOptionalInt32(existing.AvailableReplicas, dcd.Status.Component.AvailableReplicas)
			existing.ComponentNames = append(existing.ComponentNames, componentReplicaResourceNames(dcd.Status.Component, dcd.Name)...)
			oldStatuses[componentName] = existing
		}
	}

	return oldStatuses, nil
}

// resolveRollingUpdateParams reads the deployment strategy annotations from a component spec
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
func (r *DynamoGraphDeploymentReconciler) buildRollingUpdateContext(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (dynamo.RollingUpdateContext, error) {
	logger := log.FromContext(ctx)

	newWorkerHash, err := nvidiacomv1beta1.ComputeDGDWorkersSpecHash(dgd)
	if err != nil {
		return dynamo.RollingUpdateContext{}, fmt.Errorf("failed to compute worker hash: %w", err)
	}
	prevWorkerHash := r.getCurrentWorkerHash(dgd)

	if prevWorkerHash == newWorkerHash {
		return dynamo.RollingUpdateContext{
			NewWorkerHash:     newWorkerHash,
			OldWorkerReplicas: make(map[string]int32),
			NewWorkerReplicas: make(map[string]int32),
		}, nil
	}

	oldStates, err := r.getOldWorkerComponentStates(ctx, dgd, newWorkerHash)
	if err != nil {
		return dynamo.RollingUpdateContext{}, fmt.Errorf("failed to get old worker component states: %w", err)
	}

	oldWorkerReplicas := make(map[string]int32)
	newWorkerReplicas := make(map[string]int32)

	for i := range dgd.Spec.Components {
		spec := &dgd.Spec.Components[i]
		componentName := spec.ComponentName
		if !dynamo.IsWorkerComponent(string(spec.ComponentType)) {
			continue
		}

		desired := int32(1)
		if spec.Replicas != nil {
			desired = *spec.Replicas
		}

		maxSurge, maxUnavailable := resolveRollingUpdateParams(dynamo.GetPodTemplateAnnotations(spec), desired)
		minAvailable := desired - maxUnavailable

		var newState dcdComponentState
		newDCDName := dynamo.GetDCDResourceName(dgd, componentName, newWorkerHash)
		newDCD := &nvidiacomv1beta1.DynamoComponentDeployment{}
		if err := r.Get(ctx, types.NamespacedName{Name: newDCDName, Namespace: dgd.Namespace}, newDCD); err == nil {
			newState = dcdComponentStateFromDCD(newDCD)
		} else if !apierrors.IsNotFound(err) {
			return dynamo.RollingUpdateContext{}, fmt.Errorf("failed to get new worker DCD %s: %w", newDCDName, err)
		}

		oldState := oldStates[componentName]

		newUnavailable := max(int32(0), newState.Spec-newState.Available)
		// maxScaledDown is the maximum number of old replicas that can be scaled down
		maxScaledDown := max(int32(0), (oldState.Spec+newState.Spec)-minAvailable-newUnavailable)
		oldUnhealthy := max(int32(0), oldState.Spec-oldState.Available)
		// availableSurplus is how many extra available replicas we have above minAvailable (min 0)
		availableSurplus := max(int32(0), (oldState.Available+newState.Available)-minAvailable)
		oldTarget := max(int32(0), oldState.Spec-min(maxScaledDown, oldUnhealthy+availableSurplus))

		// Surge budget uses Spec (declared intent) like K8s Deployment controller; scheduler enforces actual resource constraints.
		scaleUpBudget := max(int32(0), desired+maxSurge-oldState.Spec-newState.Spec)
		newTarget := min(desired, newState.Spec+scaleUpBudget)

		oldWorkerReplicas[componentName] = oldTarget
		newWorkerReplicas[componentName] = newTarget

		logger.V(1).Info("Rolling update replica calculation",
			"component", componentName,
			"desired", desired,
			"maxSurge", maxSurge,
			"maxUnavailable", maxUnavailable,
			"minAvailable", minAvailable,
			"old", oldState,
			"new", newState,
			"oldTarget", oldTarget,
			"newTarget", newTarget)
	}

	return dynamo.RollingUpdateContext{
		NewWorkerHash:     newWorkerHash,
		OldWorkerReplicas: oldWorkerReplicas,
		NewWorkerReplicas: newWorkerReplicas,
	}, nil
}

// mergeWorkerComponentStatuses merges old worker component statuses into the existing component statuses.
// For each worker component present in both maps, it aggregates replica counts so that the status
// reflects the total across old and new worker DCDs during a rolling update.
func mergeWorkerComponentStatuses(
	componentStatuses map[string]nvidiacomv1beta1.ComponentReplicaStatus,
	oldWorkerStatuses map[string]nvidiacomv1beta1.ComponentReplicaStatus,
) {
	for componentName, oldStatus := range oldWorkerStatuses {
		newStatus, exists := componentStatuses[componentName]
		if !exists {
			continue
		}

		// Build sorted ComponentNames from old and new DCD names.
		componentNames := append(slices.Clone(oldStatus.ComponentNames), newStatus.ComponentNames...)
		slices.Sort(componentNames)
		newStatus.ComponentNames = componentNames

		// Aggregate replica counts
		newStatus.Replicas += oldStatus.Replicas
		// UpdatedReplicas stays as-is (only new are "updated")
		newStatus.ReadyReplicas = addOptionalInt32(newStatus.ReadyReplicas, oldStatus.ReadyReplicas)
		newStatus.AvailableReplicas = addOptionalInt32(newStatus.AvailableReplicas, oldStatus.AvailableReplicas)

		componentStatuses[componentName] = newStatus
	}
}

func componentReplicaResourceNames(status *nvidiacomv1beta1.ComponentReplicaStatus, fallback string) []string {
	if status == nil {
		return nil
	}
	if len(status.ComponentNames) > 0 {
		return slices.Clone(status.ComponentNames)
	}
	if fallback == "" {
		return nil
	}
	return []string{fallback}
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
