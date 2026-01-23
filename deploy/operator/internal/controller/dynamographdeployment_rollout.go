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

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/proxy"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/rollout"
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
// The algorithm follows maxSurge=1, maxUnavailable=0:
// 1. Scale up one new worker
// 2. Wait for it to become ready
// 3. Scale down one old worker
// 4. Update HAProxy weights proportionally
// 5. Repeat until all workers are migrated
func (r *DynamoGraphDeploymentReconciler) continueRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rolloutStatus *nvidiacomv1alpha1.RolloutStatus,
	oldNamespace, newNamespace string,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	workerServices := r.getWorkerServices(dgd)
	if len(workerServices) == 0 {
		logger.Info("No worker services found, completing rollout")
		return r.completeRollout(ctx, dgd, rolloutStatus, newNamespace)
	}

	// Count ready workers in old and new namespaces
	oldReadyWorkers, err := r.countReadyWorkersInNamespace(ctx, dgd, oldNamespace)
	if err != nil {
		logger.Error(err, "Failed to count old workers")
		// Continue anyway - we may be in a state where old DCDs don't exist yet
		oldReadyWorkers = 0
	}

	newReadyWorkers, err := r.countReadyWorkersInNamespace(ctx, dgd, newNamespace)
	if err != nil {
		logger.Error(err, "Failed to count new workers")
		newReadyWorkers = 0
	}

	// Get desired total replicas
	desiredReplicas := r.getDesiredWorkerReplicas(dgd)

	logger.Info("Rolling update progress",
		"oldReadyWorkers", oldReadyWorkers,
		"newReadyWorkers", newReadyWorkers,
		"desiredReplicas", desiredReplicas,
		"oldNamespace", oldNamespace,
		"newNamespace", newNamespace)

	// Check if rollout is complete
	if newReadyWorkers >= desiredReplicas && oldReadyWorkers == 0 {
		return r.completeRollout(ctx, dgd, rolloutStatus, newNamespace)
	}

	// Calculate and update traffic weights based on ready workers
	totalReady := oldReadyWorkers + newReadyWorkers
	if totalReady > 0 {
		newWeight := (newReadyWorkers * 100) / totalReady
		oldWeight := 100 - newWeight

		// Only update weights if they changed
		if rolloutStatus.TrafficWeightOld != oldWeight || rolloutStatus.TrafficWeightNew != newWeight {
			rolloutStatus.TrafficWeightOld = oldWeight
			rolloutStatus.TrafficWeightNew = newWeight

			// Update HAProxy weights
			if err := r.updateProxyWeights(ctx, dgd, oldWeight, newWeight); err != nil {
				logger.Error(err, "Failed to update proxy weights", "oldWeight", oldWeight, "newWeight", newWeight)
				// Continue anyway - the weights will be updated on next reconciliation
			} else {
				logger.Info("Updated proxy weights", "oldWeight", oldWeight, "newWeight", newWeight)
			}
		}
	}

	// Update status
	if err := r.Status().Update(ctx, dgd); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to update rollout status: %w", err)
	}

	// Requeue to continue monitoring progress
	// The normal DCD reconciliation will handle creating/scaling the new workers
	return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
}

// completeRollout marks the rollout as completed and updates status.
func (r *DynamoGraphDeploymentReconciler) completeRollout(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	rolloutStatus *nvidiacomv1alpha1.RolloutStatus,
	newNamespace string,
) (ctrl.Result, error) {
	rolloutStatus.Phase = nvidiacomv1alpha1.RolloutPhaseCompleted
	rolloutStatus.TrafficWeightOld = 0
	rolloutStatus.TrafficWeightNew = 100
	now := metav1.Now()
	rolloutStatus.EndTime = &now

	// Ensure HAProxy has 100% traffic to new
	if err := r.updateProxyWeights(ctx, dgd, 0, 100); err != nil {
		log.FromContext(ctx).Error(err, "Failed to set final proxy weights")
	}

	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateCompleted",
		"Rolling update completed, traffic shifted to namespace %s", newNamespace)

	if err := r.Status().Update(ctx, dgd); err != nil {
		return ctrl.Result{}, fmt.Errorf("failed to update rollout status: %w", err)
	}

	return ctrl.Result{Requeue: true}, nil
}

// countReadyWorkersInNamespace counts ready worker replicas across all worker DCDs
// in the specified Dynamo namespace.
func (r *DynamoGraphDeploymentReconciler) countReadyWorkersInNamespace(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	dynamoNamespace string,
) (int32, error) {
	// List DCDs that belong to this DGD and have the specified Dynamo namespace
	dcdList := &nvidiacomv1alpha1.DynamoComponentDeploymentList{}
	listOpts := []client.ListOption{
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			consts.KubeLabelDynamoNamespace:           dynamoNamespace,
		},
	}

	if err := r.List(ctx, dcdList, listOpts...); err != nil {
		return 0, fmt.Errorf("failed to list DCDs: %w", err)
	}

	var totalReady int32
	for _, dcd := range dcdList.Items {
		// Only count worker components
		if !dynamo.IsWorkerComponent(dcd.Spec.ComponentType) {
			continue
		}

		// Check DCD status for ready replicas (via Service status)
		if dcd.Status.Service != nil && dcd.Status.Service.ReadyReplicas != nil {
			totalReady += *dcd.Status.Service.ReadyReplicas
		}
	}

	return totalReady, nil
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

// updateProxyWeights updates the HAProxy backend weights via the runtime API.
func (r *DynamoGraphDeploymentReconciler) updateProxyWeights(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	oldWeight, newWeight int32,
) error {
	// Get the HAProxy service address
	// The proxy service is named <dgd-name>-traffic-proxy in the same namespace
	proxyServiceName := fmt.Sprintf("%s-traffic-proxy", dgd.Name)
	proxyHost := fmt.Sprintf("%s.%s.svc.cluster.local", proxyServiceName, dgd.Namespace)

	// Create HAProxy client connecting to the runtime API port
	haproxyClient := proxy.NewHAProxyClientTCP(proxyHost, consts.HAProxyRuntimePort)

	// Update weights
	if err := haproxyClient.UpdateWeights(ctx, oldWeight, newWeight); err != nil {
		return fmt.Errorf("failed to update HAProxy weights: %w", err)
	}

	return nil
}

// finalizeRollingUpdate cleans up after a completed rolling update.
// This deletes old DCDs from the old namespace and updates the active worker hash.
func (r *DynamoGraphDeploymentReconciler) finalizeRollingUpdate(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	newWorkerHash string,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Get the old worker hash to identify old resources
	oldWorkerHash := r.getCurrentActiveWorkerHash(dgd)
	if oldWorkerHash == "" {
		// No old hash means this is likely a first deployment, nothing to clean up
		logger.Info("No old worker hash found, skipping cleanup")
		return r.completeFinalization(ctx, dgd, newWorkerHash)
	}

	if oldWorkerHash == newWorkerHash {
		// Hashes are the same, nothing to clean up
		logger.Info("Old and new worker hashes are the same, skipping cleanup")
		return r.completeFinalization(ctx, dgd, newWorkerHash)
	}

	oldNamespace := dynamo.ComputeHashedDynamoNamespaceWithHash(dgd, oldWorkerHash)

	// Delete old DCDs from the old namespace
	if err := r.deleteOldDCDs(ctx, dgd, oldNamespace); err != nil {
		logger.Error(err, "Failed to delete old DCDs", "oldNamespace", oldNamespace)
		// Don't fail the whole finalization, just log the error
		// The old resources will be orphaned but won't affect the new deployment
		r.Recorder.Eventf(dgd, corev1.EventTypeWarning, "CleanupPartialFailure",
			"Failed to delete some old resources from namespace %s: %v", oldNamespace, err)
	}

	logger.Info("Old resources cleaned up", "oldNamespace", oldNamespace, "oldWorkerHash", oldWorkerHash)
	r.Recorder.Eventf(dgd, corev1.EventTypeNormal, "RollingUpdateCleanup",
		"Cleaned up old resources from namespace %s", oldNamespace)

	return r.completeFinalization(ctx, dgd, newWorkerHash)
}

// completeFinalization updates the active worker hash after cleanup.
// The rollout status is preserved (not cleared) so users can see completion info.
// It will be reset when a new rollout starts.
func (r *DynamoGraphDeploymentReconciler) completeFinalization(
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

	return ctrl.Result{}, nil
}

// deleteOldDCDs deletes all DCDs belonging to this DGD that have the old Dynamo namespace.
func (r *DynamoGraphDeploymentReconciler) deleteOldDCDs(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	oldDynamoNamespace string,
) error {
	logger := log.FromContext(ctx)

	// List all DCDs that belong to this DGD and have the old Dynamo namespace
	dcdList := &nvidiacomv1alpha1.DynamoComponentDeploymentList{}
	listOpts := []client.ListOption{
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			consts.KubeLabelDynamoNamespace:           oldDynamoNamespace,
		},
	}

	if err := r.List(ctx, dcdList, listOpts...); err != nil {
		return fmt.Errorf("failed to list old DCDs: %w", err)
	}

	if len(dcdList.Items) == 0 {
		logger.Info("No old DCDs found to delete", "oldDynamoNamespace", oldDynamoNamespace)
		return nil
	}

	logger.Info("Deleting old DCDs", "count", len(dcdList.Items), "oldDynamoNamespace", oldDynamoNamespace)

	var deleteErrors []error
	for i := range dcdList.Items {
		dcd := &dcdList.Items[i]
		logger.Info("Deleting old DCD", "name", dcd.Name, "dynamoNamespace", oldDynamoNamespace)

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

// reconcileTrafficProxy ensures the HAProxy traffic proxy resources exist and are configured.
// The proxy is used for traffic shifting during rolling updates.
func (r *DynamoGraphDeploymentReconciler) reconcileTrafficProxy(
	ctx context.Context,
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) error {
	logger := log.FromContext(ctx)

	// Build the proxy configuration
	proxyConfig := r.buildProxyConfig(dgd)

	// Generate proxy resources
	proxyDeployment := dynamo.GenerateProxyDeployment(proxyConfig)
	proxyService := dynamo.GenerateProxyService(proxyConfig)
	proxyRuntimeService := dynamo.GenerateProxyRuntimeService(proxyConfig)
	proxyConfigMap := dynamo.GenerateProxyConfigMap(proxyConfig)

	// Sync ConfigMap first (Deployment mounts it)
	_, _, err := commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*corev1.ConfigMap, bool, error) {
		return proxyConfigMap, false, nil
	})
	if err != nil {
		return fmt.Errorf("failed to sync proxy ConfigMap: %w", err)
	}

	// Sync main Service (for frontend traffic)
	_, _, err = commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*corev1.Service, bool, error) {
		return proxyService, false, nil
	})
	if err != nil {
		return fmt.Errorf("failed to sync proxy Service: %w", err)
	}

	// Sync runtime Service (for controller weight updates)
	_, _, err = commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*corev1.Service, bool, error) {
		return proxyRuntimeService, false, nil
	})
	if err != nil {
		return fmt.Errorf("failed to sync proxy runtime Service: %w", err)
	}

	// Sync Deployment
	_, _, err = commoncontroller.SyncResource(ctx, r, dgd, func(ctx context.Context) (*appsv1.Deployment, bool, error) {
		return proxyDeployment, false, nil
	})
	if err != nil {
		return fmt.Errorf("failed to sync proxy Deployment: %w", err)
	}

	logger.V(1).Info("Traffic proxy reconciled successfully", "name", dynamo.GetProxyName(dgd.Name))
	return nil
}

// buildProxyConfig constructs the ProxyConfig from DGD spec and platform defaults.
func (r *DynamoGraphDeploymentReconciler) buildProxyConfig(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) *dynamo.ProxyConfig {
	// Default values from platform config
	replicas := r.Config.TrafficProxy.Replicas
	image := r.Config.TrafficProxy.Image
	resources := r.Config.TrafficProxy.Resources

	// Override with DGD-specific settings if provided
	if dgd.Spec.TrafficProxy != nil {
		if dgd.Spec.TrafficProxy.Replicas != nil {
			replicas = *dgd.Spec.TrafficProxy.Replicas
		}
		if dgd.Spec.TrafficProxy.Resources != nil {
			resources = *dgd.Spec.TrafficProxy.Resources
		}
	}

	// Get frontend service name for backend configuration
	frontendServiceName := r.getFrontendServiceName(dgd)

	// Configure backends based on rollout state
	var oldBackend, newBackend *dynamo.BackendConfig

	if r.isRollingUpdateInProgress(dgd) && dgd.Status.Rollout != nil {
		// During rollout, configure both backends with current weights
		oldWorkerHash := r.getCurrentActiveWorkerHash(dgd)
		newWorkerHash := dynamo.ComputeWorkerSpecHash(dgd)

		oldBackend = &dynamo.BackendConfig{
			ServiceName: r.getFrontendServiceNameWithHash(dgd, oldWorkerHash),
			ServicePort: consts.DynamoServicePort,
			Weight:      dgd.Status.Rollout.TrafficWeightOld,
		}
		newBackend = &dynamo.BackendConfig{
			ServiceName: r.getFrontendServiceNameWithHash(dgd, newWorkerHash),
			ServicePort: consts.DynamoServicePort,
			Weight:      dgd.Status.Rollout.TrafficWeightNew,
		}
	} else {
		// Normal operation - single backend at 100%
		oldBackend = &dynamo.BackendConfig{
			ServiceName: frontendServiceName,
			ServicePort: consts.DynamoServicePort,
			Weight:      100,
		}
	}

	// Build tolerations - merge platform defaults with DGD-specific
	var tolerations []corev1.Toleration
	tolerations = append(tolerations, r.Config.TrafficProxy.Tolerations...)
	if dgd.Spec.TrafficProxy != nil {
		tolerations = append(tolerations, dgd.Spec.TrafficProxy.Tolerations...)
	}

	// Affinity - DGD overrides platform default
	var affinity *corev1.Affinity
	if dgd.Spec.TrafficProxy != nil && dgd.Spec.TrafficProxy.Affinity != nil {
		affinity = dgd.Spec.TrafficProxy.Affinity
	} else {
		affinity = r.Config.TrafficProxy.Affinity
	}

	return &dynamo.ProxyConfig{
		DGDName:     dgd.Name,
		Namespace:   dgd.Namespace,
		Replicas:    replicas,
		Image:       image,
		Resources:   resources,
		Tolerations: tolerations,
		Affinity:    affinity,
		OldBackend:  oldBackend,
		NewBackend:  newBackend,
		Labels: map[string]string{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		},
	}
}

// getFrontendServiceName returns the frontend service name for normal operation.
func (r *DynamoGraphDeploymentReconciler) getFrontendServiceName(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
) string {
	// Find the frontend service
	for name, spec := range dgd.Spec.Services {
		if spec != nil && spec.ComponentType == consts.ComponentTypeFrontend {
			return fmt.Sprintf("%s-%s", dgd.Name, name)
		}
	}
	// Fallback to default frontend naming
	return fmt.Sprintf("%s-frontend", dgd.Name)
}

// getFrontendServiceNameWithHash returns the frontend service name for a specific worker hash.
// During rolling updates, frontends are also versioned by the worker hash.
func (r *DynamoGraphDeploymentReconciler) getFrontendServiceNameWithHash(
	dgd *nvidiacomv1alpha1.DynamoGraphDeployment,
	workerHash string,
) string {
	baseName := r.getFrontendServiceName(dgd)
	return fmt.Sprintf("%s-%s", baseName, workerHash[:8])
}
