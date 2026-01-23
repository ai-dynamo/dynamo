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
