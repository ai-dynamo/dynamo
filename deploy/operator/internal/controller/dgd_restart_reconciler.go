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
	"sort"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// dgdRestartReconciler derives restart state from the DGD and a
// pathway-specific progress resolver. It performs no Kubernetes I/O. Restart
// status remains owned by the workload program until the outer controller
// persists it.
type dgdRestartReconciler struct{}

func newDGDRestartReconciler() *dgdRestartReconciler {
	return &dgdRestartReconciler{}
}

// Resolve derives restart state after shared resources and pathway-specific
// input validation have completed.
func (r *dgdRestartReconciler) Resolve(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
	resolveProgress restartProgressResolver,
) programRestart {
	statusView := dgd.DeepCopy()
	statusView.Status = *status
	restartStatus := r.computeRestartStatusWithProgressResolver(ctx, statusView, resolveProgress)
	return programRestart{
		State:  dynamo.DetermineRestartState(statusView, restartStatus),
		Status: restartStatus,
	}
}

func isRestartAlreadyProcessed(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	if dgd.Spec.Restart == nil || dgd.Spec.Restart.ID == "" {
		return true
	}

	if dgd.Status.Restart == nil || dgd.Status.Restart.ObservedID == "" {
		return false
	}

	return dgd.Spec.Restart.ID == dgd.Status.Restart.ObservedID &&
		(dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseCompleted ||
			dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseFailed ||
			dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseSuperseded)
}

// isNewRestartRequest checks if the current spec.restart.id represents a new restart request.
func isNewRestartRequest(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	if dgd.Status.Restart == nil || dgd.Status.Restart.ObservedID == "" || dgd.Spec.Restart.ID == "" {
		return true
	}
	return dgd.Spec.Restart.ID != dgd.Status.Restart.ObservedID
}

// computeParallelRestartStatus handles parallel restart where all components restart together.
func (r *dgdRestartReconciler) computeParallelRestartStatus(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	resolveProgress restartProgressResolver,
) *nvidiacomv1beta1.RestartStatus {
	logger := log.FromContext(ctx)
	specID := dgd.Spec.Restart.ID

	var componentsToCheck []string
	if isNewRestartRequest(dgd) {
		logger.Info("New restart request detected, resetting to all components", "specID", specID)
		componentsToCheck = make([]string, 0, len(dgd.Spec.Components))
		for i := range dgd.Spec.Components {
			componentsToCheck = append(componentsToCheck, dgd.Spec.Components[i].ComponentName)
		}
		sort.Strings(componentsToCheck)

		if len(componentsToCheck) > 0 {
			return &nvidiacomv1beta1.RestartStatus{
				ObservedID: specID,
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: componentsToCheck,
			}
		}
	} else if dgd.Status.Restart != nil && len(dgd.Status.Restart.InProgress) > 0 {
		componentsToCheck = dgd.Status.Restart.InProgress
	} else {
		componentsToCheck = make([]string, 0, len(dgd.Spec.Components))
		for i := range dgd.Spec.Components {
			componentsToCheck = append(componentsToCheck, dgd.Spec.Components[i].ComponentName)
		}
		sort.Strings(componentsToCheck)
	}

	if len(componentsToCheck) == 0 {
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
		}
	}

	updatedInProgress := resolveProgress(ctx, dgd, componentsToCheck)
	if len(updatedInProgress) == 0 {
		logger.Info("Restart completed for all components")
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
		}
	}

	return &nvidiacomv1beta1.RestartStatus{
		ObservedID: specID,
		Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
		InProgress: updatedInProgress,
	}
}

// computeSequentialRestartStatus handles sequential restart where components restart one at a time.
func (r *dgdRestartReconciler) computeSequentialRestartStatus(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	order []string,
	resolveProgress restartProgressResolver,
) *nvidiacomv1beta1.RestartStatus {
	logger := log.FromContext(ctx)
	specID := dgd.Spec.Restart.ID
	if len(order) == 0 {
		logger.Info("Sequential restart completed with no components", "specID", specID)
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
		}
	}

	var currentComponent string
	if isNewRestartRequest(dgd) {
		logger.Info("New restart request detected, starting from first component", "specID", specID, "firstComponent", order[0])
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{order[0]},
		}
	}

	if dgd.Status.Restart != nil && len(dgd.Status.Restart.InProgress) > 0 {
		currentComponent = dgd.Status.Restart.InProgress[0]
	}
	if currentComponent == "" {
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{order[0]},
		}
	}

	updatedInProgress := resolveProgress(ctx, dgd, []string{currentComponent})
	if len(updatedInProgress) > 0 {
		logger.Info("Component restart not completed", "component", currentComponent, "updatedInProgress", updatedInProgress)
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{currentComponent},
		}
	}

	logger.Info("Component restart completed", "component", currentComponent)
	nextComponent, currentFound := getNextComponentInOrder(order, currentComponent)
	if !currentFound {
		logger.Info("Current restart component is no longer in order, restarting sequence from first component", "component", currentComponent, "firstComponent", order[0])
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{order[0]},
		}
	}

	if nextComponent == "" {
		logger.Info("Restart completed for all components")
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
		}
	}

	logger.Info("Starting next component restart", "component", nextComponent)
	return &nvidiacomv1beta1.RestartStatus{
		ObservedID: specID,
		Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
		InProgress: []string{nextComponent},
	}
}

// getNextComponentInOrder returns the component after the current component.
// The boolean reports whether currentComponent was found in order.
func getNextComponentInOrder(order []string, currentComponent string) (string, bool) {
	for i, componentName := range order {
		if componentName != currentComponent {
			continue
		}
		if i+1 < len(order) {
			return order[i+1], true
		}
		return "", true
	}
	return "", false
}

func (r *dgdRestartReconciler) computeRestartStatusWithProgressResolver(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	resolveProgress restartProgressResolver,
) *nvidiacomv1beta1.RestartStatus {
	if dgd.Spec.Restart == nil || dgd.Spec.Restart.ID == "" {
		if dgd.Status.Restart != nil &&
			(dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseCompleted ||
				dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseFailed ||
				dgd.Status.Restart.Phase == nvidiacomv1beta1.RestartPhaseSuperseded) {
			return dgd.Status.Restart
		}
		return nil
	}

	if isRestartAlreadyProcessed(dgd) {
		return dgd.Status.Restart
	}
	if rollingUpdateInProgress(dgd.Status.RollingUpdate) {
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: dgd.Spec.Restart.ID,
			Phase:      nvidiacomv1beta1.RestartPhaseSuperseded,
		}
	}

	if dynamo.IsParallelRestart(dgd) {
		return r.computeParallelRestartStatus(ctx, dgd, resolveProgress)
	}
	return r.computeSequentialRestartStatus(ctx, dgd, dynamo.GetRestartOrder(dgd), resolveProgress)
}
