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

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type disaggregatedSetWorkloadsReconciler struct {
	client.Client
	Config                   *configv1alpha1.OperatorConfiguration
	RuntimeConfig            *commoncontroller.RuntimeConfig
	Recorder                 events.EventRecorder
	DockerSecretRetriever    DockerSecretRetriever
	rollout                  *dgdWorkerRolloutReconciler
	renderer                 *dcdWorkloadRenderer
	componentRestartProgress *componentRestartProgressResolver
}

func (r *DynamoGraphDeploymentReconciler) newDisaggregatedSetWorkloadsReconciler(
	rollout *dgdWorkerRolloutReconciler,
) *disaggregatedSetWorkloadsReconciler {
	workloads := newDisaggregatedSetWorkloadsReconciler(
		r.Client,
		r.Recorder,
		r.Config,
		r.RuntimeConfig,
		r.DockerSecretRetriever,
		rollout,
	)
	return workloads
}
func newDisaggregatedSetWorkloadsReconciler(
	k8sClient client.Client,
	recorder events.EventRecorder,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	dockerSecretRetriever DockerSecretRetriever,
	rollout *dgdWorkerRolloutReconciler,
) *disaggregatedSetWorkloadsReconciler {
	return &disaggregatedSetWorkloadsReconciler{
		Client:                   k8sClient,
		Config:                   config,
		RuntimeConfig:            runtimeConfig,
		Recorder:                 recorder,
		DockerSecretRetriever:    dockerSecretRetriever,
		rollout:                  rollout,
		renderer:                 newDCDWorkloadRenderer(k8sClient, config, runtimeConfig, dockerSecretRetriever),
		componentRestartProgress: newComponentRestartProgressResolver(k8sClient),
	}
}

func (r *disaggregatedSetWorkloadsReconciler) GetRecorder() events.EventRecorder {
	return r.Recorder
}

func (r *disaggregatedSetWorkloadsReconciler) ResolveRestart(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	status *nvidiacomv1beta1.DynamoGraphDeploymentStatus,
) programRestart {
	statusView := dgd.DeepCopy()
	statusView.Status = *status
	restartStatus := r.computeRestartStatus(ctx, statusView)
	return programRestart{
		State:  coalesceDisaggregatedSetRestartState(statusView, dynamo.DetermineRestartState(statusView, restartStatus)),
		Status: restartStatus,
	}
}

func (r *disaggregatedSetWorkloadsReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) (ReconcileResult, error) {
	return r.reconcileDisaggregatedSetResources(ctx, dgd, restartState, checkpointInfos)
}

func (r *disaggregatedSetWorkloadsReconciler) computeRestartStatus(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
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
		return r.computeParallelRestartStatus(ctx, dgd)
	}
	return r.computeSequentialRestartStatus(ctx, dgd, dynamo.GetRestartOrder(dgd))
}

func (r *disaggregatedSetWorkloadsReconciler) computeParallelRestartStatus(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) *nvidiacomv1beta1.RestartStatus {
	specID := dgd.Spec.Restart.ID
	var componentsToCheck []string
	if isNewRestartRequest(dgd) {
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

	updatedInProgress := r.getUpdatedInProgressForDisaggregatedSet(ctx, dgd, componentsToCheck)
	if len(updatedInProgress) == 0 {
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

func (r *disaggregatedSetWorkloadsReconciler) computeSequentialRestartStatus(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	order []string,
) *nvidiacomv1beta1.RestartStatus {
	specID := dgd.Spec.Restart.ID
	if len(order) == 0 {
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
		}
	}
	if isNewRestartRequest(dgd) {
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{order[0]},
		}
	}

	currentComponent := ""
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

	updatedInProgress := r.getUpdatedInProgressForDisaggregatedSet(ctx, dgd, []string{currentComponent})
	if len(updatedInProgress) > 0 {
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{currentComponent},
		}
	}

	nextComponent, currentFound := r.getNextSequentialRestartComponent(dgd, order, currentComponent)
	if !currentFound {
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			InProgress: []string{order[0]},
		}
	}
	if nextComponent == "" {
		return &nvidiacomv1beta1.RestartStatus{
			ObservedID: specID,
			Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
		}
	}
	return &nvidiacomv1beta1.RestartStatus{
		ObservedID: specID,
		Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
		InProgress: []string{nextComponent},
	}
}

func (r *disaggregatedSetWorkloadsReconciler) getNextSequentialRestartComponent(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	order []string,
	currentComponent string,
) (string, bool) {
	nextComponent, currentFound := getNextComponentInOrder(order, currentComponent)
	if !currentFound || nextComponent == "" {
		return nextComponent, currentFound
	}
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		return nextComponent, currentFound
	}
	if _, selected := selection.componentToRole[currentComponent]; !selected {
		return nextComponent, currentFound
	}
	for nextComponent != "" {
		if _, selected := selection.componentToRole[nextComponent]; !selected {
			return nextComponent, true
		}
		nextComponent, _ = getNextComponentInOrder(order, nextComponent)
	}
	return "", true
}
