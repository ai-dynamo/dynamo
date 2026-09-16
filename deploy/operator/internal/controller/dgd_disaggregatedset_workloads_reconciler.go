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
	"fmt"
	"maps"
	"sort"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
)

type disaggregatedSetWorkloadsReconciler struct {
	rollout                  *dgdWorkerRolloutReconciler
	renderer                 *disaggregatedSetWorkloadRenderer
	resources                *disaggregatedSetResourceReconciler
	stableResources          *disaggregatedSetStableResourcesReconciler
	auxiliaryDCDs            *disaggregatedSetAuxiliaryDCDReconciler
	readiness                *disaggregatedSetReadinessResolver
	componentRestartProgress *componentRestartProgressResolver
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
	resources := []Resource{}
	logger := log.FromContext(ctx)

	workerHashTransition, err := r.rollout.planUnsupportedWorkerHashTransition(dgd)
	if err != nil {
		return ReconcileResult{}, failWorkloadProgram(reasonRollingUpdateFailed, err)
	}
	rollingUpdateCtx, err := r.rollout.buildRollingUpdateContext(ctx, dgd)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to build rolling update context: %w", err)
	}
	selection, reason := selectDisaggregatedSetComponents(dgd)
	if reason != "" {
		return ReconcileResult{}, fmt.Errorf("failed to select DisaggregatedSet roles: %s", reason)
	}

	existingRestartAnnotations, err := getExistingRestartAnnotationsDCD(ctx, r.componentRestartProgress.reader, dgd)
	if err != nil {
		logger.Error(err, "failed to get existing restart annotations")
		return ReconcileResult{}, fmt.Errorf("failed to get existing restart annotations: %w", err)
	}
	existingDSRestartAnnotations, err := r.resources.RestartAnnotations(ctx, dgd, selection)
	if err != nil {
		logger.Error(err, "failed to get existing DisaggregatedSet restart annotations")
		return ReconcileResult{}, fmt.Errorf("failed to get existing DisaggregatedSet restart annotations: %w", err)
	}
	maps.Copy(existingRestartAnnotations, existingDSRestartAnnotations)

	normalizedComponents, err := dynamo.NormalizeDynamoGraphDeploymentComponents(
		dgd,
		restartState,
		existingRestartAnnotations,
		rollingUpdateCtx,
	)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to normalize components for DisaggregatedSet path: %w", err)
	}

	checkpointGated, err := r.applyDisaggregatedSetCheckpointStartupPolicies(normalizedComponents, checkpointInfos, selection)
	if err != nil {
		return ReconcileResult{}, err
	}

	desiredDS, err := r.renderer.Render(ctx, dgd, normalizedComponents, selection, rollingUpdateCtx, checkpointInfos)
	if err != nil {
		return ReconcileResult{}, err
	}
	syncedDS, dsModified, err := r.resources.Reconcile(ctx, dgd, desiredDS)
	if err != nil {
		return ReconcileResult{}, err
	}

	readiness, err := r.readiness.Resolve(ctx, syncedDS, selection)
	if err != nil {
		return ReconcileResult{}, err
	}
	dsReady := readiness.Ready && !dsModified && !checkpointGated
	targetRevision, err := disaggregatedSetTargetRevision(syncedDS)
	if err != nil {
		return ReconcileResult{}, err
	}

	nonSelectedComponents := make(map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec)
	for componentName, component := range normalizedComponents {
		if _, selected := selection.componentToRole[componentName]; !selected {
			nonSelectedComponents[componentName] = component
		}
	}
	dcds, err := dynamo.GenerateDynamoComponentsDeploymentsFromNormalized(dgd, nonSelectedComponents, rollingUpdateCtx)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to lower non-DisaggregatedSet components: %w", err)
	}

	selectedServiceNames, err := r.stableResources.Reconcile(
		ctx,
		dgd,
		dcds,
		normalizedComponents,
		selection,
		targetRevision,
		dsReady,
		rollingUpdateCtx,
	)
	if err != nil {
		return ReconcileResult{}, err
	}

	syncedDSResource, err := commoncontroller.NewResourceWithComponentStatuses(
		syncedDS,
		func() (bool, string, map[string]nvidiacomv1beta1.ComponentReplicaStatus) {
			if dsModified {
				return false, "DisaggregatedSet spec was updated; waiting for controller status", readiness.ComponentStatuses
			}
			if checkpointGated {
				return false, "DisaggregatedSet roles are waiting for checkpoint readiness", readiness.ComponentStatuses
			}
			return dsReady, readiness.Reason, readiness.ComponentStatuses
		},
	)
	if err != nil {
		return ReconcileResult{}, err
	}
	resources = append(resources, syncedDSResource)

	nonSelectedResources, nonSelectedDCDsModified, err := r.auxiliaryDCDs.Reconcile(ctx, dgd, dcds, selection)
	if err != nil {
		return ReconcileResult{}, err
	}
	resources = append(resources, nonSelectedResources...)

	if workerHashTransition.needsCommit() && !dsModified && !nonSelectedDCDsModified {
		observed, err := disaggregatedSetPathwayObservesWorkerHash(
			dgd,
			syncedDS,
			selection,
			dcds,
			rollingUpdateCtx.NewWorkerHash,
		)
		if err != nil {
			return ReconcileResult{}, failWorkloadProgram(reasonRollingUpdateFailed, err)
		}
		if observed {
			if err := r.rollout.commitUnsupportedWorkerHashTransition(ctx, dgd, workerHashTransition, false); err != nil {
				return ReconcileResult{}, failWorkloadProgram(
					reasonRollingUpdateFailed,
					fmt.Errorf("project observed DisaggregatedSet worker hash: %w", err),
				)
			}
		}
	}

	if dsReady {
		if err := r.auxiliaryDCDs.DeleteSelected(ctx, dgd, selection); err != nil {
			return ReconcileResult{}, err
		}
	}

	result := checkResourcesReadiness(resources)
	if result.State == nvidiacomv1beta1.DGDStateSuccessful {
		if err := r.stableResources.DeleteStale(ctx, dgd, selectedServiceNames); err != nil {
			return ReconcileResult{}, err
		}
	}
	return result, nil
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

	updatedInProgress := r.readiness.updatedInProgressForDisaggregatedSet(
		ctx,
		dgd,
		componentsToCheck,
		r.componentRestartProgress,
	)
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

	updatedInProgress := r.readiness.updatedInProgressForDisaggregatedSet(
		ctx,
		dgd,
		[]string{currentComponent},
		r.componentRestartProgress,
	)
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

func (r *disaggregatedSetWorkloadsReconciler) applyDisaggregatedSetCheckpointStartupPolicies(
	components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
	selection disaggregatedSetSelection,
) (bool, error) {
	for _, componentName := range sortedComponentNames(components) {
		if err := applyCheckpointStartupPolicy(components[componentName], checkpointInfos[componentName]); err != nil {
			return false, fmt.Errorf("failed to apply checkpoint startup policy for %s: %w", componentName, err)
		}
	}

	gateSelectedRoles := false
	for componentName := range selection.componentToRole {
		info := checkpointInfos[componentName]
		if info != nil &&
			info.Enabled &&
			info.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint &&
			!info.Ready {
			gateSelectedRoles = true
			break
		}
	}
	for componentName := range selection.componentToRole {
		component := components[componentName]
		if component == nil {
			return false, fmt.Errorf("generated DynamoComponentDeployment missing for selected component %q", componentName)
		}
		if gateSelectedRoles {
			component.Replicas = ptr.To(int32(0))
		}
		selection.desiredReplicas[componentName] = desiredComponentReplicas(component)
	}
	return gateSelectedRoles, nil
}

func sortedComponentNames(components map[string]*nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) []string {
	names := make([]string, 0, len(components))
	for name := range components {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func disaggregatedSetPathwayObservesWorkerHash(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	ds *unstructured.Unstructured,
	selection disaggregatedSetSelection,
	dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	targetHash string,
) (bool, error) {
	typedDS := &disaggregatedsetv1.DisaggregatedSet{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(ds.Object, typedDS); err != nil {
		return false, fmt.Errorf("decode DisaggregatedSet while observing worker hash: %w", err)
	}
	roles := make(map[string]*disaggregatedsetv1.DisaggregatedRoleSpec, len(typedDS.Spec.Roles))
	for i := range typedDS.Spec.Roles {
		role := &typedDS.Spec.Roles[i]
		roles[role.Name] = role
	}

	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		if roleName, selected := selection.componentToRole[component.ComponentName]; selected {
			role := roles[roleName]
			if role == nil || role.Spec.LeaderWorkerTemplate.WorkerTemplate.Labels[consts.KubeLabelDynamoWorkerHash] != targetHash {
				return false, nil
			}
			if leader := role.Spec.LeaderWorkerTemplate.LeaderTemplate; leader != nil &&
				leader.Labels[consts.KubeLabelDynamoWorkerHash] != targetHash {
				return false, nil
			}
			continue
		}
		dcd := dcds[component.ComponentName]
		if dcd == nil || dcd.Labels[consts.KubeLabelDynamoWorkerHash] != targetHash {
			return false, nil
		}
	}
	return true, nil
}
