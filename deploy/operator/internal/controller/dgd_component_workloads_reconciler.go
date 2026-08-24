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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// componentWorkloadsReconciler owns the component pathway's complete DCD graph
// reconciliation without depending on the top-level DGD reconciler.
type componentWorkloadsReconciler struct {
	syncer  dgdResourceSyncer
	rollout *dgdWorkerRolloutReconciler
}

func newComponentWorkloadsReconciler(
	kubeClient client.Client,
	recorder events.EventRecorder,
	rollout *dgdWorkerRolloutReconciler,
) *componentWorkloadsReconciler {
	return &componentWorkloadsReconciler{
		syncer:  newDGDResourceSyncer(kubeClient, recorder),
		rollout: rollout,
	}
}

func (r *componentWorkloadsReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	restartState *dynamo.RestartState,
	checkpointInfos map[string]*checkpoint.CheckpointInfo,
) (ReconcileResult, error) {
	resources := []Resource{}
	logger := log.FromContext(ctx)

	rollingUpdateCtx, err := r.rollout.buildRollingUpdateContext(ctx, dgd)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to build rolling update context: %w", err)
	}

	existingRestartAnnotations, err := r.getExistingRestartAnnotationsDCD(ctx, dgd)
	if err != nil {
		logger.Error(err, "failed to get existing restart annotations")
		return ReconcileResult{}, fmt.Errorf("failed to get existing restart annotations: %w", err)
	}
	if rollingUpdateCtx.InProgress() {
		logger.Info("Rolling update in progress",
			"newWorkerHash", rollingUpdateCtx.NewWorkerHash,
			"oldWorkerComponentReplicas", rollingUpdateCtx.OldWorkerReplicaTargetsByComponent)
	}

	dcds, err := dynamo.GenerateDynamoComponentsDeployments(
		dgd,
		restartState,
		existingRestartAnnotations,
		rollingUpdateCtx,
	)
	if err != nil {
		logger.Error(err, "failed to generate the DynamoComponentsDeployments")
		return ReconcileResult{}, fmt.Errorf("failed to generate the DynamoComponentsDeployments: %w", err)
	}

	for key, dcd := range dcds {
		if err := r.applyCheckpointStartupPolicy(dcd, checkpointInfos[key]); err != nil {
			return ReconcileResult{}, fmt.Errorf("failed to apply checkpoint startup policy for %s: %w", key, err)
		}
		logger.Info("Reconciling DynamoComponentDeployment", "key", key, "name", dcd.Name)
		if err := r.preserveExistingBackendFramework(ctx, dcd); err != nil {
			logger.Error(err, "failed to preserve existing DynamoComponentDeployment backendFramework", "name", dcd.Name)
			return ReconcileResult{}, fmt.Errorf("failed to preserve existing DynamoComponentDeployment backendFramework: %w", err)
		}
		_, syncedDCD, err := commoncontroller.SyncResource(
			ctx,
			&r.syncer,
			dgd,
			func(context.Context) (*nvidiacomv1beta1.DynamoComponentDeployment, bool, error) {
				return dcd, false, nil
			},
		)
		if err != nil {
			logger.Error(err, "failed to sync the DynamoComponentDeployment", "name", dcd.Name)
			return ReconcileResult{}, fmt.Errorf("failed to sync the DynamoComponentDeployment: %w", err)
		}
		resources = append(resources, syncedDCD)
	}

	if rollingUpdateCtx.InProgress() {
		if err := r.rollout.scaleOldWorkerDCDs(ctx, dgd, rollingUpdateCtx); err != nil {
			logger.Error(err, "failed to scale old worker DCDs")
			return ReconcileResult{}, fmt.Errorf("failed to scale old worker DCDs: %w", err)
		}
	}

	result := checkResourcesReadiness(resources)
	if rollingUpdateCtx.InProgress() {
		oldWorkerStatuses, err := r.rollout.aggregateOldWorkerComponentStatuses(ctx, dgd, rollingUpdateCtx)
		if err != nil {
			logger.Error(err, "failed to aggregate old worker component statuses")
		} else if len(oldWorkerStatuses) > 0 {
			mergeWorkerComponentStatuses(result.ComponentStatus, oldWorkerStatuses)
		}
	}

	return result, nil
}

func (r *componentWorkloadsReconciler) getExistingRestartAnnotationsDCD(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (map[string]string, error) {
	return getExistingRestartAnnotationsDCD(ctx, r.syncer, dgd)
}

func getExistingRestartAnnotationsDCD(
	ctx context.Context,
	reader client.Reader,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) (map[string]string, error) {
	logger := log.FromContext(ctx)
	hashes, err := desiredWorkerHashes(dgd)
	if err != nil {
		return nil, err
	}
	workerHashes := activeWorkerHashCandidates(dgd, hashes)

	restartAnnotations := make(map[string]string)
	for i := range dgd.Spec.Components {
		componentName := dgd.Spec.Components[i].ComponentName
		existingDCD := &nvidiacomv1beta1.DynamoComponentDeployment{}
		for _, workerHash := range workerHashes {
			dcdName := dynamo.GetDCDResourceName(dgd, componentName, workerHash)
			err := reader.Get(
				ctx,
				types.NamespacedName{Name: dcdName, Namespace: dgd.Namespace},
				existingDCD,
			)
			if err == nil {
				break
			}
			if !apierrors.IsNotFound(err) {
				return nil, fmt.Errorf("failed to get DynamoComponentDeployment: %w", err)
			}
			logger.Info("DynamoComponentDeployment not found", "dcdName", dcdName)
		}
		if existingDCD.Name == "" {
			continue
		}
		restartAt := dynamo.GetPodTemplateAnnotations(
			&existingDCD.Spec.DynamoComponentDeploymentSharedSpec,
		)[consts.RestartAnnotation]
		if restartAt != "" {
			restartAnnotations[componentName] = restartAt
		}
	}
	return restartAnnotations, nil
}

func (r *componentWorkloadsReconciler) applyCheckpointStartupPolicy(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	checkpointInfo *checkpoint.CheckpointInfo,
) error {
	if dcd == nil {
		return nil
	}
	return applyCheckpointStartupPolicy(&dcd.Spec.DynamoComponentDeploymentSharedSpec, checkpointInfo)
}

func applyCheckpointStartupPolicy(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	checkpointInfo *checkpoint.CheckpointInfo,
) error {
	if component == nil || checkpointInfo == nil || !checkpointInfo.Enabled {
		return nil
	}

	if checkpointInfo.Exists && checkpointInfo.CheckpointName != "" {
		if component.Experimental == nil {
			component.Experimental = &nvidiacomv1beta1.ExperimentalSpec{}
		}
		if component.Experimental.Checkpoint == nil {
			component.Experimental.Checkpoint = &nvidiacomv1beta1.ComponentCheckpointConfig{}
		}
		checkpointName := checkpointInfo.CheckpointName
		component.Experimental.Checkpoint.Enabled = true
		component.Experimental.Checkpoint.CheckpointRef = &checkpointName
		component.Experimental.Checkpoint.Identity = nil
		component.Experimental.Checkpoint.Job = nil
		startupPolicy := checkpointInfo.StartupPolicy
		if startupPolicy == "" {
			startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
		}
		component.Experimental.Checkpoint.StartupPolicy = nvidiacomv1beta1.CheckpointStartupPolicy(startupPolicy)
	}

	if checkpointInfo.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint && !checkpointInfo.Ready {
		component.Replicas = ptr.To(int32(0))
		return nil
	}
	if checkpointInfo.StartupPolicy != "" &&
		checkpointInfo.StartupPolicy != nvidiacomv1alpha1.CheckpointStartupPolicyImmediate {
		return nil
	}

	labels := dynamo.GetPodTemplateLabels(component)
	if labels == nil {
		if component.PodTemplate == nil {
			component.PodTemplate = &corev1.PodTemplateSpec{}
		}
		if component.PodTemplate.Labels == nil {
			component.PodTemplate.Labels = map[string]string{}
		}
		labels = component.PodTemplate.Labels
	}
	annotations := dynamo.GetPodTemplateAnnotations(component)
	if annotations == nil {
		if component.PodTemplate == nil {
			component.PodTemplate = &corev1.PodTemplateSpec{}
		}
		if component.PodTemplate.Annotations == nil {
			component.PodTemplate.Annotations = map[string]string{}
		}
		annotations = component.PodTemplate.Annotations
	}
	return checkpoint.ApplyRestoreCandidateMetadata(labels, annotations, checkpointInfo)
}

func (r *componentWorkloadsReconciler) preserveExistingBackendFramework(
	ctx context.Context,
	desired *nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	return preserveExistingBackendFramework(ctx, r.syncer, desired)
}

func preserveExistingBackendFramework(
	ctx context.Context,
	reader client.Reader,
	desired *nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	existing := &nvidiacomv1beta1.DynamoComponentDeployment{}
	err := reader.Get(
		ctx,
		types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace},
		existing,
	)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf(
			"failed to get existing DynamoComponentDeployment %s/%s: %w",
			desired.Namespace,
			desired.Name,
			err,
		)
	}

	desired.Spec.BackendFramework = existing.Spec.BackendFramework
	return nil
}
