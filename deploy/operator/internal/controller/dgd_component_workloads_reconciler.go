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
		// Keyed by declared component name, so a synthesized elastic-EP follower finds
		// nothing -- which is what it should find. Inheriting the leader's checkpoint
		// would set Checkpoint.Enabled and a CheckpointRef on the follower, making it a
		// CRIU restore target for an engine process it never runs (its command is a bare
		// `ray start --block`), and would gate it on a waitForCheckpoint it has no use
		// for. Unlike the GMS claim template, this one must NOT resolve to the leader.
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
			err := r.syncer.Get(
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
	if dcd == nil || checkpointInfo == nil || !checkpointInfo.Enabled {
		return nil
	}

	if checkpointInfo.Exists && checkpointInfo.CheckpointName != "" {
		if dcd.Spec.Experimental == nil {
			dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{}
		}
		if dcd.Spec.Experimental.Checkpoint == nil {
			dcd.Spec.Experimental.Checkpoint = &nvidiacomv1beta1.ComponentCheckpointConfig{}
		}
		checkpointName := checkpointInfo.CheckpointName
		dcd.Spec.Experimental.Checkpoint.Enabled = true
		dcd.Spec.Experimental.Checkpoint.CheckpointRef = &checkpointName
		dcd.Spec.Experimental.Checkpoint.Identity = nil
		dcd.Spec.Experimental.Checkpoint.Job = nil
		startupPolicy := checkpointInfo.StartupPolicy
		if startupPolicy == "" {
			startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
		}
		dcd.Spec.Experimental.Checkpoint.StartupPolicy = nvidiacomv1beta1.CheckpointStartupPolicy(startupPolicy)
	}

	if checkpointInfo.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint && !checkpointInfo.Ready {
		dcd.Spec.Replicas = ptr.To(int32(0))
		return nil
	}
	if checkpointInfo.StartupPolicy != "" &&
		checkpointInfo.StartupPolicy != nvidiacomv1alpha1.CheckpointStartupPolicyImmediate {
		return nil
	}

	labels := dynamo.GetPodTemplateLabels(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
	if labels == nil {
		if dcd.Spec.PodTemplate == nil {
			dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{}
		}
		if dcd.Spec.PodTemplate.Labels == nil {
			dcd.Spec.PodTemplate.Labels = map[string]string{}
		}
		labels = dcd.Spec.PodTemplate.Labels
	}
	annotations := dynamo.GetPodTemplateAnnotations(&dcd.Spec.DynamoComponentDeploymentSharedSpec)
	if annotations == nil {
		if dcd.Spec.PodTemplate == nil {
			dcd.Spec.PodTemplate = &corev1.PodTemplateSpec{}
		}
		if dcd.Spec.PodTemplate.Annotations == nil {
			dcd.Spec.PodTemplate.Annotations = map[string]string{}
		}
		annotations = dcd.Spec.PodTemplate.Annotations
	}
	return checkpoint.ApplyRestoreCandidateMetadata(labels, annotations, checkpointInfo)
}

func (r *componentWorkloadsReconciler) preserveExistingBackendFramework(
	ctx context.Context,
	desired *nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	existing := &nvidiacomv1beta1.DynamoComponentDeployment{}
	err := r.syncer.Get(
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
