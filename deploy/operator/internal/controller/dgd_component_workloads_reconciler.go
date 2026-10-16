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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/client-go/util/retry"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// componentWorkloadsReconciler owns the component pathway's complete DCD graph
// reconciliation without depending on the top-level DGD reconciler.
type componentWorkloadsReconciler struct {
	syncer  dgdResourceSyncer
	reader  client.Reader
	rollout *dgdWorkerRolloutReconciler
}

func newComponentWorkloadsReconciler(
	kubeClient client.Client,
	apiReader client.Reader,
	recorder events.EventRecorder,
	rollout *dgdWorkerRolloutReconciler,
) *componentWorkloadsReconciler {
	return &componentWorkloadsReconciler{
		syncer:  newDGDResourceSyncer(kubeClient, recorder),
		reader:  apiReader,
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
		if err := preserveExistingBackendFramework(ctx, r.syncer, dcd); err != nil {
			logger.Error(err, "failed to preserve existing DynamoComponentDeployment state", "name", dcd.Name)
			return ReconcileResult{}, fmt.Errorf("failed to preserve existing DynamoComponentDeployment state: %w", err)
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

		// Establish identity only after SyncResource has persisted a no-surge spec.
		apiReader := r.reader
		if apiReader == nil {
			apiReader = r.syncer
		}
		if _, err := ensureDCDGraphUIDLabel(ctx, apiReader, r.syncer, dgd, syncedDCD); err != nil {
			logger.Error(err, "failed to establish DynamoComponentDeployment graph identity", "name", dcd.Name)
			return ReconcileResult{}, fmt.Errorf("failed to establish DynamoComponentDeployment graph identity: %w", err)
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
		for _, podTemplate := range dynamo.ComponentPodTemplates(&existingDCD.Spec.DynamoComponentDeploymentSharedSpec) {
			if restartAt := podTemplate.Annotations[consts.RestartAnnotation]; restartAt != "" {
				restartAnnotations[componentName] = restartAt
				break
			}
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

	// Artifact identity is independent of startup policy. Preserve the automatic
	// SnapshotJob handoff even while WaitForCheckpoint keeps replicas gated.
	if checkpointInfo.AutomaticSnapshotJob != nil {
		if err := applyRestoreCandidateMetadataToComponent(component, checkpointInfo); err != nil {
			return err
		}
	}

	if checkpointInfo.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint && !checkpointInfo.Ready {
		component.Replicas = ptr.To(int32(0))
		return nil
	}
	if checkpointInfo.StartupPolicy != "" &&
		checkpointInfo.StartupPolicy != nvidiacomv1alpha1.CheckpointStartupPolicyImmediate {
		return nil
	}
	if checkpointInfo.AutomaticSnapshotJob != nil {
		return nil
	}
	return applyRestoreCandidateMetadataToComponent(component, checkpointInfo)
}

func applyRestoreCandidateMetadataToComponent(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
	checkpointInfo *checkpoint.CheckpointInfo,
) error {
	for _, podTemplate := range dynamo.EnsureComponentPodTemplates(component) {
		if err := checkpoint.ApplyRestoreCandidateMetadata(podTemplate.Annotations, checkpointInfo); err != nil {
			return err
		}
	}
	return nil
}

// preserveExistingBackendFramework carries forward immutable server state that
// must not be overwritten by a generated DCD.
func preserveExistingBackendFramework(
	ctx context.Context,
	reader client.Reader,
	desired *nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	// Read the immutable value from the existing child when it is present.
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

// ensureDCDGraphUIDLabel establishes Pod identity only after the live DCD spec
// guarantees that projecting the label cannot create surge capacity. dgd and
// dcd must be non-nil.
func ensureDCDGraphUIDLabel(
	ctx context.Context,
	apiReader client.Reader,
	kubeClient client.Client,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
) (bool, error) {
	// New objects already carry this label; only legacy DCDs need migration.
	if dgd.UID == "" {
		return false, nil
	}
	if dcd.Labels[consts.KubeLabelDynamoGraphDeploymentUID] == string(dgd.UID) {
		return false, nil
	}
	if !allowsNoSurgeUIDLabelMigration(dcd) {
		log.FromContext(ctx).V(1).Info(
			"Deferring DGD UID label migration until the component uses a no-surge Deployment strategy",
			"name", dcd.Name,
		)
		return false, nil
	}

	// A conflict proves the cached resourceVersion stale. Only then bypass the
	// cache so each bounded retry uses a new optimistic-lock precondition.
	reader := client.Reader(kubeClient)
	modified := false
	err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		// Revalidate strategy and ownership from the same object that is patched.
		current := &nvidiacomv1beta1.DynamoComponentDeployment{}
		if err := reader.Get(
			ctx,
			types.NamespacedName{Name: dcd.Name, Namespace: dcd.Namespace},
			current,
		); err != nil {
			return err
		}
		if current.Labels[consts.KubeLabelDynamoGraphDeploymentUID] == string(dgd.UID) {
			return nil
		}
		if !allowsNoSurgeUIDLabelMigration(current) {
			return nil
		}
		if err := commoncontroller.CheckControllerOwnership(
			current,
			dgd,
			kubeClient.Scheme(),
		); err != nil {
			return err
		}

		// Patch only the protected identity label under resourceVersion CAS.
		before := current.DeepCopy()
		current.Labels = maps.Clone(current.Labels)
		if current.Labels == nil {
			current.Labels = make(map[string]string)
		}
		current.Labels[consts.KubeLabelDynamoGraphDeploymentUID] = string(dgd.UID)
		err := kubeClient.Patch(
			ctx,
			current,
			client.MergeFromWithOptions(
				before,
				client.MergeFromWithOptimisticLock{},
			),
		)
		if apierrors.IsConflict(err) {
			reader = apiReader
		}
		if err == nil {
			modified = true
		}
		return err
	})
	if err != nil {
		return false, fmt.Errorf(
			"failed to label DynamoComponentDeployment %s/%s: %w",
			dcd.Namespace,
			dcd.Name,
			err,
		)
	}
	return modified, nil
}

func allowsNoSurgeUIDLabelMigration(dcd *nvidiacomv1beta1.DynamoComponentDeployment) bool {
	if dcd.Status.Component == nil || dcd.Status.Component.ComponentKind != nvidiacomv1beta1.ComponentKindDeployment {
		return false
	}
	annotations := getResourceAnnotations(dcd)
	if deploymentStrategyFromAnnotations(annotations) == common.DeploymentStrategyRecreate {
		return true
	}
	maxSurge := annotations[KubeAnnotationDeploymentRollingUpdateMaxSurge]
	return maxSurge == "0" || maxSurge == "0%"
}
