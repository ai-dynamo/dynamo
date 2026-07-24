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
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// graphReconcileState is ephemeral state shared by the common DGD flow and the
// selected workload program during one reconciliation. It is intentionally
// small: dependencies belong to concrete programs, and provider-native API
// objects must not be added here.
type graphReconcileState struct {
	DGD             *nvidiacomv1beta1.DynamoGraphDeployment
	HasMultinode    bool
	RestartState    *dynamo.RestartState
	CheckpointInfos map[string]*checkpoint.CheckpointInfo
	Result          ReconcileResult
}

// workloadProgram owns the complete graph-workload state machine for one
// pathway. The common DGD controller selects one program and invokes it once;
// it does not drive provider rendering, rollout, readiness, or cleanup through
// lifecycle callbacks.
type workloadProgram interface {
	Reconcile(context.Context, *graphReconcileState) error
}

// groveReconcileFunc is the temporary strangler seam around the existing Grove
// pathway. It disappears when Grove reconciliation moves into groveProgram.
type groveReconcileFunc func(
	context.Context,
	*nvidiacomv1beta1.DynamoGraphDeployment,
	*dynamo.RestartState,
	map[string]*checkpoint.CheckpointInfo,
) (ReconcileResult, error)

type componentProgram struct {
	// The DGD reconciler temporarily supplies shared controller dependencies and
	// managed rolling-update helpers. Later extractions can narrow this without
	// moving component-path orchestration back into the common controller flow.
	reconciler *DynamoGraphDeploymentReconciler
	lwsEnabled bool
}

func (p *componentProgram) Reconcile(ctx context.Context, state *graphReconcileState) error {
	log.FromContext(ctx).Info(
		"Reconciling Dynamo components deployments",
		"hasMultinode", state.HasMultinode,
		"lwsEnabled", p.lwsEnabled,
	)

	result, err := p.reconcile(ctx, state)
	if err != nil {
		return err
	}
	state.Result = result
	return nil
}

// reconcile owns the component pathway's complete DCD graph reconciliation.
// Managed rolling-update helpers remain on the DGD reconciler until their
// dedicated extraction; this program owns when they participate in the flow.
func (p *componentProgram) reconcile(
	ctx context.Context,
	state *graphReconcileState,
) (ReconcileResult, error) {
	r := p.reconciler
	dynamoDeployment := state.DGD
	resources := []Resource{}
	logger := log.FromContext(ctx)

	rollingUpdateCtx, err := r.buildRollingUpdateContext(ctx, dynamoDeployment)
	if err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to build rolling update context: %w", err)
	}

	existingRestartAnnotations, err := r.getExistingRestartAnnotationsDCD(ctx, dynamoDeployment)
	if err != nil {
		logger.Error(err, "failed to get existing restart annotations")
		return ReconcileResult{}, fmt.Errorf("failed to get existing restart annotations: %w", err)
	}
	if rollingUpdateCtx.InProgress() {
		logger.Info("Rolling update in progress",
			"newWorkerHash", rollingUpdateCtx.NewWorkerHash,
			"oldWorkerComponentReplicas", rollingUpdateCtx.OldWorkerReplicaTargetsByComponent)
	}

	// Generate all DCDs, including the desired generations during a managed
	// rolling update.
	dynamoComponentsDeployments, err := dynamo.GenerateDynamoComponentsDeployments(
		dynamoDeployment,
		state.RestartState,
		existingRestartAnnotations,
		rollingUpdateCtx,
	)
	if err != nil {
		logger.Error(err, "failed to generate the DynamoComponentsDeployments")
		return ReconcileResult{}, fmt.Errorf("failed to generate the DynamoComponentsDeployments: %w", err)
	}

	// Apply resolved checkpoint policy and synchronize every desired DCD.
	for key, dcd := range dynamoComponentsDeployments {
		if err := p.applyCheckpointStartupPolicy(dcd, state.CheckpointInfos[key]); err != nil {
			return ReconcileResult{}, fmt.Errorf("failed to apply checkpoint startup policy for %s: %w", key, err)
		}
		logger.Info("Reconciling DynamoComponentDeployment", "key", key, "name", dcd.Name)
		if err := p.preserveExistingBackendFramework(ctx, dcd); err != nil {
			logger.Error(err, "failed to preserve existing DynamoComponentDeployment backendFramework", "name", dcd.Name)
			return ReconcileResult{}, fmt.Errorf("failed to preserve existing DynamoComponentDeployment backendFramework: %w", err)
		}
		_, syncedDCD, err := commoncontroller.SyncResource(ctx, r, dynamoDeployment, func(context.Context) (*nvidiacomv1beta1.DynamoComponentDeployment, bool, error) {
			return dcd, false, nil
		})
		if err != nil {
			logger.Error(err, "failed to sync the DynamoComponentDeployment", "name", dcd.Name)
			return ReconcileResult{}, fmt.Errorf("failed to sync the DynamoComponentDeployment: %w", err)
		}
		resources = append(resources, syncedDCD)
	}

	// Old worker DCDs are scaled through direct patches so their stored specs
	// are not overwritten with the new generation's desired spec.
	if rollingUpdateCtx.InProgress() {
		if err := r.scaleOldWorkerDCDs(ctx, dynamoDeployment, rollingUpdateCtx); err != nil {
			logger.Error(err, "failed to scale old worker DCDs")
			return ReconcileResult{}, fmt.Errorf("failed to scale old worker DCDs: %w", err)
		}
	}

	result := r.checkResourcesReadiness(resources)

	// Include old worker generations in status while a managed rolling update
	// is active. A transient aggregation error remains non-fatal so readiness
	// can continue with the statuses already collected above.
	if rollingUpdateCtx.InProgress() {
		oldWorkerStatuses, err := r.aggregateOldWorkerComponentStatuses(ctx, dynamoDeployment, rollingUpdateCtx)
		if err != nil {
			logger.Error(err, "failed to aggregate old worker component statuses")
		} else if len(oldWorkerStatuses) > 0 {
			mergeWorkerComponentStatuses(result.ComponentStatus, oldWorkerStatuses)
		}
	}

	return result, nil
}

func (p *componentProgram) applyCheckpointStartupPolicy(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	checkpointInfo *checkpoint.CheckpointInfo,
) error {
	if dcd == nil || checkpointInfo == nil || !checkpointInfo.Enabled {
		return nil
	}

	// DGD-managed automatic checkpoints deliberately bypass identity lookup and
	// are resolved by the exact DynamoCheckpoint CR created for this
	// DGD/component generation. Propagate that resolved reference into the child
	// DCD so the DCD controller does not independently fall back to legacy
	// identity-based reuse logic.
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
	if checkpointInfo.StartupPolicy == "" ||
		checkpointInfo.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyImmediate {
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
	return nil
}

func (p *componentProgram) preserveExistingBackendFramework(
	ctx context.Context,
	desired *nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	r := p.reconciler
	existing := &nvidiacomv1beta1.DynamoComponentDeployment{}
	err := r.Get(ctx, types.NamespacedName{Name: desired.Name, Namespace: desired.Namespace}, existing)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to get existing DynamoComponentDeployment %s/%s: %w", desired.Namespace, desired.Name, err)
	}

	// backendFramework is immutable on DCDs. Older generated children may have
	// an empty value, so preserve the stored value on update while allowing new
	// children to be created with the inferred backend.
	desired.Spec.BackendFramework = existing.Spec.BackendFramework
	return nil
}

type groveProgram struct {
	reconcile  groveReconcileFunc
	lwsEnabled bool
}

func (p *groveProgram) Reconcile(ctx context.Context, state *graphReconcileState) error {
	log.FromContext(ctx).Info(
		"Reconciling Grove resources",
		"hasMultinode", state.HasMultinode,
		"lwsEnabled", p.lwsEnabled,
	)

	result, err := p.reconcile(
		ctx,
		state.DGD,
		state.RestartState,
		state.CheckpointInfos,
	)
	if err != nil {
		return err
	}
	state.Result = result
	return nil
}

func (r *DynamoGraphDeploymentReconciler) selectWorkloadProgram(
	state *graphReconcileState,
) workloadProgram {
	if r.isGrovePathway(state.DGD) {
		return &groveProgram{
			reconcile:  r.reconcileGroveResources,
			lwsEnabled: r.RuntimeConfig.Gate.Enabled(features.LWS),
		}
	}
	return &componentProgram{
		reconciler: r,
		lwsEnabled: r.RuntimeConfig.Gate.Enabled(features.LWS),
	}
}
