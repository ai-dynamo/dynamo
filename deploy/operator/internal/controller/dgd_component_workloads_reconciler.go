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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// componentWorkloadsReconciler owns the component pathway's complete DCD graph
// reconciliation without depending on the top-level DGD reconciler.
// This reconciler deliberately does not carry features.ElasticEPRayPoC. Nothing on this
// pathway branches on it: follower synthesis is ungated (it is the deployment's declared
// width, not capacity the PoC invents), and a follower's replica count freezes at whatever
// is live regardless of the gate. The gate's remaining job is the admission rule in
// internal/webhook/validation. Holding a copy here that nothing reads is how a gate field
// silently stops meaning anything.
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
		// checkpointInfos is keyed by declared component name, so a synthesized elastic-EP
		// follower looks up nil -- which is correct. Its command is a bare
		// `ray start --block`, so inheriting the leader's checkpoint would make it a CRIU
		// restore target for an engine it never runs. The GMS claim template is the
		// opposite case and does resolve to the leader; see dynamo.ElasticEPComponentIdentity.
		if err := r.applyCheckpointStartupPolicy(dcd, checkpointInfos[key]); err != nil {
			return ReconcileResult{}, fmt.Errorf("failed to apply checkpoint startup policy for %s: %w", key, err)
		}
		logger.Info("Reconciling DynamoComponentDeployment", "key", key, "name", dcd.Name)
		if err := r.preserveExistingDCDState(ctx, dcd); err != nil {
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
		resources = append(resources, syncedDCD)
	}

	if err := r.deleteOrphanedElasticEPFollowers(ctx, dgd, dcds); err != nil {
		return ReconcileResult{}, fmt.Errorf("failed to delete orphaned elastic-EP followers: %w", err)
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

	// Artifact identity is independent of startup policy. Preserve the automatic
	// SnapshotJob handoff even while WaitForCheckpoint keeps replicas gated.
	if checkpointInfo.AutomaticSnapshotJob != nil {
		if err := applyRestoreCandidateMetadataToDCD(dcd, checkpointInfo); err != nil {
			return err
		}
	}

	if checkpointInfo.StartupPolicy == nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint && !checkpointInfo.Ready {
		dcd.Spec.Replicas = ptr.To(int32(0))
		return nil
	}
	if checkpointInfo.StartupPolicy != "" &&
		checkpointInfo.StartupPolicy != nvidiacomv1alpha1.CheckpointStartupPolicyImmediate {
		return nil
	}
	if checkpointInfo.AutomaticSnapshotJob != nil {
		return nil
	}
	return applyRestoreCandidateMetadataToDCD(dcd, checkpointInfo)
}

func applyRestoreCandidateMetadataToDCD(
	dcd *nvidiacomv1beta1.DynamoComponentDeployment,
	checkpointInfo *checkpoint.CheckpointInfo,
) error {
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
	return checkpoint.ApplyRestoreCandidateMetadata(annotations, checkpointInfo)
}

// preserveExistingDCDState carries forward immutable server state that must not
// be overwritten by a generated DCD.
func (r *componentWorkloadsReconciler) preserveExistingDCDState(
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

	// A synthesized follower's replica count is seeded once, at creation, and never
	// re-asserted afterwards.
	//
	// Synthesis is ungated and re-derives the follower on every pass, stamping the
	// declared launch width (`--data-parallel-size` minus the leader's own rank). That
	// stamp is what a follower is CREATED with; it is not a target the operator drives
	// the count back to. Once the object exists, the live value wins in both directions:
	//
	//   scaled up   3 -> 5   stays 5
	//   scaled down 3 -> 1   stays 1
	//
	// This holds at either gate position, which makes the gate deliberately absent from
	// this decision. "Gate off" means the count stops changing -- it freezes wherever it
	// is -- not that the operator drags it back to the declared width. Dragging it back
	// would be the worse of the two failures in both directions: downward it deletes pods
	// that may hold live engine ranks (nothing calls scale_elastic_ep to drain them
	// first, and DYN-3838 / DYN-2660 record what that leaves behind), and upward it
	// re-adds capacity an operator deliberately removed.
	//
	// Without the preservation at all, generation classifies any external scale as a
	// manual change: a cluster reverted `replicas: 1` within two seconds, logging
	// "Manual changes detected ... will be overwritten".
	if existing.GetAnnotations()[consts.KubeAnnotationElasticEPFollower] == consts.KubeLabelValueTrue &&
		existing.Spec.Replicas != nil {
		desired.Spec.Replicas = existing.Spec.Replicas
	}
	return nil
}

// deleteOrphanedElasticEPFollowers removes synthesized elastic-EP follower DCDs that
// generation no longer produces.
//
// A follower is derived, never declared, so nothing else will ever clean it up. The
// rollout path prunes worker DCDs by comparing their hash label against the current
// worker generation, and a follower's label still matches -- the hash is deliberately
// gate-independent -- so it would survive as an orphan owned by no one.
//
// Three things strand a follower this way: an administrator disabling
// features.ElasticEPRayPoC, a user removing the elastic-EP flags from the leader, and a
// user deleting the leader component outright. Comparing against what generation
// actually produced covers all three without asking why.
func (r *componentWorkloadsReconciler) deleteOrphanedElasticEPFollowers(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	generated map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
) error {
	logger := log.FromContext(ctx)

	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	if err := r.syncer.List(ctx, dcdList,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name},
	); err != nil {
		return fmt.Errorf("failed to list DynamoComponentDeployments: %w", err)
	}

	wanted := make(map[string]struct{}, len(generated))
	for _, dcd := range generated {
		if dcd != nil {
			wanted[dcd.Name] = struct{}{}
		}
	}

	var deleteErrors []error
	for i := range dcdList.Items {
		existing := &dcdList.Items[i]
		if existing.GetAnnotations()[consts.KubeAnnotationElasticEPFollower] != consts.KubeLabelValueTrue {
			continue
		}
		if _, keep := wanted[existing.Name]; keep {
			continue
		}
		// Prove ownership before destroying anything. The list is narrowed only by the
		// DGD-name label and the follower annotation, and both are mutable and settable
		// by anyone -- so without this a standalone or foreign-owned DCD that happens to
		// carry those two values lands in this loop and is deleted by a DGD that does not
		// control it.
		if !metav1.IsControlledBy(existing, dgd) {
			logger.Info(
				"Skipping a follower-marked DynamoComponentDeployment this DynamoGraphDeployment does not control",
				"name", existing.Name,
			)
			continue
		}
		// Only release a follower that is provably empty. Nothing in the operator calls
		// scale_elastic_ep -- there is no engine-control client in the tree -- so deleting
		// a follower that still holds ranks leaves the engine committed to a DP size whose
		// members are gone: DYN-3838 records the leader surviving at restart=0 with
		// inference stopped, and DYN-2660 records the orphaned placement group then
		// blocking every later scale-up until the pod restarts. DYN-3686 classifies that
		// state as a fault on the recovery path, not a scale-down signal, so the operator
		// must not manufacture it.
		//
		// Today this is vacuous for the disable case by construction -- a follower rests at
		// zero and only an external scale client moves it -- so it is not dead code, it is
		// the precondition a Phase 7 drain will satisfy. It also gives "turning the gate
		// off stops scaling" its literal meaning: running capacity is left alone.
		if replicas := existing.Spec.Replicas; replicas != nil && *replicas > 0 {
			logger.Info(
				"Refusing to delete an elastic-EP follower that still has replicas; scale it to zero first",
				"name", existing.Name, "replicas", *replicas,
			)
			if recorder := r.syncer.GetRecorder(); recorder != nil {
				recorder.Eventf(
					dgd, nil, corev1.EventTypeWarning, "ElasticEPFollowerNotReleased", "Delete",
					"follower %s still has %d replicas and may hold live engine ranks; scale it to zero before it can be removed",
					existing.Name, *replicas,
				)
			}
			continue
		}
		logger.Info("Deleting orphaned elastic-EP follower", "name", existing.Name)
		// UID and resourceVersion preconditions, so this cannot lose a delete race. The
		// name is reused across generations, so between the List above and this call the
		// object may already have been replaced by a follower that generation does want.
		// Without the preconditions that replacement is what gets deleted; with them the
		// API server refuses and the next reconcile re-evaluates.
		preconditions := client.Preconditions{
			UID:             &existing.UID,
			ResourceVersion: &existing.ResourceVersion,
		}
		if err := r.syncer.Delete(ctx, existing, preconditions); err != nil &&
			!apierrors.IsNotFound(err) && !apierrors.IsConflict(err) {
			deleteErrors = append(deleteErrors, fmt.Errorf("delete %s: %w", existing.Name, err))
		}
	}
	if len(deleteErrors) > 0 {
		return fmt.Errorf("failed to delete %d orphaned followers: %v", len(deleteErrors), deleteErrors)
	}
	return nil
}
