// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	appsv1 "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// powerCapRolloutBlocked reports a managed rollout state that disables cap-only preservation.
func powerCapRolloutBlocked(status *v1beta1.RollingUpdateStatus) bool {
	return rollingUpdateInProgress(status) || (status != nil && status.Phase == v1beta1.RollingUpdatePhaseFailed)
}

type componentPowerCapPreservation struct {
	hash string
	caps map[string]string
}

// resolvePowerCapPreservation proves the exception from owned live DCDs and their owned Deployments.
// It renders without a new restart and with the retained DCDs' restart stamps: restarts stamp the
// retained DCDs in place and are not worker identity.
// dgd must be non-nil.
func (r *componentWorkloadsReconciler) resolvePowerCapPreservation(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment, checkpoints map[string]*checkpoint.CheckpointInfo) (*componentPowerCapPreservation, error) {
	if len(dynamo.DesiredWorkerPowerCaps(dgd)) == 0 || powerCapRolloutBlocked(dgd.Status.RollingUpdate) {
		return nil, nil
	}
	current := currentWorkerHashes(dgd)
	hash := current.v1
	if hash == "" {
		hash = current.v2
	}
	if hash == "" {
		return nil, nil
	}
	restarts, err := getRestartAnnotationsForWorkerHashes(ctx, r.syncer, dgd, []string{hash})
	if err != nil {
		return nil, err
	}
	desired, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, restarts, dynamo.RollingUpdateContext{NewWorkerHash: hash})
	if err != nil {
		return nil, err
	}

	// A second observed worker generation invalidates the exception even before rollout status catches up.
	workers := &v1beta1.DynamoComponentDeploymentList{}
	if err := r.syncer.List(ctx, workers, client.InNamespace(dgd.Namespace), client.MatchingLabels{consts.KubeLabelDynamoGraphDeploymentName: dgd.Name}); err != nil {
		return nil, err
	}
	for i := range workers.Items {
		worker := &workers.Items[i]
		if metav1.IsControlledBy(worker, dgd) && dynamo.IsWorkerComponent(string(worker.Spec.ComponentType)) && dynamo.GetDCDEffectiveWorkerHash(worker) != hash {
			return nil, nil
		}
	}

	return r.resolveWorkerPowerCapPreservation(ctx, dgd, desired, checkpoints, hash)
}

// resolveWorkerPowerCapPreservation compares every worker's DCD intent with its live DCD.
// A preserved DCD worker spec means the cap-only edit adds no change to the Deployment's desired template,
// so the API-server-defaulted stored template is never compared.
// It aligns each desired worker entry in place (checkpoint startup policy, stored backendFramework and
// retained cap) before comparing. dgd and desired worker entries must be non-nil.
func (r *componentWorkloadsReconciler) resolveWorkerPowerCapPreservation(ctx context.Context, dgd *v1beta1.DynamoGraphDeployment, desired map[string]*v1beta1.DynamoComponentDeployment, checkpoints map[string]*checkpoint.CheckpointInfo, hash string) (*componentPowerCapPreservation, error) {
	retained := &componentPowerCapPreservation{hash: hash, caps: map[string]string{}}
	changed := false
	for name, candidate := range desired {
		if !dynamo.IsWorkerComponent(string(candidate.Spec.ComponentType)) {
			continue
		}
		live := &v1beta1.DynamoComponentDeployment{}
		if err := r.syncer.Get(ctx, client.ObjectKeyFromObject(candidate), live); err != nil {
			if apierrors.IsNotFound(err) {
				return nil, nil
			}
			return nil, err
		}
		if !metav1.IsControlledBy(live, dgd) || live.DeletionTimestamp != nil {
			return nil, nil
		}
		if err := r.applyCheckpointStartupPolicy(candidate, checkpoints[name]); err != nil {
			return nil, err
		}
		if candidate.Spec.PodTemplate != nil && live.Spec.PodTemplate != nil {
			want, wantPresent := candidate.Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit]
			old, oldPresent := live.Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit]
			if wantPresent != oldPresent {
				return nil, nil
			}
			if oldPresent {
				changed = changed || want != old
				retained.caps[name] = old
				candidate.Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = old
			}
		}
		// Ordinary reconciliation keeps the stored backendFramework, so it is not worker drift.
		candidate.Spec.BackendFramework = live.Spec.BackendFramework
		if !dynamo.WorkerSpecsEqual(candidate, live) {
			return nil, nil
		}

		// An absent or foreign Deployment is created or reconciled from desired through normal rollout.
		deployment := &appsv1.Deployment{}
		if err := r.syncer.Get(ctx, client.ObjectKeyFromObject(live), deployment); err != nil {
			if apierrors.IsNotFound(err) {
				return nil, nil
			}
			return nil, err
		}
		if !metav1.IsControlledBy(deployment, live) {
			return nil, nil
		}
	}
	if !changed {
		return nil, nil
	}

	// Only caps may differ from the recorded generation, across every input the worker hash covers.
	identity, err := dynamo.WorkerHashWithCaps(dgd, retained.caps)
	if err != nil {
		return nil, err
	}
	if identity != hash && identity != currentWorkerHashes(dgd).v2 {
		return nil, nil
	}
	return retained, nil
}
