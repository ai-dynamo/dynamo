// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"fmt"
	"math"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// retireLPXAttempt retires only the recorded batch, preserving other engines in
// its shared PCS. deployment and attempt must be nonnil; requests are owned by deployment.
func (r *graphReconciler) retireLPXAttempt(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	attempt *nvidiacomv1beta1.LPXAttemptStatus,
	requests []lpxv1alpha1.LPUPipelineRequest,
) (bool, error) {
	// Partition the shared workload using exact request identities, including replacements.
	identities := make(map[string]types.UID, len(attempt.Requests))
	for _, row := range attempt.Requests {
		identities[row.Name] = row.UID
	}
	pcsName := dynamo.PCSNameForLPX(deployment)
	var batch, retained []*lpxv1alpha1.LPUPipelineRequest
	for index := range requests {
		request := &requests[index]
		owner := metav1.GetControllerOf(request)
		if owner == nil || owner.UID != attempt.PodCliqueSetUID {
			continue
		}
		pcsName = owner.Name
		if uid := identities[request.Name]; uid != "" && uid == request.UID {
			batch = append(batch, request)
		} else {
			retained = append(retained, request)
		}
	}

	// Preserve the shared PCS when a later scheduling batch fails during scale-out.
	if len(retained) > 0 {
		return r.retireLPXScaleOut(ctx, deployment, pcsName, attempt.PodCliqueSetUID, retained, batch)
	}
	pending, err := r.deleteLPXPodCliqueSet(ctx, deployment, pcsName, attempt.PodCliqueSetUID)
	return pending || len(batch) > 0, err
}

// retireLPXScaleOut removes trailing engines through Grove before deleting their
// requests, whose scheduler finalizers may wait for pods to disappear. deployment
// must be nonnil and retained must contain at least one request outside the batch.
func (r *graphReconciler) retireLPXScaleOut(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	pcsName string,
	pcsUID types.UID,
	retained, batch []*lpxv1alpha1.LPUPipelineRequest,
) (bool, error) {
	// Derive the surviving prefix from immutable request targets, independent of source edits.
	groupName, replicas, err := lpxRetainedEngineScale(retained, batch)
	if err != nil {
		return false, err
	}

	// An authoritative owner read fences a replaced PCS before changing its group's scale.
	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := r.apiReader.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: pcsName}, pcs); err != nil {
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	}
	if pcs.UID != pcsUID {
		return true, nil
	}
	if !metav1.IsControlledBy(pcs, deployment) {
		return false, fmt.Errorf("refusing to scale PodCliqueSet %q without the exact LPXGraphDeployment controller owner", pcs.Name)
	}
	if !pcs.DeletionTimestamp.IsZero() {
		return true, nil
	}

	// Wait for Grove to create the group, then lower its scale without ever growing it.
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: groupName}, group); err != nil {
		return true, client.IgnoreNotFound(err)
	}
	if !metav1.IsControlledBy(group, pcs) {
		return false, fmt.Errorf("LPX scaling group lacks the current PCS owner")
	}
	if !group.DeletionTimestamp.IsZero() {
		return true, nil
	}
	if err := r.validateLPXDeploymentAuthority(ctx, deployment); err != nil {
		return false, err
	}
	if group.Spec.Replicas > replicas {
		scale := &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{ResourceVersion: group.ResourceVersion},
			Spec:       autoscalingv1.ScaleSpec{Replicas: replicas},
		}
		if err := r.SubResource("scale").Update(ctx, group, client.WithSubResourceBody(scale)); err != nil {
			return false, err
		}
		return true, nil
	}

	// Keep request cleanup fenced by exact UIDs until the scheduler releases each finalizer.
	if err := r.deleteStaleLPXRequests(ctx, deployment, batch); err != nil {
		return false, err
	}
	return len(batch) > 0, nil
}

// lpxRetainedEngineScale requires nonnil requests and a nonempty retained slice.
// Grove removes a suffix when scaling down; reject targets that would retire any
// engine outside the failed batch, including a model sharing the same replica.
func lpxRetainedEngineScale(retained, batch []*lpxv1alpha1.LPUPipelineRequest) (string, int32, error) {
	// Retain every replica with scheduler intent outside the failed batch.
	groupName := ""
	var replicas int64
	for _, request := range retained {
		scope := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef
		if scope == nil || scope.Name == "" || scope.ReplicaIndex < 0 || scope.ReplicaIndex >= math.MaxInt32 {
			return "", 0, fmt.Errorf("LPX request %q lacks a valid scaling group target", request.Name)
		}
		if groupName != "" && scope.Name != groupName {
			return "", 0, fmt.Errorf("LPX requests target different scaling groups")
		}
		groupName = scope.Name
		replicas = max(replicas, scope.ReplicaIndex+1)
	}

	// A failed scale-out must lie entirely beyond the surviving engines.
	for _, request := range batch {
		scope := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef
		if scope == nil || scope.Name != groupName || scope.ReplicaIndex < replicas {
			return "", 0, fmt.Errorf("cannot retire LPX request %q without disturbing an engine outside its scheduling batch", request.Name)
		}
	}
	return groupName, int32(replicas), nil
}
