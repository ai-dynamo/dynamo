// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"fmt"
	"math"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// retireExpiredLPXRequests lowers the one live Grove group before deleting the
// affected engine suffix. pcs is nil after its disappearance, which leaves its
// requests to owner garbage collection.
func (r *graphReconciler) retireExpiredLPXRequests(
	ctx context.Context,
	deployment *nvidiacomv1alpha1.LPXGraphDeployment,
	source *nvidiacomv1beta1.DynamoGraphDeployment,
	pcs *grovev1alpha1.PodCliqueSet,
	requests []lpxv1alpha1.LPUPipelineRequest,
	expired []*lpxv1alpha1.LPUPipelineRequest,
) error {
	if len(expired) == 0 {
		return nil
	}
	owner := metav1.GetControllerOf(expired[0])
	if owner == nil || owner.Name == "" || owner.UID == "" {
		return fmt.Errorf("LPX request %q lacks an exact PodCliqueSet owner", expired[0].Name)
	}

	// A missing, replaced, or deleting PCS leaves its requests to owner GC.
	if pcs == nil || pcs.Name != owner.Name || pcs.UID != owner.UID || !pcs.DeletionTimestamp.IsZero() {
		return nil
	}
	if !metav1.IsControlledBy(pcs, deployment) {
		return fmt.Errorf("refusing to scale PodCliqueSet %q without the exact LPXGraphDeployment controller owner", pcs.Name)
	}
	for _, request := range expired {
		if !metav1.IsControlledBy(request, pcs) {
			return fmt.Errorf("expired LPX request %q does not have the current PodCliqueSet owner", request.Name)
		}
	}

	groupName, _, err := lpxScalingGroupTarget(expired[0])
	if err != nil {
		return err
	}
	group := &grovev1alpha1.PodCliqueScalingGroup{}
	if err := r.Get(ctx, client.ObjectKey{Namespace: deployment.Namespace, Name: groupName}, group); err != nil {
		if apierrors.IsNotFound(err) {
			return nil
		}
		return err
	}
	if !metav1.IsControlledBy(group, pcs) {
		return fmt.Errorf("LPX scaling group lacks the current PCS owner")
	}
	if !group.DeletionTimestamp.IsZero() {
		return nil
	}
	replicas, retiring, err := lpxExpiredEngineSuffix(requests, expired, group.Spec.Replicas)
	if err != nil {
		return err
	}
	if len(retiring) == 0 {
		return nil
	}

	if err := r.validateLPXDeploymentAuthority(ctx, deployment); err != nil {
		return err
	}
	// Omitted replicas leave capacity externally managed; expire only scheduler
	// intent so a later input revision can reuse the unchanged Grove ordinal.
	component := lpx.ServingComponent(source)
	if component == nil {
		return fmt.Errorf("cannot determine replica ownership without an LPX serving component")
	}
	if component.Replicas != nil && group.Spec.Replicas > replicas {
		scale := &autoscalingv1.Scale{
			ObjectMeta: metav1.ObjectMeta{ResourceVersion: group.ResourceVersion},
			Spec:       autoscalingv1.ScaleSpec{Replicas: replicas},
		}
		if err := r.SubResource("scale").Update(ctx, group, client.WithSubResourceBody(scale)); err != nil {
			return err
		}
	}

	// Grove and the scheduler converge asynchronously after the ordered writes.
	return r.deleteStaleLPXRequests(ctx, deployment, retiring)
}

// lpxExpiredEngineSuffix returns complete engine replicas only when every
// expired replica belongs to one contiguous trailing suffix. It orders the
// expired requests last so transient sibling cleanup leaves durable expiry
// evidence for the next reconciliation. An interior failure blocks all
// scale-down so a healthy higher ordinal is never removed.
func lpxExpiredEngineSuffix(
	requests []lpxv1alpha1.LPUPipelineRequest,
	expired []*lpxv1alpha1.LPUPipelineRequest,
	replicas int32,
) (int32, []*lpxv1alpha1.LPUPipelineRequest, error) {
	if len(expired) == 0 {
		return replicas, nil, nil
	}
	groupName, _, err := lpxScalingGroupTarget(expired[0])
	if err != nil {
		return 0, nil, err
	}
	failedReplicas := make(map[int64]struct{}, len(expired))
	for _, request := range expired {
		name, replica, err := lpxScalingGroupTarget(request)
		if err != nil {
			return 0, nil, err
		}
		if name != groupName {
			return 0, nil, fmt.Errorf("expired LPX requests target different scaling groups")
		}
		failedReplicas[replica] = struct{}{}
	}

	targetReplicas := int64(replicas)
	for targetReplicas > 0 {
		if _, failed := failedReplicas[targetReplicas-1]; !failed {
			break
		}
		targetReplicas--
	}
	for replica := range failedReplicas {
		if replica < targetReplicas {
			return replicas, nil, nil
		}
	}

	expiredNames := make(map[string]struct{}, len(expired))
	for _, request := range expired {
		expiredNames[request.Name] = struct{}{}
	}
	retiring := make([]*lpxv1alpha1.LPUPipelineRequest, 0, len(requests))
	failed := make([]*lpxv1alpha1.LPUPipelineRequest, 0, len(expired))
	for index := range requests {
		request := &requests[index]
		name, replica, err := lpxScalingGroupTarget(request)
		if err != nil {
			return 0, nil, err
		}
		if name != groupName {
			return 0, nil, fmt.Errorf("LPX requests target different scaling groups")
		}
		if replica < targetReplicas {
			continue
		}
		if _, expired := expiredNames[request.Name]; expired {
			failed = append(failed, request)
		} else {
			retiring = append(retiring, request)
		}
	}
	return int32(targetReplicas), append(retiring, failed...), nil
}

func lpxScalingGroupTarget(request *lpxv1alpha1.LPUPipelineRequest) (string, int64, error) {
	target := request.Spec.MaterializationTarget.PodCliqueScalingGroupRef
	if target == nil || target.Name == "" || target.ReplicaIndex < 0 || target.ReplicaIndex >= math.MaxInt32 {
		return "", 0, fmt.Errorf("LPX request %q lacks a valid scaling group target", request.Name)
	}
	return target.Name, target.ReplicaIndex, nil
}
