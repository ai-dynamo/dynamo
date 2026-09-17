// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// mergeLPXChildStatus composes a current LGD observation into the ordinary
// Grove result. The outer DGD controller remains the only status writer.
func mergeLPXChildStatus(
	source *v1beta1.DynamoGraphDeployment,
	child *v1alpha1.LPXGraphDeployment,
	ordinary ReconcileResult,
) (ReconcileResult, *v1beta1.DynamoGraphDeploymentLPXStatus) {
	components := lpx.Components(source)
	if len(components) == 0 && child == nil {
		return ordinary, nil
	}
	if ordinary.ComponentStatus == nil {
		ordinary.ComponentStatus = make(map[string]v1beta1.ComponentReplicaStatus)
	}
	serving := lpx.ServingComponent(source)
	for _, component := range components {
		kind := v1beta1.ComponentKindPodCliqueScalingGroup
		if component != serving {
			kind = v1beta1.ComponentKindPodClique
		}
		ordinary.ComponentStatus[component.ComponentName] = v1beta1.ComponentReplicaStatus{ComponentKind: kind}
	}

	current := len(components) > 0 && child != nil && child.DeletionTimestamp.IsZero()
	observed := current && child.Status.ObservedGeneration == child.Generation
	var status *v1beta1.DynamoGraphDeploymentLPXStatus
	if observed {
		status = &v1beta1.DynamoGraphDeploymentLPXStatus{
			ModelDownload: child.Status.ModelDownload.DeepCopy(),
			Placement:     child.Status.Placement.DeepCopy(),
		}
		for _, component := range components {
			if observedStatus, found := child.Status.Components[component.ComponentName]; found {
				ordinary.ComponentStatus[component.ComponentName] = *observedStatus.DeepCopy()
			}
		}
	}

	if ordinary.State != v1beta1.DGDStateFailed && current {
		failed := meta.FindStatusCondition(child.Status.Conditions, "Failed")
		if failed != nil && failed.Status == metav1.ConditionTrue && failed.ObservedGeneration == child.Generation {
			ordinary.State = v1beta1.DGDStateFailed
			ordinary.Reason, ordinary.Message = Reason(failed.Reason), Message(failed.Message)
			return ordinary, status
		}
	}
	if observed {
		ready := meta.FindStatusCondition(child.Status.Conditions, "Ready")
		if ready != nil && ready.ObservedGeneration == child.Generation && ready.Status == metav1.ConditionTrue {
			return ordinary, status
		}
		if ordinary.State != v1beta1.DGDStateFailed && ready != nil && ready.ObservedGeneration == child.Generation {
			ordinary.State = v1beta1.DGDStatePending
			ordinary.Reason, ordinary.Message = Reason(ready.Reason), Message(ready.Message)
			return ordinary, status
		}
	}
	if ordinary.State != v1beta1.DGDStateFailed {
		ordinary.State = v1beta1.DGDStatePending
		ordinary.Reason = "LPXChildPending"
		ordinary.Message = "Waiting for the current LPX engine revision and readiness"
	}
	return ordinary, status
}

// lpxPlacementProjection retains an independent ordinary-DGD placement value,
// but mirrors current LPX placement and clears stale mirrors.
func lpxPlacementProjection(
	source *v1beta1.DynamoGraphDeployment,
	current *v1beta1.PlacementStatus,
	previousLPX *v1beta1.DynamoGraphDeploymentLPXStatus,
	nextLPX *v1beta1.DynamoGraphDeploymentLPXStatus,
) *v1beta1.PlacementStatus {
	if nextLPX != nil {
		return nextLPX.Placement
	}
	if len(lpx.Components(source)) > 0 {
		return nil
	}
	if previousLPX != nil && apiequality.Semantic.DeepEqual(current, previousLPX.Placement) {
		return nil
	}
	return current
}
