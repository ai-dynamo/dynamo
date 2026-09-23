// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// mergeLPXChildStatus composes a current LGD observation into the ordinary
// Grove result. The outer DGD controller remains the only status writer.
func mergeLPXChildStatus(
	source *v1beta1.DynamoGraphDeployment,
	child *v1alpha1.LPXGraphDeployment,
	ordinary ReconcileResult,
) ReconcileResult {
	components := lpx.Components(source)
	if len(components) == 0 && child == nil {
		return ordinary
	}
	if ordinary.ComponentStatus == nil {
		ordinary.ComponentStatus = make(map[string]v1beta1.ComponentReplicaStatus)
	}
	for _, component := range components {
		kind := v1beta1.ComponentKindPodCliqueScalingGroup
		if component.ComponentRole(v1beta1.ComponentRoleLPXConductor) == nil {
			kind = v1beta1.ComponentKindPodClique
		}
		ordinary.ComponentStatus[component.ComponentName] = v1beta1.ComponentReplicaStatus{ComponentKind: kind}
	}

	current := len(components) > 0 && child != nil && child.DeletionTimestamp.IsZero()
	observed := current && child.Status.ObservedGeneration == child.Generation
	if observed {
		for _, component := range components {
			if observedStatus, found := child.Status.Components[component.ComponentName]; found {
				ordinary.ComponentStatus[component.ComponentName] = *observedStatus.DeepCopy()
			}
		}
	}

	// A current failure is actionable even if reconciliation did not finish.
	if current {
		ready := meta.FindStatusCondition(child.Status.Conditions, "Ready")
		if ordinary.State != v1beta1.DGDStateFailed && ready != nil &&
			ready.ObservedGeneration == child.Generation && ready.Status == metav1.ConditionFalse &&
			ready.Reason == v1alpha1.LPXReadyReasonFailed {
			ordinary.State = v1beta1.DGDStateFailed
			ordinary.Reason, ordinary.Message = Reason(ready.Reason), Message(ready.Message)
			return ordinary
		}
		if observed && ready != nil && ready.ObservedGeneration == child.Generation && ready.Status == metav1.ConditionTrue {
			return ordinary
		}
		if observed && ordinary.State != v1beta1.DGDStateFailed && ready != nil && ready.ObservedGeneration == child.Generation {
			ordinary.State = v1beta1.DGDStatePending
			ordinary.Reason, ordinary.Message = Reason(ready.Reason), Message(ready.Message)
			return ordinary
		}
	}
	if ordinary.State != v1beta1.DGDStateFailed {
		ordinary.State = v1beta1.DGDStatePending
		ordinary.Reason = "LPXChildPending"
		ordinary.Message = "Waiting for the current LPX engine revision and readiness"
	}
	return ordinary
}
