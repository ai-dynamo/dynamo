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

// mergeLPXChildStatus composes a current LGD observation into the managed
// Grove result. The outer DGD controller remains the only status writer.
// dgd selects LPX and child is the nonnil result of a successful handoff.
// The result reuses the child's component status.
func mergeLPXChildStatus(
	dgd *v1beta1.DynamoGraphDeployment,
	child *v1alpha1.LPXGraphDeployment,
	result ReconcileResult,
) ReconcileResult {
	components := lpx.Components(dgd)
	if result.ComponentStatus == nil {
		result.ComponentStatus = make(map[string]v1beta1.ComponentReplicaStatus)
	}
	current := child.DeletionTimestamp.IsZero()
	observed := current && child.Status.ObservedGeneration == child.Generation
	for _, component := range components {
		if observed {
			if observedStatus, found := child.Status.Components[component.ComponentName]; found {
				result.ComponentStatus[component.ComponentName] = observedStatus.ComponentReplicaStatus
				continue
			}
		}
		kind := v1beta1.ComponentKindPodCliqueScalingGroup
		if component.ComponentRole(v1beta1.ComponentRoleLPXConductor) == nil {
			kind = v1beta1.ComponentKindPodClique
		}
		result.ComponentStatus[component.ComponentName] = v1beta1.ComponentReplicaStatus{ComponentKind: kind}
	}

	// A current failure is actionable even if reconciliation did not finish.
	if current {
		ready := meta.FindStatusCondition(child.Status.Conditions, "Ready")
		if result.State != v1beta1.DGDStateFailed && ready != nil &&
			ready.ObservedGeneration == child.Generation && ready.Status == metav1.ConditionFalse &&
			ready.Reason == v1alpha1.LPXReadyReasonFailed {
			result.State = v1beta1.DGDStateFailed
			result.Reason, result.Message = Reason(ready.Reason), Message(ready.Message)
			return result
		}
		if observed && ready != nil && ready.ObservedGeneration == child.Generation && ready.Status == metav1.ConditionTrue {
			return result
		}
		if observed && result.State != v1beta1.DGDStateFailed && ready != nil && ready.ObservedGeneration == child.Generation {
			result.State = v1beta1.DGDStatePending
			result.Reason, result.Message = Reason(ready.Reason), Message(ready.Message)
			return result
		}
	}
	if result.State != v1beta1.DGDStateFailed {
		result.State = v1beta1.DGDStatePending
		result.Reason = "LPXChildPending"
		result.Message = "Waiting for the current LPX engine revision and readiness"
	}
	return result
}
