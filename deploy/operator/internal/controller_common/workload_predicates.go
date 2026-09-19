// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller_common

import (
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// GenerationOrDeletionChangedPredicate observes desired-state changes and finalizer entry.
func GenerationOrDeletionChangedPredicate() predicate.Predicate {
	return predicate.Or(
		predicate.GenerationChangedPredicate{},
		predicate.Funcs{
			CreateFunc:  func(event.CreateEvent) bool { return false },
			DeleteFunc:  func(event.DeleteEvent) bool { return false },
			GenericFunc: func(event.GenericEvent) bool { return false },
			UpdateFunc: func(update event.UpdateEvent) bool {
				return update.ObjectOld.GetDeletionTimestamp().IsZero() &&
					!update.ObjectNew.GetDeletionTimestamp().IsZero()
			},
		},
	)
}

// groveScheduledConditionChanged reports whether the scheduling conditions
// consumed by Grove readiness changed.
func groveScheduledConditionChanged(oldConditions, newConditions []metav1.Condition) bool {
	for _, conditionType := range []string{
		groveconstants.ConditionTypePodCliqueScheduled,
		groveconstants.ConditionTypeMinAvailableBreached,
	} {
		oldCondition := meta.FindStatusCondition(oldConditions, conditionType)
		newCondition := meta.FindStatusCondition(newConditions, conditionType)
		if (oldCondition == nil) != (newCondition == nil) {
			return true
		}
		if oldCondition != nil &&
			newCondition != nil &&
			(oldCondition.Status != newCondition.Status ||
				oldCondition.Reason != newCondition.Reason ||
				oldCondition.Message != newCondition.Message) {
			return true
		}
	}
	return false
}

// PodCliqueStatusChangeIsSignificant mirrors every PodClique field consumed by
// Grove readiness and capacity classification.
func PodCliqueStatusChangeIsSignificant(
	oldPodClique *grovev1alpha1.PodClique,
	newPodClique *grovev1alpha1.PodClique,
) bool {
	return oldPodClique.Status.ReadyReplicas != newPodClique.Status.ReadyReplicas ||
		oldPodClique.Status.UpdatedReplicas != newPodClique.Status.UpdatedReplicas ||
		oldPodClique.Status.Replicas != newPodClique.Status.Replicas ||
		oldPodClique.Status.ScheduledReplicas != newPodClique.Status.ScheduledReplicas ||
		oldPodClique.Status.ScheduleGatedReplicas != newPodClique.Status.ScheduleGatedReplicas ||
		oldPodClique.Spec.Replicas != newPodClique.Spec.Replicas ||
		!ptr.Equal(oldPodClique.Status.ObservedGeneration, newPodClique.Status.ObservedGeneration) ||
		!ptr.Equal(oldPodClique.Status.CurrentPodCliqueSetGenerationHash, newPodClique.Status.CurrentPodCliqueSetGenerationHash) ||
		(oldPodClique.Status.UpdateProgress != nil && oldPodClique.Status.UpdateProgress.UpdateEndedAt != nil) !=
			(newPodClique.Status.UpdateProgress != nil && newPodClique.Status.UpdateProgress.UpdateEndedAt != nil) ||
		groveScheduledConditionChanged(oldPodClique.Status.Conditions, newPodClique.Status.Conditions)
}

// PodCliqueScalingGroupStatusChangeIsSignificant mirrors every PodCliqueScalingGroup field
// consumed by Grove readiness and capacity classification.
func PodCliqueScalingGroupStatusChangeIsSignificant(
	oldScalingGroup *grovev1alpha1.PodCliqueScalingGroup,
	newScalingGroup *grovev1alpha1.PodCliqueScalingGroup,
) bool {
	return oldScalingGroup.Status.AvailableReplicas != newScalingGroup.Status.AvailableReplicas ||
		oldScalingGroup.Status.UpdatedReplicas != newScalingGroup.Status.UpdatedReplicas ||
		oldScalingGroup.Status.Replicas != newScalingGroup.Status.Replicas ||
		oldScalingGroup.Status.ScheduledReplicas != newScalingGroup.Status.ScheduledReplicas ||
		oldScalingGroup.Spec.Replicas != newScalingGroup.Spec.Replicas ||
		!ptr.Equal(oldScalingGroup.Status.ObservedGeneration, newScalingGroup.Status.ObservedGeneration) ||
		!ptr.Equal(oldScalingGroup.Status.CurrentPodCliqueSetGenerationHash, newScalingGroup.Status.CurrentPodCliqueSetGenerationHash) ||
		(oldScalingGroup.Status.UpdateProgress != nil && oldScalingGroup.Status.UpdateProgress.UpdateEndedAt != nil) !=
			(newScalingGroup.Status.UpdateProgress != nil && newScalingGroup.Status.UpdateProgress.UpdateEndedAt != nil) ||
		groveScheduledConditionChanged(oldScalingGroup.Status.Conditions, newScalingGroup.Status.Conditions)
}
