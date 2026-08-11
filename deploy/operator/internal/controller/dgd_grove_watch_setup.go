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

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	groveconstants "github.com/ai-dynamo/grove/operator/api/common/constants"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
)

// groveWatchSetup contains Grove-specific watch registration, event
// significance, and request mapping. The DGD setup remains the composition
// root and invokes this only when Grove is available.
type groveWatchSetup struct {
	reader client.Reader
}

// newGroveWatchSetup wires Grove-owned watch predicates and request mapping.
func newGroveWatchSetup(reader client.Reader) *groveWatchSetup {
	return &groveWatchSetup{reader: reader}
}

func (s *groveWatchSetup) addTo(ctrlBuilder *builder.Builder) *builder.Builder {
	return ctrlBuilder.
		Owns(&grovev1alpha1.PodCliqueSet{}, builder.WithPredicates(predicate.Funcs{
			// Creation is caused by DGD reconciliation and does not need to
			// enqueue the owner again.
			CreateFunc:  func(event.CreateEvent) bool { return false },
			DeleteFunc:  func(event.DeleteEvent) bool { return true },
			UpdateFunc:  func(event.UpdateEvent) bool { return true },
			GenericFunc: func(event.GenericEvent) bool { return true },
		})).
		Watches(
			&grovev1alpha1.PodClique{},
			handler.EnqueueRequestsFromMapFunc(s.mapPodCliqueToRequests),
			builder.WithPredicates(podCliqueEventPredicates()),
		).
		// PodCliqueScalingGroup status can settle after the final PodClique
		// update, so it needs an independent readiness watch.
		Watches(
			&grovev1alpha1.PodCliqueScalingGroup{},
			handler.EnqueueRequestsFromMapFunc(s.mapPodCliqueScalingGroupToRequests),
			builder.WithPredicates(pcsgEventPredicates()),
		)
}

func podCliqueEventPredicates() predicate.Funcs {
	return predicate.Funcs{
		CreateFunc: func(event.CreateEvent) bool { return false },
		DeleteFunc: func(event.DeleteEvent) bool { return false },
		UpdateFunc: func(updateEvent event.UpdateEvent) bool {
			oldPodClique, oldOK := updateEvent.ObjectOld.(*grovev1alpha1.PodClique)
			newPodClique, newOK := updateEvent.ObjectNew.(*grovev1alpha1.PodClique)
			return oldOK &&
				newOK &&
				podCliqueStatusChangeIsSignificant(oldPodClique, newPodClique)
		},
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}

func pcsgEventPredicates() predicate.Funcs {
	return predicate.Funcs{
		CreateFunc: func(event.CreateEvent) bool { return false },
		DeleteFunc: func(event.DeleteEvent) bool { return false },
		UpdateFunc: func(updateEvent event.UpdateEvent) bool {
			oldScalingGroup, oldOK := updateEvent.ObjectOld.(*grovev1alpha1.PodCliqueScalingGroup)
			newScalingGroup, newOK := updateEvent.ObjectNew.(*grovev1alpha1.PodCliqueScalingGroup)
			return oldOK &&
				newOK &&
				pcsgStatusChangeIsSignificant(oldScalingGroup, newScalingGroup)
		},
		GenericFunc: func(event.GenericEvent) bool { return false },
	}
}

func (s *groveWatchSetup) mapPodCliqueToRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	podClique, ok := obj.(*grovev1alpha1.PodClique)
	if !ok {
		return nil
	}

	dgdName := podClique.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName]
	if dgdName == "" {
		log.FromContext(ctx).V(1).Info(
			"PodClique missing DGD label",
			"podClique", podClique.Name,
			"namespace", podClique.Namespace,
		)
		return nil
	}

	return []ctrl.Request{{
		NamespacedName: types.NamespacedName{
			Name:      dgdName,
			Namespace: podClique.Namespace,
		},
	}}
}

// mapPodCliqueScalingGroupToRequests walks PCSG -> PCS -> DGD because the PCS
// name can be truncated and therefore cannot safely stand in for the DGD name.
func (s *groveWatchSetup) mapPodCliqueScalingGroupToRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	pcsg, ok := obj.(*grovev1alpha1.PodCliqueScalingGroup)
	if !ok {
		return nil
	}

	controllerRef := metav1.GetControllerOf(pcsg)
	if controllerRef == nil ||
		controllerRef.Kind != "PodCliqueSet" ||
		controllerRef.APIVersion != grovev1alpha1.SchemeGroupVersion.String() {
		log.FromContext(ctx).V(1).Info(
			"PodCliqueScalingGroup missing PodCliqueSet controller ownerReference",
			"podCliqueScalingGroup", pcsg.Name,
			"namespace", pcsg.Namespace,
		)
		return nil
	}

	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := s.reader.Get(ctx, types.NamespacedName{
		Name:      controllerRef.Name,
		Namespace: pcsg.Namespace,
	}, pcs); err != nil {
		log.FromContext(ctx).V(1).Info(
			"failed to look up PodCliqueSet for PCSG",
			"podCliqueScalingGroup", pcsg.Name,
			"pcsName", controllerRef.Name,
			"error", err,
		)
		return nil
	}

	pcsOwnerRef := metav1.GetControllerOf(pcs)
	if pcsOwnerRef == nil || pcsOwnerRef.Kind != consts.ResourceTypeDynamoGraphDeployment {
		log.FromContext(ctx).V(1).Info(
			"PodCliqueSet missing DynamoGraphDeployment controller ownerReference",
			"pcsName", pcs.Name,
			"namespace", pcs.Namespace,
		)
		return nil
	}

	return []ctrl.Request{{
		NamespacedName: types.NamespacedName{
			Name:      pcsOwnerRef.Name,
			Namespace: pcsg.Namespace,
		},
	}}
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
				oldCondition.Reason != newCondition.Reason) {
			return true
		}
	}
	return false
}

// podCliqueStatusChangeIsSignificant mirrors every PodClique field consumed by
// Grove readiness and capacity classification.
func podCliqueStatusChangeIsSignificant(
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
		groveScheduledConditionChanged(oldPodClique.Status.Conditions, newPodClique.Status.Conditions)
}

// pcsgStatusChangeIsSignificant mirrors every PodCliqueScalingGroup field
// consumed by Grove readiness and capacity classification.
func pcsgStatusChangeIsSignificant(
	oldScalingGroup *grovev1alpha1.PodCliqueScalingGroup,
	newScalingGroup *grovev1alpha1.PodCliqueScalingGroup,
) bool {
	return oldScalingGroup.Status.AvailableReplicas != newScalingGroup.Status.AvailableReplicas ||
		oldScalingGroup.Status.UpdatedReplicas != newScalingGroup.Status.UpdatedReplicas ||
		oldScalingGroup.Status.Replicas != newScalingGroup.Status.Replicas ||
		oldScalingGroup.Status.ScheduledReplicas != newScalingGroup.Status.ScheduledReplicas ||
		oldScalingGroup.Spec.Replicas != newScalingGroup.Spec.Replicas ||
		!ptr.Equal(oldScalingGroup.Status.ObservedGeneration, newScalingGroup.Status.ObservedGeneration) ||
		groveScheduledConditionChanged(oldScalingGroup.Status.Conditions, newScalingGroup.Status.Conditions)
}
