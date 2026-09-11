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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
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
			CreateFunc:  func(event.CreateEvent) bool { return true },
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
				commoncontroller.PodCliqueStatusChangeIsSignificant(oldPodClique, newPodClique)
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
				commoncontroller.PodCliqueScalingGroupStatusChangeIsSignificant(oldScalingGroup, newScalingGroup)
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

	pcs, found := s.resolveOwningPodCliqueSet(ctx, podClique)
	if !found {
		return nil
	}

	return mapPodCliqueSetToDGDRequest(pcs)
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

	pcs, found := s.resolvePodCliqueSetOwner(ctx, pcsg)
	if !found {
		return nil
	}
	return mapPodCliqueSetToDGDRequest(pcs)
}

func (s *groveWatchSetup) resolveOwningPodCliqueSet(
	ctx context.Context,
	podClique *grovev1alpha1.PodClique,
) (*grovev1alpha1.PodCliqueSet, bool) {
	controllerRef := metav1.GetControllerOf(podClique)
	if controllerRef == nil || controllerRef.APIVersion != grovev1alpha1.SchemeGroupVersion.String() {
		return nil, false
	}

	switch controllerRef.Kind {
	case "PodCliqueSet":
		return s.getPodCliqueSet(ctx, podClique.Namespace, controllerRef)
	case "PodCliqueScalingGroup":
		pcsg := &grovev1alpha1.PodCliqueScalingGroup{}
		if err := s.reader.Get(ctx, types.NamespacedName{
			Name:      controllerRef.Name,
			Namespace: podClique.Namespace,
		}, pcsg); err != nil || pcsg.UID != controllerRef.UID {
			return nil, false
		}
		return s.resolvePodCliqueSetOwner(ctx, pcsg)
	default:
		return nil, false
	}
}

func (s *groveWatchSetup) resolvePodCliqueSetOwner(
	ctx context.Context,
	pcsg *grovev1alpha1.PodCliqueScalingGroup,
) (*grovev1alpha1.PodCliqueSet, bool) {
	controllerRef := metav1.GetControllerOf(pcsg)
	if controllerRef == nil ||
		controllerRef.APIVersion != grovev1alpha1.SchemeGroupVersion.String() ||
		controllerRef.Kind != "PodCliqueSet" {
		return nil, false
	}
	return s.getPodCliqueSet(ctx, pcsg.Namespace, controllerRef)
}

func (s *groveWatchSetup) getPodCliqueSet(
	ctx context.Context,
	namespace string,
	controllerRef *metav1.OwnerReference,
) (*grovev1alpha1.PodCliqueSet, bool) {
	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := s.reader.Get(ctx, types.NamespacedName{
		Name:      controllerRef.Name,
		Namespace: namespace,
	}, pcs); err != nil {
		return nil, false
	}
	return pcs, pcs.UID == controllerRef.UID
}

func mapPodCliqueSetToDGDRequest(pcs *grovev1alpha1.PodCliqueSet) []ctrl.Request {
	pcsOwnerRef := metav1.GetControllerOf(pcs)
	if pcsOwnerRef == nil ||
		pcsOwnerRef.Name == "" ||
		pcsOwnerRef.UID == "" {
		return nil
	}

	// Accept any served DGD version from Dynamo's API group.
	groupVersion, err := schema.ParseGroupVersion(pcsOwnerRef.APIVersion)
	if err != nil ||
		groupVersion.Group != nvidiacomv1beta1.GroupVersion.Group ||
		groupVersion.Version == "" ||
		pcsOwnerRef.Kind != nvidiacomv1beta1.DynamoGraphDeploymentGVK.Kind {
		return nil
	}

	return []ctrl.Request{{
		NamespacedName: types.NamespacedName{
			Name:      pcsOwnerRef.Name,
			Namespace: pcs.Namespace,
		},
	}}
}
