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
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	disaggregatedsetv1 "sigs.k8s.io/lws/api/disaggregatedset/v1"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

type disaggregatedSetWatchSetup struct {
	reader client.Reader
}

func newDisaggregatedSetWatchSetup(reader client.Reader) *disaggregatedSetWatchSetup {
	return &disaggregatedSetWatchSetup{reader: reader}
}

func (s *disaggregatedSetWatchSetup) addTo(ctrlBuilder *builder.Builder) *builder.Builder {
	return ctrlBuilder.Owns(newDisaggregatedSetObject(), builder.WithPredicates(predicate.Funcs{
		CreateFunc: func(event.CreateEvent) bool { return false },
		DeleteFunc: func(event.DeleteEvent) bool { return true },
		UpdateFunc: func(update event.UpdateEvent) bool {
			return disaggregatedSetStatusChanged(update.ObjectOld, update.ObjectNew)
		},
		GenericFunc: func(event.GenericEvent) bool { return true },
	})).Watches(
		&leaderworkersetv1.LeaderWorkerSet{},
		handler.EnqueueRequestsFromMapFunc(s.mapChildLWSToDGD),
		builder.WithPredicates(predicate.Funcs{
			CreateFunc: func(event.CreateEvent) bool { return true },
			DeleteFunc: func(event.DeleteEvent) bool { return true },
			UpdateFunc: func(update event.UpdateEvent) bool {
				return leaderWorkerSetStatusChanged(update.ObjectOld, update.ObjectNew)
			},
			GenericFunc: func(event.GenericEvent) bool { return false },
		}),
	)
}

func disaggregatedSetStatusChanged(oldObj, newObj client.Object) bool {
	oldDS, okOld := oldObj.(*unstructured.Unstructured)
	newDS, okNew := newObj.(*unstructured.Unstructured)
	if !okOld || !okNew {
		return false
	}
	return oldDS.GetGeneration() != newDS.GetGeneration() ||
		!equality.Semantic.DeepEqual(oldDS.Object["status"], newDS.Object["status"]) ||
		!equality.Semantic.DeepEqual(oldDS.GetLabels(), newDS.GetLabels()) ||
		!equality.Semantic.DeepEqual(oldDS.GetOwnerReferences(), newDS.GetOwnerReferences())
}

func leaderWorkerSetStatusChanged(oldObj, newObj client.Object) bool {
	oldLWS, okOld := oldObj.(*leaderworkersetv1.LeaderWorkerSet)
	newLWS, okNew := newObj.(*leaderworkersetv1.LeaderWorkerSet)
	if !okOld || !okNew {
		return false
	}
	return oldLWS.Generation != newLWS.Generation ||
		!equality.Semantic.DeepEqual(oldLWS.Status, newLWS.Status) ||
		!equality.Semantic.DeepEqual(oldLWS.GetLabels(), newLWS.GetLabels()) ||
		!equality.Semantic.DeepEqual(oldLWS.GetOwnerReferences(), newLWS.GetOwnerReferences())
}

func (s *disaggregatedSetWatchSetup) mapChildLWSToDGD(ctx context.Context, obj client.Object) []ctrl.Request {
	childOwner := metav1.GetControllerOf(obj)
	if childOwner == nil ||
		childOwner.APIVersion != disaggregatedsetv1.GroupVersion.String() ||
		childOwner.Kind != disaggregatedSetGVK.Kind ||
		childOwner.Name == "" ||
		childOwner.UID == "" {
		return nil
	}

	ds := newDisaggregatedSetObject()
	if err := s.reader.Get(ctx, types.NamespacedName{Name: childOwner.Name, Namespace: obj.GetNamespace()}, ds); err != nil {
		if !apierrors.IsNotFound(err) {
			log.FromContext(ctx).Error(err, "failed to map DisaggregatedSet child LeaderWorkerSet", "leaderWorkerSet", obj.GetName())
		}
		return nil
	}
	if ds.GetUID() != childOwner.UID {
		return nil
	}

	owner := metav1.GetControllerOf(ds)
	if owner == nil || owner.APIVersion != nvidiacomv1beta1.GroupVersion.String() || owner.Kind != dynamoGraphDeploymentKind {
		return nil
	}

	return []ctrl.Request{{
		NamespacedName: types.NamespacedName{Name: owner.Name, Namespace: ds.GetNamespace()},
	}}
}
