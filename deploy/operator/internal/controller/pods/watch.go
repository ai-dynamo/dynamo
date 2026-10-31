/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pods

import (
	"context"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	poddiagnostics "github.com/ai-dynamo/dynamo/deploy/operator/internal/pods"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// OwnerName returns the name, in the Pod's namespace, of the object that a Pod
// belongs to, and whether it has one.
type OwnerName func(client.Object) (string, bool)

// DCDName returns the DCD whose Deployment selects obj. A DCD's Deployment
// shares its name and selects Pods by that name. Grove Pods carry the same label
// but belong to no DCD.
func DCDName(obj client.Object) (string, bool) {
	labels := obj.GetLabels()
	name := labels[commonconsts.KubeLabelDynamoSelector]
	return name, name != "" && labels[grovecommon.LabelPodClique] == ""
}

// DGDName returns the graph whose component obj is labeled for.
func DGDName(obj client.Object) (string, bool) {
	labels := obj.GetLabels()
	name := labels[commonconsts.KubeLabelDynamoGraphDeploymentName]
	return name, name != "" && labels[commonconsts.KubeLabelDynamoComponent] != ""
}

// EnqueueOwners enqueues the owner of the Pod in an event. The mapped handler
// maps both sides of an update, so a Pod's previous owner is enqueued too.
func EnqueueOwners(ownerName OwnerName) handler.EventHandler {
	return handler.EnqueueRequestsFromMapFunc(func(_ context.Context, obj client.Object) []reconcile.Request {
		name, ok := ownerName(obj)
		if !ok {
			return nil
		}
		return []reconcile.Request{{Namespace: obj.GetNamespace(), Name: name}}
	})
}

// CrashLoopPredicate admits the Pod events that can change an owner's
// crash-loop diagnosis. A container restart changes no workload status.
func CrashLoopPredicate(ownerName OwnerName) predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			pod, ok := e.Object.(*corev1.Pod)
			if !ok {
				return false
			}
			_, owned := ownerName(pod)
			_, crashLooping := poddiagnostics.CrashLoopOf(pod)
			return owned && crashLooping
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			pod, ok := e.Object.(*corev1.Pod)
			if !ok {
				return false
			}
			_, owned := ownerName(pod)
			_, crashLooping := poddiagnostics.CrashLoopOf(pod)
			return owned && crashLooping
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			// A Pod joining, leaving or moving between owners can change both owners' diagnoses.
			oldOwner, oldOwned := ownerName(e.ObjectOld)
			newOwner, newOwned := ownerName(e.ObjectNew)
			if oldOwned != newOwned || (newOwned && oldOwner != newOwner) {
				return true
			}
			if !newOwned {
				return false
			}
			oldPod, oldOK := e.ObjectOld.(*corev1.Pod)
			newPod, newOK := e.ObjectNew.(*corev1.Pod)
			return oldOK && newOK && poddiagnostics.CrashLoopChanged(oldPod, newPod)
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}
