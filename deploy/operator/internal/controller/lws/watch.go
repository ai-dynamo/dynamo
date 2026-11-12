/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"context"
	"strconv"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

// EnqueueDCDForPod maps Pod events to the DCD that currently owns the hinted LWS.
//
// The official set-name label is only a routing hint. Missing parents return no
// request so a later StatefulSet/LWS event can unblock convergence. Mapping never
// counts or attributes faults; Observe performs the strict owner/revision checks.
func EnqueueDCDForPod(reader client.Reader) handler.EventHandler {
	return handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []reconcile.Request {
		return mapSetNameHintToDCD(ctx, reader, obj.GetNamespace(), setNameHint(obj))
	})
}

// EnqueueDCDForStatefulSet maps StatefulSet dependency events to a DCD.
//
// Prefer a label hint so an initially missing StatefulSet can still enqueue once
// it appears, instead of requiring a full owner chain that would suppress the
// missing-parent event needed for convergence.
func EnqueueDCDForStatefulSet(reader client.Reader) handler.EventHandler {
	return handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, obj client.Object) []reconcile.Request {
		setName := setNameHint(obj)
		if setName == "" {
			setName = leaderWorkerSetNameFromOwner(obj)
		}
		return mapSetNameHintToDCD(ctx, reader, obj.GetNamespace(), setName)
	})
}

// PodObservationPredicate admits Pod events that can change current LWS evidence.
func PodObservationPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			// Any set-name-hinted create can unblock a waiting failed sibling.
			_, ok := e.Object.(*corev1.Pod)
			return ok && setNameHint(e.Object) != ""
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			return setNameHint(e.Object) != ""
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldPod, oldOK := e.ObjectOld.(*corev1.Pod)
			newPod, newOK := e.ObjectNew.(*corev1.Pod)
			if !oldOK || !newOK {
				return false
			}
			if setNameHint(oldPod) == "" && setNameHint(newPod) == "" {
				return false
			}
			return podObservationChanged(oldPod, newPod)
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}

// StatefulSetObservationPredicate admits STS events that can unblock Unknown evidence.
func StatefulSetObservationPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			return setNameHint(e.Object) != "" || leaderWorkerSetNameFromOwner(e.Object) != ""
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			return setNameHint(e.Object) != "" || leaderWorkerSetNameFromOwner(e.Object) != ""
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldSTS, oldOK := e.ObjectOld.(*appsv1.StatefulSet)
			newSTS, newOK := e.ObjectNew.(*appsv1.StatefulSet)
			if !oldOK || !newOK {
				return false
			}
			if setNameHint(oldSTS) == "" && setNameHint(newSTS) == "" &&
				leaderWorkerSetNameFromOwner(oldSTS) == "" && leaderWorkerSetNameFromOwner(newSTS) == "" {
				return false
			}
			return statefulSetObservationChanged(oldSTS, newSTS)
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}

func mapSetNameHintToDCD(
	ctx context.Context,
	reader client.Reader,
	namespace, setName string,
) []reconcile.Request {
	if setName == "" {
		return nil
	}

	lws := &leaderworkersetv1.LeaderWorkerSet{}
	if err := reader.Get(ctx, types.NamespacedName{Namespace: namespace, Name: setName}, lws); err != nil {
		return nil
	}
	owner := metav1.GetControllerOf(lws)
	if owner == nil || owner.Controller == nil || !*owner.Controller {
		return nil
	}
	if owner.Kind != nvidiacomv1beta1.DynamoComponentDeploymentGVK.Kind {
		return nil
	}
	if owner.APIVersion != nvidiacomv1beta1.GroupVersion.String() &&
		owner.APIVersion != nvidiacomv1alpha1.GroupVersion.String() {
		return nil
	}
	if owner.Name == "" || owner.UID == "" {
		return nil
	}
	return []reconcile.Request{{Namespace: namespace, Name: owner.Name}}
}

func setNameHint(obj client.Object) string {
	labels := obj.GetLabels()
	if labels == nil {
		return ""
	}
	return labels[leaderworkersetv1.SetNameLabelKey]
}

func leaderWorkerSetNameFromOwner(obj client.Object) string {
	owner := metav1.GetControllerOf(obj)
	if owner == nil || owner.Controller == nil || !*owner.Controller {
		return ""
	}
	if owner.APIVersion != leaderWorkerSetAPIVersion || owner.Kind != leaderWorkerSetKind {
		return ""
	}
	return owner.Name
}

func podObservationChanged(oldPod, newPod *corev1.Pod) bool {
	if oldPod.UID != newPod.UID {
		return true
	}
	if !apiequality.Semantic.DeepEqual(oldPod.DeletionTimestamp, newPod.DeletionTimestamp) {
		return true
	}
	if !apiequality.Semantic.DeepEqual(oldPod.OwnerReferences, newPod.OwnerReferences) {
		return true
	}
	if observationLabels(oldPod.Labels) != observationLabels(newPod.Labels) {
		return true
	}
	if readyObservation(oldPod) != readyObservation(newPod) {
		return true
	}
	return mainTerminationObservation(oldPod) != mainTerminationObservation(newPod)
}

func statefulSetObservationChanged(oldSTS, newSTS *appsv1.StatefulSet) bool {
	if oldSTS.UID != newSTS.UID {
		return true
	}
	if oldSTS.Generation != newSTS.Generation {
		return true
	}
	if !apiequality.Semantic.DeepEqual(oldSTS.DeletionTimestamp, newSTS.DeletionTimestamp) {
		return true
	}
	if !apiequality.Semantic.DeepEqual(oldSTS.OwnerReferences, newSTS.OwnerReferences) {
		return true
	}
	if observationLabels(oldSTS.Labels) != observationLabels(newSTS.Labels) {
		return true
	}
	return templateRevision(oldSTS) != templateRevision(newSTS)
}

func observationLabels(labels map[string]string) string {
	if labels == nil {
		return ""
	}
	return labels[leaderworkersetv1.SetNameLabelKey] + "|" +
		labels[leaderworkersetv1.GroupIndexLabelKey] + "|" +
		labels[leaderworkersetv1.WorkerIndexLabelKey] + "|" +
		labels[leaderworkersetv1.RevisionKey]
}

func templateRevision(sts *appsv1.StatefulSet) string {
	if sts.Spec.Template.Labels == nil {
		return ""
	}
	return sts.Spec.Template.Labels[leaderworkersetv1.RevisionKey]
}

func readyObservation(pod *corev1.Pod) string {
	for i := range pod.Status.Conditions {
		if pod.Status.Conditions[i].Type != corev1.PodReady {
			continue
		}
		return string(pod.Status.Conditions[i].Status) + "|" + pod.Status.Conditions[i].LastTransitionTime.String()
	}
	return ""
}

func mainTerminationObservation(pod *corev1.Pod) string {
	main := mainContainerStatus(pod)
	if main == nil || main.State.Terminated == nil {
		return ""
	}
	return main.Name + "|" +
		strconv.FormatInt(int64(main.State.Terminated.ExitCode), 10) + "|" +
		main.State.Terminated.FinishedAt.String() + "|" +
		strconv.FormatInt(int64(main.RestartCount), 10)
}
