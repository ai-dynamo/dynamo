/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/workermetadata"
)

// componentServing counts, by owning DCD, the ready pods of one worker
// component that the ready frontends of its DGD serve.
type componentServing struct {
	// byEveryFrontend counts pods that every ready frontend serves.
	byEveryFrontend map[string]int32
	// byAnyFrontend counts pods that at least one ready frontend serves.
	byAnyFrontend map[string]int32
}

// frontendServing reports which ready pods of a worker component the ready
// frontends of dgd serve.
//
// Pod readiness does not mean a frontend can route to a worker: the frontend
// first discovers the worker, builds a pipeline for its model card, and needs
// the worker's generation to form a complete serving topology, such as both
// prefill and decode. Frontends publish the workers they serve in their
// DynamoWorkerMetadata resources.
//
// The second result is false when no ready frontend publishes admission
// records, as with frontends from earlier releases or without Kubernetes
// discovery. While any ready frontend publishes them, a frontend that does not
// serves no worker.
func (r *dgdWorkerRolloutReconciler) frontendServing(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
) (componentServing, bool, error) {
	// Find the frontends that currently receive traffic.
	frontendPods := &corev1.PodList{}
	if err := r.List(ctx, frontendPods,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			consts.KubeLabelDynamoComponentType:       consts.ComponentTypeFrontend,
		},
	); err != nil {
		return componentServing{}, false, fmt.Errorf("list frontend pods: %w", err)
	}
	var readyFrontends []*corev1.Pod
	for i := range frontendPods.Items {
		if podServesTraffic(&frontendPods.Items[i]) {
			readyFrontends = append(readyFrontends, &frontendPods.Items[i])
		}
	}
	if len(readyFrontends) == 0 {
		return componentServing{}, false, nil
	}

	// Index the discovery records in the namespace by the pod that published them.
	metadataList := workermetadata.NewList()
	if err := r.List(ctx, metadataList, client.InNamespace(dgd.Namespace)); err != nil {
		if apierrors.IsNotFound(err) || apimeta.IsNoMatchError(err) {
			return componentServing{}, false, nil
		}
		return componentServing{}, false, fmt.Errorf("list DynamoWorkerMetadata: %w", err)
	}
	metadataByPod := make(map[string][]*unstructured.Unstructured)
	for i := range metadataList.Items {
		if podName, ok := workermetadata.OwningPod(&metadataList.Items[i]); ok {
			metadataByPod[podName] = append(metadataByPod[podName], &metadataList.Items[i])
		}
	}

	// Read which workers each ready frontend serves.
	admissions := make([]workermetadata.FrontendAdmission, 0, len(readyFrontends))
	anyCapable := false
	for _, pod := range readyFrontends {
		admission := workermetadata.FrontendAdmission{Members: make(map[string]struct{})}
		for _, metadata := range metadataByPod[pod.Name] {
			workermetadata.AddFrontendAdmission(metadata, &admission)
		}
		anyCapable = anyCapable || admission.Capable
		admissions = append(admissions, admission)
	}
	if !anyCapable {
		return componentServing{}, false, nil
	}

	componentPods := &corev1.PodList{}
	if err := r.List(ctx, componentPods,
		client.InNamespace(dgd.Namespace),
		client.MatchingFields{dgdComponentPodIndex: dgdComponentPodIndexValue(dgd.Name, componentName)},
	); err != nil {
		return componentServing{}, false, fmt.Errorf("list pods for component %s: %w", componentName, err)
	}

	// Collect each pod's model cards. A worker can become Ready before it
	// publishes its cards, so once any pod of the component publishes cards, a
	// pod without them is not served yet; otherwise it has nothing to serve.
	cardsByPod := make(map[string]map[string]struct{}, len(componentPods.Items))
	registersModels := false
	for i := range componentPods.Items {
		pod := &componentPods.Items[i]
		cards := make(map[string]struct{})
		for _, metadata := range metadataByPod[pod.Name] {
			workermetadata.AddBaseModelCards(metadata, cards)
		}
		cardsByPod[pod.Name] = cards
		registersModels = registersModels || len(cards) > 0
	}

	// Count each DCD's ready pods by how many frontends serve them.
	serving := componentServing{
		byEveryFrontend: make(map[string]int32),
		byAnyFrontend:   make(map[string]int32),
	}
	for i := range componentPods.Items {
		pod := &componentPods.Items[i]
		if !podServesTraffic(pod) {
			continue
		}
		dcdName := pod.Labels[consts.KubeLabelDynamoSelector]
		cards := cardsByPod[pod.Name]
		if len(cards) == 0 {
			if !registersModels {
				serving.byEveryFrontend[dcdName]++
				serving.byAnyFrontend[dcdName]++
			}
			continue
		}
		servingFrontends := 0
		for _, admission := range admissions {
			if admission.Capable && servesAll(admission, cards) {
				servingFrontends++
			}
		}
		if servingFrontends == len(admissions) {
			serving.byEveryFrontend[dcdName]++
		}
		if servingFrontends > 0 {
			serving.byAnyFrontend[dcdName]++
		}
	}
	return serving, true, nil
}

// servesAll reports whether admission lists every card in cards.
func servesAll(admission workermetadata.FrontendAdmission, cards map[string]struct{}) bool {
	for card := range cards {
		if _, ok := admission.Members[card]; !ok {
			return false
		}
	}
	return true
}

func isDGDFrontendPod(obj client.Object) bool {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return false
	}
	return pod.Labels[consts.KubeLabelDynamoGraphDeploymentName] != "" &&
		pod.Labels[consts.KubeLabelDynamoComponentType] == consts.ComponentTypeFrontend
}

// dgdFrontendPodEventPredicate admits frontend events that change which
// frontends must serve a replacement worker before a rollout retires an old one.
func dgdFrontendPodEventPredicate() predicate.Predicate {
	return predicate.Funcs{
		CreateFunc: func(e event.CreateEvent) bool {
			return isDGDFrontendPod(e.Object)
		},
		DeleteFunc: func(e event.DeleteEvent) bool {
			return isDGDFrontendPod(e.Object)
		},
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldFrontend := isDGDFrontendPod(e.ObjectOld)
			newFrontend := isDGDFrontendPod(e.ObjectNew)
			if oldFrontend != newFrontend {
				return true
			}
			if !newFrontend {
				return false
			}
			return podServesTraffic(e.ObjectOld.(*corev1.Pod)) != podServesTraffic(e.ObjectNew.(*corev1.Pod))
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}

func mapDGDFrontendPodToRequests(_ context.Context, obj client.Object) []ctrl.Request {
	if !isDGDFrontendPod(obj) {
		return nil
	}
	dgdName := obj.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName]
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: obj.GetNamespace(), Name: dgdName}}}
}

// workerMetadataChangedPredicate admits DynamoWorkerMetadata events whose
// projected discovery records changed. The cache holds projections, so changes
// confined to dropped fields, such as a model card's contents, are ignored.
func workerMetadataChangedPredicate() predicate.Predicate {
	return predicate.Funcs{
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldMetadata, oldOK := e.ObjectOld.(*unstructured.Unstructured)
			newMetadata, newOK := e.ObjectNew.(*unstructured.Unstructured)
			if !oldOK || !newOK {
				return true
			}
			return !equality.Semantic.DeepEqual(oldMetadata.Object["spec"], newMetadata.Object["spec"])
		},
		GenericFunc: func(event.GenericEvent) bool {
			return false
		},
	}
}

// mapWorkerMetadataToDGDRequests enqueues the DGD whose pod published obj.
func (r *DynamoGraphDeploymentReconciler) mapWorkerMetadataToDGDRequests(ctx context.Context, obj client.Object) []ctrl.Request {
	metadata, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return nil
	}
	podName, ok := workermetadata.OwningPod(metadata)
	if !ok {
		return nil
	}
	pod := &corev1.Pod{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: obj.GetNamespace(), Name: podName}, pod); err != nil {
		return nil
	}
	dgdName := pod.Labels[consts.KubeLabelDynamoGraphDeploymentName]
	if dgdName == "" {
		return nil
	}
	return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: obj.GetNamespace(), Name: dgdName}}}
}

// podServesTraffic reports whether pod is Ready and neither terminating nor
// finished.
func podServesTraffic(pod *corev1.Pod) bool {
	if pod.DeletionTimestamp != nil || isTerminalPhase(pod.Status.Phase) {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}
