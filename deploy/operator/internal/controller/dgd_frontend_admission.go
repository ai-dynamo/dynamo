/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

const (
	frontendModelAdmissionTopic    = "frontend-model-admission"
	frontendModelAdmissionProtocol = "v1"
)

var (
	dynamoWorkerMetadataGVK = schema.GroupVersionKind{
		Group: "nvidia.com", Version: "v1alpha1", Kind: "DynamoWorkerMetadata",
	}
	dynamoWorkerMetadataListGVK = schema.GroupVersionKind{
		Group: "nvidia.com", Version: "v1alpha1", Kind: "DynamoWorkerMetadataList",
	}
)

type frontendAdmissionSnapshot struct {
	capable bool
	members map[string]struct{}
}

func newDynamoWorkerMetadata() *unstructured.Unstructured {
	metadata := &unstructured.Unstructured{}
	metadata.SetGroupVersionKind(dynamoWorkerMetadataGVK)
	return metadata
}

func newDynamoWorkerMetadataList() *unstructured.UnstructuredList {
	metadata := &unstructured.UnstructuredList{}
	metadata.SetGroupVersionKind(dynamoWorkerMetadataListGVK)
	return metadata
}

func (r *DynamoGraphDeploymentReconciler) mapDynamoWorkerMetadataToDGDRequests(
	ctx context.Context,
	obj client.Object,
) []ctrl.Request {
	for _, owner := range obj.GetOwnerReferences() {
		if owner.APIVersion != "v1" || owner.Kind != "Pod" || owner.Name == "" {
			continue
		}
		pod := &corev1.Pod{}
		if err := r.Get(ctx, types.NamespacedName{Namespace: obj.GetNamespace(), Name: owner.Name}, pod); err != nil {
			return nil
		}
		dgdName := pod.Labels[consts.KubeLabelDynamoGraphDeploymentName]
		if dgdName == "" {
			return nil
		}
		return []ctrl.Request{{NamespacedName: types.NamespacedName{
			Namespace: obj.GetNamespace(),
			Name:      dgdName,
		}}}
	}
	return nil
}

func dgdHasServingFrontend(dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if string(component.ComponentType) != consts.ComponentTypeFrontend {
			continue
		}
		replicas := int32(1)
		if component.Replicas != nil {
			replicas = *component.Replicas
		}
		if replicas > 0 {
			return true
		}
	}
	return false
}

func podReadyForTraffic(pod *corev1.Pod) bool {
	if !pod.DeletionTimestamp.IsZero() || isTerminalPhase(pod.Status.Phase) {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func metadataByOwningPod(items []unstructured.Unstructured) map[string][]unstructured.Unstructured {
	byPod := make(map[string][]unstructured.Unstructured)
	for i := range items {
		metadata := items[i]
		for _, owner := range metadata.GetOwnerReferences() {
			if owner.APIVersion == "v1" && owner.Kind == "Pod" && owner.Name != "" {
				byPod[owner.Name] = append(byPod[owner.Name], metadata)
				break
			}
		}
	}
	return byPod
}

func frontendAdmissionFromMetadata(
	items []unstructured.Unstructured,
	runtimeNamespace string,
) frontendAdmissionSnapshot {
	snapshot := frontendAdmissionSnapshot{members: make(map[string]struct{})}
	for i := range items {
		sources, found, _ := unstructured.NestedMap(
			items[i].Object, "spec", "data", "event_sources",
		)
		if !found {
			continue
		}
		for _, rawSource := range sources {
			source, ok := rawSource.(map[string]any)
			if !ok || source["topic"] != frontendModelAdmissionTopic {
				continue
			}
			metadata, ok := source["metadata"].(map[string]any)
			if !ok || metadata["protocol"] != frontendModelAdmissionProtocol {
				continue
			}
			if capability, ok := metadata["capability"].(bool); ok && capability {
				snapshot.capable = true
			}
			scope, ok := source["scope"].(map[string]any)
			if !ok || scope["kind"] != "namespace" || scope["name"] != runtimeNamespace {
				continue
			}
			members, ok := metadata["members"].([]any)
			if !ok {
				continue
			}
			for _, member := range members {
				if member, ok := member.(string); ok {
					snapshot.members[member] = struct{}{}
				}
			}
		}
	}
	return snapshot
}

func baseModelCardsFromMetadata(
	items []unstructured.Unstructured,
	runtimeNamespace string,
) map[string]struct{} {
	cards := make(map[string]struct{})
	for i := range items {
		modelCards, found, _ := unstructured.NestedMap(
			items[i].Object, "spec", "data", "model_cards",
		)
		if !found {
			continue
		}
		for path, rawCard := range modelCards {
			card, ok := rawCard.(map[string]any)
			if !ok || card["type"] != "Model" || card["namespace"] != runtimeNamespace {
				continue
			}
			if suffix, present := card["model_suffix"]; present && suffix != nil {
				continue
			}
			cards[path] = struct{}{}
		}
	}
	return cards
}

func containsAllMembers(admitted, required map[string]struct{}) bool {
	for member := range required {
		if _, ok := admitted[member]; !ok {
			return false
		}
	}
	return true
}

// frontendAdmittedReplicas returns how many locally available replacement
// workers every traffic-serving frontend has committed. The second return
// value is false for older frontends that do not advertise this protocol, so
// an operator-only upgrade preserves the legacy rollout behavior.
func (r *dgdWorkerRolloutReconciler) frontendAdmittedReplicas(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	newDCD *nvidiacomv1beta1.DynamoComponentDeployment,
	locallyAvailable int32,
) (int32, bool, error) {
	if locallyAvailable == 0 || !dgdHasServingFrontend(dgd) {
		return locallyAvailable, false, nil
	}

	metadataList := newDynamoWorkerMetadataList()
	if err := r.List(ctx, metadataList, client.InNamespace(dgd.Namespace)); err != nil {
		if apierrors.IsNotFound(err) || apimeta.IsNoMatchError(err) {
			return locallyAvailable, false, nil
		}
		return 0, true, err
	}
	metadataByPod := metadataByOwningPod(metadataList.Items)
	runtimeNamespace := dynamo.GetDCDRuntimeNamespace(newDCD)
	if newDCD.Status.Component != nil && newDCD.Status.Component.RuntimeNamespace != "" {
		runtimeNamespace = newDCD.Status.Component.RuntimeNamespace
	}

	frontendPods := &corev1.PodList{}
	if err := r.List(ctx, frontendPods,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			consts.KubeLabelDynamoComponentType:       consts.ComponentTypeFrontend,
		},
	); err != nil {
		return 0, true, err
	}
	frontends := make([]frontendAdmissionSnapshot, 0, len(frontendPods.Items))
	protocolActive := false
	for i := range frontendPods.Items {
		pod := &frontendPods.Items[i]
		if !podReadyForTraffic(pod) {
			continue
		}
		snapshot := frontendAdmissionFromMetadata(metadataByPod[pod.Name], runtimeNamespace)
		protocolActive = protocolActive || snapshot.capable
		frontends = append(frontends, snapshot)
	}
	if !protocolActive {
		return locallyAvailable, false, nil
	}
	for _, frontend := range frontends {
		if !frontend.capable {
			return 0, true, nil
		}
	}

	workerPods := &corev1.PodList{}
	if err := r.List(ctx, workerPods,
		client.InNamespace(dgd.Namespace),
		client.MatchingLabels{
			consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
			consts.KubeLabelDynamoComponent:           dynamo.GetDCDComponentName(newDCD),
			consts.KubeLabelDynamoWorkerHash:          newDCD.Labels[consts.KubeLabelDynamoWorkerHash],
		},
	); err != nil {
		return 0, true, err
	}

	var admitted int32
	for i := range workerPods.Items {
		pod := &workerPods.Items[i]
		if !podReadyForTraffic(pod) {
			continue
		}
		cards := baseModelCardsFromMetadata(metadataByPod[pod.Name], runtimeNamespace)
		if len(cards) == 0 {
			continue
		}
		admittedByAll := true
		for _, frontend := range frontends {
			if !containsAllMembers(frontend.members, cards) {
				admittedByAll = false
				break
			}
		}
		if admittedByAll {
			admitted++
		}
	}
	return min(admitted, locallyAvailable), true, nil
}
