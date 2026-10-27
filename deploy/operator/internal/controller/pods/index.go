/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package pods observes the Pods of Dynamo workloads through the manager's
// informer cache. It owns the Pod indexes together with the queries that use them.
package pods

import (
	"context"
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	poddiagnostics "github.com/ai-dynamo/dynamo/deploy/operator/internal/pods"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// ComponentIndex indexes Pods by the graph and component they are labeled for.
	ComponentIndex = ".metadata.dgdComponent"

	// SelectorIndex indexes Pods by the Dynamo selector label that a DCD's
	// Deployment selects on.
	SelectorIndex = ".metadata.dynamoSelector"
)

// ComponentIndexValues returns the ComponentIndex value of a Pod labeled for a
// graph component, and none for any other object.
func ComponentIndexValues(obj client.Object) []string {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return nil
	}
	graph := pod.Labels[commonconsts.KubeLabelDynamoGraphDeploymentName]
	component := pod.Labels[commonconsts.KubeLabelDynamoComponent]
	if graph == "" || component == "" {
		return nil
	}
	return []string{componentIndexValue(graph, component)}
}

func componentIndexValue(graph, component string) string {
	// Label values cannot contain '/', so this encoding is collision-free.
	return graph + "/" + component
}

// SelectorIndexValues returns the SelectorIndex value of a Pod carrying the
// Dynamo selector label, and none for any other object.
func SelectorIndexValues(obj client.Object) []string {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return nil
	}
	if selector := pod.Labels[commonconsts.KubeLabelDynamoSelector]; selector != "" {
		return []string{selector}
	}
	return nil
}

// ListComponentPods lists the Pods in namespace labeled for component of graph.
// reader must be the manager's cache with ComponentIndex registered.
func ListComponentPods(
	ctx context.Context,
	reader client.Reader,
	namespace, graph, component string,
) ([]corev1.Pod, error) {
	podList := &corev1.PodList{}
	if err := reader.List(
		ctx,
		podList,
		client.InNamespace(namespace),
		client.MatchingFields{ComponentIndex: componentIndexValue(graph, component)},
	); err != nil {
		return nil, fmt.Errorf("failed to list pods for DGD %s component %s: %w", graph, component, err)
	}
	return podList.Items, nil
}

// ComponentCrashLoops returns, by component name, the crash loop found among
// the Pods labeled for each of components of graph.
func ComponentCrashLoops(
	ctx context.Context,
	reader client.Reader,
	namespace, graph string,
	components []string,
) (map[string]poddiagnostics.CrashLoop, error) {
	crashLoops := map[string]poddiagnostics.CrashLoop{}
	for _, component := range components {
		componentPods, err := ListComponentPods(ctx, reader, namespace, graph, component)
		if err != nil {
			return nil, err
		}
		if crashLoop, found := poddiagnostics.FindCrashLoop(componentPods); found {
			crashLoops[component] = crashLoop
		}
	}
	return crashLoops, nil
}

// FindDeploymentCrashLoop reports a crash-looping container among the Pods the
// Deployment selects, including Pods of older revisions during a rollout. reader
// must be the manager's cache with SelectorIndex registered, and the Deployment
// must select on the Dynamo selector label.
func FindDeploymentCrashLoop(
	ctx context.Context,
	reader client.Reader,
	deployment *appsv1.Deployment,
) (poddiagnostics.CrashLoop, bool, error) {
	selector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		return poddiagnostics.CrashLoop{}, false, fmt.Errorf("parse Deployment %s selector: %w", deployment.Name, err)
	}
	key := deployment.Spec.Selector.MatchLabels[commonconsts.KubeLabelDynamoSelector]
	if key == "" {
		return poddiagnostics.CrashLoop{}, false, fmt.Errorf(
			"deployment %s does not select Pods by label %s", deployment.Name, commonconsts.KubeLabelDynamoSelector,
		)
	}

	// The index narrows the candidates; the Deployment's own selector decides membership.
	podList := &corev1.PodList{}
	if err := reader.List(
		ctx,
		podList,
		client.InNamespace(deployment.Namespace),
		client.MatchingFields{SelectorIndex: key},
	); err != nil {
		return poddiagnostics.CrashLoop{}, false, fmt.Errorf("list Pods of Deployment %s: %w", deployment.Name, err)
	}
	selected := make([]corev1.Pod, 0, len(podList.Items))
	for i := range podList.Items {
		if selector.Matches(labels.Set(podList.Items[i].Labels)) {
			selected = append(selected, podList.Items[i])
		}
	}

	crashLoop, found := poddiagnostics.FindCrashLoop(selected)
	return crashLoop, found, nil
}
