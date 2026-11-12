/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package lws observes LeaderWorkerSet Pods through the manager informer cache.
//
// Watch mappers may use the official set-name label only as a routing hint to a
// currently DCD-owned LeaderWorkerSet. That hint enqueues reconciliation and must
// never count or attribute faults. Actual failure evidence always requires the
// strict controller-owner and revision checks in Observe.
package lws

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

const (
	// SetNameIndex indexes Pods by the official LeaderWorkerSet name label.
	SetNameIndex = ".metadata.lwsSetName"
)

// SetNameIndexValues returns the SetNameIndex value of a Pod labeled for an LWS.
func SetNameIndexValues(obj client.Object) []string {
	pod, ok := obj.(*corev1.Pod)
	if !ok || pod == nil {
		return nil
	}
	if name := pod.Labels[leaderworkersetv1.SetNameLabelKey]; name != "" {
		return []string{name}
	}
	return nil
}

// ListPodsBySetName lists Pods in namespace labeled for setName.
// reader must be the manager's cache with SetNameIndex registered.
func ListPodsBySetName(
	ctx context.Context,
	reader client.Reader,
	namespace, setName string,
) ([]corev1.Pod, error) {
	podList := &corev1.PodList{}
	if err := reader.List(
		ctx,
		podList,
		client.InNamespace(namespace),
		client.MatchingFields{SetNameIndex: setName},
	); err != nil {
		return nil, fmt.Errorf("list Pods for LeaderWorkerSet %s/%s: %w", namespace, setName, err)
	}
	return podList.Items, nil
}
