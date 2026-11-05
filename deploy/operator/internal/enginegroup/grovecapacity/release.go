/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package grovecapacity

import (
	"fmt"
	"reflect"
	"slices"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	corev1 "k8s.io/api/core/v1"
)

// releaseIntent pins the expected physical outcome independently of victim-selection mechanics.
// The selected release implementation persists it before any deletion or scale-in effect.
type releaseIntent struct {
	Fence    enginegroup.ReleaseFence  `json:"fence"`
	Retained []enginegroup.CapacityRef `json:"retained"`
	Replicas int32                     `json:"replicas"`
}

func (r releaseIntent) observe(pods []corev1.Pod, replicas int32) (bool, error) {
	// Losing any retained UID is an unplanned loss, even if the resulting pod count is correct.
	for _, ref := range r.Retained {
		if !slices.ContainsFunc(pods, func(pod corev1.Pod) bool {
			return pod.Name == ref.Name && enginegroup.PodUID(pod.UID) == ref.UID && pod.DeletionTimestamp.IsZero()
		}) {
			return false, fmt.Errorf("%w: Grove released or replaced retained Pod %s (UID %s)", enginegroup.ErrRecoveryRequired, ref.Name, ref.UID)
		}
	}

	// A replacement or unexpected occupant cannot satisfy the original release intent.
	victimPresent := false
	for _, pod := range pods {
		if slices.ContainsFunc(r.Retained, func(ref enginegroup.CapacityRef) bool {
			return ref.Name == pod.Name && ref.UID == enginegroup.PodUID(pod.UID)
		}) {
			continue
		}
		if pod.Name != r.Fence.CapacityRefs[0].Name || enginegroup.PodUID(pod.UID) != r.Fence.CapacityRefs[0].UID {
			return false, fmt.Errorf("%w: an unplanned UID occupies capacity during release", enginegroup.ErrRecoveryRequired)
		}
		victimPresent = true
	}
	return !victimPresent && replicas == r.Replicas, nil
}

func observeReleaseFences(state journalState, pods []corev1.Pod, replicas int32) ([]enginegroup.ReleaseFence, error) {
	// The journal preserves previous receipts when a later accepted target replaces the current release.
	fences := slices.Clone(state.ReleaseFences)
	if state.Release != nil {
		complete, err := state.Release.observe(pods, replicas)
		if err != nil {
			return nil, err
		}
		if complete && !hasReleaseFence(fences, state.Release.Fence) {
			fences = append(fences, state.Release.Fence)
		}
	}
	return fences, nil
}

func hasReleaseFence(fences []enginegroup.ReleaseFence, expected enginegroup.ReleaseFence) bool {
	return slices.ContainsFunc(fences, func(fence enginegroup.ReleaseFence) bool { return reflect.DeepEqual(fence, expected) })
}

func releaseFencesAfterTarget(fences []enginegroup.ReleaseFence, target enginegroup.CapacityTarget) []enginegroup.ReleaseFence {
	// Reopening a slot invalidates only that slot's old occupant authorization.
	return slices.DeleteFunc(slices.Clone(fences), func(fence enginegroup.ReleaseFence) bool {
		return slices.ContainsFunc(target.Replicas, func(replica enginegroup.CapacityReplicaTarget) bool { return replica.ReplicaID == fence.ReplicaID })
	})
}
