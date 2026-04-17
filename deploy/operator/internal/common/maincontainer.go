// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package common

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// ResolveMainContainer returns a pointer to the workload container in podSpec.
// It prefers the container named commonconsts.MainContainerName and falls back
// to Containers[0] so that pods rendered before the naming convention (or by
// non-operator tooling in tests) still resolve to a sensible target.
// Returns nil only when podSpec has no containers.
//
// Keep this helper on the operator side. Snapshot tooling has its own copy in
// snapshotprotocol because deploy/snapshot is a separate Go module and cannot
// import deploy/operator/internal. The two implementations must match.
func ResolveMainContainer(podSpec *corev1.PodSpec) *corev1.Container {
	if podSpec == nil || len(podSpec.Containers) == 0 {
		return nil
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == commonconsts.MainContainerName {
			return &podSpec.Containers[i]
		}
	}
	return &podSpec.Containers[0]
}
