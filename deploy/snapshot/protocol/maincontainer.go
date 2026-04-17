// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import corev1 "k8s.io/api/core/v1"

// MainContainerName is the conventional name of the workload container that
// checkpoint/restore operates on. The operator always sets this name on the
// pods it emits. Keeping the constant in the snapshot protocol package makes
// it the shared contract between the operator (which builds pods) and the
// snapshot agent / snapshotctl (which checkpoint them).
const MainContainerName = "main"

// ResolveMainContainer returns a pointer to the workload container in podSpec.
// It prefers the container named MainContainerName and falls back to
// Containers[0] so that pods rendered before the naming convention (or by
// non-operator tooling in tests) still resolve to a sensible target.
// Returns nil only when podSpec has no containers.
func ResolveMainContainer(podSpec *corev1.PodSpec) *corev1.Container {
	if podSpec == nil || len(podSpec.Containers) == 0 {
		return nil
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == MainContainerName {
			return &podSpec.Containers[i]
		}
	}
	return &podSpec.Containers[0]
}

// ResolveMainContainerName returns the name of the workload container in pod.
// Used by the snapshot agent, which reads pods from an informer cache and
// needs to send signals to the matching ContainerStatus entry by name.
// Returns "" only when pod has no containers.
func ResolveMainContainerName(pod *corev1.Pod) string {
	container := ResolveMainContainer(&pod.Spec)
	if container == nil {
		return ""
	}
	return container.Name
}
