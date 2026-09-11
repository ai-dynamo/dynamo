/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"

	corev1 "k8s.io/api/core/v1"
)

func appendVolumeIfMissing(volumes []corev1.Volume, volume corev1.Volume) []corev1.Volume {
	for _, existing := range volumes {
		if existing.Name == volume.Name {
			return volumes
		}
	}
	return append(volumes, volume)
}

func setVolumeByName(volumes []corev1.Volume, volume corev1.Volume) []corev1.Volume {
	return replaceConflicts(volumes, volume, func(existing corev1.Volume) bool {
		return existing.Name == volume.Name
	})
}

func setVolumeMount(volumeMounts []corev1.VolumeMount, volumeMount corev1.VolumeMount) []corev1.VolumeMount {
	return replaceConflicts(volumeMounts, volumeMount, func(existing corev1.VolumeMount) bool {
		return existing.MountPath == volumeMount.MountPath
	})
}

func replaceConflicts[T any](items []T, replacement T, conflicts func(T) bool) []T {
	// Preserve the first conflict's position while removing later conflicts, or append when none exists.
	first := slices.IndexFunc(items, conflicts)
	if first < 0 {
		return append(items, replacement)
	}

	out := append(items[:first], replacement)
	for _, item := range items[first+1:] {
		if !conflicts(item) {
			out = append(out, item)
		}
	}
	return out
}
