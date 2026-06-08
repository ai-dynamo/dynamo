/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package common

import corev1 "k8s.io/api/core/v1"

// FindContainerByName returns the first container with the given name.
func FindContainerByName(containers []corev1.Container, name string) *corev1.Container {
	for i := range containers {
		if containers[i].Name == name {
			return &containers[i]
		}
	}
	return nil
}

// HasPodResourceClaim reports whether the pod spec references a claim by name.
func HasPodResourceClaim(podSpec *corev1.PodSpec, name string) bool {
	for i := range podSpec.ResourceClaims {
		if podSpec.ResourceClaims[i].Name == name {
			return true
		}
	}
	return false
}

// HasContainerResourceClaim reports whether the container references a claim by name.
func HasContainerResourceClaim(container *corev1.Container, name string) bool {
	for i := range container.Resources.Claims {
		if container.Resources.Claims[i].Name == name {
			return true
		}
	}
	return false
}

// HasVolume reports whether a volume exists by name.
func HasVolume(volumes []corev1.Volume, name string) bool {
	for i := range volumes {
		if volumes[i].Name == name {
			return true
		}
	}
	return false
}

// HasVolumeMount reports whether a volume mount exists by name and path.
func HasVolumeMount(mounts []corev1.VolumeMount, name, mountPath string) bool {
	for i := range mounts {
		if mounts[i].Name == name && mounts[i].MountPath == mountPath {
			return true
		}
	}
	return false
}

// HasEnvValue reports whether an environment variable exists with the given value.
func HasEnvValue(env []corev1.EnvVar, name, value string) bool {
	for i := range env {
		if env[i].Name == name && env[i].Value == value {
			return true
		}
	}
	return false
}
