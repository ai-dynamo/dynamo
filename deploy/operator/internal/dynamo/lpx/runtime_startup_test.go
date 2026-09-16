/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func TestRuntimeConfigStorage(t *testing.T) {
	for _, security := range []*corev1.SecurityContext{nil, {ReadOnlyRootFilesystem: ptr.To(false)}, {ReadOnlyRootFilesystem: ptr.To(true)}} {
		t.Log("Allow container-local expansion unless the retained root filesystem is read-only")
		pod := corev1.PodSpec{}
		container := corev1.Container{SecurityContext: security, VolumeMounts: []corev1.VolumeMount{{Name: lpuConfigVolumeName, MountPath: lpuConfigMountPath}}}
		before := container.DeepCopy()
		err := validateRuntimeConfigStorage(&pod, &container, "lpu_servers")
		if security != nil && ptr.Deref(security.ReadOnlyRootFilesystem, false) {
			require.ErrorContains(t, err, "writable storage")
		} else {
			require.NoError(t, err)
		}
		require.Empty(t, pod.Volumes)
		require.Equal(t, *before, container)
	}

	for _, source := range []corev1.VolumeSource{
		{EmptyDir: &corev1.EmptyDirVolumeSource{}},
		{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "shared"}},
		{Secret: &corev1.SecretVolumeSource{SecretName: "readonly"}},
	} {
		t.Log("Reuse only a writable Pod-local workspace for expanded runtime configuration")
		pod := corev1.PodSpec{Volumes: []corev1.Volume{{Name: "scratch", VolumeSource: source}}}
		container := corev1.Container{SecurityContext: &corev1.SecurityContext{ReadOnlyRootFilesystem: ptr.To(true)}, VolumeMounts: []corev1.VolumeMount{
			{Name: "scratch", MountPath: "/tmp"},
			{Name: lpuConfigVolumeName, MountPath: lpuConfigMountPath},
		}}
		authoredPod, authoredContainer := pod.DeepCopy(), container.DeepCopy()
		err := validateRuntimeConfigStorage(&pod, &container, "lpu_servers")
		if source.EmptyDir == nil {
			require.ErrorContains(t, err, "Pod-local emptyDir")
		} else {
			require.NoError(t, err)
			require.Len(t, pod.Volumes, 1)
		}
		require.Equal(t, *authoredPod, pod)
		require.Equal(t, *authoredContainer, container)
		container.VolumeMounts[0].ReadOnly = true
		require.ErrorContains(t, validateRuntimeConfigStorage(&pod, &container, "lpu_servers"), "writable storage")
	}
}

func TestNativeNovaConfigStorage(t *testing.T) {
	t.Log("Mount a file at the obsolete expansion path alongside the original configuration")
	pod := renderTestPodSpec()
	pod.Containers[0].VolumeMounts = append(pod.Containers[0].VolumeMounts,
		corev1.VolumeMount{Name: "unrelated", MountPath: "/tmp/datacenter.toml", ReadOnly: true},
	)
	pod.Volumes = append(pod.Volumes, corev1.Volume{Name: "unrelated", VolumeSource: corev1.VolumeSource{
		ConfigMap: &corev1.ConfigMapVolumeSource{LocalObjectReference: corev1.LocalObjectReference{Name: "unrelated"}},
	}})

	t.Log("Launch Nova against /configs without trying to overwrite the unrelated mount")
	require.NoError(t, configureNodeLocalConductorRuntime(&pod, BuildFamilyXT, "agent", "ssh-secret"))
	require.Equal(t, []string{"/bin/nova"}, pod.Containers[0].Command)
	requireFlagValue(t, pod.Containers[0].Args, "--datacenter-config-filepath", "/configs/datacenter.toml")

	t.Log("Continue rejecting an overlapping Cyborg expansion destination")
	pod.Containers[0].VolumeMounts = append(pod.Containers[0].VolumeMounts,
		corev1.VolumeMount{Name: "unrelated", MountPath: "/tmp/lpu_servers", ReadOnly: true},
	)
	require.ErrorContains(t, validateRuntimeConfigStorage(&pod, &pod.Containers[0], "lpu_servers"), "cannot overwrite")
}
