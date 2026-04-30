// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package checkpoint

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// EnsurePodInfoVolume installs the canonical DownwardAPI volume carrying
// checkpoint/restore metadata onto podSpec. If a volume with the same name
// already exists, it is overwritten so that the volume source type and item
// list always match what the snapshot tooling expects to read at
// /etc/podinfo/*. The name is a dynamo-internal constant, so replacing any
// user-supplied entry with the canonical definition is the right default.
func EnsurePodInfoVolume(podSpec *corev1.PodSpec) {
	canonical := corev1.Volume{
		Name: commonconsts.PodInfoVolumeName,
		VolumeSource: corev1.VolumeSource{
			DownwardAPI: &corev1.DownwardAPIVolumeSource{
				Items: []corev1.DownwardAPIVolumeFile{
					{Path: "pod_name", FieldRef: &corev1.ObjectFieldSelector{FieldPath: commonconsts.PodInfoFieldPodName}},
					{Path: "pod_uid", FieldRef: &corev1.ObjectFieldSelector{FieldPath: commonconsts.PodInfoFieldPodUID}},
					{Path: "pod_namespace", FieldRef: &corev1.ObjectFieldSelector{FieldPath: commonconsts.PodInfoFieldPodNamespace}},
					{Path: commonconsts.PodInfoFileDynNamespace, FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.labels['" + commonconsts.KubeLabelDynamoNamespace + "']"}},
					{Path: commonconsts.PodInfoFileDynNamespaceWorkerSuffix, FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.labels['" + commonconsts.KubeLabelDynamoWorkerHash + "']"}},
					{Path: commonconsts.PodInfoFileDynComponent, FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.labels['" + commonconsts.KubeLabelDynamoComponentType + "']"}},
					{Path: commonconsts.PodInfoFileDynParentDGDName, FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.labels['" + commonconsts.KubeLabelDynamoGraphDeploymentName + "']"}},
					{Path: commonconsts.PodInfoFileDynParentDGDNamespace, FieldRef: &corev1.ObjectFieldSelector{FieldPath: commonconsts.PodInfoFieldPodNamespace}},
				},
			},
		},
	}

	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].Name == commonconsts.PodInfoVolumeName {
			podSpec.Volumes[i] = canonical
			return
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, canonical)
}

func EnsurePodInfoMount(container *corev1.Container) {
	for _, mount := range container.VolumeMounts {
		if mount.Name == commonconsts.PodInfoVolumeName {
			return
		}
	}

	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      commonconsts.PodInfoVolumeName,
		MountPath: commonconsts.PodInfoMountPath,
		ReadOnly:  true,
	})
}
