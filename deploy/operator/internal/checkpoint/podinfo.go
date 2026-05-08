// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package checkpoint

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

func EnsurePodInfoVolume(podSpec *corev1.PodSpec) {
	for _, volume := range podSpec.Volumes {
		if volume.Name == commonconsts.PodInfoVolumeName {
			return
		}
	}

	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: commonconsts.PodInfoVolumeName,
		VolumeSource: corev1.VolumeSource{
			DownwardAPI: &corev1.DownwardAPIVolumeSource{
				Items: []corev1.DownwardAPIVolumeFile{
					{
						Path: "pod_name",
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: commonconsts.PodInfoFieldPodName,
						},
					},
					{
						Path: "pod_uid",
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: commonconsts.PodInfoFieldPodUID,
						},
					},
					{
						Path: "pod_namespace",
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: commonconsts.PodInfoFieldPodNamespace,
						},
					},
					{
						Path: commonconsts.PodInfoFileDynNamespace,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.labels['" + commonconsts.KubeLabelDynamoNamespace + "']",
						},
					},
					{
						Path: commonconsts.PodInfoFileDynNamespaceWorkerSuffix,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.labels['" + commonconsts.KubeLabelDynamoWorkerHash + "']",
						},
					},
					{
						Path: commonconsts.PodInfoFileDynComponent,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.labels['" + commonconsts.KubeLabelDynamoComponentType + "']",
						},
					},
					{
						Path: commonconsts.PodInfoFileDynParentDGDName,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.labels['" + commonconsts.KubeLabelDynamoGraphDeploymentName + "']",
						},
					},
					{
						Path: commonconsts.PodInfoFileDynParentDGDNamespace,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: commonconsts.PodInfoFieldPodNamespace,
						},
					},
				},
			},
		},
	})
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
