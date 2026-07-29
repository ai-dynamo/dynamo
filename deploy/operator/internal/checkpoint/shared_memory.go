/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func sharedMemorySize(src *nvidiacomv1alpha1.SharedMemorySpec) *resource.Quantity {
	if src == nil || (!src.Disabled && src.Size.IsZero()) {
		return nil
	}
	if src.Disabled {
		zero := resource.MustParse("0")
		return &zero
	}
	dst := src.Size.DeepCopy()
	return &dst
}

func buildSharedMemoryVolumeAndMount(sizeSpec *resource.Quantity) (*corev1.Volume, *corev1.VolumeMount) {
	size := resource.MustParse(commonconsts.DefaultSharedMemorySize)
	if sizeSpec != nil {
		if sizeSpec.Sign() == 0 {
			return nil, nil
		}
		size = sizeSpec.DeepCopy()
	}

	volume := &corev1.Volume{
		Name: commonconsts.KubeValueNameSharedMemory,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{
				Medium:    corev1.StorageMediumMemory,
				SizeLimit: &size,
			},
		},
	}
	volumeMount := &corev1.VolumeMount{
		Name:      commonconsts.KubeValueNameSharedMemory,
		MountPath: commonconsts.DefaultSharedMemoryMountPath,
	}

	return volume, volumeMount
}

// ApplySharedMemoryVolumeAndMount applies the checkpoint Job's /dev/shm
// compatibility setting to the target container.
func ApplySharedMemoryVolumeAndMount(
	podSpec *corev1.PodSpec,
	mainContainer *corev1.Container,
	sharedMemory *nvidiacomv1alpha1.SharedMemorySpec,
) {
	volume, volumeMount := buildSharedMemoryVolumeAndMount(sharedMemorySize(sharedMemory))
	if volume == nil || volumeMount == nil {
		return
	}

	volumes := make([]corev1.Volume, 0, len(podSpec.Volumes)+1)
	for _, existingVolume := range podSpec.Volumes {
		if existingVolume.Name != volume.Name {
			volumes = append(volumes, existingVolume)
		}
	}
	podSpec.Volumes = append(volumes, *volume)

	mounts := make([]corev1.VolumeMount, 0, len(mainContainer.VolumeMounts)+1)
	for _, existingMount := range mainContainer.VolumeMounts {
		if existingMount.Name != volumeMount.Name && existingMount.MountPath != volumeMount.MountPath {
			mounts = append(mounts, existingMount)
		}
	}
	mainContainer.VolumeMounts = append(mounts, *volumeMount)
}
