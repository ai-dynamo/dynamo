/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func buildSharedMemoryVolumeAndMount(spec *v1alpha1.SharedMemorySpec) (*corev1.Volume, *corev1.VolumeMount) {
	size := resource.MustParse(commonconsts.DefaultSharedMemorySize)
	if spec != nil {
		if spec.Disabled {
			return nil, nil
		}
		if !spec.Size.IsZero() {
			size = spec.Size
		}
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

func ApplySharedMemoryVolumeAndMount(podSpec *corev1.PodSpec, mainContainer *corev1.Container, spec *v1alpha1.SharedMemorySpec) {
	volume, volumeMount := buildSharedMemoryVolumeAndMount(spec)
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
