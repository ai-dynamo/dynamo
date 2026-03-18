/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package checkpoint

import (
	"fmt"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func ApplyCheckpointSourcePodMetadata(
	labels map[string]string,
	annotations map[string]string,
	hash string,
	location string,
	storageType nvidiacomv1alpha1.DynamoCheckpointStorageType,
) {
	delete(labels, commonconsts.KubeLabelIsRestoreTarget)
	delete(labels, commonconsts.KubeLabelCheckpointHash)
	delete(annotations, commonconsts.KubeAnnotationCheckpointLocation)
	delete(annotations, commonconsts.KubeAnnotationCheckpointStorageType)

	labels[commonconsts.KubeLabelIsCheckpointSource] = commonconsts.KubeLabelValueTrue
	if hash != "" {
		labels[commonconsts.KubeLabelCheckpointHash] = hash
	}
	if location != "" {
		annotations[commonconsts.KubeAnnotationCheckpointLocation] = location
	}
	if storageType != "" {
		annotations[commonconsts.KubeAnnotationCheckpointStorageType] = string(storageType)
	}
}

func ApplyRestorePodMetadata(labels map[string]string, annotations map[string]string, checkpointInfo *CheckpointInfo) {
	delete(labels, commonconsts.KubeLabelIsRestoreTarget)
	delete(labels, commonconsts.KubeLabelCheckpointHash)
	delete(annotations, commonconsts.KubeAnnotationCheckpointLocation)
	delete(annotations, commonconsts.KubeAnnotationCheckpointStorageType)

	if checkpointInfo == nil || !checkpointInfo.Enabled || !checkpointInfo.Ready {
		return
	}

	labels[commonconsts.KubeLabelIsRestoreTarget] = commonconsts.KubeLabelValueTrue
	if checkpointInfo.Hash != "" {
		labels[commonconsts.KubeLabelCheckpointHash] = checkpointInfo.Hash
	}
	if checkpointInfo.Location != "" {
		annotations[commonconsts.KubeAnnotationCheckpointLocation] = checkpointInfo.Location
	}
	if checkpointInfo.StorageType != "" {
		annotations[commonconsts.KubeAnnotationCheckpointStorageType] = string(checkpointInfo.StorageType)
	}
}

func InjectCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) {
	for _, volume := range podSpec.Volumes {
		if volume.Name == commonconsts.CheckpointVolumeName {
			return
		}
	}

	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: commonconsts.CheckpointVolumeName,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
				ReadOnly:  false,
			},
		},
	})
}

func InjectCheckpointVolumeMount(container *corev1.Container, basePath string) {
	for _, mount := range container.VolumeMounts {
		if mount.Name == commonconsts.CheckpointVolumeName {
			return
		}
	}

	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      commonconsts.CheckpointVolumeName,
		MountPath: basePath,
		ReadOnly:  false,
	})
}

func InjectPodInfoVolume(podSpec *corev1.PodSpec) {
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

func InjectPodInfoVolumeMount(container *corev1.Container) {
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

func InjectCheckpointIntoPodSpec(
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
	checkpointConfig *configv1alpha1.CheckpointConfiguration,
) error {
	if checkpointInfo == nil || !checkpointInfo.Enabled {
		return nil
	}

	info := checkpointInfo
	if info.Hash == "" {
		if info.Identity == nil {
			return fmt.Errorf("checkpoint enabled but identity is nil and hash is not set")
		}

		hash, err := ComputeIdentityHash(*info.Identity)
		if err != nil {
			return fmt.Errorf("failed to compute identity hash: %w", err)
		}
		info.Hash = hash
	}

	var mainContainer *corev1.Container
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == commonconsts.MainContainerName {
			mainContainer = &podSpec.Containers[i]
			break
		}
	}
	if mainContainer == nil && len(podSpec.Containers) > 0 {
		mainContainer = &podSpec.Containers[0]
	}
	if mainContainer == nil {
		return fmt.Errorf("no container found to inject checkpoint config")
	}

	if info.Ready {
		mainContainer.Command = []string{"sleep", "infinity"}
		mainContainer.Args = nil
	}

	if podSpec.SecurityContext == nil {
		podSpec.SecurityContext = &corev1.PodSecurityContext{}
	}
	podSpec.SecurityContext.SeccompProfile = &corev1.SeccompProfile{
		Type:             corev1.SeccompProfileTypeLocalhost,
		LocalhostProfile: ptr.To(commonconsts.SeccompProfilePath),
	}

	storageType := configv1alpha1.CheckpointStorageTypePVC
	var storageConfig *configv1alpha1.CheckpointStorageConfiguration
	if checkpointConfig != nil {
		storageConfig = &checkpointConfig.Storage
		if storageConfig.Type != "" {
			storageType = storageConfig.Type
		}
	}

	switch storageType {
	case configv1alpha1.CheckpointStorageTypeS3:
		info.StorageType = nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType)
		if storageConfig == nil || storageConfig.S3.URI == "" {
			return fmt.Errorf("S3 storage type selected but no S3 URI configured (set checkpoint.storage.s3.uri)")
		}
		info.Location = fmt.Sprintf("%s/%s.tar", storageConfig.S3.URI, info.Hash)
	case configv1alpha1.CheckpointStorageTypeOCI:
		info.StorageType = nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType)
		if storageConfig == nil || storageConfig.OCI.URI == "" {
			return fmt.Errorf("OCI storage type selected but no OCI URI configured (set checkpoint.storage.oci.uri)")
		}
		info.Location = fmt.Sprintf("%s:%s", storageConfig.OCI.URI, info.Hash)
	default:
		info.StorageType = nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType)
		basePath := ""
		if storageConfig != nil && storageConfig.PVC.BasePath != "" {
			basePath = storageConfig.PVC.BasePath
		}
		if storageConfig == nil || storageConfig.PVC.PVCName == "" {
			return fmt.Errorf("PVC storage type selected but no PVC name configured (set checkpoint.storage.pvc.pvcName)")
		}
		if basePath == "" {
			return fmt.Errorf("PVC storage type selected but no PVC base path configured (set checkpoint.storage.pvc.basePath)")
		}
		info.Location = fmt.Sprintf("%s/%s", basePath, info.Hash)
		InjectCheckpointVolume(podSpec, storageConfig.PVC.PVCName)
		InjectCheckpointVolumeMount(mainContainer, basePath)
	}

	InjectPodInfoVolume(podSpec)
	InjectPodInfoVolumeMount(mainContainer)
	return nil
}
