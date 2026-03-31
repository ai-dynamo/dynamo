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
	snapshotworkload "github.com/ai-dynamo/dynamo/deploy/snapshot/workload"
	corev1 "k8s.io/api/core/v1"
)

func ApplyRestorePodMetadata(labels map[string]string, annotations map[string]string, checkpointInfo *CheckpointInfo) {
	enabled := checkpointInfo != nil && checkpointInfo.Enabled && checkpointInfo.Ready
	hash := ""
	artifactVersion := ""
	if enabled {
		hash = checkpointInfo.Hash
		artifactVersion = checkpointInfo.ArtifactVersion
	}
	snapshotworkload.ApplyRestoreTargetMetadata(labels, annotations, enabled, hash, artifactVersion)
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
	if checkpointConfig == nil {
		return fmt.Errorf("checkpoint config is required")
	}
	if checkpointConfig.Storage.PVC.PVCName == "" {
		return fmt.Errorf("checkpoint pvc name is required")
	}

	storageConfig := snapshotworkload.Storage{
		Type:     snapshotworkload.StorageTypePVC,
		PVCName:  checkpointConfig.Storage.PVC.PVCName,
		BasePath: checkpointConfig.Storage.PVC.BasePath,
	}
	resolvedStorage, err := snapshotworkload.ResolveRestoreStorage(
		info.Hash,
		info.ArtifactVersion,
		info.Location,
		storageConfig,
	)
	if err != nil {
		return err
	}
	info.StorageType = nvidiacomv1alpha1.DynamoCheckpointStorageType(resolvedStorage.Type)
	info.Location = resolvedStorage.Location
	snapshotworkload.PrepareRestorePodSpec(
		podSpec,
		mainContainer,
		resolvedStorage,
		commonconsts.SeccompProfilePath,
		info.Ready,
	)

	hasPodInfoVolume := false
	for _, volume := range podSpec.Volumes {
		if volume.Name == commonconsts.PodInfoVolumeName {
			hasPodInfoVolume = true
			break
		}
	}
	if !hasPodInfoVolume {
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

	for _, mount := range mainContainer.VolumeMounts {
		if mount.Name == commonconsts.PodInfoVolumeName {
			return nil
		}
	}
	mainContainer.VolumeMounts = append(mainContainer.VolumeMounts, corev1.VolumeMount{
		Name:      commonconsts.PodInfoVolumeName,
		MountPath: commonconsts.PodInfoMountPath,
		ReadOnly:  true,
	})
	return nil
}
