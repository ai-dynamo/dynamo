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
	"context"
	"fmt"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

func ApplyRestorePodMetadata(labels map[string]string, annotations map[string]string, checkpointInfo *CheckpointInfo) {
	_ = ApplyRestorePodMetadataWithStorageConfig(
		labels,
		annotations,
		checkpointInfo,
		configv1alpha1.CheckpointStorageConfiguration{},
	)
}

func ApplyRestorePodMetadataWithStorageConfig(
	labels map[string]string,
	annotations map[string]string,
	checkpointInfo *CheckpointInfo,
	storageConfig configv1alpha1.CheckpointStorageConfiguration,
) error {
	enabled := checkpointInfo != nil && checkpointInfo.Enabled && checkpointInfo.Ready
	hash := ""
	artifactVersion := ""
	var (
		storage snapshotprotocol.Storage
		ok      bool
		err     error
	)
	if enabled {
		if labels == nil {
			return fmt.Errorf("checkpoint restore labels map is required when checkpoint restore metadata is enabled")
		}
		if annotations == nil {
			return fmt.Errorf("checkpoint restore annotations map is required when checkpoint restore metadata is enabled")
		}
		hash = checkpointInfo.Hash
		artifactVersion = checkpointInfo.ArtifactVersion
		storage, ok, err = StorageFromConfig(storageConfig)
		if err != nil {
			return err
		}
	}

	snapshotprotocol.ApplyRestoreTargetMetadata(labels, annotations, enabled, hash, artifactVersion)
	if annotations != nil {
		delete(annotations, snapshotprotocol.TargetContainersAnnotation)
		delete(annotations, snapshotprotocol.CheckpointStorageTypeAnnotation)
		delete(annotations, snapshotprotocol.CheckpointStorageBasePathAnnotation)
	}
	if !enabled {
		return nil
	}

	targets := checkpointInfo.RestoreTargetContainers
	if len(targets) == 0 {
		targets = []string{commonconsts.MainContainerName}
	}
	annotations[snapshotprotocol.TargetContainersAnnotation] = snapshotprotocol.FormatTargetContainers(targets)
	if ok {
		snapshotprotocol.ApplyCheckpointStorageMetadata(annotations, storage)
	}
	return nil
}

func RequireMainContainer(podSpec *corev1.PodSpec) (*corev1.Container, error) {
	if podSpec == nil {
		return nil, fmt.Errorf("pod spec is nil")
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == commonconsts.MainContainerName {
			return &podSpec.Containers[i], nil
		}
	}
	return nil, fmt.Errorf("pod spec has no container named %q", commonconsts.MainContainerName)
}

func InjectCheckpointIntoPodSpec(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
	seccompProfile string,
) error {
	return injectCheckpointIntoPodSpec(
		ctx,
		reader,
		namespace,
		podSpec,
		checkpointInfo,
		configv1alpha1.CheckpointStorageConfiguration{},
		seccompProfile,
	)
}

func InjectCheckpointIntoPodSpecWithStorageConfig(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
	storageConfig configv1alpha1.CheckpointStorageConfiguration,
	seccompProfile string,
) error {
	return injectCheckpointIntoPodSpec(
		ctx,
		reader,
		namespace,
		podSpec,
		checkpointInfo,
		storageConfig,
		seccompProfile,
	)
}

func injectCheckpointIntoPodSpec(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
	storageConfig configv1alpha1.CheckpointStorageConfiguration,
	seccompProfile string,
) error {
	// Only mutate the worker pod spec once the checkpoint is Ready. Before
	// the checkpoint exists, the worker must cold-start normally without
	// the snapshot-control volume, DYN_SNAPSHOT_CONTROL_DIR, checkpoint PVC
	// mount, or localhost seccomp profile.
	if checkpointInfo == nil || !checkpointInfo.Enabled || !checkpointInfo.Ready {
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

	if reader == nil {
		return fmt.Errorf("checkpoint client is required")
	}
	targets := info.RestoreTargetContainers
	if len(targets) == 0 {
		targets = []string{commonconsts.MainContainerName}
	}
	annotations := map[string]string{
		snapshotprotocol.TargetContainersAnnotation: snapshotprotocol.FormatTargetContainers(targets),
	}

	storage, err := ResolveStorage(
		ctx,
		reader,
		namespace,
		info.Hash,
		info.ArtifactVersion,
		storageConfig,
	)
	if err != nil {
		return err
	}
	if err := snapshotprotocol.PrepareRestorePodSpec(
		podSpec,
		annotations,
		storage,
		seccompProfile,
		info.Ready,
	); err != nil {
		return err
	}

	EnsurePodInfoVolume(podSpec)
	for _, name := range targets {
		container := findPodSpecContainer(podSpec, name)
		if container == nil {
			return fmt.Errorf("checkpoint restore target %q does not exist in pod spec", name)
		}
		EnsurePodInfoMount(container)
	}
	if info.Ready && info.GPUMemoryService != nil && info.GPUMemoryService.Enabled {
		mainContainer, err := RequireMainContainer(podSpec)
		if err != nil {
			return fmt.Errorf("gpuMemoryService enabled: %w", err)
		}
		EnsureGMSRestoreSidecars(podSpec, mainContainer, storage)
	}

	return nil
}

func findPodSpecContainer(podSpec *corev1.PodSpec, name string) *corev1.Container {
	if podSpec == nil {
		return nil
	}
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == name {
			return &podSpec.Containers[i]
		}
	}
	return nil
}
