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
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
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
		delete(annotations, commonconsts.CheckpointRestoreCandidateAnnotation)
		delete(annotations, commonconsts.CheckpointNameAnnotation)
		delete(annotations, commonconsts.CheckpointStartupPolicyAnnotation)
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

func ApplyRestoreCandidateMetadata(labels map[string]string, annotations map[string]string, checkpointInfo *CheckpointInfo) error {
	if labels == nil {
		return fmt.Errorf("checkpoint restore candidate labels map is required")
	}
	if annotations == nil {
		return fmt.Errorf("checkpoint restore candidate annotations map is required")
	}
	delete(labels, snapshotprotocol.CheckpointIDLabel)
	delete(labels, snapshotprotocol.RestoreTargetLabel)
	delete(labels, snapshotprotocol.CheckpointSourceLabel)
	delete(annotations, snapshotprotocol.CheckpointArtifactVersionAnnotation)
	delete(annotations, snapshotprotocol.CheckpointStatusAnnotation)
	delete(annotations, snapshotprotocol.CheckpointStorageTypeAnnotation)
	delete(annotations, snapshotprotocol.CheckpointStorageBasePathAnnotation)
	delete(annotations, commonconsts.CheckpointRestoreCandidateAnnotation)
	delete(annotations, commonconsts.CheckpointNameAnnotation)
	delete(annotations, commonconsts.CheckpointStartupPolicyAnnotation)
	delete(annotations, snapshotprotocol.TargetContainersAnnotation)
	if checkpointInfo == nil || !checkpointInfo.Enabled || !checkpointInfo.Exists || checkpointInfo.CheckpointName == "" {
		return nil
	}

	targets := checkpointInfo.RestoreTargetContainers
	if len(targets) == 0 {
		targets = []string{commonconsts.MainContainerName}
	}
	annotations[commonconsts.CheckpointRestoreCandidateAnnotation] = commonconsts.KubeLabelValueTrue
	annotations[commonconsts.CheckpointNameAnnotation] = checkpointInfo.CheckpointName
	startupPolicy := checkpointInfo.StartupPolicy
	if startupPolicy == "" {
		startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
	}
	annotations[commonconsts.CheckpointStartupPolicyAnnotation] = string(startupPolicy)
	annotations[snapshotprotocol.TargetContainersAnnotation] = snapshotprotocol.FormatTargetContainers(targets)
	return nil
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

//nolint:gocyclo
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
	if reader == nil {
		return fmt.Errorf("checkpoint client is required")
	}

	info := *checkpointInfo
	if info.Hash == "" {
		if info.CheckpointName == "" {
			if info.Identity == nil {
				return fmt.Errorf("checkpoint enabled but identity is nil and hash is not set")
			}

			hash, err := ComputeIdentityHash(*info.Identity)
			if err != nil {
				return fmt.Errorf("failed to compute identity hash: %w", err)
			}
			info.Hash = hash
		} else {
			ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{}
			if err := reader.Get(ctx, ctrlclient.ObjectKey{Namespace: namespace, Name: info.CheckpointName}, ckpt); err != nil {
				return fmt.Errorf("failed to get checkpoint %s/%s: %w", namespace, info.CheckpointName, err)
			}
			hash, err := CheckpointID(ckpt)
			if err != nil {
				return err
			}
			info.Hash = hash
			if info.ArtifactVersion == "" {
				info.ArtifactVersion = checkpointArtifactVersion(ckpt)
			}
			if info.GPUMemoryService == nil {
				info.GPUMemoryService = ckpt.Spec.GPUMemoryService
			}
		}
	}

	if info.ArtifactVersion == "" {
		info.ArtifactVersion = snapshotprotocol.DefaultCheckpointArtifactVersion
	}

	if info.Hash == "" {
		return fmt.Errorf("checkpoint enabled but hash is not set")
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
	targetContainers := make([]*corev1.Container, 0, len(targets))
	for _, name := range targets {
		var container *corev1.Container
		for i := range podSpec.Containers {
			if podSpec.Containers[i].Name == name {
				container = &podSpec.Containers[i]
				break
			}
		}
		if container == nil {
			return fmt.Errorf("checkpoint restore target %q does not exist in pod spec", name)
		}
		EnsurePodInfoMount(container)
		targetContainers = append(targetContainers, container)
	}
	if info.Ready && info.GPUMemoryService != nil && info.GPUMemoryService.Enabled {
		switch info.GPUMemoryService.Mode {
		case "", nvidiacomv1alpha1.GMSModeIntraPod:
			EnsureIntraPodGPUMemoryService(podSpec, targetContainers, info.GPUMemoryService.ExtraClientContainers)
		case nvidiacomv1alpha1.GMSModeInterPod:
			return fmt.Errorf("gpuMemoryService checkpoint restore for mode %q is not implemented", info.GPUMemoryService.Mode)
		default:
			return fmt.Errorf("gpuMemoryService checkpoint restore has unsupported mode %q", info.GPUMemoryService.Mode)
		}
	}

	return nil
}

// EnsureIntraPodGPUMemoryService wires the in-pod GMS server sidecar and
// socket clients for checkpoint create/restore pod specs.
func EnsureIntraPodGPUMemoryService(
	podSpec *corev1.PodSpec,
	targetContainers []*corev1.Container,
	extraClientContainerNames []string,
) {
	if len(targetContainers) == 0 {
		return
	}
	gms.EnsureServerSidecar(podSpec, targetContainers[0])
	for _, container := range targetContainers {
		gms.EnsureClient(podSpec, container)
	}
	for _, name := range extraClientContainerNames {
		var container *corev1.Container
		for i := range podSpec.Containers {
			if podSpec.Containers[i].Name == name {
				container = &podSpec.Containers[i]
				break
			}
		}
		if container == nil {
			continue
		}
		gms.EnsureClient(podSpec, container)
	}
}
