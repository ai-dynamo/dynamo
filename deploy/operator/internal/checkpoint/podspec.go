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
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpointjob"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
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

// ResolvedPodSpecRestore contains the Kubernetes-backed checkpoint inputs
// needed to mutate a rendered PodSpec. Its fields stay private so callers can
// pass the resolved value between observation and rendering without depending
// on checkpoint storage internals.
type ResolvedPodSpecRestore struct {
	info    CheckpointInfo
	storage snapshotprotocol.Storage
}

// ResolvePodSpecRestore performs the reads needed before checkpoint restore
// configuration can be applied to a PodSpec. A nil result means that no
// restore mutation is required for the supplied checkpoint.
func ResolvePodSpecRestore(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	checkpointInfo *CheckpointInfo,
	storageConfig configv1alpha1.CheckpointStorageConfiguration,
) (*ResolvedPodSpecRestore, error) {
	if checkpointInfo == nil || !checkpointInfo.Enabled || !checkpointInfo.Ready {
		return nil, nil
	}
	if reader == nil {
		return nil, fmt.Errorf("checkpoint client is required")
	}

	info := *checkpointInfo
	if info.Hash == "" && info.CheckpointName != "" {
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{}
		if err := reader.Get(ctx, ctrlclient.ObjectKey{Namespace: namespace, Name: info.CheckpointName}, ckpt); err != nil {
			return nil, fmt.Errorf("failed to get checkpoint %s/%s: %w", namespace, info.CheckpointName, err)
		}
		hash, err := CheckpointID(ckpt)
		if err != nil {
			return nil, err
		}
		info.Hash = hash
		if info.ArtifactVersion == "" {
			info.ArtifactVersion = checkpointArtifactVersion(ckpt)
		}
		if info.GPUMemoryService == nil {
			info.GPUMemoryService = ckpt.Spec.GPUMemoryService
		}
	}

	if info.Hash == "" {
		return nil, fmt.Errorf("checkpoint is ready but hash is not set")
	}

	if info.ArtifactVersion == "" {
		info.ArtifactVersion = snapshotprotocol.DefaultCheckpointArtifactVersion
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
		return nil, err
	}

	return &ResolvedPodSpecRestore{
		info:    info,
		storage: storage,
	}, nil
}

// InjectResolvedCheckpointIntoPodSpec applies a previously resolved checkpoint
// restore to a rendered PodSpec without performing Kubernetes reads.
func InjectResolvedCheckpointIntoPodSpec(
	podSpec *corev1.PodSpec,
	restore *ResolvedPodSpecRestore,
	seccompProfile string,
) error {
	if restore == nil {
		return nil
	}

	targets := restore.info.RestoreTargetContainers
	if len(targets) == 0 {
		targets = []string{commonconsts.MainContainerName}
	}
	annotations := map[string]string{
		snapshotprotocol.TargetContainersAnnotation: snapshotprotocol.FormatTargetContainers(targets),
	}
	if err := snapshotprotocol.PrepareRestorePodSpec(
		podSpec,
		annotations,
		restore.storage,
		seccompProfile,
		restore.info.Ready,
	); err != nil {
		return err
	}

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
		targetContainers = append(targetContainers, container)
	}
	if restore.info.Ready && restore.info.GPUMemoryService != nil && restore.info.GPUMemoryService.Enabled {
		switch restore.info.GPUMemoryService.Mode {
		case "", nvidiacomv1alpha1.GMSModeIntraPod:
			EnsureIntraPodGPUMemoryService(podSpec, targetContainers, restore.info.GPUMemoryService.ExtraClientContainers, true)
		case nvidiacomv1alpha1.GMSModeInterPod:
			return fmt.Errorf("gpuMemoryService checkpoint restore for mode %q is not implemented", restore.info.GPUMemoryService.Mode)
		default:
			return fmt.Errorf("gpuMemoryService checkpoint restore has unsupported mode %q", restore.info.GPUMemoryService.Mode)
		}
	}

	return nil
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
	restore, err := ResolvePodSpecRestore(ctx, reader, namespace, checkpointInfo, storageConfig)
	if err != nil {
		return err
	}
	return InjectResolvedCheckpointIntoPodSpec(podSpec, restore, seccompProfile)
}

// EnsureIntraPodGPUMemoryService wires the in-pod GMS server sidecar and
// socket clients for checkpoint create/restore pod specs. Checkpoint jobs
// and restores are snapshot-coupled, so useV1 selects the V1 server.
func EnsureIntraPodGPUMemoryService(
	podSpec *corev1.PodSpec,
	targetContainers []*corev1.Container,
	extraClientContainerNames []string,
	useV1 bool,
) {
	if len(targetContainers) == 0 {
		return
	}
	gms.EnsureServerSidecar(podSpec, targetContainers[0], useV1)
	for _, container := range targetContainers {
		gms.EnsureClient(podSpec, container)
		if useV1 {
			gms.EnableV1(container)
		}
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
		if useV1 {
			gms.EnableV1(container)
		}
	}
}
