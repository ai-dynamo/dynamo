// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

//nolint:gocyclo
func buildCheckpointJob(
	ctx context.Context,
	kubeClient ctrlclient.Client,
	config *configv1alpha1.OperatorConfiguration,
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
	jobName string,
) (*batchv1.Job, error) {
	podTemplate := ckpt.Spec.Job.PodTemplateSpec.DeepCopy()
	hash := ckpt.Status.CheckpointID
	if hash == "" {
		hash = ckpt.Status.IdentityHash
	}
	if hash == "" {
		var err error
		hash, err = checkpoint.CheckpointID(ckpt)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve checkpoint ID: %w", err)
		}
	}

	if podTemplate.Labels == nil {
		podTemplate.Labels = make(map[string]string)
	}
	if podTemplate.Annotations == nil {
		podTemplate.Annotations = make(map[string]string)
	}
	targetContainerName := ckpt.Spec.Job.TargetContainerName
	if targetContainerName == "" {
		targetContainerName = consts.MainContainerName
	}
	podTemplate.Annotations[snapshotprotocol.TargetContainersAnnotation] = snapshotprotocol.FormatTargetContainers([]string{targetContainerName})

	checkpoint.EnsurePodInfoVolume(&podTemplate.Spec)

	if len(podTemplate.Spec.Containers) == 0 {
		return nil, fmt.Errorf("checkpoint job requires at least one container")
	}
	var targetContainer *corev1.Container
	for i := range podTemplate.Spec.Containers {
		if podTemplate.Spec.Containers[i].Name == targetContainerName {
			targetContainer = &podTemplate.Spec.Containers[i]
			break
		}
	}
	if targetContainer == nil {
		return nil, fmt.Errorf("checkpoint job pod template: pod spec has no container named %q", targetContainerName)
	}
	checkpoint.EnsurePodInfoMount(targetContainer)
	checkpoint.ApplySharedMemoryVolumeAndMount(&podTemplate.Spec, targetContainer, ckpt.Spec.Job.SharedMemory)
	// NewCheckpointJob handles control volume + readiness probe from the
	// snapshot contract.

	if err := checkpoint.EnsureStoragePVC(ctx, kubeClient, ckpt.Namespace, config.Checkpoint.Storage); err != nil {
		return nil, err
	}
	if storage, ok, err := checkpoint.StorageFromConfig(config.Checkpoint.Storage); err != nil {
		return nil, err
	} else if ok {
		snapshotprotocol.InjectCheckpointVolume(&podTemplate.Spec, storage.PVCName)
		snapshotprotocol.InjectCheckpointVolumeMount(targetContainer, storage.BasePath)
		if podTemplate.Annotations == nil {
			podTemplate.Annotations = map[string]string{}
		}
		snapshotprotocol.ApplyCheckpointStorageMetadata(podTemplate.Annotations, storage)
	}

	activeDeadlineSeconds := ckpt.Spec.Job.ActiveDeadlineSeconds
	if activeDeadlineSeconds == nil {
		defaultDeadline := int64(3600)
		activeDeadlineSeconds = &defaultDeadline
	}

	// Wrap with cuda-checkpoint --launch-job for multi-GPU jobs.
	// Use checkpoint identity (not container limits) because prepared templates
	// may already have DRA/GMS wiring that removes scalar GPU resources.
	tp := ckpt.Spec.Identity.TensorParallelSize
	pp := ckpt.Spec.Identity.PipelineParallelSize
	if tp == 0 {
		tp = 1
	}
	if pp == 0 {
		pp = 1
	}
	wrapLaunchJob := tp*pp > 1

	ttlSecondsAfterFinish := snapshotprotocol.DefaultCheckpointJobTTLSeconds

	return snapshotprotocol.NewCheckpointJob(podTemplate, snapshotprotocol.CheckpointJobOptions{
		Namespace:             ckpt.Namespace,
		CheckpointID:          hash,
		ArtifactVersion:       snapshotprotocol.ArtifactVersion(ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation]),
		SeccompProfile:        config.Checkpoint.EffectiveSeccompProfile(),
		Name:                  jobName,
		ActiveDeadlineSeconds: activeDeadlineSeconds,
		TTLSecondsAfterFinish: &ttlSecondsAfterFinish,
		WrapLaunchJob:         wrapLaunchJob,
	})
}
