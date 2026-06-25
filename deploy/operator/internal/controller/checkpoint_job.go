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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
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
	// Stamp the owning checkpoint so the controller's source-pod watch can map the Job's pod back to
	// this DynamoCheckpoint. Operator-side only: NewCheckpointJob is shared with snapshotctl.
	podTemplate.Labels[consts.SnapshotOwnerLabel] = ckpt.Name
	targetContainerName := ckpt.Spec.Job.TargetContainerName
	if targetContainerName == "" {
		targetContainerName = consts.MainContainerName
	}
	podTemplate.Annotations[snapshotprotocol.TargetContainersAnnotation] = snapshotprotocol.FormatTargetContainers([]string{targetContainerName})

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

	// Decide whether cuda-checkpoint needs --launch-job from the rendered pod
	// template, not from deprecated identity fields. Regular pods expose GPUs
	// as scalar resources; GMS/DRA-prepared checkpoint pods expose them through
	// ResourceClaims. For DRA claims, only count requests using the configured
	// GMS DeviceClassName (or Dynamo's default) so unrelated device claims do
	// not trigger checkpoint launch wrapping.
	gpuCount, err := dra.ExtractGPUCountFromResourceRequirements(targetContainer.Resources)
	if err != nil {
		return nil, err
	}
	if kubeClient != nil {
		gpuDeviceClassName := dra.DefaultDeviceClassName
		if ckpt.Spec.GPUMemoryService != nil && ckpt.Spec.GPUMemoryService.DeviceClassName != "" {
			gpuDeviceClassName = ckpt.Spec.GPUMemoryService.DeviceClassName
		}
		deviceCount := func(allocation resourcev1.DeviceAllocationMode, count int64) int {
			if allocation == resourcev1.DeviceAllocationModeAll {
				// AllocationModeAll is resolved by the scheduler, so the
				// exact count is unavailable here. Assume multi-GPU so
				// checkpoint launch wrapping is not missed.
				return 2
			}
			if count == 0 {
				return 1
			}
			return int(count)
		}

		for _, containerClaim := range targetContainer.Resources.Claims {
			var claimSpec *resourcev1.ResourceClaimSpec
			for i := range podTemplate.Spec.ResourceClaims {
				podClaim := podTemplate.Spec.ResourceClaims[i]
				if podClaim.Name != containerClaim.Name {
					continue
				}
				switch {
				case podClaim.ResourceClaimTemplateName != nil && *podClaim.ResourceClaimTemplateName != "":
					template := &resourcev1.ResourceClaimTemplate{}
					name := *podClaim.ResourceClaimTemplateName
					if err := kubeClient.Get(ctx, ctrlclient.ObjectKey{Namespace: ckpt.Namespace, Name: name}, template); err != nil {
						return nil, fmt.Errorf("failed to get ResourceClaimTemplate %s/%s for checkpoint GPU count: %w", ckpt.Namespace, name, err)
					}
					claimSpec = &template.Spec.Spec
				case podClaim.ResourceClaimName != nil && *podClaim.ResourceClaimName != "":
					claim := &resourcev1.ResourceClaim{}
					name := *podClaim.ResourceClaimName
					if err := kubeClient.Get(ctx, ctrlclient.ObjectKey{Namespace: ckpt.Namespace, Name: name}, claim); err != nil {
						return nil, fmt.Errorf("failed to get ResourceClaim %s/%s for checkpoint GPU count: %w", ckpt.Namespace, name, err)
					}
					claimSpec = &claim.Spec
				}
				break
			}
			if claimSpec == nil {
				continue
			}

			for _, request := range claimSpec.Devices.Requests {
				if containerClaim.Request != "" && request.Name != containerClaim.Request {
					continue
				}
				if request.Exactly != nil {
					if request.Exactly.DeviceClassName == gpuDeviceClassName {
						gpuCount += deviceCount(request.Exactly.AllocationMode, request.Exactly.Count)
					}
					continue
				}

				requestGPUCount := 0
				for _, subRequest := range request.FirstAvailable {
					if subRequest.DeviceClassName == gpuDeviceClassName {
						requestGPUCount = max(requestGPUCount, deviceCount(subRequest.AllocationMode, subRequest.Count))
					}
				}
				gpuCount += requestGPUCount
			}
		}
	}
	wrapLaunchJob := gpuCount > 1

	activeDeadlineSeconds := ckpt.Spec.Job.ActiveDeadlineSeconds
	if activeDeadlineSeconds == nil {
		defaultDeadline := int64(3600)
		activeDeadlineSeconds = &defaultDeadline
	}

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
