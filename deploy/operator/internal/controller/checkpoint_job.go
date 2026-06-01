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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/discovery"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

func buildCheckpointWorkerDefaultEnv(
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
	podTemplate *corev1.PodTemplateSpec,
) []corev1.EnvVar {
	componentType := consts.ComponentTypeWorker
	dynamoNamespace := consts.GlobalDynamoNamespace
	parentGraphDeploymentName := podTemplate.Labels[consts.KubeLabelDynamoGraphDeploymentName]
	workerHashSuffix := podTemplate.Labels[consts.KubeLabelDynamoWorkerHash]
	discoveryBackend := configv1alpha1.DiscoveryBackendKubernetes

	if podTemplate.Labels[consts.KubeLabelDynamoNamespace] != "" {
		dynamoNamespace = podTemplate.Labels[consts.KubeLabelDynamoNamespace]
	}
	if podTemplate.Labels[consts.KubeLabelDynamoComponentType] != "" &&
		dynamo.IsWorkerComponent(podTemplate.Labels[consts.KubeLabelDynamoComponentType]) {
		componentType = podTemplate.Labels[consts.KubeLabelDynamoComponentType]
	}

	defaultContainer, _ := dynamo.NewWorkerDefaults().GetBaseContainer(dynamo.ComponentContext{
		ComponentType:                  componentType,
		DynamoNamespace:                dynamoNamespace,
		ParentGraphDeploymentName:      parentGraphDeploymentName,
		ParentGraphDeploymentNamespace: ckpt.Namespace,
		Discovery: dynamo.DiscoveryContext{
			Backend: discoveryBackend,
			Mode:    configv1alpha1.KubeDiscoveryModePod,
		},
		WorkerHashSuffix: workerHashSuffix,
	})
	return defaultContainer.Env
}

//nolint:gocyclo
func buildCheckpointJob(
	ctx context.Context,
	kubeClient ctrlclient.Client,
	config *configv1alpha1.OperatorConfiguration,
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
	jobName string,
) (*batchv1.Job, error) {
	podTemplate := ckpt.Spec.Job.PodTemplateSpec.DeepCopy()
	hash := ckpt.Status.IdentityHash
	if hash == "" {
		var err error
		hash, err = checkpoint.ComputeIdentityHash(ckpt.Spec.Identity)
		if err != nil {
			return nil, fmt.Errorf("failed to compute identity hash: %w", err)
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
	if podTemplate.Spec.ServiceAccountName == "" {
		podTemplate.Spec.ServiceAccountName = discovery.GetK8sDiscoveryServiceAccountName(ckpt.Name)
	}

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
	targetContainer.Env = dynamo.MergeEnvs(
		buildCheckpointWorkerDefaultEnv(ckpt, podTemplate),
		targetContainer.Env,
	)
	dynamo.AddStandardEnvVars(targetContainer, config)

	checkpoint.EnsurePodInfoMount(targetContainer)
	dynamo.ApplySharedMemoryVolumeAndMount(&podTemplate.Spec, targetContainer, dynamo.ToBetaSharedMemorySize(ckpt.Spec.Job.SharedMemory))
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

	if ckpt.Spec.GPUMemoryService != nil && ckpt.Spec.GPUMemoryService.Enabled {
		switch ckpt.Spec.GPUMemoryService.Mode {
		case "", nvidiacomv1alpha1.GMSModeIntraPod:
			claimTemplateName := dra.ResourceClaimTemplateName("checkpoint-"+hash, "worker")
			foundToleration := false
			for i := range podTemplate.Spec.Tolerations {
				toleration := podTemplate.Spec.Tolerations[i]
				if toleration.Key == consts.KubeResourceGPUNvidia && toleration.Effect == corev1.TaintEffectNoSchedule {
					foundToleration = true
					break
				}
			}
			if !foundToleration {
				podTemplate.Spec.Tolerations = append(podTemplate.Spec.Tolerations, corev1.Toleration{
					Key:      consts.KubeResourceGPUNvidia,
					Operator: corev1.TolerationOpExists,
					Effect:   corev1.TaintEffectNoSchedule,
				})
			}

			podClaim := corev1.PodResourceClaim{
				Name:                      dra.ClaimName,
				ResourceClaimTemplateName: &claimTemplateName,
			}
			foundPodClaim := false
			for i := range podTemplate.Spec.ResourceClaims {
				if podTemplate.Spec.ResourceClaims[i].Name == dra.ClaimName {
					podTemplate.Spec.ResourceClaims[i] = podClaim
					foundPodClaim = true
					break
				}
			}
			if !foundPodClaim {
				podTemplate.Spec.ResourceClaims = append(podTemplate.Spec.ResourceClaims, podClaim)
			}

			gms.EnsureServerSidecar(&podTemplate.Spec, targetContainer)
			for _, name := range ckpt.Spec.GPUMemoryService.ExtraClientContainers {
				var container *corev1.Container
				for i := range podTemplate.Spec.Containers {
					if podTemplate.Spec.Containers[i].Name == name {
						container = &podTemplate.Spec.Containers[i]
						break
					}
				}
				if container == nil {
					continue
				}
				gms.EnsureClient(&podTemplate.Spec, container)
			}
		case nvidiacomv1alpha1.GMSModeInterPod:
			return nil, fmt.Errorf("gpuMemoryService checkpoint jobs for mode %q are not implemented", ckpt.Spec.GPUMemoryService.Mode)
		default:
			return nil, fmt.Errorf("gpuMemoryService checkpoint job has unsupported mode %q", ckpt.Spec.GPUMemoryService.Mode)
		}
	}

	activeDeadlineSeconds := ckpt.Spec.Job.ActiveDeadlineSeconds
	if activeDeadlineSeconds == nil {
		defaultDeadline := int64(3600)
		activeDeadlineSeconds = &defaultDeadline
	}

	// Wrap with cuda-checkpoint --launch-job for multi-GPU jobs (TP*PP > 1).
	// Use checkpoint identity (not container limits) because DRA may have
	// already removed nvidia.com/gpu from the template.
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
