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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
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
		DiscoveryBackend:               discoveryBackend,
		WorkerHashSuffix:               workerHashSuffix,
	})
	return defaultContainer.Env
}

func buildCheckpointJob(
	ctx context.Context,
	reader ctrlclient.Reader,
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
	if podTemplate.Spec.ServiceAccountName == "" {
		podTemplate.Spec.ServiceAccountName = discovery.GetK8sDiscoveryServiceAccountName(ckpt.Name)
	}

	checkpoint.EnsurePodInfoVolume(&podTemplate.Spec)

	mainContainer, err := snapshotprotocol.ResolveCheckpointWorkerContainer(&podTemplate.Spec)
	if err != nil {
		return nil, err
	}
	mainContainer.Env = dynamo.MergeEnvs(
		buildCheckpointWorkerDefaultEnv(ckpt, podTemplate),
		mainContainer.Env,
	)
	dynamo.AddStandardEnvVars(mainContainer, config)
	mainContainer.Env = append(mainContainer.Env, corev1.EnvVar{
		Name:  consts.EnvReadyForCheckpointFile,
		Value: config.Checkpoint.ReadyForCheckpointFilePath,
	})
	mainContainer.ReadinessProbe = &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{
			Exec: &corev1.ExecAction{
				Command: []string{"cat", config.Checkpoint.ReadyForCheckpointFilePath},
			},
		},
		InitialDelaySeconds: 15,
		PeriodSeconds:       2,
	}
	mainContainer.LivenessProbe = nil
	mainContainer.StartupProbe = nil
	checkpoint.EnsurePodInfoMount(mainContainer)
	dynamo.ApplySharedMemoryVolumeAndMount(&podTemplate.Spec, mainContainer, ckpt.Spec.Job.SharedMemory)

	if ckpt.Spec.GPUMemoryService != nil && ckpt.Spec.GPUMemoryService.Enabled {
		storage, err := checkpoint.ResolveGMSCheckpointStorage(
			ctx,
			reader,
			ckpt.Namespace,
			hash,
			ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation],
		)
		if err != nil {
			return nil, err
		}
		if err := checkpoint.EnsureGMSCheckpointJobSidecars(&podTemplate.Spec, mainContainer, storage); err != nil {
			return nil, err
		}
	}

	activeDeadlineSeconds := ckpt.Spec.Job.ActiveDeadlineSeconds
	if activeDeadlineSeconds == nil {
		defaultDeadline := int64(3600)
		activeDeadlineSeconds = &defaultDeadline
	}

	wrapLaunchJob := false
	if gpus, ok := mainContainer.Resources.Limits[corev1.ResourceName(consts.KubeResourceGPUNvidia)]; ok {
		wrapLaunchJob = gpus.Cmp(*resource.NewQuantity(1, resource.DecimalSI)) > 0
	}
	ttlSecondsAfterFinish := snapshotprotocol.DefaultCheckpointJobTTLSeconds

	return snapshotprotocol.NewCheckpointJob(podTemplate, snapshotprotocol.CheckpointJobOptions{
		Namespace:             ckpt.Namespace,
		CheckpointID:          hash,
		ArtifactVersion:       snapshotprotocol.ArtifactVersion(ckpt.Annotations[snapshotprotocol.CheckpointArtifactVersionAnnotation]),
		SeccompProfile:        snapshotprotocol.DefaultSeccompLocalhostProfile,
		Name:                  jobName,
		ActiveDeadlineSeconds: activeDeadlineSeconds,
		TTLSecondsAfterFinish: &ttlSecondsAfterFinish,
		WrapLaunchJob:         wrapLaunchJob,
	})
}
