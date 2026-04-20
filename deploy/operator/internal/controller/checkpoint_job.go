// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"fmt"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/discovery"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
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

	if len(podTemplate.Spec.Containers) == 0 {
		return nil, fmt.Errorf("checkpoint job requires at least one container")
	}
	mainContainer := &podTemplate.Spec.Containers[0]
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

	// The snapshot agent sends SIGUSR1 to PID 1 of the main container after
	checkpoint.EnsurePodInfoMount(mainContainer)
	dynamo.ApplySharedMemoryVolumeAndMount(&podTemplate.Spec, mainContainer, ckpt.Spec.Job.SharedMemory)

	if ckpt.Spec.GPUMemoryService != nil && ckpt.Spec.GPUMemoryService.Enabled {
		claimTemplateName := dra.ResourceClaimTemplateName("checkpoint-"+hash, "worker")
		if err := dra.ApplyClaim(&podTemplate.Spec, claimTemplateName); err != nil {
			return nil, fmt.Errorf("failed to apply DRA claim for GMS checkpoint: %w", err)
		}
		storage, err := snapshotprotocol.DiscoverAndResolveStorage(
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
		// Re-acquire pointer: append in EnsureGMSCheckpointJobSidecars may
		// have reallocated the Containers slice.
		mainContainer = &podTemplate.Spec.Containers[0]
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

	// For single-GPU jobs (no cuda-checkpoint wrapper), unwrap /bin/sh -c so
	// the actual process is PID 1 and receives SIGUSR1 from the snapshot agent.
	if !wrapLaunchJob && len(mainContainer.Command) >= 2 &&
		mainContainer.Command[len(mainContainer.Command)-1] == "-c" &&
		len(mainContainer.Args) == 1 {
		parts := strings.Fields(mainContainer.Args[0])
		mainContainer.Command = parts[:1]
		mainContainer.Args = parts[1:]
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
