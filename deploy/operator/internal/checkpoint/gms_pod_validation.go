/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	corev1 "k8s.io/api/core/v1"
)

// ValidatePreparedGPUMemoryServicePodTemplate verifies that a GMS-enabled
// DynamoCheckpoint already carries the pod wiring normally prepared by the
// Dynamo layer.
func ValidatePreparedGPUMemoryServicePodTemplate(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) error {
	if ckpt == nil || ckpt.Spec.GPUMemoryService == nil || !ckpt.Spec.GPUMemoryService.Enabled {
		return nil
	}

	switch ckpt.Spec.GPUMemoryService.Mode {
	case "", nvidiacomv1alpha1.GMSModeIntraPod:
	case nvidiacomv1alpha1.GMSModeInterPod:
		return fmt.Errorf("gpuMemoryService checkpoint jobs for mode %q are not implemented", ckpt.Spec.GPUMemoryService.Mode)
	default:
		return fmt.Errorf("gpuMemoryService checkpoint job has unsupported mode %q", ckpt.Spec.GPUMemoryService.Mode)
	}

	podSpec := &ckpt.Spec.Job.PodTemplateSpec.Spec
	targetContainerName := ckpt.Spec.Job.TargetContainerName
	if targetContainerName == "" {
		// Operator-created checkpoints always set Job.TargetContainerName to the
		// component's resolved main-container name; this literal default only
		// applies to hand-authored CRs.
		targetContainerName = consts.MainContainerName
	}
	targetContainer := common.FindContainerByName(podSpec.Containers, targetContainerName)
	if targetContainer == nil {
		return fmt.Errorf("gpuMemoryService checkpoint pod template has no target container %q", targetContainerName)
	}

	if !common.HasPodResourceClaim(podSpec, dra.ClaimName) {
		return fmt.Errorf("gpuMemoryService checkpoint pod template is missing pod resource claim %q", dra.ClaimName)
	}
	if !common.HasVolume(podSpec.Volumes, gms.SharedVolumeName) {
		return fmt.Errorf("gpuMemoryService checkpoint pod template is missing shared volume %q", gms.SharedVolumeName)
	}
	if err := validateGMSClientContainer(targetContainerName, targetContainer); err != nil {
		return err
	}

	server := common.FindContainerByName(podSpec.InitContainers, gms.ServerContainerName)
	if server == nil {
		return fmt.Errorf("gpuMemoryService checkpoint pod template is missing init sidecar %q", gms.ServerContainerName)
	}
	if err := validateGMSClientContainer(gms.ServerContainerName, server); err != nil {
		return err
	}

	for _, name := range ckpt.Spec.GPUMemoryService.ExtraClientContainers {
		container := common.FindContainerByName(podSpec.Containers, name)
		if container == nil {
			return fmt.Errorf("gpuMemoryService checkpoint pod template is missing extra client container %q", name)
		}
		if err := validateGMSClientContainer(name, container); err != nil {
			return err
		}
	}
	return nil
}

func validateGMSClientContainer(name string, container *corev1.Container) error {
	if !common.HasContainerResourceClaim(container, dra.ClaimName) {
		return fmt.Errorf("gpuMemoryService checkpoint container %q is missing resource claim %q", name, dra.ClaimName)
	}
	if !common.HasVolumeMount(container.VolumeMounts, gms.SharedVolumeName, gms.SharedMountPath) {
		return fmt.Errorf("gpuMemoryService checkpoint container %q is missing %s mount at %s", name, gms.SharedVolumeName, gms.SharedMountPath)
	}
	if !common.HasEnvValue(container.Env, gms.EnvSocketDir, gms.SharedMountPath) {
		return fmt.Errorf("gpuMemoryService checkpoint container %q is missing %s=%s", name, gms.EnvSocketDir, gms.SharedMountPath)
	}
	return nil
}
