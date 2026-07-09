/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation

import (
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestValidateDynamoCheckpointGMSSnapshotRejectsUnpreparedTemplate(t *testing.T) {
	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{Enabled: true},
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName}},
					},
				},
			},
		},
	}

	err := validateDynamoCheckpointGMSSnapshot(ckpt)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "gpuMemoryService is metadata-only")
	assert.Contains(t, err.Error(), "missing pod resource claim")
}

func TestValidateDynamoCheckpointGMSSnapshotAllowsPreparedTemplate(t *testing.T) {
	claimTemplateName := "checkpoint-test-worker-gpu"
	clientContainer := func(name string) corev1.Container {
		return corev1.Container{
			Name: name,
			Env: []corev1.EnvVar{
				{Name: gms.EnvSocketDir, Value: gms.SharedMountPath},
			},
			VolumeMounts: []corev1.VolumeMount{
				{Name: gms.SharedVolumeName, MountPath: gms.SharedMountPath},
			},
			Resources: corev1.ResourceRequirements{
				Claims: []corev1.ResourceClaim{{Name: dra.ClaimName}},
			},
		}
	}
	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			GPUMemoryService: &nvidiacomv1alpha1.GPUMemoryServiceSpec{
				Enabled:               true,
				ExtraClientContainers: []string{"saver"},
			},
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				TargetContainerName: consts.MainContainerName,
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						ResourceClaims: []corev1.PodResourceClaim{{
							Name:                      dra.ClaimName,
							ResourceClaimTemplateName: &claimTemplateName,
						}},
						Volumes: []corev1.Volume{{
							Name:         gms.SharedVolumeName,
							VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}},
						}},
						InitContainers: []corev1.Container{clientContainer(gms.ServerContainerName)},
						Containers: []corev1.Container{
							clientContainer(consts.MainContainerName),
							clientContainer("saver"),
						},
					},
				},
			},
		},
	}

	require.NoError(t, validateDynamoCheckpointGMSSnapshot(ckpt))
}
