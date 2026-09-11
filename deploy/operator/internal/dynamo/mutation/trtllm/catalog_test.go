/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package trtllm

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/mutation"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/workload"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestWorkerCatalogCombinesWiringAndLaunch(t *testing.T) {
	container := &corev1.Container{
		Name:           "main",
		LivenessProbe:  &corev1.Probe{},
		ReadinessProbe: &corev1.Probe{},
		StartupProbe:   &corev1.Probe{},
	}
	mutations := mutation.Concat(
		MultinodeWiring("mpi-secret"),
		WorkerLaunch([]string{"/bin/sh", "-c"}, []string{"exec sshd"}),
	)
	require.NoError(t, mutations.Apply(nil, workload.RoleWorker, container))
	require.Nil(t, container.LivenessProbe)
	require.NotNil(t, container.ReadinessProbe)
	require.Nil(t, container.StartupProbe)
	require.Equal(t, []string{"/bin/sh", "-c"}, container.Command)
	require.Equal(t, []string{"exec sshd"}, container.Args)
	require.Equal(t, []corev1.VolumeMount{{Name: "mpi-secret", MountPath: "/ssh-pk", ReadOnly: true}}, container.VolumeMounts)
	require.Equal(t, []corev1.EnvVar{MPIEnvironment()}, container.Env)
}
