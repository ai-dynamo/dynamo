/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package gms

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestEnsureServerSidecar(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{{
			Name:  "main",
			Image: "test-image:latest",
			Resources: corev1.ResourceRequirements{
				Claims: []corev1.ResourceClaim{{Name: dra.ClaimName}},
			},
		}},
	}

	EnsureServerSidecar(podSpec, &podSpec.Containers[0])

	require.Len(t, podSpec.InitContainers, 1)
	server := &podSpec.InitContainers[0]

	assert.Equal(t, ServerContainerName, server.Name)
	assert.Equal(t, []string{"python3", "-m", ServerModule}, server.Command)
	assert.Equal(t, corev1.ContainerRestartPolicyAlways, *server.RestartPolicy)
	assert.Nil(t, server.StartupProbe, "clients drive readiness via connect-retry")
	require.Len(t, server.Resources.Claims, 1)
	assert.Equal(t, dra.ClaimName, server.Resources.Claims[0].Name)

	// Shared volume and env on main
	assert.Equal(t, SharedMountPath, envValue(t, &podSpec.Containers[0], "GMS_SOCKET_DIR"))
	var hasVolume bool
	for _, v := range podSpec.Volumes {
		if v.Name == SharedVolumeName {
			hasVolume = true
		}
	}
	assert.True(t, hasVolume)
}

func TestEnsureServerSidecarIdempotent(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Image: "test:latest"}},
	}
	EnsureServerSidecar(podSpec, &podSpec.Containers[0])
	EnsureServerSidecar(podSpec, &podSpec.Containers[0])

	assert.Len(t, podSpec.InitContainers, 1)
}

func TestEnsureServerSidecarDoesNotAddCheckpointControl(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Image: "test:latest"}},
	}
	EnsureServerSidecar(podSpec, &podSpec.Containers[0])

	for _, v := range podSpec.Volumes {
		if v.Name == "gms-control" {
			t.Fatal("should not add checkpoint control volume")
		}
	}
}

func TestEnsureClientOverridesOperatorOwnedSocketFields(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Volumes: []corev1.Volume{{
			Name:         SharedVolumeName,
			VolumeSource: corev1.VolumeSource{HostPath: &corev1.HostPathVolumeSource{Path: "/wrong"}},
		}},
		Containers: []corev1.Container{{
			Name: "main",
			Env: []corev1.EnvVar{{
				Name:      EnvSocketDir,
				ValueFrom: &corev1.EnvVarSource{},
			}},
			VolumeMounts: []corev1.VolumeMount{{
				Name:      SharedVolumeName,
				MountPath: "/wrong",
			}},
		}},
	}

	EnsureClient(podSpec, &podSpec.Containers[0])

	require.NotNil(t, podSpec.Volumes[0].EmptyDir)
	assert.Equal(t, SharedMountPath, podSpec.Containers[0].VolumeMounts[0].MountPath)
	assert.Equal(t, SharedMountPath, podSpec.Containers[0].Env[0].Value)
	assert.Nil(t, podSpec.Containers[0].Env[0].ValueFrom)
	require.Len(t, podSpec.Containers[0].Resources.Claims, 1)
	assert.Equal(t, dra.ClaimName, podSpec.Containers[0].Resources.Claims[0].Name)
}

func envValue(t *testing.T, container *corev1.Container, name string) string {
	t.Helper()
	for _, env := range container.Env {
		if env.Name == name {
			return env.Value
		}
	}
	t.Fatalf("env %s not found", name)
	return ""
}
