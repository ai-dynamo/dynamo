/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package gms

import (
	"path/filepath"
	"testing"

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
				Claims: []corev1.ResourceClaim{{Name: DRAClaimName}},
			},
		}},
	}

	EnsureServerSidecar(podSpec, &podSpec.Containers[0])

	require.Len(t, podSpec.Containers, 1)
	require.Len(t, podSpec.InitContainers, 1)

	main := &podSpec.Containers[0]
	server := &podSpec.InitContainers[0]

	assert.Equal(t, ServerContainerName, server.Name)
	assert.Equal(t, []string{"python3", "-m", serverSidecarModule}, server.Command)
	assert.Equal(t, SharedMountPath, envValue(t, main, "TMPDIR"))
	assert.Equal(t, SharedMountPath, envValue(t, main, "GMS_SOCKET_DIR"))
	assert.Equal(t, SharedMountPath, envValue(t, server, "TMPDIR"))
	assert.Equal(t, SharedMountPath, envValue(t, server, "GMS_SOCKET_DIR"))

	assert.Equal(t, corev1.ContainerRestartPolicyAlways, *server.RestartPolicy)
	require.NotNil(t, server.StartupProbe)
	assert.Equal(t, []string{"test", "-f", filepath.Join(SharedMountPath, readyFile)},
		server.StartupProbe.Exec.Command)

	// DRA claim copied from main
	assert.Len(t, server.Resources.Claims, 1)
	assert.Equal(t, DRAClaimName, server.Resources.Claims[0].Name)
}

func TestEnsureServerSidecarDoesNotAddCheckpointControl(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Image: "test:latest"}},
	}

	EnsureServerSidecar(podSpec, &podSpec.Containers[0])

	for _, volume := range podSpec.Volumes {
		if volume.Name == ControlVolumeName {
			t.Fatal("runtime shaping should not add checkpoint control volume")
		}
	}
	server := FindServerContainer(podSpec)
	require.NotNil(t, server)
	for _, env := range server.Env {
		if env.Name == "GMS_CONTROL_DIR" {
			t.Fatal("server should not have checkpoint control env")
		}
	}
}

func TestEnsureServerSidecarIdempotent(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Image: "test:latest"}},
	}

	EnsureServerSidecar(podSpec, &podSpec.Containers[0])
	EnsureServerSidecar(podSpec, &podSpec.Containers[0])

	assert.Len(t, podSpec.InitContainers, 1)
	volumeCount := 0
	for _, v := range podSpec.Volumes {
		if v.Name == SharedVolumeName {
			volumeCount++
		}
	}
	assert.Equal(t, 1, volumeCount)
}

func TestFindServerContainer(t *testing.T) {
	podSpec := &corev1.PodSpec{
		Containers: []corev1.Container{{Name: "main", Image: "test:latest"}},
	}
	assert.Nil(t, FindServerContainer(podSpec))

	EnsureServerSidecar(podSpec, &podSpec.Containers[0])
	assert.NotNil(t, FindServerContainer(podSpec))
	assert.Equal(t, ServerContainerName, FindServerContainer(podSpec).Name)
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
