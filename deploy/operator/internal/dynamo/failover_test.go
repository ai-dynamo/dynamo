/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"strconv"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// failoverPodSpec returns a pod spec that has already been transformed by
// applyGPUMemoryService (DRA claims, shared volume, TMPDIR set), including
// a frontend sidecar to verify sidecar preservation.
func failoverPodSpec() corev1.PodSpec {
	httpPort := intstr.FromString("system")
	return corev1.PodSpec{
		Containers: []corev1.Container{
			{
				Name:    "main",
				Image:   "test-image:latest",
				Command: []string{"python3", "-m", "dynamo.vllm"},
				Env: []corev1.EnvVar{
					{Name: "DYN_SYSTEM_PORT", Value: "9090"},
					{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
					{Name: "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", Value: "true"},
					{Name: "DYN_HEALTH_CHECK_ENABLED", Value: "true"},
					{Name: commonconsts.DynamoDiscoveryBackendEnvVar, Value: "kubernetes"},
					{Name: "TMPDIR", Value: gms.SharedMountPath},
				},
				Ports: []corev1.ContainerPort{
					{Name: "system", ContainerPort: 9090, Protocol: corev1.ProtocolTCP},
				},
				StartupProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: httpPort},
					},
				},
				LivenessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: httpPort},
					},
				},
				ReadinessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						HTTPGet: &corev1.HTTPGetAction{Path: "/health", Port: httpPort},
					},
				},
				Resources: corev1.ResourceRequirements{
					Claims: []corev1.ResourceClaim{{Name: dra.ClaimName}},
				},
				VolumeMounts: []corev1.VolumeMount{
					{Name: gms.SharedVolumeName, MountPath: gms.SharedMountPath},
				},
			},
			{
				Name:  "frontend-sidecar",
				Image: "test-image:latest",
			},
		},
	}
}

// --- buildFailoverPod ---

func TestBuildFailoverPod_TwoEnginesPlusSidecar(t *testing.T) {
	ps := failoverPodSpec()
	err := buildFailoverPod(&ps, 1, BackendFrameworkVLLM)
	require.NoError(t, err)

	// 2 engines + 1 preserved sidecar
	assert.Len(t, ps.Containers, 3)
	assert.Equal(t, "engine-0", ps.Containers[0].Name)
	assert.Equal(t, "engine-1", ps.Containers[1].Name)
	assert.Equal(t, "frontend-sidecar", ps.Containers[2].Name)
}

func TestBuildFailoverPod_EmptyContainers(t *testing.T) {
	ps := corev1.PodSpec{}
	err := buildFailoverPod(&ps, 1, BackendFrameworkVLLM)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least one container")
}

func TestBuildFailoverPod_RejectsNonVLLM(t *testing.T) {
	ps := failoverPodSpec()
	err := buildFailoverPod(&ps, 1, BackendFrameworkSGLang)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "currently supported only for vLLM")
}

func TestBuildFailoverPod_EngineEnvVars(t *testing.T) {
	ps := failoverPodSpec()
	err := buildFailoverPod(&ps, 1, BackendFrameworkVLLM)
	require.NoError(t, err)

	for i := range 2 {
		engine := ps.Containers[i]
		env := envToMap(engine.Env)
		assert.Equal(t, strconv.Itoa(i), env["ENGINE_ID"], "engine-%d ENGINE_ID", i)
		assert.Equal(t, fmt.Sprintf("engine-%d", i), env["CONTAINER_NAME"], "engine-%d CONTAINER_NAME", i)
		assert.Equal(t, failoverLockFile, env["FAILOVER_LOCK_PATH"], "engine-%d FAILOVER_LOCK_PATH", i)
		assert.Equal(t, "true", env["DYN_VLLM_GMS_SHADOW_MODE"], "engine-%d shadow mode", i)
		assert.Equal(t, "notready", env["DYN_SYSTEM_STARTING_HEALTH_STATUS"], "engine-%d starting health", i)
		assert.Equal(t, "true", env["DYN_SYSTEM_ENABLED"], "engine-%d system enabled", i)

		// Removed env vars should not be present
		_, hasOldHealth := env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"]
		assert.False(t, hasOldHealth, "engine-%d should not have DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", i)
		_, hasHealthCheck := env["DYN_HEALTH_CHECK_ENABLED"]
		assert.False(t, hasHealthCheck, "engine-%d should not have DYN_HEALTH_CHECK_ENABLED", i)
	}
}

func TestBuildFailoverPod_StaggeredPorts(t *testing.T) {
	ps := failoverPodSpec()
	err := buildFailoverPod(&ps, 1, BackendFrameworkVLLM)
	require.NoError(t, err)

	for i := range 2 {
		engine := ps.Containers[i]
		env := envToMap(engine.Env)
		assert.Equal(t, strconv.Itoa(commonconsts.DynamoSystemPort+i), env["DYN_SYSTEM_PORT"])
		require.Len(t, engine.Ports, 1)
		assert.Equal(t, int32(commonconsts.DynamoSystemPort+i), engine.Ports[0].ContainerPort)
		assert.Equal(t, fmt.Sprintf("system-%d", i), engine.Ports[0].Name)
	}
}

func TestBuildFailoverPod_ProbesRetargetedToNamedPort(t *testing.T) {
	ps := failoverPodSpec()
	err := buildFailoverPod(&ps, 1, BackendFrameworkVLLM)
	require.NoError(t, err)

	for i := range 2 {
		engine := ps.Containers[i]
		expectedPort := intstr.FromString(fmt.Sprintf("system-%d", i))
		if engine.StartupProbe != nil && engine.StartupProbe.HTTPGet != nil {
			assert.Equal(t, expectedPort, engine.StartupProbe.HTTPGet.Port)
		}
		if engine.LivenessProbe != nil && engine.LivenessProbe.HTTPGet != nil {
			assert.Equal(t, expectedPort, engine.LivenessProbe.HTTPGet.Port)
		}
		if engine.ReadinessProbe != nil && engine.ReadinessProbe.HTTPGet != nil {
			assert.Equal(t, expectedPort, engine.ReadinessProbe.HTTPGet.Port)
		}
	}
}

func TestBuildFailoverPod_PreservesDRAClaim(t *testing.T) {
	ps := failoverPodSpec()
	err := buildFailoverPod(&ps, 1, BackendFrameworkVLLM)
	require.NoError(t, err)

	for i := range 2 {
		engine := ps.Containers[i]
		require.Len(t, engine.Resources.Claims, 1, "engine-%d should retain DRA claim", i)
		assert.Equal(t, dra.ClaimName, engine.Resources.Claims[0].Name)
	}
}

func TestBuildFailoverPod_PreservesDiscoveryBackend(t *testing.T) {
	ps := failoverPodSpec()
	err := buildFailoverPod(&ps, 1, BackendFrameworkVLLM)
	require.NoError(t, err)

	for i := range 2 {
		env := envToMap(ps.Containers[i].Env)
		assert.Equal(t, "kubernetes", env[commonconsts.DynamoDiscoveryBackendEnvVar])
	}
}

func TestBuildFailoverPod_MultinodeNNODES(t *testing.T) {
	ps := failoverPodSpec()
	err := buildFailoverPod(&ps, 4, BackendFrameworkVLLM)
	require.NoError(t, err)

	for i := range 2 {
		env := envToMap(ps.Containers[i].Env)
		assert.Equal(t, "4", env["NNODES"], "engine-%d should have NNODES=4", i)
	}
}

func TestBuildFailoverPod_SingleNodeNoNNODES(t *testing.T) {
	ps := failoverPodSpec()
	err := buildFailoverPod(&ps, 1, BackendFrameworkVLLM)
	require.NoError(t, err)

	for i := range 2 {
		env := envToMap(ps.Containers[i].Env)
		_, has := env["NNODES"]
		assert.False(t, has, "engine-%d should not have NNODES for single-node", i)
	}
}

// --- isFailoverEnabled ---

func TestIsFailoverEnabled(t *testing.T) {
	assert.True(t, isFailoverEnabled(&v1alpha1.DynamoComponentDeploymentSharedSpec{
		Failover: &v1alpha1.FailoverSpec{Enabled: true},
	}))
	assert.False(t, isFailoverEnabled(&v1alpha1.DynamoComponentDeploymentSharedSpec{
		Failover: &v1alpha1.FailoverSpec{Enabled: false},
	}))
	assert.False(t, isFailoverEnabled(&v1alpha1.DynamoComponentDeploymentSharedSpec{}))
}

func envToMap(envs []corev1.EnvVar) map[string]string {
	m := make(map[string]string, len(envs))
	for _, e := range envs {
		m[e.Name] = e.Value
	}
	return m
}
