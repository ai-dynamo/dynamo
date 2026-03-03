/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"strconv"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func failoverComponent(gpuCount int) *v1alpha1.DynamoComponentDeploymentSharedSpec {
	return &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType: commonconsts.ComponentTypeWorker,
		Failover:      &v1alpha1.FailoverSpec{Enabled: true},
		Resources: &v1alpha1.Resources{
			Limits: &v1alpha1.ResourceItem{GPU: strconv.Itoa(gpuCount)},
		},
	}
}

func basePodSpec() corev1.PodSpec {
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
					{Name: "ETCD_ENDPOINTS", Value: "http://etcd:2379"},
					{Name: "DYN_NAMESPACE", Value: "default_test"},
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
					Limits: corev1.ResourceList{
						corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("2"),
					},
				},
			},
		},
	}
}

// --- buildFailoverPod ---

func TestBuildFailoverPod_EtcdRequired(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failover requires etcd discovery")
}

func TestBuildFailoverPod_EmptyContainers(t *testing.T) {
	ps := corev1.PodSpec{}
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least one container")
}

func TestBuildFailoverPod_TwoEngines(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	assert.Len(t, ps.Containers, 2, "should have 2 engine containers")
	assert.Equal(t, "engine-0", ps.Containers[0].Name)
	assert.Equal(t, "engine-1", ps.Containers[1].Name)
}

func TestBuildFailoverPod_GMSSidecar(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	require.Len(t, ps.InitContainers, 1, "should have 1 init container (GMS sidecar)")
	gms := ps.InitContainers[0]
	assert.Equal(t, "gms-weights", gms.Name)
	assert.Equal(t, "test-image:latest", gms.Image)
	assert.Equal(t, []string{"bash", "-c"}, gms.Command)
	assert.Contains(t, gms.Args[0], "gpu_memory_service --device")
	assert.NotNil(t, gms.RestartPolicy)
	assert.Equal(t, corev1.ContainerRestartPolicyAlways, *gms.RestartPolicy)
}

func TestBuildFailoverPod_GMSSidecarTMPDIR(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	gms := ps.InitContainers[0]
	found := false
	for _, env := range gms.Env {
		if env.Name == "TMPDIR" {
			assert.Equal(t, "/shared", env.Value)
			found = true
		}
	}
	assert.True(t, found, "GMS sidecar should have TMPDIR=/shared")
}

func TestBuildFailoverPod_SharedVolume(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	var foundVol bool
	for _, v := range ps.Volumes {
		if v.Name == failoverSharedVolumeName {
			assert.NotNil(t, v.EmptyDir)
			foundVol = true
		}
	}
	assert.True(t, foundVol, "should have failover-shared volume")
}

func TestBuildFailoverPod_GPUToleration(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	var found bool
	for _, tol := range ps.Tolerations {
		if tol.Key == commonconsts.KubeResourceGPUNvidia && tol.Effect == corev1.TaintEffectNoSchedule {
			assert.Equal(t, corev1.TolerationOpExists, tol.Operator)
			found = true
		}
	}
	assert.True(t, found, "should have nvidia.com/gpu NoSchedule toleration")
}

func TestBuildFailoverPod_DRAResourceClaim(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	require.Len(t, ps.ResourceClaims, 1)
	assert.Equal(t, failoverDRAClaimName, ps.ResourceClaims[0].Name)
	assert.Equal(t, "myapp-worker-gpu", *ps.ResourceClaims[0].ResourceClaimTemplateName)
}

// --- buildEngineContainer ---

func TestBuildEngineContainer_EnvVars(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	for i, engine := range ps.Containers {
		envMap := envToMap(engine.Env)

		assert.Equal(t, strconv.Itoa(i), envMap["ENGINE_ID"], "engine-%d ENGINE_ID", i)
		assert.Equal(t, "/shared", envMap["TMPDIR"], "engine-%d TMPDIR", i)
		assert.Equal(t, "/shared/failover.lock", envMap["FAILOVER_LOCK_PATH"], "engine-%d FAILOVER_LOCK_PATH", i)
		assert.Equal(t, "shadow", envMap["DYN_VLLM_GMS_MODE"], "engine-%d DYN_VLLM_GMS_MODE", i)
		assert.Equal(t, "true", envMap["DYN_SYSTEM_ENABLED"], "engine-%d DYN_SYSTEM_ENABLED", i)
		assert.Equal(t, "notready", envMap["DYN_SYSTEM_STARTING_HEALTH_STATUS"], "engine-%d DYN_SYSTEM_STARTING_HEALTH_STATUS", i)
		assert.Equal(t, "etcd", envMap[commonconsts.DynamoDiscoveryBackendEnvVar], "engine-%d DYN_DISCOVERY_BACKEND should be etcd", i)

		_, hasOldSocketDir := envMap["GMS_SOCKET_DIR"]
		assert.False(t, hasOldSocketDir, "engine-%d should not have GMS_SOCKET_DIR", i)

		_, hasOldHealthStatus := envMap["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"]
		assert.False(t, hasOldHealthStatus, "engine-%d should not have DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS", i)
	}
}

func TestBuildEngineContainer_StaggeredPorts(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	for i, engine := range ps.Containers {
		envMap := envToMap(engine.Env)

		expectedSystemPort := strconv.Itoa(commonconsts.DynamoSystemPort + i)
		assert.Equal(t, expectedSystemPort, envMap["DYN_SYSTEM_PORT"], "engine-%d system port", i)

		expectedNixlPort := strconv.Itoa(5600 + i)
		assert.Equal(t, expectedNixlPort, envMap["VLLM_NIXL_SIDE_CHANNEL_PORT"], "engine-%d nixl port", i)

		expectedKvPort := strconv.Itoa(20080 + i)
		assert.Equal(t, expectedKvPort, envMap["DYN_VLLM_KV_EVENT_PORT"], "engine-%d kv event port", i)

		require.Len(t, engine.Ports, 1)
		assert.Equal(t, int32(commonconsts.DynamoSystemPort+i), engine.Ports[0].ContainerPort)
		assert.Equal(t, "system-"+strconv.Itoa(i), engine.Ports[0].Name)
	}
}

func TestBuildEngineContainer_DiscoveryBackendOverride(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	for _, engine := range ps.Containers {
		envMap := envToMap(engine.Env)
		assert.Equal(t, "etcd", envMap[commonconsts.DynamoDiscoveryBackendEnvVar])

		count := 0
		for _, env := range engine.Env {
			if env.Name == commonconsts.DynamoDiscoveryBackendEnvVar {
				count++
			}
		}
		assert.Equal(t, 1, count, "DYN_DISCOVERY_BACKEND should appear exactly once")
	}
}

func TestBuildEngineContainer_GPUResourcesRemoved(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	gpuResource := corev1.ResourceName(commonconsts.KubeResourceGPUNvidia)
	for i, engine := range ps.Containers {
		_, hasLimitsGPU := engine.Resources.Limits[gpuResource]
		assert.False(t, hasLimitsGPU, "engine-%d should not have GPU limits", i)

		_, hasRequestsGPU := engine.Resources.Requests[gpuResource]
		assert.False(t, hasRequestsGPU, "engine-%d should not have GPU requests", i)

		require.Len(t, engine.Resources.Claims, 1)
		assert.Equal(t, failoverDRAClaimName, engine.Resources.Claims[0].Name)
	}
}

func TestBuildEngineContainer_ProbesUseNamedPort(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	for i, engine := range ps.Containers {
		expectedPort := intstr.FromString("system-" + strconv.Itoa(i))
		if engine.StartupProbe != nil && engine.StartupProbe.HTTPGet != nil {
			assert.Equal(t, expectedPort, engine.StartupProbe.HTTPGet.Port, "engine-%d startup probe port", i)
		}
		if engine.LivenessProbe != nil && engine.LivenessProbe.HTTPGet != nil {
			assert.Equal(t, expectedPort, engine.LivenessProbe.HTTPGet.Port, "engine-%d liveness probe port", i)
		}
		if engine.ReadinessProbe != nil && engine.ReadinessProbe.HTTPGet != nil {
			assert.Equal(t, expectedPort, engine.ReadinessProbe.HTTPGet.Port, "engine-%d readiness probe port", i)
		}
	}
}

func TestBuildEngineContainer_PreservesNonRemovedEnvVars(t *testing.T) {
	ps := basePodSpec()
	component := failoverComponent(2)
	err := buildFailoverPod(&ps, component, "myapp", "Worker", "http://etcd:2379")
	require.NoError(t, err)

	for _, engine := range ps.Containers {
		envMap := envToMap(engine.Env)
		assert.Equal(t, "http://etcd:2379", envMap["ETCD_ENDPOINTS"], "should preserve ETCD_ENDPOINTS")
		assert.Equal(t, "default_test", envMap["DYN_NAMESPACE"], "should preserve DYN_NAMESPACE")
	}
}

// --- GMS sidecar helpers ---

func TestGmsWrapperScript(t *testing.T) {
	script := gmsWrapperScript(3)
	assert.Contains(t, script, "for dev in 0 1 2")
	assert.Contains(t, script, `python3 -m gpu_memory_service --device "$dev"`)
	assert.Contains(t, script, "trap cleanup SIGTERM SIGINT")
	assert.Contains(t, script, "wait -n")
}

func TestGmsReadyCheckCommand(t *testing.T) {
	cmd := gmsReadyCheckCommand(2)
	assert.Equal(t, "sh", cmd[0])
	assert.Equal(t, "-c", cmd[1])
	assert.Contains(t, cmd[2], "gms_*.sock")
	assert.Contains(t, cmd[2], "-ge 2")
}

func TestGmsReadyCheckCommand_SingleGPU(t *testing.T) {
	cmd := gmsReadyCheckCommand(1)
	assert.Equal(t, "sh", cmd[0])
	assert.Equal(t, "-c", cmd[1])
	assert.Contains(t, cmd[2], "-ge 1")
}

// --- GenerateFailoverResourceClaimTemplate ---

func TestGenerateFailoverResourceClaimTemplate_Enabled(t *testing.T) {
	component := failoverComponent(4)
	tmpl, toDelete, err := GenerateFailoverResourceClaimTemplate("myapp", "default", "Worker", component)

	require.NoError(t, err)
	assert.False(t, toDelete)
	assert.Equal(t, "myapp-worker-gpu", tmpl.Name)
	assert.Equal(t, "default", tmpl.Namespace)

	require.Len(t, tmpl.Spec.Spec.Devices.Requests, 1)
	req := tmpl.Spec.Spec.Devices.Requests[0]
	assert.Equal(t, "gpus", req.Name)
	require.NotNil(t, req.Exactly)
	assert.Equal(t, "gpu.nvidia.com", req.Exactly.DeviceClassName)
	assert.Equal(t, resourcev1.DeviceAllocationModeExactCount, req.Exactly.AllocationMode)
	assert.Equal(t, int64(4), req.Exactly.Count)
}

func TestGenerateFailoverResourceClaimTemplate_CustomGPUType(t *testing.T) {
	component := failoverComponent(2)
	component.Resources.Limits.GPUType = "gpu.intel.com/xe"
	tmpl, toDelete, err := GenerateFailoverResourceClaimTemplate("myapp", "default", "Worker", component)

	require.NoError(t, err)
	assert.False(t, toDelete)
	assert.Equal(t, "gpu.intel.com/xe", tmpl.Spec.Spec.Devices.Requests[0].Exactly.DeviceClassName)
}

func TestGenerateFailoverResourceClaimTemplate_DisabledReturnsDelete(t *testing.T) {
	component := &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType: commonconsts.ComponentTypeWorker,
	}
	tmpl, toDelete, err := GenerateFailoverResourceClaimTemplate("myapp", "default", "Worker", component)

	require.NoError(t, err)
	assert.True(t, toDelete)
	assert.Equal(t, "myapp-worker-gpu", tmpl.Name)
}

func TestGenerateFailoverResourceClaimTemplate_NoGPUCountError(t *testing.T) {
	component := &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType: commonconsts.ComponentTypeWorker,
		Failover:      &v1alpha1.FailoverSpec{Enabled: true},
	}
	_, _, err := GenerateFailoverResourceClaimTemplate("myapp", "default", "Worker", component)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "resources must be specified")
}

// --- FailoverResourceClaimTemplateName ---

func TestFailoverResourceClaimTemplateName(t *testing.T) {
	assert.Equal(t, "myapp-worker-gpu", FailoverResourceClaimTemplateName("myapp", "Worker"))
	assert.Equal(t, "app-vllmdecodeworker-gpu", FailoverResourceClaimTemplateName("app", "VllmDecodeWorker"))
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

// --- getGPUCount ---

func TestGetGPUCount(t *testing.T) {
	tests := []struct {
		name      string
		component *v1alpha1.DynamoComponentDeploymentSharedSpec
		want      int
		wantErr   bool
	}{
		{
			name:      "from limits",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{Resources: &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPU: "4"}}},
			want:      4,
		},
		{
			name:      "from requests",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{Resources: &v1alpha1.Resources{Requests: &v1alpha1.ResourceItem{GPU: "2"}}},
			want:      2,
		},
		{
			name:      "no resources",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			wantErr:   true,
		},
		{
			name:      "invalid GPU string",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{Resources: &v1alpha1.Resources{Limits: &v1alpha1.ResourceItem{GPU: "abc"}}},
			wantErr:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := getGPUCount(tt.component)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

// helpers

func envToMap(envs []corev1.EnvVar) map[string]string {
	m := make(map[string]string, len(envs))
	for _, e := range envs {
		m[e.Name] = e.Value
	}
	return m
}

