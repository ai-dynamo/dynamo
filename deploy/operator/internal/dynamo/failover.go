/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

const (
	failoverSharedVolumeName = "failover-shared"
	failoverSharedMountPath  = "/shared"
	failoverLockFile         = "/shared/failover.lock"
	failoverDRAClaimName     = "shared-gpu"
	failoverEngineCount      = 2
)

func isFailoverEnabled(component *v1alpha1.DynamoComponentDeploymentSharedSpec) bool {
	return component.Failover != nil && component.Failover.Enabled
}

// getGPUCount extracts the GPU count from the component resource spec.
func getGPUCount(component *v1alpha1.DynamoComponentDeploymentSharedSpec) (int, error) {
	if component.Resources == nil {
		return 0, fmt.Errorf("resources must be specified for failover workers")
	}

	gpuStr := ""
	if component.Resources.Limits != nil && component.Resources.Limits.GPU != "" {
		gpuStr = component.Resources.Limits.GPU
	} else if component.Resources.Requests != nil && component.Resources.Requests.GPU != "" {
		gpuStr = component.Resources.Requests.GPU
	}

	if gpuStr == "" {
		return 0, fmt.Errorf("GPU count must be specified for failover workers")
	}

	count, err := strconv.Atoi(gpuStr)
	if err != nil {
		return 0, fmt.Errorf("invalid GPU count %q: %w", gpuStr, err)
	}
	return count, nil
}

// buildFailoverPod transforms a single-container worker pod spec into a
// multi-container failover pod with two engine containers and a GMS weight sidecar.
//
// The transformation:
//  1. Clones the main container into engine-0 and engine-1 with staggered system ports
//  2. Adds a GMS weight sidecar as an init container (restartPolicy: Always)
//  3. Adds a shared emptyDir volume for GMS UDS sockets and the flock file
//  4. Sets up DRA resource claims so all containers share GPU access
//  5. Injects failover-specific env vars (ENGINE_ID, GMS_SOCKET_DIR, FAILOVER_LOCK_PATH, etc.)
func buildFailoverPod(
	podSpec *corev1.PodSpec,
	component *v1alpha1.DynamoComponentDeploymentSharedSpec,
	parentName string,
	serviceName string,
) error {
	if len(podSpec.Containers) == 0 {
		return fmt.Errorf("pod spec must have at least one container for failover transformation")
	}

	gpuCount, err := getGPUCount(component)
	if err != nil {
		return err
	}

	mainContainer := podSpec.Containers[0]

	engines := make([]corev1.Container, failoverEngineCount)
	for i := range failoverEngineCount {
		engines[i] = buildEngineContainer(mainContainer, i, commonconsts.DynamoSystemPort+i)
	}

	gmsSidecar := buildGMSSidecar(mainContainer.Image, gpuCount)

	podSpec.Containers = engines
	podSpec.InitContainers = append(podSpec.InitContainers, gmsSidecar)
	podSpec.Volumes = append(podSpec.Volumes, failoverSharedVolume())

	claimTemplateName := fmt.Sprintf("%s-%s-gpu", parentName, strings.ToLower(serviceName))
	podSpec.ResourceClaims = append(podSpec.ResourceClaims, corev1.PodResourceClaim{
		Name:                      failoverDRAClaimName,
		ResourceClaimTemplateName: &claimTemplateName,
	})

	return nil
}

// buildEngineContainer clones the main container with ENGINE_ID and failover env vars.
// Each engine gets a unique system port and named port for probe targeting.
func buildEngineContainer(base corev1.Container, engineID int, systemPort int) corev1.Container {
	engine := *base.DeepCopy()
	engine.Name = fmt.Sprintf("engine-%d", engineID)

	portName := fmt.Sprintf("system-%d", engineID)

	engine.Ports = []corev1.ContainerPort{
		{
			Protocol:      corev1.ProtocolTCP,
			Name:          portName,
			ContainerPort: int32(systemPort),
		},
	}

	// Env vars to remove: replaced by failover-specific values or intentionally omitted.
	// DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS is omitted to activate Branch 3 in SystemHealth.
	removeSet := map[string]bool{
		"DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS": true,
		"DYN_SYSTEM_PORT":                       true,
		"DYN_SYSTEM_ENABLED":                    true,
		"DYN_HEALTH_CHECK_ENABLED":              true,
	}

	var filtered []corev1.EnvVar
	for _, env := range engine.Env {
		if !removeSet[env.Name] {
			filtered = append(filtered, env)
		}
	}

	failoverEnvs := []corev1.EnvVar{
		{Name: "ENGINE_ID", Value: strconv.Itoa(engineID)},
		{Name: "GMS_SOCKET_DIR", Value: failoverSharedMountPath},
		{Name: "FAILOVER_LOCK_PATH", Value: failoverLockFile},
		{Name: "DYN_SYSTEM_STARTING_HEALTH_STATUS", Value: "notready"},
		{Name: "DYN_SYSTEM_PORT", Value: strconv.Itoa(systemPort)},
		{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
		{Name: "DYN_VLLM_GMS_MODE", Value: "shadow"},
	}
	engine.Env = append(filtered, failoverEnvs...)

	engine.VolumeMounts = append(engine.VolumeMounts, corev1.VolumeMount{
		Name:      failoverSharedVolumeName,
		MountPath: failoverSharedMountPath,
	})

	portRef := intstr.FromString(portName)
	if engine.StartupProbe != nil && engine.StartupProbe.HTTPGet != nil {
		engine.StartupProbe.HTTPGet.Port = portRef
	}
	if engine.LivenessProbe != nil && engine.LivenessProbe.HTTPGet != nil {
		engine.LivenessProbe.HTTPGet.Port = portRef
	}
	if engine.ReadinessProbe != nil && engine.ReadinessProbe.HTTPGet != nil {
		engine.ReadinessProbe.HTTPGet.Port = portRef
	}

	removeGPUResources(&engine)
	engine.Resources.Claims = append(engine.Resources.Claims, corev1.ResourceClaim{
		Name: failoverDRAClaimName,
	})

	return engine
}

// removeGPUResources strips nvidia.com/gpu from container resource limits and requests.
// GPU allocation is handled by DRA when failover is enabled.
func removeGPUResources(container *corev1.Container) {
	gpuResource := corev1.ResourceName(commonconsts.KubeResourceGPUNvidia)
	delete(container.Resources.Limits, gpuResource)
	delete(container.Resources.Requests, gpuResource)
}

// buildGMSSidecar creates the GMS weight server as a sidecar init container
// (restartPolicy: Always). kubelet starts it before regular containers and
// keeps it running for the pod's lifetime.
func buildGMSSidecar(image string, gpuCount int) corev1.Container {
	devices := make([]string, gpuCount)
	for i := range gpuCount {
		devices[i] = strconv.Itoa(i)
	}
	deviceList := strings.Join(devices, ",")

	return corev1.Container{
		Name:          "gms-weights",
		Image:         image,
		Command:       []string{"python3", "-m", "gpu_memory_service"},
		Args:          []string{"--socket-dir", failoverSharedMountPath, "--devices", deviceList},
		RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
		VolumeMounts: []corev1.VolumeMount{
			{
				Name:      failoverSharedVolumeName,
				MountPath: failoverSharedMountPath,
			},
		},
		StartupProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				Exec: &corev1.ExecAction{
					Command: gmsReadyCheckCommand(gpuCount),
				},
			},
			PeriodSeconds:    2,
			FailureThreshold: 150, // 2s * 150 = 5 min
		},
		Resources: corev1.ResourceRequirements{
			Claims: []corev1.ResourceClaim{
				{Name: failoverDRAClaimName},
			},
		},
	}
}

// gmsReadyCheckCommand returns the exec probe command that verifies all
// GMS UDS sockets exist on the shared volume.
func gmsReadyCheckCommand(gpuCount int) []string {
	if gpuCount == 1 {
		return []string{"test", "-S", fmt.Sprintf("%s/gms_0.sock", failoverSharedMountPath)}
	}
	checks := make([]string, gpuCount)
	for i := range gpuCount {
		checks[i] = fmt.Sprintf("test -S %s/gms_%d.sock", failoverSharedMountPath, i)
	}
	return []string{"sh", "-c", strings.Join(checks, " && ")}
}

func failoverSharedVolume() corev1.Volume {
	return corev1.Volume{
		Name: failoverSharedVolumeName,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	}
}
