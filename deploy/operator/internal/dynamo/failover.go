/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"path/filepath"
	"strconv"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

var failoverLockFile = filepath.Join(gmsSharedMountPath, "failover.lock")

const (
	failoverEngineCount = 2
)

func isFailoverEnabled(component *v1alpha1.DynamoComponentDeploymentSharedSpec) bool {
	return component.Failover != nil && component.Failover.Enabled
}

// buildFailoverPod clones the main container into two engine containers (active + standby).
// This runs AFTER applyGPUMemoryService, so the main container already has DRA claims,
// shared volume mount, and TMPDIR set. This function only handles engine duplication
// and failover-specific env vars.
//
// Non-main containers (e.g. frontend sidecar) are preserved in the final pod spec.
func buildFailoverPod(
	podSpec *corev1.PodSpec,
	numberOfNodes int32,
	backendFramework BackendFramework,
) error {
	if len(podSpec.Containers) == 0 {
		return fmt.Errorf("pod spec must have at least one container for failover transformation")
	}

	mainContainer := podSpec.Containers[0]
	sidecars := podSpec.Containers[1:]

	engines := make([]corev1.Container, failoverEngineCount)
	for i := range failoverEngineCount {
		engines[i] = buildEngineContainer(mainContainer, i, commonconsts.DynamoSystemPort+i)
	}

	podSpec.Containers = append(engines, sidecars...)

	// Backend-specific overrides
	switch backendFramework {
	case BackendFrameworkVLLM:
		applyVLLMOverrides(podSpec, numberOfNodes)
	default:
		return fmt.Errorf("failover is currently supported only for vLLM (detected: %s)", backendFramework)
	}

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
	removeSet := map[string]bool{
		"DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS": true,
		"DYN_SYSTEM_PORT":                       true,
		"DYN_SYSTEM_ENABLED":                    true,
		"DYN_HEALTH_CHECK_ENABLED":              true,
		"CONTAINER_NAME":                        true,
	}

	var filtered []corev1.EnvVar
	for _, env := range engine.Env {
		if !removeSet[env.Name] {
			filtered = append(filtered, env)
		}
	}

	containerName := fmt.Sprintf("engine-%d", engineID)
	failoverEnvs := []corev1.EnvVar{
		{Name: "ENGINE_ID", Value: strconv.Itoa(engineID)},
		{Name: "CONTAINER_NAME", Value: containerName},
		{Name: "FAILOVER_LOCK_PATH", Value: failoverLockFile},
		{Name: "DYN_SYSTEM_STARTING_HEALTH_STATUS", Value: "notready"},
		{Name: "DYN_SYSTEM_PORT", Value: strconv.Itoa(systemPort)},
		{Name: "DYN_SYSTEM_ENABLED", Value: "true"},
	}
	engine.Env = append(filtered, failoverEnvs...)

	// Retarget HTTP probes to this engine's named port. Each engine runs its
	// system server on a staggered port (e.g. 9090, 9091), and the probes
	// inherited from the base container still reference the original port name.
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

	return engine
}
