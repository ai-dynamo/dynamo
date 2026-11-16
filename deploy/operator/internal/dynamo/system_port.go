// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// configureSystemPort aligns a merged runtime container with its authored override.
// container must be non-nil; authored may be nil when no template was supplied.
func configureSystemPort(container, authored *corev1.Container, hostNetwork bool) error {
	// Leave defaults and intentionally cleared declarations unchanged.
	if authored == nil || findEnvVar(authored.Env, "DYN_SYSTEM_PORT") == nil {
		return nil
	}
	if authored.Ports != nil && len(authored.Ports) == 0 {
		container.Ports = []corev1.ContainerPort{}
		return nil
	}

	// Sourced values require a scheduling-time declaration, not a guessed default.
	env := findEnvVar(container.Env, "DYN_SYSTEM_PORT")
	var port int32
	if env.ValueFrom != nil || strings.Contains(env.Value, "$(") {
		declared := findContainerPort(authored, commonconsts.DynamoSystemPortName)
		if declared == nil || declared.ContainerPort < 1 || declared.ContainerPort > 32767 {
			return fmt.Errorf("DYN_SYSTEM_PORT from valueFrom or environment expansion requires an explicit system container port between 1 and 32767")
		}
		port = declared.ContainerPort
	} else {
		var err error
		port, err = parseSystemPort(env.Value)
		if err != nil {
			return fmt.Errorf("invalid DYN_SYSTEM_PORT: %w", err)
		}
	}

	// Disabled and ephemeral listeners cannot back generated fixed-port probes.
	if port <= 0 {
		if findContainerPort(authored, commonconsts.DynamoSystemPortName) == nil {
			container.Ports = slices.DeleteFunc(container.Ports, func(p corev1.ContainerPort) bool {
				return p.Name == commonconsts.DynamoSystemPortName
			})
		}
		if authored.StartupProbe == nil {
			container.StartupProbe = nil
		}
		if authored.ReadinessProbe == nil {
			container.ReadinessProbe = nil
		}
		if authored.LivenessProbe == nil {
			container.LivenessProbe = nil
		}
		return nil
	}

	// Keep named references and existing host reservations on the actual listener.
	declared := findContainerPort(container, commonconsts.DynamoSystemPortName)
	if declared == nil {
		container.Ports = append(container.Ports, corev1.ContainerPort{Name: commonconsts.DynamoSystemPortName, Protocol: corev1.ProtocolTCP})
		declared = &container.Ports[len(container.Ports)-1]
	}
	declared.ContainerPort = port
	if hostNetwork {
		declared.HostPort = port
	}
	return nil
}

// parseSystemPort accepts the runtime's signed port range, including disabled and ephemeral listeners.
func parseSystemPort(value string) (int32, error) {
	value = strings.TrimSpace(value)
	radix := 10
	if strings.HasPrefix(value, "0x") || strings.HasPrefix(value, "0X") {
		radix = 16
		value = value[2:]
	}
	port, err := strconv.ParseInt(value, radix, 16)
	return int32(port), err
}
