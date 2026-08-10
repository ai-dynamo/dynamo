/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

const (
	vllmMasterPortFlag   = "--master-port"
	vllmMasterPortStride = 100
)

// applyVLLMOverrides injects vLLM-specific env vars into all engine containers.
// Port staggering (NIXL side channel, KV event, master port) prevents collisions
// between engines sharing the same pod network namespace.
// For multinode deployments, it also injects NNODES so engines know the group size.
func applyVLLMOverrides(podSpec *corev1.PodSpec, numberOfNodes int32) error {
	engineCount := 0
	for i := range podSpec.Containers {
		if strings.HasPrefix(podSpec.Containers[i].Name, "engine-") {
			engineCount++
		}
	}
	if engineCount == failoverEngineCount {
		applyLegacyVLLMOverrides(podSpec, numberOfNodes)
		return nil
	}

	usedPorts := make(map[int]string)
	for i := range podSpec.Containers {
		c := &podSpec.Containers[i]
		if !strings.HasPrefix(c.Name, "engine-") {
			continue
		}

		engineID, _ := strconv.Atoi(strings.TrimPrefix(c.Name, "engine-"))

		c.Env = append(c.Env,
			corev1.EnvVar{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "true"},
			corev1.EnvVar{Name: "VLLM_NIXL_SIDE_CHANNEL_PORT", Value: strconv.Itoa(5600 + engineID)},
			corev1.EnvVar{Name: "DYN_VLLM_KV_EVENT_PORT", Value: strconv.Itoa(20080 + engineID)},
		)

		masterPort, found, err := tokenizedVLLMMasterPort(c)
		if err != nil {
			return fmt.Errorf("%s: %w", c.Name, err)
		}
		if !found {
			masterPort = 29500
		}
		if engineCount > failoverEngineCount && !found && engineID > 0 && isShellWrapped(c) {
			return fmt.Errorf("%s: cannot inject a unique %s into a shell-wrapped command", c.Name, vllmMasterPortFlag)
		}
		masterPort += engineID * vllmMasterPortStride
		for _, candidate := range []struct {
			name string
			port int
		}{
			{name: "system", port: commonconsts.DynamoSystemPort + engineID},
			{name: "forward-pass metric", port: commonconsts.DynamoFPMBasePort + engineID},
			{name: "NIXL side channel", port: 5600 + engineID},
			{name: "KV event", port: 20080 + engineID},
			{name: "vLLM master", port: masterPort},
		} {
			if candidate.port < 1 || candidate.port > 65535 {
				return fmt.Errorf("%s %s port %d must be between 1 and 65535", c.Name, candidate.name, candidate.port)
			}
			if owner, ok := usedPorts[candidate.port]; ok {
				return fmt.Errorf("%s %s port %d collides with %s", c.Name, candidate.name, candidate.port, owner)
			}
			usedPorts[candidate.port] = c.Name + " " + candidate.name
		}
		if found || engineID > 0 {
			if err := setTokenizedVLLMMasterPort(c, masterPort, found); err != nil {
				return fmt.Errorf("%s: %w", c.Name, err)
			}
		}

		if numberOfNodes > 1 {
			c.Env = append(c.Env,
				corev1.EnvVar{Name: "NNODES", Value: strconv.Itoa(int(numberOfNodes))},
			)
		}
	}
	return nil
}

// applyLegacyVLLMOverrides preserves the two-engine command rewriting behavior,
// including shell-wrapped launch commands supported before two-shadow failover.
func applyLegacyVLLMOverrides(podSpec *corev1.PodSpec, numberOfNodes int32) {
	for i := range podSpec.Containers {
		c := &podSpec.Containers[i]
		if !strings.HasPrefix(c.Name, "engine-") {
			continue
		}

		engineID, _ := strconv.Atoi(strings.TrimPrefix(c.Name, "engine-"))

		c.Env = append(c.Env,
			corev1.EnvVar{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "true"},
			corev1.EnvVar{Name: "VLLM_NIXL_SIDE_CHANNEL_PORT", Value: strconv.Itoa(5600 + engineID)},
			corev1.EnvVar{Name: "DYN_VLLM_KV_EVENT_PORT", Value: strconv.Itoa(20080 + engineID)},
		)

		if engineID > 0 {
			if hasMasterPortFlag(c) {
				staggerMasterPort(c, engineID)
			} else {
				c.Args = append(c.Args, vllmMasterPortFlag, strconv.Itoa(29500+engineID*vllmMasterPortStride))
			}
		}

		if numberOfNodes > 1 {
			c.Env = append(c.Env,
				corev1.EnvVar{Name: "NNODES", Value: strconv.Itoa(int(numberOfNodes))},
			)
		}
	}
}

func hasMasterPortFlag(container *corev1.Container) bool {
	for _, arg := range container.Args {
		if arg == vllmMasterPortFlag || strings.Contains(arg, vllmMasterPortFlag+" ") {
			return true
		}
	}
	for _, cmd := range container.Command {
		if strings.Contains(cmd, vllmMasterPortFlag+" ") {
			return true
		}
	}
	return false
}

func staggerMasterPort(container *corev1.Container, engineID int) {
	staggerFlagValue(container, vllmMasterPortFlag, engineID*vllmMasterPortStride)
}

func staggerFlagValue(container *corev1.Container, flag string, offset int) {
	for i, arg := range container.Args {
		if arg == flag && i+1 < len(container.Args) {
			if port, err := strconv.Atoi(container.Args[i+1]); err == nil {
				container.Args[i+1] = strconv.Itoa(port + offset)
				return
			}
		}
	}

	for i, arg := range container.Args {
		if strings.Contains(arg, flag+" ") {
			parts := strings.Split(arg, flag+" ")
			if len(parts) < 2 {
				continue
			}
			var portString string
			for _, character := range parts[1] {
				if character < '0' || character > '9' {
					break
				}
				portString += string(character)
			}
			if port, err := strconv.Atoi(portString); err == nil {
				container.Args[i] = strings.Replace(arg, flag+" "+portString, flag+" "+strconv.Itoa(port+offset), 1)
				return
			}
		}
	}

	for i, command := range container.Command {
		if strings.Contains(command, flag+" ") {
			parts := strings.Split(command, flag+" ")
			if len(parts) < 2 {
				continue
			}
			var portString string
			for _, character := range parts[1] {
				if character < '0' || character > '9' {
					break
				}
				portString += string(character)
			}
			if port, err := strconv.Atoi(portString); err == nil {
				container.Command[i] = strings.Replace(command, flag+" "+portString, flag+" "+strconv.Itoa(port+offset), 1)
				return
			}
		}
	}
}

func tokenizedVLLMMasterPort(container *corev1.Container) (int, bool, error) {
	tokens := append(append([]string{}, container.Command...), container.Args...)
	value := ""
	found := false
	for i, token := range tokens {
		switch {
		case token == vllmMasterPortFlag:
			if found {
				return 0, false, fmt.Errorf("%s must appear at most once", vllmMasterPortFlag)
			}
			if i+1 >= len(tokens) || strings.HasPrefix(tokens[i+1], "--") {
				return 0, false, fmt.Errorf("%s requires a value", vllmMasterPortFlag)
			}
			value, found = tokens[i+1], true
		case strings.HasPrefix(token, vllmMasterPortFlag+"="):
			if found {
				return 0, false, fmt.Errorf("%s must appear at most once", vllmMasterPortFlag)
			}
			value, found = strings.TrimPrefix(token, vllmMasterPortFlag+"="), true
			if value == "" {
				return 0, false, fmt.Errorf("%s requires a value", vllmMasterPortFlag)
			}
		case strings.Contains(token, vllmMasterPortFlag):
			return 0, false, fmt.Errorf(
				"%s must use a separate argument or --master-port=PORT",
				vllmMasterPortFlag,
			)
		}
	}
	if !found {
		return 0, false, nil
	}
	port, err := strconv.Atoi(value)
	if err != nil || port < 1 || port > 65535 {
		return 0, false, fmt.Errorf("%s must be an integer between 1 and 65535", vllmMasterPortFlag)
	}
	return port, true, nil
}

func isShellWrapped(container *corev1.Container) bool {
	if len(container.Command) < 2 || container.Command[1] != "-c" {
		return false
	}
	switch filepath.Base(container.Command[0]) {
	case "sh", "bash":
		return true
	default:
		return false
	}
}

func setTokenizedVLLMMasterPort(container *corev1.Container, port int, found bool) error {
	if !found {
		container.Args = append(container.Args, vllmMasterPortFlag, strconv.Itoa(port))
		return nil
	}
	value := strconv.Itoa(port)
	for i, token := range container.Command {
		switch {
		case token == vllmMasterPortFlag && i+1 < len(container.Command):
			container.Command[i+1] = value
			return nil
		case token == vllmMasterPortFlag:
			container.Args[0] = value
			return nil
		case strings.HasPrefix(token, vllmMasterPortFlag+"="):
			container.Command[i] = vllmMasterPortFlag + "=" + value
			return nil
		}
	}
	updated, err := upsertTokenizedVLLMFlag(container.Args, vllmMasterPortFlag, value)
	if err != nil {
		return err
	}
	container.Args = updated
	return nil
}
