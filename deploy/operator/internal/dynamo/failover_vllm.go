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

	twoShadowIntraPodFailoverProfileMessage = "two shadows currently require a single-node direct vLLM launch without Ray or data parallel"
)

type vllmEnginePorts struct {
	master int
	system int
	fpm    int
	nixl   int
	kv     int
}

// TwoShadowIntraPodFailoverProfileError validates the deliberately narrow
// launch profile whose listeners the operator can isolate across three engines.
func TwoShadowIntraPodFailoverProfileError(
	container *corev1.Container,
	numberOfNodes int32,
	backendFramework string,
	vllmExecutorBackend string,
) error {
	if backendFramework == "" && isDirectVLLMLaunch(container) {
		backendFramework = string(BackendFrameworkVLLM)
	}
	if numberOfNodes != 1 ||
		!strings.EqualFold(backendFramework, string(BackendFrameworkVLLM)) ||
		!isDirectVLLMLaunch(container) ||
		strings.EqualFold(vllmExecutorBackend, "ray") {
		return fmt.Errorf("%s", twoShadowIntraPodFailoverProfileMessage)
	}

	tokens := append(append([]string{}, container.Command...), container.Args...)
	for i, token := range tokens {
		if token == enableElasticEPFlag || strings.HasPrefix(token, enableElasticEPFlag+"=") ||
			strings.HasPrefix(token, "--data-parallel-") {
			return fmt.Errorf("%s", twoShadowIntraPodFailoverProfileMessage)
		}
		if token == distributedExecutorFlag {
			if i+1 < len(tokens) && strings.EqualFold(tokens[i+1], "ray") {
				return fmt.Errorf("%s", twoShadowIntraPodFailoverProfileMessage)
			}
		} else if strings.EqualFold(token, distributedExecutorFlag+"=ray") {
			return fmt.Errorf("%s", twoShadowIntraPodFailoverProfileMessage)
		}
	}
	return nil
}

func isDirectVLLMLaunch(container *corev1.Container) bool {
	if container == nil {
		return false
	}
	tokens := append(append([]string{}, container.Command...), container.Args...)
	return len(tokens) >= 3 &&
		isPythonExecutable(tokens[0]) &&
		tokens[1] == "-m" &&
		tokens[2] == "dynamo.vllm"
}

func isPythonExecutable(command string) bool {
	name := filepath.Base(command)
	if !strings.HasPrefix(name, "python") {
		return false
	}
	suffix := strings.TrimPrefix(name, "python")
	if suffix == "" {
		return true
	}
	for _, part := range strings.Split(suffix, ".") {
		if part == "" {
			return false
		}
		for _, r := range part {
			if r < '0' || r > '9' {
				return false
			}
		}
	}
	return true
}

func threeEngineVLLMPorts(container *corev1.Container) ([]vllmEnginePorts, error) {
	masterBase, found, err := tokenizedVLLMMasterPort(container)
	if err != nil {
		return nil, err
	}
	if !found {
		masterBase = 29500
	}

	const engineCount = 3
	ports := make([]vllmEnginePorts, engineCount)
	used := make(map[int]string, engineCount*5)
	for i := range ports {
		ports[i] = vllmEnginePorts{
			master: masterBase + i*vllmMasterPortStride,
			system: commonconsts.DynamoSystemPort + i,
			fpm:    commonconsts.DynamoFPMBasePort + i,
			nixl:   5600 + i,
			kv:     20080 + i,
		}
		for _, candidate := range []struct {
			name string
			port int
		}{
			{name: "master", port: ports[i].master},
			{name: "system", port: ports[i].system},
			{name: "FPM", port: ports[i].fpm},
			{name: "NIXL", port: ports[i].nixl},
			{name: "KV", port: ports[i].kv},
		} {
			if candidate.port < 1 || candidate.port > 65535 {
				return nil, fmt.Errorf("invalid vLLM %s port %d for engine-%d: must be between 1 and 65535", candidate.name, candidate.port, i)
			}
			if owner, exists := used[candidate.port]; exists {
				return nil, fmt.Errorf("vLLM port %d for engine-%d %s collides with %s", candidate.port, i, candidate.name, owner)
			}
			used[candidate.port] = fmt.Sprintf("engine-%d %s", i, candidate.name)
		}
	}
	return ports, nil
}

func tokenizedVLLMMasterPort(container *corev1.Container) (int, bool, error) {
	if container == nil {
		return 0, false, fmt.Errorf("container is nil")
	}
	var values []string
	tokens := append(append([]string{}, container.Command...), container.Args...)
	for i, token := range tokens {
		switch {
		case token == vllmMasterPortFlag:
			if i+1 >= len(tokens) {
				return 0, false, fmt.Errorf("%s requires a value", vllmMasterPortFlag)
			}
			values = append(values, tokens[i+1])
		case strings.HasPrefix(token, vllmMasterPortFlag+"="):
			values = append(values, strings.TrimPrefix(token, vllmMasterPortFlag+"="))
		}
	}
	if len(values) == 0 {
		return 0, false, nil
	}
	if len(values) > 1 {
		return 0, false, fmt.Errorf("%s appears more than once", vllmMasterPortFlag)
	}
	port, err := strconv.Atoi(values[0])
	if err != nil {
		return 0, false, fmt.Errorf("invalid vLLM %s value %q: %w", vllmMasterPortFlag, values[0], err)
	}
	return port, true, nil
}

func setTokenizedVLLMMasterPort(container *corev1.Container, port int) {
	value := strconv.Itoa(port)
	for i, token := range container.Command {
		switch {
		case token == vllmMasterPortFlag && i+1 < len(container.Command):
			container.Command[i+1] = value
			return
		case token == vllmMasterPortFlag:
			container.Args[0] = value
			return
		case strings.HasPrefix(token, vllmMasterPortFlag+"="):
			container.Command[i] = vllmMasterPortFlag + "=" + value
			return
		}
	}
	for i, token := range container.Args {
		switch {
		case token == vllmMasterPortFlag:
			container.Args[i+1] = value
			return
		case strings.HasPrefix(token, vllmMasterPortFlag+"="):
			container.Args[i] = vllmMasterPortFlag + "=" + value
			return
		}
	}
	container.Args = append(container.Args, vllmMasterPortFlag, value)
}

func applyThreeEngineVLLMOverrides(podSpec *corev1.PodSpec, ports []vllmEnginePorts) {
	for i := range ports {
		engine := &podSpec.Containers[i]
		setTokenizedVLLMMasterPort(engine, ports[i].master)
		engine.Env = MergeEnvs(engine.Env, []corev1.EnvVar{
			corev1.EnvVar{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "true"},
			corev1.EnvVar{Name: "VLLM_NIXL_SIDE_CHANNEL_PORT", Value: strconv.Itoa(ports[i].nixl)},
			corev1.EnvVar{Name: "DYN_VLLM_KV_EVENT_PORT", Value: strconv.Itoa(ports[i].kv)},
		})
	}
}

// applyVLLMOverrides injects vLLM-specific env vars into all engine containers.
// Port staggering (NIXL side channel, KV event, master port) prevents collisions
// between engines sharing the same pod network namespace.
// For multinode deployments, it also injects NNODES so engines know the group size.
func applyVLLMOverrides(podSpec *corev1.PodSpec, numberOfNodes int32) {
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

		// Stagger --master-port for TP so each engine group uses a distinct
		// torch.distributed TCP store. engine-0 keeps the default (29500),
		// engine-1 gets 29500 + stride.
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

// hasMasterPortFlag checks if --master-port appears in the container args or command.
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
	offset := engineID * vllmMasterPortStride
	staggerFlagValue(container, vllmMasterPortFlag, offset)
}

// staggerFlagValue finds a --flag VALUE pair in container args and adds offset
// to the integer value. Handles both separate-token args (["--flag", "29500"])
// and shell-wrapped args (["sh", "-c", "... --flag 29500 ..."]).
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
			var portStr string
			for _, ch := range parts[1] {
				if ch >= '0' && ch <= '9' {
					portStr += string(ch)
				} else {
					break
				}
			}
			if port, err := strconv.Atoi(portStr); err == nil {
				container.Args[i] = strings.Replace(arg, flag+" "+portStr, flag+" "+strconv.Itoa(port+offset), 1)
				return
			}
		}
	}

	for i, cmd := range container.Command {
		if strings.Contains(cmd, flag+" ") {
			parts := strings.Split(cmd, flag+" ")
			if len(parts) < 2 {
				continue
			}
			var portStr string
			for _, ch := range parts[1] {
				if ch >= '0' && ch <= '9' {
					portStr += string(ch)
				} else {
					break
				}
			}
			if port, err := strconv.Atoi(portStr); err == nil {
				container.Command[i] = strings.Replace(cmd, flag+" "+portStr, flag+" "+strconv.Itoa(port+offset), 1)
				return
			}
		}
	}
}
