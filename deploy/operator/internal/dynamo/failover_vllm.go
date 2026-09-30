/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"strconv"
	"strings"

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
