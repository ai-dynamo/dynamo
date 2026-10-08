/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

const (
	vllmMasterPortFlag   = "--master-port"
	vllmMasterPortStride = 100
)

// buildColdStartFailoverPod prepares snapshot-less failover, supported only for
// vLLM with GMS V0. Engines initialize concurrently in shadow mode and need
// separate collective/KV ports, unlike the restored snapshot engine pair.
// Mutates podSpec, which must not be nil and must already have GMS resources.
func buildColdStartFailoverPod(podSpec *corev1.PodSpec, numberOfNodes int32, backendFramework BackendFramework) error {
	// Reject other backends before changing the pod.
	if backendFramework != BackendFrameworkVLLM {
		return fmt.Errorf("cold-start failover is currently supported only for vLLM (detected: %s)", backendFramework)
	}
	if err := buildFailoverEnginePair(podSpec); err != nil {
		return err
	}

	// Only the two cloned engines receive legacy shadow initialization settings.
	for engineID := range failoverEngineCount {
		c := &podSpec.Containers[engineID]
		c.Env = append(c.Env, corev1.EnvVar{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "true"})
		c.Env = append(c.Env,
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
	return nil
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

	// Preserve Args precedence when looking inside shell-wrapped launch strings.
	if !staggerEmbeddedFlagValue(container.Args, flag, offset) {
		staggerEmbeddedFlagValue(container.Command, flag, offset)
	}
}

// staggerEmbeddedFlagValue offsets the first embedded flag value it can parse.
func staggerEmbeddedFlagValue(tokens []string, flag string, offset int) bool {
	for i, token := range tokens {
		parts := strings.Split(token, flag+" ")
		if len(parts) < 2 {
			continue
		}

		// Read only the integer prefix, leaving trailing launch arguments intact.
		var portStr string
		for _, ch := range parts[1] {
			if ch < '0' || ch > '9' {
				break
			}
			portStr += string(ch)
		}
		if port, err := strconv.Atoi(portStr); err == nil {
			tokens[i] = strings.Replace(token, flag+" "+portStr, flag+" "+strconv.Itoa(port+offset), 1)
			return true
		}
	}
	return false
}
