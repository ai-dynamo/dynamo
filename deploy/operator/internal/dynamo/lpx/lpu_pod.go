/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

const (
	lpuAgentContainerName       = "agent"
	defaultLPUAgentCPUs   int64 = 62
)

func updateWorkerPodSpec(podSpec *corev1.PodSpec) {
	// TODO can we remove some of this when we get the device-plugin?
	podSpec.HostIPC = true
	podSpec.HostNetwork = true
	podSpec.DNSPolicy = corev1.DNSClusterFirstWithHostNet
}

// ensureLPUNodeTolerations adds standard LPU-node tolerations to podSpec, which must be nonnil.
func ensureLPUNodeTolerations(podSpec *corev1.PodSpec) {
	// Derive each complete toleration from the platform's ordered taint keys.
	for _, key := range [...]string{"lpu.nvidia.com/node", "lpu.nvidia.com/lpu", "lpu.nvidia.com/node-v2"} {
		toleration := corev1.Toleration{Key: key, Operator: corev1.TolerationOpExists}
		if !slices.Contains(podSpec.Tolerations, toleration) {
			podSpec.Tolerations = append(podSpec.Tolerations, toleration)
		}
	}
}

func applyLPUWorkerContainerBase(container *corev1.Container) {
	// Workers need privileged device access and an interactive runtime session.
	container.SecurityContext = &corev1.SecurityContext{
		Privileged: ptr.To(true),
		RunAsUser:  ptr.To(int64(0)),
	}
	container.TTY = true
	container.Stdin = true

	// Agent scheduling already initialized device resources; add runtime CPU and hugepage requirements.
	cpuRequest := *resource.NewQuantity(defaultLPUAgentCPUs, resource.DecimalSI)
	if cpuLimit, ok := container.Resources.Limits[corev1.ResourceCPU]; ok && cpuLimit.Cmp(cpuRequest) < 0 {
		cpuRequest = cpuLimit.DeepCopy()
	}
	container.Resources.Requests[corev1.ResourceCPU] = cpuRequest
	container.Resources.Requests[corev1.ResourceHugePagesPrefix+"2Mi"] = resource.MustParse("4096Mi")
	container.Resources.Limits[corev1.ResourceHugePagesPrefix+"2Mi"] = resource.MustParse("4096Mi")
}
