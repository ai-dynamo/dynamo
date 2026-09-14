/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

const (
	lpuAgentContainerName       = "agent"
	defaultLPUAgentCPUs   int64 = 62
)

func validateLPUHostDeviceVolumes(podSpec *corev1.PodSpec) error {
	// Keep the existing compatibility checks without supplying missing device volumes.
	for _, volume := range podSpec.Volumes {
		var path string
		switch volume.Name {
		case "host-dev":
			path = "/dev"
		case "host-sys":
			path = "/sys"
		default:
			continue
		}
		hostPath := volume.HostPath
		if hostPath == nil || hostPath.Path != path ||
			hostPath.Type != nil && *hostPath.Type != corev1.HostPathUnset &&
				*hostPath.Type != corev1.HostPathDirectory && *hostPath.Type != corev1.HostPathDirectoryOrCreate {
			return fmt.Errorf("selected LPX podTemplate volume %q must use directory hostPath %q", volume.Name, path)
		}
	}
	return nil
}

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

func applyLPUWorkerContainerBase(container *corev1.Container, preserveCommand bool) {
	if !preserveCommand {
		container.Command = []string{"/bin/bash"}
	}
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
