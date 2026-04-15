// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
)

const checkpointWorkerContainerName = "main"

func ResolveCheckpointWorkerContainer(podSpec *corev1.PodSpec) (*corev1.Container, error) {
	if podSpec == nil || len(podSpec.Containers) == 0 {
		return nil, fmt.Errorf("checkpoint job requires at least one container")
	}
	if len(podSpec.Containers) == 1 {
		return &podSpec.Containers[0], nil
	}

	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == checkpointWorkerContainerName {
			return &podSpec.Containers[i], nil
		}
	}

	return nil, fmt.Errorf("checkpoint job requires a container named %q when multiple containers are present", checkpointWorkerContainerName)
}
