// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
)

const WorkerContainerName = "main"

func WorkerContainerIndex(containers []corev1.Container) (int, error) {
	if len(containers) == 0 {
		return -1, fmt.Errorf("pod spec has no containers")
	}
	if len(containers) == 1 {
		return 0, nil
	}

	workerIndex := -1
	for i, container := range containers {
		if container.Name != WorkerContainerName {
			continue
		}
		if workerIndex >= 0 {
			return -1, fmt.Errorf("pod spec has multiple %q containers", WorkerContainerName)
		}
		workerIndex = i
	}
	if workerIndex >= 0 {
		return workerIndex, nil
	}
	return -1, fmt.Errorf("pod spec has %d containers; expected one named %q", len(containers), WorkerContainerName)
}
