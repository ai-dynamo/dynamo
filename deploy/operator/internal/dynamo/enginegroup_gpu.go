/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// resolveEngineGroupGPUAllocation proves static GPU limits belong exclusively to the main container.
// podSpec and main must not be nil; main must be the Pod's main container.
func resolveEngineGroupGPUAllocation(podSpec *corev1.PodSpec, main *corev1.Container) (int64, error) {
	// Current native launch profiles require Pod networking and declarative scalar GPU geometry.
	if podSpec.HostNetwork {
		return 0, fmt.Errorf("Engine Groups require Pod networking")
	}
	if len(podSpec.ResourceClaims) != 0 || len(main.Resources.Claims) != 0 {
		return 0, fmt.Errorf("Engine Groups do not yet support DRA GPU allocation")
	}
	containers := append(append([]corev1.Container(nil), podSpec.InitContainers...), podSpec.Containers...)
	for _, container := range containers {
		if container.Name == consts.MainContainerName {
			continue
		}
		gpuLimit := container.Resources.Limits[consts.KubeResourceGPUNvidia]
		if !gpuLimit.IsZero() || len(container.Resources.Claims) != 0 {
			return 0, fmt.Errorf("Engine Group GPUs must belong exclusively to the main container")
		}
	}

	// A fixed allocation requires a positive integral main-container GPU limit.
	limit := main.Resources.Limits[consts.KubeResourceGPUNvidia]
	gpus, exact := limit.AsInt64()
	if !exact || gpus < 1 {
		return 0, fmt.Errorf("Engine Group main container requires a positive whole-GPU limit")
	}
	return gpus, nil
}
