/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"strings"

	corev1 "k8s.io/api/core/v1"
)

const sglangDisablePiecewiseCudaGraphFlag = "--disable-piecewise-cuda-graph"

func applySGLangFailoverOverrides(podSpec *corev1.PodSpec) {
	for i := range podSpec.Containers {
		c := &podSpec.Containers[i]
		if !strings.HasPrefix(c.Name, "engine-") {
			continue
		}
		if containerCommandLineHasFlag(c, sglangDisablePiecewiseCudaGraphFlag) {
			continue
		}
		c.Args = append(c.Args, sglangDisablePiecewiseCudaGraphFlag)
	}
}
