/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
)

// EffectiveCyborgGPUCount returns the effective exact nvidia.com/gpu count
// consumed by hybrid Cyborg inventory. Limits-only resources are valid because
// Kubernetes defaults their request; an explicit request must equal its limit.
// resources is not mutated.
func EffectiveCyborgGPUCount(resources corev1.ResourceRequirements) (int, error) {
	// Select only the classic scalar projected by hybrid Cyborg inventory.
	gpuName := corev1.ResourceName(commonconsts.KubeResourceGPUNvidia)
	limit, hasLimit := resources.Limits[gpuName]
	request, hasRequest := resources.Requests[gpuName]

	// Leave DRA claims and unrelated scalar resources to their owning paths.
	if !hasLimit && !hasRequest {
		return 0, nil
	}

	// Validate an explicit limit before treating it as the effective request.
	limitCount := 0
	if hasLimit {
		limitValue := limit.Value()
		if limit.CmpInt64(limitValue) != 0 {
			err := fmt.Errorf("GPU resource %q quantity %q must be a whole number", gpuName, limit.String())
			return 0, fmt.Errorf("Cyborg GPU limit: %w", err)
		}
		limitCount = int(limitValue)
		if limitCount <= 0 {
			return 0, fmt.Errorf("Cyborg GPU limit %q must be positive, got %q", gpuName, limit.String())
		}
	}

	// Validate an explicit request and require Kubernetes extended-resource equality.
	if hasRequest {
		requestValue := request.Value()
		if request.CmpInt64(requestValue) != 0 {
			err := fmt.Errorf("GPU resource %q quantity %q must be a whole number", gpuName, request.String())
			return 0, fmt.Errorf("Cyborg GPU request: %w", err)
		}
		requestCount := int(requestValue)
		if requestCount <= 0 {
			return 0, fmt.Errorf("Cyborg GPU request %q must be positive, got %q", gpuName, request.String())
		}
		if !hasLimit {
			return 0, fmt.Errorf("Cyborg GPU request %q requires an equal limit", gpuName)
		}
		if request.Cmp(limit) != 0 {
			return 0, fmt.Errorf(
				"Cyborg GPU request %q must equal its limit, got request %q and limit %q",
				gpuName,
				request.String(),
				limit.String(),
			)
		}
	}

	// Kubernetes defaults a missing request from the validated limit.
	return limitCount, nil
}
