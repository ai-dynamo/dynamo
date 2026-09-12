// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package features

import (
	"context"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	groveschedulerv1alpha1 "github.com/ai-dynamo/grove/scheduler/api/core/v1alpha1"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
)

// LPX enables LPX scheduler integration.
//
// Owner: @andrewpaprotsky
// Experimental since: N/A
// Beta since: N/A
// GA since: N/A
// Configuration: lpx.enabled
// Auto-detection: N/A; API availability is verified when explicitly enabled
// Requires: LPX Scheduler serving scheduling.lpu.nvidia.com/v1alpha1
// LpuPipelineRequest resources and scheduler.grove.io/v1alpha1 PodGang resources
// Default: false
const LPX Name = "lpx"

// resolveLPX verifies API availability for explicitly enabled LPX integration.
// A nil config retains the native discovery-unavailable error.
func resolveLPX(ctx context.Context, config *rest.Config) (bool, error) {
	// Discover the exact request resource before resolving the enabled gate.
	resource := lpxv1alpha1.GroupVersion.WithResource("lpupipelinerequests")
	available, err := detectAPIAvailability(ctx, config, resource.Group, resource.Version, resource.Resource)
	if err != nil {
		return false, err
	}

	if _, err := resolve(ptr.To(true), available,
		"LPX is explicitly enabled in config but the scheduling.lpu.nvidia.com/v1alpha1 LpuPipelineRequest API was not detected in the cluster"); err != nil {
		return false, err
	}

	// LPX also watches scheduler-scoped PodGangs, which external Grove installs may omit.
	resource = groveschedulerv1alpha1.SchemeGroupVersion.WithResource("podgangs")
	available, err = detectAPIAvailability(ctx, config, resource.Group, resource.Version, resource.Resource)
	if err != nil {
		return false, err
	}

	return resolve(ptr.To(true), available,
		"LPX is explicitly enabled in config but the scheduler.grove.io/v1alpha1 PodGang API was not detected in the cluster")
}
