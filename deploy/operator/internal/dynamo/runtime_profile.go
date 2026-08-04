/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	runtimefeatures "github.com/ai-dynamo/dynamo/deploy/operator/internal/features/runtime"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
)

// runtimeProfileForComponent resolves and evaluates the component's runtime gates.
func runtimeProfileForComponent(component *v1beta1.DynamoComponentDeploymentSharedSpec) runtimefeatures.RuntimeProfile {
	// Keep legacy behavior when no component inputs are available.
	if component == nil {
		return runtimefeatures.RuntimeProfile{}
	}

	// Read the effective main image used by runtime version resolution.
	image := ""
	if main := GetMainContainer(component); main != nil {
		image = main.Image
	}

	// Unresolvable legacy images keep every runtime-dependent default disabled.
	version, err := runtimeversion.Resolve(image, component.RuntimeVersionOverride)
	if err != nil {
		return runtimefeatures.RuntimeProfile{}
	}

	// Evaluate the centralized gate set for the resolved compatibility version.
	return runtimefeatures.ProfileForVersion(&version)
}
