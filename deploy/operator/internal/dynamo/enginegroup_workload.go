/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"

	api "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// RenderComponentEngineGroup selects the backend and declares a new logical world.
// dgd, component and component.EngineGroup must not be nil. Inputs are not mutated.
func RenderComponentEngineGroup(dgd *api.DynamoGraphDeployment, component *api.DynamoComponentDeploymentSharedSpec, worldIndex int32) (*api.DynamoGraphDeploymentEngineGroup, error) {
	// Backend-owned validation precedes declaration; the lifecycle controller consumes only the resulting CR.
	backend, err := getBackendFrameworkFromComponent(component, dgd)
	if err != nil {
		return nil, err
	}
	switch backend {
	default:
		return nil, fmt.Errorf("Engine Group declaration is not implemented for backend %q", backend)
	}
}
