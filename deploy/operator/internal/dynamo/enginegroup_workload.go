/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"crypto/sha256"
	"fmt"
	"strconv"
	"strings"

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
	case BackendFrameworkVLLM:
		return renderVLLMComponentEngineGroup(dgd, component, worldIndex)
	default:
		return nil, fmt.Errorf("Engine Group declaration is not implemented for backend %q", backend)
	}
}

// EngineGroupNameForComponent returns the stable name of one indexed world, independent of workload revisions.
// worldIndex must be nonnegative and identifies a world, not a capacity slot within it.
func EngineGroupNameForComponent(dgdName, componentName string, worldIndex int32) string {
	name := strings.ToLower(dgdName + "-" + componentName + "-" + strconv.FormatInt(int64(worldIndex), 10))
	if len(name) <= 63 {
		return name
	}

	// Keep names usable as label values without losing uniqueness after truncation.
	digest := sha256.Sum256([]byte(name))
	return fmt.Sprintf("%s-%x", strings.TrimRight(name[:50], "-."), digest[:6])
}
