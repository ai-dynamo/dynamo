/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"k8s.io/utils/ptr"
)

const (
	draftStageName     = "draft"
	runtimeModelDraft  = "draft"
	runtimeModelTarget = "target"
)

// Components returns LPX components. In a supported pair, the worker-only draft
// precedes the serving component. dgd is non-nil and is not mutated.
// Returned component pointers remain read-only.
func Components(dgd *dynamov1beta1.DynamoGraphDeployment) []*dynamov1beta1.DynamoComponentDeploymentSharedSpec {
	components := make([]*dynamov1beta1.DynamoComponentDeploymentSharedSpec, 0, 2)
	for index := range dgd.Spec.Components {
		component := &dgd.Spec.Components[index]
		if component.IsLPX() {
			components = append(components, component)
		}
	}
	if len(components) == 2 && components[0].ComponentRole(dynamov1beta1.ComponentRoleLeader) != nil {
		components[0], components[1] = components[1], components[0]
	}
	return components
}

// ServingComponent returns the sole LPX component or the component declaring
// the shared leader role. It returns nil when a pair has no leader owner.
// dgd is non-nil and the returned component is read-only.
func ServingComponent(dgd *dynamov1beta1.DynamoGraphDeployment) *dynamov1beta1.DynamoComponentDeploymentSharedSpec {
	var soleComponent *dynamov1beta1.DynamoComponentDeploymentSharedSpec
	componentCount := 0
	for index := range dgd.Spec.Components {
		component := &dgd.Spec.Components[index]
		if !component.IsLPX() {
			continue
		}
		if component.ComponentRole(dynamov1beta1.ComponentRoleLeader) != nil {
			return component
		}
		soleComponent = component
		componentCount++
	}
	if componentCount == 1 {
		return soleComponent
	}
	return nil
}

// SelectedModelNames returns build-backed runtime model identities, in canonical
// stage and draft-instance order, for a supported runtime.
// dgd must be non-nil and is not mutated.
func SelectedModelNames(dgd *dynamov1beta1.DynamoGraphDeployment) ([]string, error) {
	components := Components(dgd)
	if err := validateLPXComposition(dgd, len(components)).ToAggregate(); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrUnsupportedRuntime, err)
	}
	var names []string
	for index, component := range components {
		names = append(names, expandedSelectedModelNames(index, len(components), int(ptr.Deref(component.Replicas, 1)))...)
	}
	return names, nil
}

func expandedSelectedModelNames(index, componentCount, draftCount int) []string {
	// Component membership retains Nova's existing default/draft/target identities.
	if componentCount == 1 {
		return []string{"default"}
	}
	if index == 1 {
		return []string{runtimeModelTarget}
	}
	names := make([]string, draftCount)
	for index := range names {
		names[index] = fmt.Sprintf("draft%d", index)
	}
	return names
}
