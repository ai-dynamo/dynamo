/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// Components returns LPX components in authored order without mutating dgd.
// dgd must be non-nil.
// Returned component pointers remain read-only.
func Components(dgd *dynamov1beta1.DynamoGraphDeployment) []*dynamov1beta1.DynamoComponentDeploymentSharedSpec {
	components := make([]*dynamov1beta1.DynamoComponentDeploymentSharedSpec, 0, len(dgd.Spec.Components))
	for index := range dgd.Spec.Components {
		component := &dgd.Spec.Components[index]
		if component.IsLPX() {
			components = append(components, component)
		}
	}
	return components
}

// ComponentGroups returns workload membership keyed by conductor component name.
// Each conductor owns a group. Admission permits an agent-only component only
// when it shares the graph's sole conductor. Member names are sorted.
// dgd must be non-nil and have passed admission; it is not mutated.
func ComponentGroups(dgd *dynamov1beta1.DynamoGraphDeployment) map[string][]string {
	components := Components(dgd)
	groups := make(map[string][]string)

	for _, component := range components {
		groupName := component.ComponentName

		// Admission makes the shared conductor unambiguous for agent-only members.
		if component.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor) == nil {
			for _, targetComponent := range components {
				if targetComponent.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor) != nil {
					groupName = targetComponent.ComponentName
					break
				}
			}
		}

		groups[groupName] = append(groups[groupName], component.ComponentName)
	}

	// Keep membership deterministic without changing the authored component order.
	for _, members := range groups {
		slices.Sort(members)
	}

	return groups
}
