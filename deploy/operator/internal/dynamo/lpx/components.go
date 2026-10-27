/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
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

// RuntimeSidecarRole names the role whose Pods host component's engine, and so
// the only role whose template may declare the Dynamo runtime sidecar, the init
// container named "runtime": the conductor role when component declares one,
// otherwise the agent role. component must be non-nil.
func RuntimeSidecarRole(component *dynamov1beta1.DynamoComponentDeploymentSharedSpec) string {
	if component.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor) != nil {
		return dynamov1beta1.ComponentRoleLPXConductor
	}
	return dynamov1beta1.ComponentRoleLPXAgent
}

// agentRuntimeSidecar returns the runtime sidecar through which component serves
// from its agent Pods: the "runtime" init container of the agent role template
// of a component without a conductor role. It returns nil when component has a
// conductor role or its agent template declares no runtime sidecar. component
// must be non-nil; the result points into component.
func agentRuntimeSidecar(component *dynamov1beta1.DynamoComponentDeploymentSharedSpec) *corev1.Container {
	agent := component.ComponentRole(dynamov1beta1.ComponentRoleLPXAgent)
	if RuntimeSidecarRole(component) != dynamov1beta1.ComponentRoleLPXAgent || agent == nil || agent.PodTemplate == nil {
		return nil
	}
	return common.FindContainerByName(agent.PodTemplate.Spec.InitContainers, commonconsts.RuntimeContainerName)
}

// ServesEndpoint reports whether component serves its own endpoint, through a
// conductor role or through its agent runtime sidecar. Admission accepts an LPX
// component that does neither only as the shared draft of a target with a
// conductor role. component must be non-nil.
func ServesEndpoint(component *dynamov1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return component.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor) != nil || agentRuntimeSidecar(component) != nil
}

// ComponentGroups returns workload membership keyed by serving component name.
// Each component that serves its own endpoint owns a group. Admission permits
// a component that does not only when it shares the graph's sole conductor.
// Member names are sorted. dgd must be non-nil and have passed admission; it
// is not mutated.
func ComponentGroups(dgd *dynamov1beta1.DynamoGraphDeployment) map[string][]string {
	components := Components(dgd)
	groups := make(map[string][]string)

	for _, component := range components {
		groupName := component.ComponentName

		// Admission makes the shared conductor unambiguous for agent-only members.
		if !ServesEndpoint(component) {
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
