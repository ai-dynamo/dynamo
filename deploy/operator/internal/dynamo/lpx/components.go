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
// the role whose Dynamo runtime sidecar, the init container named "runtime",
// serves it: the conductor role when component declares one, otherwise the
// agent role. Beside a conductor that declares the sidecar, the agent role may
// declare one too, only to publish topology; see PublishesTopology. component
// must be non-nil.
func RuntimeSidecarRole(component *dynamov1beta1.DynamoComponentDeploymentSharedSpec) string {
	if component.ComponentRole(dynamov1beta1.ComponentRoleLPXConductor) != nil {
		return dynamov1beta1.ComponentRoleLPXConductor
	}
	return dynamov1beta1.ComponentRoleLPXAgent
}

// roleRuntimeSidecar returns the "runtime" init container of the named role's
// template, or nil when component lacks the role, its template, or the sidecar.
// component must be non-nil; the result points into component.
func roleRuntimeSidecar(component *dynamov1beta1.DynamoComponentDeploymentSharedSpec, name string) *corev1.Container {
	role := component.ComponentRole(name)
	if role == nil || role.PodTemplate == nil {
		return nil
	}
	return common.FindContainerByName(role.PodTemplate.Spec.InitContainers, commonconsts.RuntimeContainerName)
}

// agentRuntimeSidecar returns the runtime sidecar through which component serves
// from its agent Pods: the "runtime" init container of the agent role template
// of a component without a conductor role. It returns nil when component has a
// conductor role or its agent template declares no runtime sidecar. component
// must be non-nil; the result points into component.
func agentRuntimeSidecar(component *dynamov1beta1.DynamoComponentDeploymentSharedSpec) *corev1.Container {
	if RuntimeSidecarRole(component) != dynamov1beta1.ComponentRoleLPXAgent {
		return nil
	}
	return roleRuntimeSidecar(component, dynamov1beta1.ComponentRoleLPXAgent)
}

// PublishesTopology reports whether component declares the LPX topology roles:
// its conductor and agent role templates both declare the "runtime" sidecar.
// The conductor's sidecar still fronts the engine; the agent's sidecar only
// publishes the topology of the Agent allocation and never serves an endpoint,
// so it changes neither ServesEndpoint nor ComponentGroups. Reconciliation
// accepts the pair only for a hybrid build. component must be non-nil.
func PublishesTopology(component *dynamov1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return roleRuntimeSidecar(component, dynamov1beta1.ComponentRoleLPXConductor) != nil &&
		roleRuntimeSidecar(component, dynamov1beta1.ComponentRoleLPXAgent) != nil
}

// RuntimeSidecarPermitted reports whether the named role of component may
// declare the Dynamo "runtime" sidecar: the role that hosts the engine, or the
// agent role beside a conductor that declares one, to publish topology.
// component must be non-nil.
func RuntimeSidecarPermitted(component *dynamov1beta1.DynamoComponentDeploymentSharedSpec, role string) bool {
	return role == RuntimeSidecarRole(component) ||
		role == dynamov1beta1.ComponentRoleLPXAgent && PublishesTopology(component)
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
