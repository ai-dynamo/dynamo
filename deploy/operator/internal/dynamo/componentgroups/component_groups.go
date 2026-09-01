/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package componentgroups provides indexed access to component-group declarations.
package componentgroups

import (
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// ComponentGroups is the set of component groups declared by a DGD.
type ComponentGroups struct {
	groups            []v1beta1.ComponentGroupSpec
	groupsByName      map[string]*v1beta1.ComponentGroupSpec
	componentsToGroup map[string]string
}

// New returns the declared component groups, or an empty set when experimental settings are absent.
func New(experimental *v1beta1.DynamoGraphDeploymentExperimentalSpec) *ComponentGroups {
	if experimental == nil {
		return nil
	}

	groupsByName := make(map[string]*v1beta1.ComponentGroupSpec, len(experimental.ComponentGroups))
	componentsToGroup := make(map[string]string)

	for i := range experimental.ComponentGroups {
		group := &experimental.ComponentGroups[i]
		groupsByName[group.Name] = group
		for _, component := range group.Components {
			componentsToGroup[component.Name] = group.Name
		}
	}

	return &ComponentGroups{groups: experimental.ComponentGroups, groupsByName: groupsByName, componentsToGroup: componentsToGroup}
}

// Groups returns the list of component groups.
func (groups *ComponentGroups) Groups() []v1beta1.ComponentGroupSpec {
	if groups == nil {
		return nil
	}

	return groups.groups
}

// Group returns the component group by name, if it exists.
func (groups *ComponentGroups) Group(name string) (v1beta1.ComponentGroupSpec, bool) {
	group, ok := groups.groupsByName[name]
	if ok {
		return *group, true
	}
	return v1beta1.ComponentGroupSpec{}, false
}

// GroupNameForComponent returns the component group that owns a component, if any.
func (groups *ComponentGroups) GroupNameForComponent(componentName string) (string, bool) {
	if groups == nil {
		return "", false
	}

	group, ok := groups.GroupForComponent(componentName)
	return group.Name, ok
}

// IsGrouped reports whether a component belongs to any component group.
func (groups *ComponentGroups) IsGrouped(componentName string) bool {
	_, ok := groups.GroupNameForComponent(componentName)
	return ok
}

// HasGroup reports whether a group exists, matching names case-insensitively.
func (groups *ComponentGroups) HasGroup(groupName string) bool {
	if groups == nil {
		return false
	}

	_, ok := groups.groupsByName[groupName]
	return ok
}

// GroupForComponent returns the component group that owns a component, if any.
// Admission validation guarantees at most one group per component.
func (groups *ComponentGroups) GroupForComponent(componentName string) (v1beta1.ComponentGroupSpec, bool) {
	if groups == nil {
		return v1beta1.ComponentGroupSpec{}, false
	}

	if groupName := groups.componentsToGroup[componentName]; groupName != "" {
		return *groups.groupsByName[groupName], true
	}

	return v1beta1.ComponentGroupSpec{}, false
}
