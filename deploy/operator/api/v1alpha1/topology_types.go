/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package v1alpha1

import (
	"fmt"
)

const (
	// ConditionTypeTopologyConstraintsEnforced indicates whether topology constraints
	// are being actively enforced by the underlying framework.
	ConditionTypeTopologyConstraintsEnforced = "TopologyConstraintsEnforced"

	// ConditionReasonAllTopologyLevelsAvailable indicates all required topology levels
	// are available in the cluster topology.
	ConditionReasonAllTopologyLevelsAvailable = "AllTopologyLevelsAvailable"
	// ConditionReasonTopologyLevelsUnavailable indicates one or more required topology
	// levels are no longer available.
	ConditionReasonTopologyLevelsUnavailable = "TopologyLevelsUnavailable"
	// ConditionReasonTopologyDefinitionNotFound indicates the topology definition
	// resource was not found by the framework.
	ConditionReasonTopologyDefinitionNotFound = "TopologyDefinitionNotFound"
)

// TopologyConstraint defines topology placement requirements for a deployment or service.
type TopologyConstraint struct {
	// PackDomain specifies the topology domain for grouping replicas.
	// Must be one of: region, zone, datacenter, block, rack, host, numa
	PackDomain TopologyDomain `json:"packDomain"`
}

// TopologyDomain represents a level in the topology hierarchy.
// These are Dynamo's own abstract vocabulary — they align with Grove today
// because both use natural, user-friendly topology terms. For non-Grove
// backends, a translation layer maps these to framework-specific values.
// +kubebuilder:validation:Enum=region;zone;datacenter;block;rack;host;numa
type TopologyDomain string

const (
	// TopologyDomainRegion represents the region level (broadest).
	TopologyDomainRegion TopologyDomain = "region"
	// TopologyDomainZone represents the zone level.
	TopologyDomainZone TopologyDomain = "zone"
	// TopologyDomainDataCenter represents the datacenter level.
	TopologyDomainDataCenter TopologyDomain = "datacenter"
	// TopologyDomainBlock represents the block level.
	TopologyDomainBlock TopologyDomain = "block"
	// TopologyDomainRack represents the rack level.
	TopologyDomainRack TopologyDomain = "rack"
	// TopologyDomainHost represents the host level.
	TopologyDomainHost TopologyDomain = "host"
	// TopologyDomainNuma represents the numa level (narrowest).
	TopologyDomainNuma TopologyDomain = "numa"
)

// topologyDomainOrder defines the hierarchical order from broadest to narrowest.
// Values are spaced apart so new levels can be inserted between existing ones
// without renumbering.
var topologyDomainOrder = map[TopologyDomain]int{
	TopologyDomainRegion:     100,
	TopologyDomainZone:       200,
	TopologyDomainDataCenter: 300,
	TopologyDomainBlock:      400,
	TopologyDomainRack:       500,
	TopologyDomainHost:       600,
	TopologyDomainNuma:       700,
}

// IsValidTopologyDomain returns true if the domain is a known TopologyDomain value.
func IsValidTopologyDomain(d TopologyDomain) bool {
	_, ok := topologyDomainOrder[d]
	return ok
}

// IsNarrowerOrEqual returns true if d is narrower than or equal to other in the topology hierarchy.
func (d TopologyDomain) IsNarrowerOrEqual(other TopologyDomain) bool {
	return topologyDomainOrder[d] >= topologyDomainOrder[other]
}

// ValidateHierarchy validates that child is narrower than or equal to parent.
// Returns an error if the child domain is broader than the parent.
func ValidateHierarchy(parent, child TopologyDomain, childPath string) error {
	if !child.IsNarrowerOrEqual(parent) {
		return fmt.Errorf("%s: topologyConstraint.packDomain %q is broader than spec-level %q; service constraints must be equal to or narrower than the deployment-level constraint",
			childPath, child, parent)
	}
	return nil
}
