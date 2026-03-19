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

import "regexp"

const (
	// ConditionTypeTopologyLevelsAvailable indicates whether the topology levels
	// referenced by the deployment's constraints are available in the cluster topology.
	ConditionTypeTopologyLevelsAvailable = "TopologyLevelsAvailable"

	// ConditionReasonAllTopologyLevelsAvailable indicates all required topology levels
	// are available in the cluster topology.
	ConditionReasonAllTopologyLevelsAvailable = "AllTopologyLevelsAvailable"
	// ConditionReasonTopologyLevelsUnavailable indicates one or more required topology
	// levels are no longer available.
	ConditionReasonTopologyLevelsUnavailable = "TopologyLevelsUnavailable"
	// ConditionReasonTopologyDefinitionNotFound indicates the topology definition
	// resource was not found by the framework.
	ConditionReasonTopologyDefinitionNotFound = "TopologyDefinitionNotFound"
	// ConditionReasonTopologyConditionPending indicates the scheduling framework
	// has not yet reported a topology condition.
	ConditionReasonTopologyConditionPending = "TopologyConditionPending"
)

// SpecTopologyConstraint defines deployment-level topology placement requirements.
// It carries both the topology profile (which ClusterTopology CR to use) and an
// optional default pack domain that services without their own constraint inherit.
type SpecTopologyConstraint struct {
	// TopologyProfile is the name of the ClusterTopology CR that defines the
	// topology hierarchy for this deployment.
	// +kubebuilder:validation:MinLength=1
	TopologyProfile string `json:"topologyProfile"`

	// PackDomain is the default topology domain to pack pods within.
	// Optional — omit when only services carry constraints.
	// +optional
	PackDomain TopologyDomain `json:"packDomain,omitempty"`
}

// TopologyConstraint defines service-level topology placement requirements.
// The topology profile is inherited from the deployment-level SpecTopologyConstraint;
// only the pack domain is specified here.
type TopologyConstraint struct {
	// PackDomain is the topology domain to pack pods within. Must match a
	// domain defined in the referenced ClusterTopology CR.
	PackDomain TopologyDomain `json:"packDomain"`
}

// TopologyDomain is a free-form topology level identifier.
// Domain names are defined by the cluster admin in the ClusterTopology CR.
// Common examples: "region", "zone", "datacenter", "block", "rack", "host", "numa".
// Must match `^[a-z0-9]([a-z0-9-]*[a-z0-9])?$` (lowercase alphanumeric,
// may contain hyphens but must not start or end with one).
// +kubebuilder:validation:Pattern=`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`
type TopologyDomain string

var topologyDomainRegex = regexp.MustCompile(`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`)

// IsValidTopologyDomainFormat returns true if the domain matches the allowed format.
func IsValidTopologyDomainFormat(d TopologyDomain) bool {
	return topologyDomainRegex.MatchString(string(d))
}
