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
// Common examples: "region", "zone", "datacenter", "block", "rack", "host", "numa".
// When used with a ClusterTopology CR, domain names are defined in the CR's
// hierarchy; when used with `spec.experimental.kvTransferPolicy.labelKey`
// alone, the value is a user-chosen logical name for the topology level.
// Must match `^[a-z0-9]([a-z0-9-]*[a-z0-9])?$` (lowercase alphanumeric,
// may contain hyphens but must not start or end with one).
// +kubebuilder:validation:Pattern=`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`
type TopologyDomain string

var topologyDomainRegex = regexp.MustCompile(`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`)

// IsValidTopologyDomainFormat returns true if the domain matches the allowed format.
func IsValidTopologyDomainFormat(d TopologyDomain) bool {
	return topologyDomainRegex.MatchString(string(d))
}

// KvTransferEnforcement controls how the selected prefill worker's topology is
// applied to decode routing.
// +kubebuilder:validation:Enum=required;preferred
type KvTransferEnforcement string

const (
	// KvTransferEnforcementRequired enforces same-domain decode worker
	// selection.
	KvTransferEnforcementRequired KvTransferEnforcement = "required"
	// KvTransferEnforcementPreferred biases decode worker selection toward the
	// same domain.
	KvTransferEnforcementPreferred KvTransferEnforcement = "preferred"
)

// KvTransferPolicy configures topology-aware routing for KV-cache transfers
// between prefill and decode workers. This graph-wide policy lives under
// `spec.experimental` while the API is incubating.
// +kubebuilder:validation:XValidation:rule="!has(self.enforcement) || self.enforcement != 'preferred' || has(self.preferredWeight)",message="preferredWeight is required when enforcement is preferred"
// +kubebuilder:validation:XValidation:rule="!has(self.preferredWeight) || (has(self.enforcement) && self.enforcement == 'preferred')",message="preferredWeight may only be set when enforcement is preferred"
type KvTransferPolicy struct {
	// LabelKey is a Kubernetes node label key (e.g.
	// "topology.kubernetes.io/zone") whose value identifies the topology
	// domain for each worker. The operator copies the node label onto worker
	// pods so the runtime can publish it as worker metadata. The label
	// should correspond to the topology level named in `domain`.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=317
	// +kubebuilder:validation:Pattern=`^(([a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?)(\.[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?)*/)?([A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?)$`
	// +kubebuilder:validation:XValidation:rule="!self.contains('/') || self.split('/')[0].size() <= 253",message="labelKey prefix must be 253 characters or less"
	LabelKey string `json:"labelKey"`

	// Domain is the logical name for the topology level to enforce
	// (e.g. "zone", "rack"). The router uses this to match workers that
	// share the same value for the label identified by `labelKey`.
	Domain TopologyDomain `json:"domain"`

	// Enforcement controls how the selected prefill worker's topology is
	// applied to decode routing. "required" only allows decode workers in the
	// same topology domain as the selected prefill worker. "preferred" keeps
	// all decode workers eligible, but biases selection toward workers in the
	// same topology domain. Defaults to "required".
	// +optional
	// +kubebuilder:default=required
	// +kubebuilder:validation:Enum=required;preferred
	Enforcement KvTransferEnforcement `json:"enforcement,omitempty"`

	// PreferredWeight is required and used only when enforcement is
	// "preferred". Higher values create a stronger same-domain routing
	// preference, but do not guarantee same-domain selection. The value is not
	// a probability; worker selection still depends on load and other routing
	// inputs. A value of 0 disables the topology preference; 1 is the strongest
	// supported preference.
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	PreferredWeight *float32 `json:"preferredWeight,omitempty"`
}
