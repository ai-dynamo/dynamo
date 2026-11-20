// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1beta1

// ComponentRoleStatus describes the lowered target configuration of one logical
// role. It does not describe the configuration of individual running Pods.
type ComponentRoleStatus struct {
	// name identifies the logical role, such as main, leader, or worker.
	// +required
	Name string `json:"name"`

	// containers contains the final application and initialization containers,
	// including native runtime sidecars, keyed by their Pod container name.
	// +optional
	// +listType=map
	// +listMapKey=name
	Containers []ComponentContainerStatus `json:"containers,omitempty"`
}

// ComponentContainerStatus projects a container's lowered launch declarations.
// Shell scripts and image entrypoints are not interpreted.
type ComponentContainerStatus struct {
	// name is the container name from the lowered PodSpec.
	// +required
	Name string `json:"name"`

	// image is the declared image; tags are not resolved to image digests.
	// +required
	Image string `json:"image"`

	// command is the final declared command, without Kubernetes or shell expansion.
	// Omission does not imply a known image entrypoint.
	// +optional
	// +listType=atomic
	Command []string `json:"command,omitempty"`

	// args is the final declared argument list, without Kubernetes or shell expansion.
	// +optional
	// +listType=atomic
	Args []string `json:"args,omitempty"`

	// env contains the final bindings after sequential Kubernetes-style expansion
	// of the lowered environment. Unavailable references, valueFrom sources, and
	// expansions exceeding the replay or export budget remain unresolved. ConfigMaps,
	// Secrets, images, and provider environments are not read. Missing names are
	// not known absent: envFrom, images, or providers may define additional names.
	// +optional
	Env map[string]EnvironmentValueStatus `json:"env,omitempty"`
}

// EnvironmentValueStatus is either a known value, including the empty string,
// or an unresolved binding. Partial expansions are never published.
// +kubebuilder:validation:XValidation:rule="has(self.value) ? !has(self.unresolved) : (has(self.unresolved) && self.unresolved)",message="specify either value or unresolved: true"
type EnvironmentValueStatus struct {
	// value is the resolved literal value. An empty string is a known value.
	// +optional
	Value *string `json:"value,omitempty"`

	// unresolved is true when the binding cannot be resolved from the lowered list
	// or its value cannot be exported within the status size budget.
	// +optional
	Unresolved bool `json:"unresolved,omitempty"`
}
