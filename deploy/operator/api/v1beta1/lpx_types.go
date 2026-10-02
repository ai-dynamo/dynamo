// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1beta1

// LPXConfig identifies the component's compiled model and configures scheduling.
type LPXConfig struct {
	// buildId references the immutable model build.
	// +kubebuilder:validation:MinLength=1
	BuildID string `json:"buildId"`

	// scheduling configures this component's LPX scheduling attempts.
	// Omission means no deadline.
	// +optional
	Scheduling *SchedulingSpec `json:"scheduling,omitempty"`

	// localPartitions selects partitions of a hybrid build that the Cyborg
	// conductor runs on its own GPU. The operator schedules LPU Agents only for
	// the remaining partitions, and schedules none when every partition is
	// local. Omission runs every partition on LPUs.
	// +optional
	LocalPartitions *LPXLocalPartitions `json:"localPartitions,omitempty"`
}

// LPXLocalPartitions selects the runtime partitions that run on the Cyborg GPU.
// Partition IDs are the compiler partition IDs of the build's runtime
// partitions. A selected prop-sync chain is identified by its first partition.
// +kubebuilder:validation:XValidation:rule="(has(self.all) && self.all) != has(self.ids)",message="set either all: true or ids"
type LPXLocalPartitions struct {
	// all runs every partition on the Cyborg GPU.
	// +optional
	All bool `json:"all,omitempty"`

	// ids lists the compiler partition IDs that run on the Cyborg GPU.
	// +optional
	// +listType=set
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:items:Minimum=0
	IDs []int32 `json:"ids,omitempty"`
}

// SchedulingSpec configures LPX scheduling attempts.
type SchedulingSpec struct {
	// attemptDeadlineSeconds limits how long each LPR may remain pending.
	// Omission means unlimited; the value is not a solver budget.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=9223372036
	AttemptDeadlineSeconds *int64 `json:"attemptDeadlineSeconds,omitempty"`
}

// HasLPXComponent returns true if any component uses the LPX integration.
func (s *DynamoGraphDeployment) HasLPXComponent() bool {
	for i := range s.Spec.Components {
		if s.Spec.Components[i].IsLPX() {
			return true
		}
	}
	return false
}

// IsLPX reports whether this shared spec uses the LPX integration.
func (s *DynamoComponentDeploymentSharedSpec) IsLPX() bool {
	return s.ComponentType == ComponentTypeLPX
}

// ManagedByExternalController reports whether a dedicated controller, rather than
// the ordinary DGD workload path, manages this component. Ownership is determined
// by the component type and does not change when its integration is disabled.
func (s *DynamoComponentDeploymentSharedSpec) ManagedByExternalController() bool {
	return s.ComponentType == ComponentTypeLPX
}
