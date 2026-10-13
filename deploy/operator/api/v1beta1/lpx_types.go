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

	// experimental groups opt-in LPX options whose API shape may change in
	// breaking ways between v1beta1 releases.
	// +optional
	Experimental *LPXExperimentalSpec `json:"experimental,omitempty"`
}

// LPXExperimentalSpec groups experimental LPX options.
type LPXExperimentalSpec struct {
	// localPartitions selects partitions of a hybrid build that the Cyborg
	// conductor runs on its own GPU. The operator schedules LPU Agents only for
	// the remaining partitions, and schedules none when every partition is
	// local. Omission runs every partition on LPUs.
	// +optional
	LocalPartitions *LPXLocalPartitions `json:"localPartitions,omitempty"`

	// checkpoint overrides the checkpoint that a hybrid build's GBuild manifest
	// names for the Cyborg conductor. Model Express downloads the selected
	// checkpoint to model storage before the workload starts, and the operator
	// sets `CYBORG_WEIGHTS_PATH` in the Cyborg container to its snapshot
	// directory. Requires a hybrid build and a configured Model Express URL.
	// Omission uses the manifest's checkpoint, if any.
	// +optional
	Checkpoint *LPXCheckpoint `json:"checkpoint,omitempty"`
}

// LocalPartitions returns the experimental local-partition selection, or nil
// when none is set. The receiver may be nil.
func (c *LPXConfig) LocalPartitions() *LPXLocalPartitions {
	if c == nil || c.Experimental == nil {
		return nil
	}
	return c.Experimental.LocalPartitions
}

// CheckpointOverride returns the experimental checkpoint override, or nil when
// none is set. The receiver may be nil.
func (c *LPXConfig) CheckpointOverride() *LPXCheckpoint {
	if c == nil || c.Experimental == nil {
		return nil
	}
	return c.Experimental.Checkpoint
}

// LPXLocalPartitionsMode selects how LPXLocalPartitions chooses partitions.
// +kubebuilder:validation:Enum=All;IDs
type LPXLocalPartitionsMode string

const (
	// LPXLocalPartitionsModeAll runs every partition on the Cyborg GPU.
	LPXLocalPartitionsModeAll LPXLocalPartitionsMode = "All"
	// LPXLocalPartitionsModeIDs runs the partitions listed in ids on the Cyborg GPU.
	LPXLocalPartitionsModeIDs LPXLocalPartitionsMode = "IDs"
)

// LPXLocalPartitions selects the runtime partitions that run on the Cyborg GPU.
// Partition IDs are the compiler partition IDs of the build's runtime
// partitions. A selected prop-sync chain is identified by its first partition.
// +kubebuilder:validation:XValidation:rule="self.mode == 'IDs' ? has(self.ids) : !has(self.ids)",message="ids is required when mode is IDs and forbidden otherwise"
type LPXLocalPartitions struct {
	// mode selects the partitions that run on the Cyborg GPU. `All` runs every
	// partition; `IDs` runs the partitions listed in ids.
	// +required
	Mode LPXLocalPartitionsMode `json:"mode"`

	// ids lists the compiler partition IDs that run on the Cyborg GPU.
	// Required when mode is `IDs` and forbidden otherwise.
	// +optional
	// +listType=set
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:items:Minimum=0
	// +kubebuilder:validation:items:Maximum=4294967295
	IDs []int64 `json:"ids,omitempty"`
}

// LPXCheckpointProvider identifies the source of an LPX checkpoint.
// +kubebuilder:validation:Enum=HuggingFace
type LPXCheckpointProvider string

const (
	// LPXCheckpointProviderHuggingFace downloads a Hugging Face Hub model repository.
	LPXCheckpointProviderHuggingFace LPXCheckpointProvider = "HuggingFace"
)

// LPXCheckpoint identifies one immutable model checkpoint.
type LPXCheckpoint struct {
	// provider selects the checkpoint source. Only `HuggingFace` is supported.
	// +required
	Provider LPXCheckpointProvider `json:"provider"`

	// model is the Hugging Face repository ID, such as `openai/gpt-oss-20b`.
	// It follows the Hub's repository-ID rules: an optional namespace and a
	// name, each at most 96 characters and starting and ending with a letter,
	// digit, or underscore.
	// +required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=193
	// +kubebuilder:validation:Pattern=`^([A-Za-z0-9_]([A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?/)?[A-Za-z0-9_]([A-Za-z0-9_.-]{0,94}[A-Za-z0-9_])?$`
	// +kubebuilder:validation:XValidation:rule="!self.contains('--') && !self.contains('..')",message="model must not contain '--' or '..'"
	// +kubebuilder:validation:XValidation:rule="!self.endsWith('.git')",message="model must not end with '.git'"
	Model string `json:"model"`

	// revision is the full 40-character commit SHA of the repository snapshot.
	// Branches and tags are not accepted because they can move.
	// +required
	// +kubebuilder:validation:Pattern=`^[0-9a-f]{40}$`
	Revision string `json:"revision"`
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
