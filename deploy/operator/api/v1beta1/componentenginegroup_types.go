/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1beta1

// ComponentEngineGroupSpec configures creation of independent engine worlds.
// Initialization and live /scale targets both count whole replica allocations.
// +kubebuilder:validation:XValidation:rule="!has(self.policy) || !has(self.policy.minSize) || self.initialSize >= self.policy.minSize",message="initialSize must be greater than or equal to policy.minSize"
// +kubebuilder:validation:XValidation:rule="!has(self.policy) || !has(self.policy.maxSize) || self.initialSize <= self.policy.maxSize",message="initialSize must be less than or equal to policy.maxSize"
type ComponentEngineGroupSpec struct {
	// initialSize is the immutable number of replica allocations in each new world.
	// It is not a GPU, EP-rank, or native-member count. The profile resolves those counts.
	// +kubebuilder:validation:Minimum=1
	InitialSize int32 `json:"initialSize"`

	// policy supplies initial bounds copied into each new Engine Group's live policy.
	// +optional
	Policy *ComponentEngineGroupPolicy `json:"policy,omitempty"`
}

// ComponentEngineGroupPolicy configures initial allocation bounds for newly created worlds.
// +kubebuilder:validation:XValidation:rule="!has(self.minSize) || !has(self.maxSize) || self.minSize <= self.maxSize",message="minSize must be less than or equal to maxSize"
type ComponentEngineGroupPolicy struct {
	// minSize seeds the generated Engine Group's policy.minReplicas.
	// +optional
	// +kubebuilder:validation:Minimum=1
	MinSize *int32 `json:"minSize,omitempty"`

	// maxSize seeds the generated Engine Group's policy.maxReplicas.
	// +optional
	// +kubebuilder:validation:Minimum=1
	MaxSize *int32 `json:"maxSize,omitempty"`
}
