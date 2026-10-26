// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

// LPXConfiguration holds LPX scheduler integration and model-registry settings.
type LPXConfiguration struct {
	// Enabled opts the operator into LPX request production. When true, startup
	// requires the scheduling.lpu.nvidia.com/v1alpha1 LpuPipelineRequest API to
	// be installed separately.
	// +kubebuilder:default=false
	Enabled bool `json:"enabled"`

	// ModelRegistryURL configures the location of the LPU model registry used by LPX.
	// Supported values are an absolute local path (or file:// URL) and gs:// URLs.
	ModelRegistryURL string `json:"modelRegistryURL"`
}
