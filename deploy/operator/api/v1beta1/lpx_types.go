// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1beta1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// LPXConfig identifies the component's compiled model and configures scheduling.
type LPXConfig struct {
	// buildId references the immutable model build.
	// +kubebuilder:validation:MinLength=1
	BuildID string `json:"buildId"`

	// scheduling configures this component's LPX scheduling attempts.
	// Omission means no deadline.
	// +optional
	Scheduling *SchedulingSpec `json:"scheduling,omitempty"`
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

// ModelDownloadStatus contains the status of remote LPU model downloads.
type ModelDownloadStatus struct {
	// builds is the sorted set of resolved remote LPU build URLs whose artifacts
	// were successfully downloaded into model-storage.
	// +optional
	Builds []string `json:"builds,omitempty"`

	// lastCheckedAt is the last time all remote LPU builds were checked with ModelExpress.
	// +optional
	LastCheckedAt *metav1.Time `json:"lastCheckedAt,omitempty"`
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
