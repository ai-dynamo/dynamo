// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1beta1

import (
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// LPXConfig identifies the component's compiled model and runtime settings.
type LPXConfig struct {
	// buildId references the immutable model build.
	// +kubebuilder:validation:MinLength=1
	BuildID string `json:"buildId"`
	// settings override the existing model configuration.
	// +optional
	// +kubebuilder:validation:Type=object
	// +nullable
	Settings *apiextensionsv1.JSON `json:"settings,omitempty"`
}

// SchedulingSpec configures LPX scheduling attempts.
type SchedulingSpec struct {
	// attemptDeadlineSeconds limits one aggregate LPX scheduling attempt.
	// Omission means unlimited; the value is not a solver budget.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=9223372036
	AttemptDeadlineSeconds *int64 `json:"attemptDeadlineSeconds,omitempty"`
}

// LPXAttemptStatus is Dynamo's durable authority record for one aggregate LPX attempt.
type LPXAttemptStatus struct {
	ObservedGeneration int64        `json:"observedGeneration"`
	PodCliqueSetUID    types.UID    `json:"podCliqueSetUID,omitempty"`
	DeadlineAt         *metav1.Time `json:"deadlineAt,omitempty"`
	ExceededAt         *metav1.Time `json:"exceededAt,omitempty"`
	DisarmedAt         *metav1.Time `json:"disarmedAt,omitempty"`
	// +listType=map
	// +listMapKey=name
	Requests []LPXAttemptRequestStatus `json:"requests"`
}

// LPXAttemptRequestStatus identifies one exact request in the aggregate attempt.
type LPXAttemptRequestStatus struct {
	Name          string    `json:"name"`
	AttemptDigest string    `json:"attemptDigest"`
	UID           types.UID `json:"uid,omitempty"`
}

// DynamoGraphDeploymentLPXStatus contains the public status of a graph's LPX workload.
// Both served DGD versions use this type, so conversion preserves its live fields natively.
type DynamoGraphDeploymentLPXStatus struct {
	// modelDownload contains the model download status for remote builds.
	// +optional
	ModelDownload *ModelDownloadStatus `json:"modelDownload,omitempty"`
	// placement contains LPX scheduling progress and placement signals.
	// +optional
	Placement *PlacementStatus `json:"placement,omitempty"`
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
