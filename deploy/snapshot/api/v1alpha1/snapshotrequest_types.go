package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type SnapshotTargetPodRef struct {
	Name string `json:"name"`
}

type SnapshotRequestSpec struct {
	Phase                        SnapshotRequestPhase    `json:"phase"`
	SnapshotID                   string                  `json:"snapshotID"`
	ArtifactVersion              string                  `json:"artifactVersion,omitempty"`
	PodTemplate                  *corev1.PodTemplateSpec `json:"podTemplate,omitempty"`
	TargetPodRef                 *SnapshotTargetPodRef   `json:"targetPodRef,omitempty"`
	DisableCudaCheckpointJobFile bool                    `json:"disableCudaCheckpointJobFile,omitempty"`
	ActiveDeadlineSeconds        *int64                  `json:"activeDeadlineSeconds,omitempty"`
	TTLSecondsAfterFinished      *int32                  `json:"ttlSecondsAfterFinished,omitempty"`
}

type SnapshotRequestStatus struct {
	State              SnapshotRequestState `json:"state,omitempty"`
	ObservedGeneration int64                `json:"observedGeneration,omitempty"`
	JobName            string               `json:"jobName,omitempty"`
	PodName            string               `json:"podName,omitempty"`
	Location           string               `json:"location,omitempty"`
	StorageType        string               `json:"storageType,omitempty"`
	Message            string               `json:"message,omitempty"`
	StartedAt          *metav1.Time         `json:"startedAt,omitempty"`
	CompletedAt        *metav1.Time         `json:"completedAt,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=sreq
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".spec.phase"
// +kubebuilder:printcolumn:name="SnapshotID",type="string",JSONPath=".spec.snapshotID"
// +kubebuilder:printcolumn:name="State",type="string",JSONPath=".status.state"
type SnapshotRequest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SnapshotRequestSpec   `json:"spec,omitempty"`
	Status SnapshotRequestStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
type SnapshotRequestList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []SnapshotRequest `json:"items"`
}

func init() {
	SchemeBuilder.Register(&SnapshotRequest{}, &SnapshotRequestList{})
}
