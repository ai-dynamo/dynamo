package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

var (
	GroupVersion  = schema.GroupVersion{Group: "snapshot.nvidia.com", Version: "v1alpha1"}
	SchemeBuilder = &scheme.Builder{GroupVersion: GroupVersion}
	AddToScheme   = SchemeBuilder.AddToScheme
)

func Kind(kind string) schema.GroupKind {
	return GroupVersion.WithKind(kind).GroupKind()
}

func Resource(resource string) schema.GroupResource {
	return GroupVersion.WithResource(resource).GroupResource()
}

type SnapshotRequestPhase string

const (
	SnapshotRequestPhaseCheckpoint SnapshotRequestPhase = "Checkpoint"
	SnapshotRequestPhaseRestore    SnapshotRequestPhase = "Restore"
)

type SnapshotRequestState string

const (
	SnapshotRequestStatePending          SnapshotRequestState = "Pending"
	SnapshotRequestStateWaitingForTarget SnapshotRequestState = "WaitingForTarget"
	SnapshotRequestStateRunning          SnapshotRequestState = "Running"
	SnapshotRequestStateSucceeded        SnapshotRequestState = "Succeeded"
	SnapshotRequestStateFailed           SnapshotRequestState = "Failed"
)

func Now() *metav1.Time {
	now := metav1.Now()
	return &now
}
