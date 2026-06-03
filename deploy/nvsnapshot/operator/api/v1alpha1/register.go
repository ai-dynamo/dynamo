// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupName is the API group all NVSnapshot v1alpha1 types belong to.
const GroupName = "nvsnapshot.io"

var (
	// SchemeGroupVersion is the GroupVersion under which the v1alpha1 types
	// register with a runtime.Scheme.
	SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: "v1alpha1"}
	// SchemeBuilder collects the AddToScheme functions for this package.
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)
	// AddToScheme registers all v1alpha1 NVSnapshot types into the given Scheme.
	AddToScheme = SchemeBuilder.AddToScheme
)

// Kind returns a Group-qualified GroupKind for an unqualified Kind name.
func Kind(kind string) schema.GroupKind {
	return SchemeGroupVersion.WithKind(kind).GroupKind()
}

// Resource returns a Group-qualified GroupResource for an unqualified resource name.
func Resource(resource string) schema.GroupResource {
	return SchemeGroupVersion.WithResource(resource).GroupResource()
}

func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&Snapshot{},
		&SnapshotList{},
		&SnapshotContent{},
		&SnapshotContentList{},
		&SnapshotJob{},
		&SnapshotJobList{},
	)
	metav1.AddToGroupVersion(scheme, SchemeGroupVersion)
	return nil
}
