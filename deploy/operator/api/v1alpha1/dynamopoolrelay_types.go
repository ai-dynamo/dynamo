/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DynamoPoolRelaySpec configures Relay Pods for the DGD with the same name and namespace.
type DynamoPoolRelaySpec struct {
	// ClusterID is the Relay's stable logical cluster ID across restarts.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	ClusterID string `json:"clusterId"`

	// Image must contain dynamo.pool_relay with file-source support.
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image"`

	// ImagePullSecrets authenticate pulls of the Relay image from a private registry.
	// +optional
	ImagePullSecrets []corev1.LocalObjectReference `json:"imagePullSecrets,omitempty"`

	// Replicas sets the number of active Relay Pods. Each watches the DGD.
	// +optional
	// +kubebuilder:default=2
	// +kubebuilder:validation:Minimum=1
	Replicas *int32 `json:"replicas,omitempty"`

	// Env sets transport and Relay tuning variables on each Pod.
	// The operator reserves identity and source settings.
	// +optional
	// +listType=map
	// +listMapKey=name
	Env []corev1.EnvVar `json:"env,omitempty"`
}

// DynamoPoolRelayStatus reports listener availability. It does not report KV or model readiness.
type DynamoPoolRelayStatus struct {
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
	// +optional
	Endpoint string `json:"endpoint,omitempty"`
	// +optional
	AvailableReplicas int32 `json:"availableReplicas,omitempty"`
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// DynamoPoolRelay deploys active-active Relay Pods for one DGD.
// The resource name matches its DGD; the Service name adds the -pool-relay suffix.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=dprelay
// +kubebuilder:printcolumn:name="Ready",type=string,JSONPath=`.status.conditions[?(@.type=="Ready")].status`
// +kubebuilder:printcolumn:name="Endpoint",type=string,JSONPath=`.status.endpoint`
type DynamoPoolRelay struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              DynamoPoolRelaySpec `json:"spec"`
	// +optional
	Status DynamoPoolRelayStatus `json:"status,omitempty"`
}

// DynamoPoolRelayList contains DynamoPoolRelay resources.
// +kubebuilder:object:root=true
type DynamoPoolRelayList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoPoolRelay `json:"items"`
}
