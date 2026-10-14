/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DynamoPoolRelaySpec configures one namespace-local Pool Relay. Conversion is
// unaffected: DynamoPoolRelay is a new, single-version API with no hub or spoke.
type DynamoPoolRelaySpec struct {
	// ClusterID is the stable discovery-cluster identity, retained across restarts.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=128
	// +kubebuilder:validation:Pattern=`^\S(.*\S)?$`
	// +kubebuilder:validation:XValidation:rule="self == oldSelf",message="clusterId is immutable; create a new relay to change identity"
	ClusterID string `json:"clusterId"`

	// Replicas is the number of active Relay processes. Each observes all selected
	// sources and maintains independent state. Defaults to two; one disables redundancy.
	// +optional
	// +kubebuilder:default=2
	// +kubebuilder:validation:Minimum=1
	Replicas *int32 `json:"replicas,omitempty"`

	// Sources selects the DGDs observed by every Relay replica.
	Sources DynamoPoolRelaySourcesSpec `json:"sources"`

	// PodTemplate configures the Relay Pod, like a DGD component. Exactly one
	// container named main with a nonempty image is required; other containers
	// are user-managed sidecars. Transport, identity and sources remain derived.
	// +kubebuilder:validation:XValidation:rule="has(self.spec) && has(self.spec.containers) && self.spec.containers.filter(c, c.name == 'main').size() == 1",message="podTemplate requires exactly one main container"
	PodTemplate corev1.PodTemplateSpec `json:"podTemplate"`

	// Service configures the generated ClusterIP ports. The port named grpc is
	// the publication endpoint. Omission exposes the main listener on 5561.
	// +optional
	Service *DynamoPoolRelayServiceSpec `json:"service,omitempty"`
}

// DynamoPoolRelaySourcesSpec groups source selection independently of publication.
type DynamoPoolRelaySourcesSpec struct {
	// DGDs selects inference graphs in the Relay's Kubernetes namespace.
	DGDs DynamoPoolRelayDGDSelection `json:"dgds"`
}

// DynamoPoolRelayDGDSelection resolves the union of label selection and explicit references.
// +kubebuilder:validation:XValidation:rule="has(self.labelSelector) || (has(self.refs) && size(self.refs) > 0)",message="supply sources.dgds.labelSelector or nonempty sources.dgds.refs"
type DynamoPoolRelayDGDSelection struct {
	// LabelSelector selects DGD metadata labels in this Kubernetes namespace.
	// An explicit empty selector selects all DGDs. Results are unioned with Refs.
	// +optional
	LabelSelector *metav1.LabelSelector `json:"labelSelector,omitempty"`

	// Refs additionally selects named DGDs in this Kubernetes namespace.
	// Missing references are reported in status and retried when they appear.
	// +optional
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=64
	Refs []DynamoPoolRelayDGDRef `json:"refs,omitempty"`
}

// DynamoPoolRelayServiceSpec configures publication without overriding ownership or selectors.
// +kubebuilder:validation:XValidation:rule="self.ports.exists(p, p.name == 'grpc')",message="service requires a port named grpc"
type DynamoPoolRelayServiceSpec struct {
	// Ports replace the default publication port. Named targets must exist in the Pod.
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=32
	Ports []corev1.ServicePort `json:"ports"`
}

// DynamoPoolRelayDGDRef names an inference graph in the Relay's Kubernetes namespace.
type DynamoPoolRelayDGDRef struct {
	// Name is the DGD metadata.name.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:Pattern=`^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$`
	Name string `json:"name"`
}

// DynamoPoolRelayStatus reports deployment availability, not pool or model readiness.
type DynamoPoolRelayStatus struct {
	// ObservedGeneration is the last successfully reconciled spec generation.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
	// Endpoint is the internal Service address of the port named grpc.
	// +optional
	Endpoint string `json:"endpoint,omitempty"`
	// DeploymentName identifies the managed Deployment for logs and inspection.
	// +optional
	DeploymentName string `json:"deploymentName,omitempty"`
	// DesiredSourcesRevision identifies the published source configuration.
	// +optional
	DesiredSourcesRevision string `json:"desiredSourcesRevision,omitempty"`
	// AppliedSourcesRevision is set only when every requested current replica
	// acknowledges DesiredSourcesRevision and no superseded active Pods remain.
	// +optional
	AppliedSourcesRevision string `json:"appliedSourcesRevision,omitempty"`
	// Replicas is the number of non-terminated Deployment replicas.
	// +optional
	Replicas int32 `json:"replicas,omitempty"`
	// UpdatedReplicas is the number of replicas at the desired Pod template.
	// +optional
	UpdatedReplicas int32 `json:"updatedReplicas,omitempty"`
	// AvailableReplicas is the number of available Deployment replicas.
	// +optional
	AvailableReplicas int32 `json:"availableReplicas,omitempty"`
	// SourcesAppliedReplicas counts current, non-terminating replicas that
	// acknowledged the desired source and connection revisions with their own Pod UID.
	// +optional
	SourcesAppliedReplicas int32 `json:"sourcesAppliedReplicas,omitempty"`

	// SelectedDGDs reports the number of active objects in the resolved union.
	// +optional
	SelectedDGDs int32 `json:"selectedDGDs,omitempty"`
	// Conditions reports Ready when all requested current replicas are available;
	// this does not imply model or KV recovery readiness.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:resource:scope=Namespaced,shortName=dprelay
// +kubebuilder:printcolumn:name="Ready",type=string,JSONPath=`.status.conditions[?(@.type=="Ready")].status`
// +kubebuilder:printcolumn:name="Cluster",type=string,JSONPath=`.spec.clusterId`
// +kubebuilder:printcolumn:name="Endpoint",type=string,JSONPath=`.status.endpoint`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// DynamoPoolRelay is an experimental, independently managed Pool Relay.
type DynamoPoolRelay struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              DynamoPoolRelaySpec `json:"spec"`
	// +optional
	Status DynamoPoolRelayStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoPoolRelayList contains DynamoPoolRelay resources.
type DynamoPoolRelayList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoPoolRelay `json:"items"`
}
