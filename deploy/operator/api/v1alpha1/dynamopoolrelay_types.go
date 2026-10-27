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
	// +kubebuilder:validation:Pattern=`^[^[:space:]\p{Z}\x85\x1c-\x1f](.*[^[:space:]\p{Z}\x85\x1c-\x1f])?$`
	ClusterID string `json:"clusterId"`

	// PodTemplate configures Relay Pods. Its main container needs an image with
	// dynamo.pool_relay and --sources-file support from the operator's revision.
	// The controller sets the main command, listener probes, source volume,
	// service account, and Relay identity and discovery environment variables.
	// Reserved env names: DYN_NAMESPACE, DYN_DISCOVERY_BACKEND,
	// DYN_KUBE_DISCOVERY_MODE, DYN_SYSTEM_PORT, DYN_TCP_RESPONSE_STREAM_HOST,
	// DYN_TCP_RESPONSE_STREAM_PORT, DYN_CLUSTER_ID, DYN_RELAY_BIND,
	// DYN_RELAY_CONNECTION_REVISION, DYN_RELAY_NAMESPACES, DYN_RELAY_WATCH_ALL,
	// DYN_RELAY_ENDPOINT_PREFIXES, POD_NAME, POD_NAMESPACE, and POD_UID.
	// The main container cannot use envFrom because it could override those values.
	// +kubebuilder:validation:Required
	PodTemplate *corev1.PodTemplateSpec `json:"podTemplate"`

	// Replicas sets the number of active Relay Pods. Each watches the DGD.
	// +optional
	// +kubebuilder:default=2
	// +kubebuilder:validation:Minimum=1
	Replicas *int32 `json:"replicas,omitempty"`
}

// DynamoPoolRelayStatus reports listener availability. It does not report KV or model readiness.
type DynamoPoolRelayStatus struct {
	// ObservedGeneration is the Relay resource generation last reconciled by the operator.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
	// Endpoint is the cluster-local `<dgd>-pool-relay.<namespace>.svc:5561` address.
	// It is empty after withdrawal or a failed reconciliation.
	// +optional
	Endpoint string `json:"endpoint,omitempty"`
	// AvailableReplicas is the number of Relay Pods the Deployment reports available.
	// +optional
	AvailableReplicas int32 `json:"availableReplicas,omitempty"`
	// Conditions includes Ready for listener availability, not model or KV readiness.
	// Its reason is ListenerAvailable, DeploymentProgressing, DGDNotFound,
	// DGDTerminating, UnsupportedDiscovery, UnsupportedSourceScope, NoWorkers,
	// InvalidConfiguration, ReconcileFailed, or ResourceOwnershipConflict.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// DynamoPoolRelay deploys active-active Relay Pods for one DGD.
// The resource name matches its DGD; the Service name adds the -pool-relay suffix.
// The DGD needs Kubernetes pod-mode discovery and at least one worker component.
// Worker and frontend components cannot use globalDynamoNamespace. The DGD name
// must produce a DNS-1035 Service name, so it starts with a lowercase letter,
// uses only lowercase letters, digits, and hyphens, and is at most 52 characters.
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
