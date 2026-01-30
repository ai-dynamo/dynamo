/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 Atalaya Tech. Inc
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES
 */

package v1alpha1

import (
	"fmt"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

const (
	DynamoGraphDeploymentConditionTypeAvailable            = "Available"
	DynamoGraphDeploymentConditionTypeDynamoComponentReady = "DynamoComponentReady"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// DynamoComponentDeploymentSpec defines the desired state of DynamoComponentDeployment
type DynamoComponentDeploymentSpec struct {
	// BackendFramework specifies the backend framework (e.g., "sglang", "vllm", "trtllm")
	// +kubebuilder:validation:Enum=sglang;vllm;trtllm
	BackendFramework string `json:"backendFramework,omitempty"`

	// DynamoComponentDeploymentSharedSpec embeds common deployment and runtime
	// settings that apply to the component (resources, scaling, ingress, etc.).
	DynamoComponentDeploymentSharedSpec `json:",inline"`
}

type DynamoComponentDeploymentSharedSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// Annotations to add to generated Kubernetes resources for this component
	// (such as Pod, Service, and Ingress when applicable).
	Annotations map[string]string `json:"annotations,omitempty"`
	// Labels to add to generated Kubernetes resources for this component.
	Labels map[string]string `json:"labels,omitempty"`

	// The name of the component
	ServiceName string `json:"serviceName,omitempty"`

	// ComponentType indicates the role of this component (for example, "main").
	ComponentType string `json:"componentType,omitempty"`

	// SubComponentType indicates the sub-role of this component (for example, "prefill").
	SubComponentType string `json:"subComponentType,omitempty"`

	// DynamoNamespace is deprecated and will be removed in a future version.
	// The DGD Kubernetes namespace and DynamoGraphDeployment name are used to construct the Dynamo namespace for each component
	// +kubebuilder:validation:Optional
	DynamoNamespace *string `json:"dynamoNamespace,omitempty"`

	// GlobalDynamoNamespace indicates that the Component will be placed in the global Dynamo namespace
	GlobalDynamoNamespace bool `json:"globalDynamoNamespace,omitempty"`

	// Resources requested and limits for this component, including CPU, memory,
	// GPUs/devices, and any runtime-specific resources.
	Resources *Resources `json:"resources,omitempty"`
	// Deprecated: This field is deprecated and ignored. Use DynamoGraphDeploymentScalingAdapter
	// with HPA, KEDA, or Planner for autoscaling instead. See docs/kubernetes/autoscaling.md
	// for migration guidance. This field will be removed in a future API version.
	Autoscaling *Autoscaling `json:"autoscaling,omitempty"`
	// Envs defines additional environment variables to inject into the component containers.
	Envs []corev1.EnvVar `json:"envs,omitempty"`
	// EnvFromSecret references a Secret whose key/value pairs will be exposed as
	// environment variables in the component containers.
	EnvFromSecret *string `json:"envFromSecret,omitempty"`
	// VolumeMounts references PVCs defined at the top level for volumes to be mounted by the component.
	VolumeMounts []VolumeMount `json:"volumeMounts,omitempty"`

	// Ingress config to expose the component outside the cluster (or through a service mesh).
	Ingress *IngressSpec `json:"ingress,omitempty"`

	// ModelRef references a model that this component serves
	// When specified, a headless service will be created for endpoint discovery
	// +optional
	ModelRef *ModelReference `json:"modelRef,omitempty"`

	// SharedMemory controls the tmpfs mounted at /dev/shm (enable/disable and size).
	SharedMemory *SharedMemorySpec `json:"sharedMemory,omitempty"`

	// +optional
	// ExtraPodMetadata adds labels/annotations to the created Pods.
	ExtraPodMetadata *ExtraPodMetadata `json:"extraPodMetadata,omitempty"`
	// +optional
	// ExtraPodSpec allows to override the main pod spec configuration.
	// It is a k8s standard PodSpec. It also contains a MainContainer (standard k8s Container) field
	// that allows overriding the main container configuration.
	ExtraPodSpec *ExtraPodSpec `json:"extraPodSpec,omitempty"`

	// LivenessProbe to detect and restart unhealthy containers.
	LivenessProbe *corev1.Probe `json:"livenessProbe,omitempty"`
	// ReadinessProbe to signal when the container is ready to receive traffic.
	ReadinessProbe *corev1.Probe `json:"readinessProbe,omitempty"`
	// Replicas is the desired number of Pods for this component.
	// When scalingAdapter is enabled, this field is managed by the
	// DynamoGraphDeploymentScalingAdapter and should not be modified directly.
	// +kubebuilder:validation:Minimum=0
	Replicas *int32 `json:"replicas,omitempty"`
	// Multinode is the configuration for multinode components.
	Multinode *MultinodeSpec `json:"multinode,omitempty"`
	// ScalingAdapter configures whether this service uses the DynamoGraphDeploymentScalingAdapter.
	// When enabled, replicas are managed via DGDSA and external autoscalers can scale
	// the service using the Scale subresource. When disabled, replicas can be modified directly.
	// +optional
	ScalingAdapter *ScalingAdapter `json:"scalingAdapter,omitempty"`

	// EPPConfig defines EPP-specific configuration options for Endpoint Picker Plugin components.
	// Only applicable when ComponentType is "epp".
	// +optional
	EPPConfig *EPPConfig `json:"eppConfig,omitempty"`
}

type MultinodeSpec struct {
	// +kubebuilder:default=2
	// Indicates the number of nodes to deploy for multinode components.
	// Total number of GPUs is NumberOfNodes * GPU limit.
	// Must be greater than 1.
	// +kubebuilder:validation:Minimum=2
	NodeCount int32 `json:"nodeCount"`
}

type IngressTLSSpec struct {
	// SecretName is the name of a Kubernetes Secret containing the TLS certificate and key.
	SecretName string `json:"secretName,omitempty"`
}

type IngressSpec struct {
	// Enabled exposes the component through an ingress or virtual service when true.
	Enabled bool `json:"enabled,omitempty"`
	// Host is the base host name to route external traffic to this component.
	Host string `json:"host,omitempty"`
	// UseVirtualService indicates whether to configure a service-mesh VirtualService instead of a standard Ingress.
	UseVirtualService bool `json:"useVirtualService,omitempty"`
	// VirtualServiceGateway optionally specifies the gateway name to attach the VirtualService to.
	VirtualServiceGateway *string `json:"virtualServiceGateway,omitempty"`
	// HostPrefix is an optional prefix added before the host.
	HostPrefix *string `json:"hostPrefix,omitempty"`
	// Annotations to set on the generated Ingress/VirtualService resources.
	Annotations map[string]string `json:"annotations,omitempty"`
	// Labels to set on the generated Ingress/VirtualService resources.
	Labels map[string]string `json:"labels,omitempty"`
	// TLS holds the TLS configuration used by the Ingress/VirtualService.
	TLS *IngressTLSSpec `json:"tls,omitempty"`
	// HostSuffix is an optional suffix appended after the host.
	HostSuffix *string `json:"hostSuffix,omitempty"`
	// IngressControllerClassName selects the ingress controller class (e.g., "nginx").
	IngressControllerClassName *string `json:"ingressControllerClassName,omitempty"`
}

func (i *IngressSpec) IsVirtualServiceEnabled() bool {
	if i == nil {
		return false
	}
	return i.Enabled && i.UseVirtualService && i.VirtualServiceGateway != nil
}

// DynamoComponentDeploymentStatus defines the observed state of DynamoComponentDeployment
type DynamoComponentDeploymentStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// ObservedGeneration is the most recent generation observed by the controller.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conditions captures the latest observed state of the component (including
	// availability and readiness) using standard Kubernetes condition types.
	Conditions []metav1.Condition `json:"conditions"`

	// PodSelector contains the labels that can be used to select Pods belonging to
	// this component deployment.
	PodSelector map[string]string `json:"podSelector,omitempty"`

	// Service contains replica status information for this service.
	// +optional
	Service *ServiceReplicaStatus `json:"service,omitempty"`
}

// +genclient
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:printcolumn:name="DynamoComponent",type="string",JSONPath=".spec.dynamoComponent",description="Dynamo component"
// +kubebuilder:printcolumn:name="Available",type="string",JSONPath=".status.conditions[?(@.type=='Available')].status",description="Available"
// +kubebuilder:printcolumn:name="Backend",type="string",JSONPath=`.spec.backendFramework`,description="Backend framework (sglang, vllm, trtllm)"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
// +kubebuilder:resource:shortName=dcd
// DynamoComponentDeployment is the Schema for the dynamocomponentdeployments API
type DynamoComponentDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired state for this Dynamo component deployment.
	Spec DynamoComponentDeploymentSpec `json:"spec,omitempty"`
	// Status reflects the current observed state of the component deployment.
	Status DynamoComponentDeploymentStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoComponentDeploymentList contains a list of DynamoComponentDeployment
type DynamoComponentDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoComponentDeployment `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoComponentDeployment{}, &DynamoComponentDeploymentList{})
}

func (s *DynamoComponentDeployment) IsReady() (bool, string) {
	ready, reason := s.Status.IsReady()
	return ready, reason
}

// GetState returns "ready" or "not_ready" based on conditions
func (d *DynamoComponentDeployment) GetState() string {
	ready, _ := d.IsReady()
	if ready {
		return commonconsts.ResourceStateReady
	}
	return commonconsts.ResourceStateNotReady
}

func (s *DynamoComponentDeployment) GetServiceStatuses() map[string]ServiceReplicaStatus {
	if s.Status.Service == nil {
		return map[string]ServiceReplicaStatus{}
	}
	return map[string]ServiceReplicaStatus{s.Spec.ServiceName: *s.Status.Service}
}

func (s *DynamoComponentDeploymentStatus) IsReady() (bool, string) {
	for _, condition := range s.Conditions {
		if condition.Type == DynamoGraphDeploymentConditionTypeAvailable && condition.Status == metav1.ConditionTrue {
			return true, ""
		}
	}
	return false, "Component deployment not ready - Available condition not true"
}

func (s *DynamoComponentDeployment) GetSpec() any {
	return s.Spec
}

func (s *DynamoComponentDeployment) SetSpec(spec any) {
	s.Spec = spec.(DynamoComponentDeploymentSpec)
}

func (s *DynamoComponentDeployment) IsFrontendComponent() bool {
	return s.Spec.ComponentType == commonconsts.ComponentTypeFrontend
}

func (s *DynamoComponentDeployment) GetDynamoDeploymentConfig() []byte {
	for _, env := range s.Spec.Envs {
		if env.Name == commonconsts.DynamoDeploymentConfigEnvVar {
			return []byte(env.Value)
		}
	}
	return nil
}

func (s *DynamoComponentDeployment) SetDynamoDeploymentConfig(config []byte) {
	for i, env := range s.Spec.Envs {
		if env.Name == commonconsts.DynamoDeploymentConfigEnvVar {
			s.Spec.Envs[i].Value = string(config)
			return
		}
	}
	s.Spec.Envs = append(s.Spec.Envs, corev1.EnvVar{
		Name:  commonconsts.DynamoDeploymentConfigEnvVar,
		Value: string(config),
	})
}

func (s *DynamoComponentDeployment) IsMultinode() bool {
	return s.GetNumberOfNodes() > 1
}

func (s *DynamoComponentDeployment) GetNumberOfNodes() int32 {
	return s.Spec.GetNumberOfNodes()
}

func (s *DynamoComponentDeploymentSharedSpec) IsMultinode() bool {
	return s.GetNumberOfNodes() > 1
}

func (s *DynamoComponentDeploymentSharedSpec) GetNumberOfNodes() int32 {
	if s.Multinode != nil {
		return s.Multinode.NodeCount
	}
	return 1
}

func (s *DynamoComponentDeployment) GetParentGraphDeploymentName() string {
	for _, ownerRef := range s.ObjectMeta.OwnerReferences {
		if ownerRef.Kind == "DynamoGraphDeployment" {
			return ownerRef.Name
		}
	}
	return ""
}

func (s *DynamoComponentDeployment) GetParentGraphDeploymentNamespace() string {
	return s.GetNamespace()
}

// GetDynamoNamespace returns the Dynamo namespace for this component.
func (s *DynamoComponentDeployment) GetDynamoNamespace() string {
	return ComputeDynamoNamespace(s.Spec.GlobalDynamoNamespace, s.GetNamespace(), s.GetParentGraphDeploymentName())
}

// ComputeDynamoNamespace is the single source of truth for computing the Dynamo namespace.
// If globalDynamoNamespace is true, returns "dynamo" (global constant).
// Otherwise, returns {k8sNamespace}-{dgdName}.
func ComputeDynamoNamespace(globalDynamoNamespace bool, k8sNamespace, dgdName string) string {
	if globalDynamoNamespace {
		return commonconsts.GlobalDynamoNamespace
	}
	return fmt.Sprintf("%s-%s", k8sNamespace, dgdName)
}

// ModelReference identifies a model served by this component
type ModelReference struct {
	// Name is the base model identifier (e.g., "llama-3-70b-instruct-v1")
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// Revision is the model revision/version (optional)
	// +optional
	Revision string `json:"revision,omitempty"`
}

// EPPConfig contains configuration for EPP (Endpoint Picker Plugin) components.
// EPP is responsible for intelligent endpoint selection and KV-aware routing.
type EPPConfig struct {
	// ConfigMapRef references a user-provided ConfigMap containing EPP configuration.
	// The ConfigMap should contain EndpointPickerConfig YAML.
	// Mutually exclusive with Config.
	// +optional
	ConfigMapRef *corev1.ConfigMapKeySelector `json:"configMapRef,omitempty"`

	// Config allows specifying EPP EndpointPickerConfig directly as a structured object.
	// The operator will marshal this to YAML and create a ConfigMap automatically.
	// Mutually exclusive with ConfigMapRef.
	// One of ConfigMapRef or Config must be specified (no default configuration).
	// Uses the upstream type from github.com/kubernetes-sigs/gateway-api-inference-extension
	// +optional
	// +kubebuilder:validation:Type=object
	// +kubebuilder:pruning:PreserveUnknownFields
	Config *apixv1alpha1.EndpointPickerConfig `json:"config,omitempty"`
}
