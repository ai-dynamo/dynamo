/*
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
 */

package v1beta1

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

const (
	// DynamoComponentDeploymentConditionTypeAvailable indicates the component is
	// available and serving traffic.
	DynamoComponentDeploymentConditionTypeAvailable = "Available"
	// DynamoComponentDeploymentConditionTypeDynamoComponentReady indicates the
	// underlying Dynamo component is ready.
	DynamoComponentDeploymentConditionTypeDynamoComponentReady = "DynamoComponentReady"

	// MainContainerName is the well-known name of the primary Dynamo workload
	// container inside a component's `podTemplate.spec.containers`. The operator
	// injects its defaults (image, command, env, ports, probes, resources,
	// volume mounts) into this container. If no container with this name is
	// present in the user-supplied `podTemplate`, the operator auto-generates
	// it. Any other container in the `podTemplate` is treated as a user-managed
	// sidecar.
	MainContainerName = "main"
)

// DynamoComponentDeploymentSpec defines the desired state of a DynamoComponentDeployment.
type DynamoComponentDeploymentSpec struct {
	// backendFramework specifies the backend framework.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=sglang;vllm;trtllm
	BackendFramework string `json:"backendFramework"`

	// DynamoComponentDeploymentSharedSpec embeds common deployment and runtime
	// settings that apply to the component.
	DynamoComponentDeploymentSharedSpec `json:",inline"`
}

// DynamoComponentDeploymentSharedSpec is the shared configuration used by both
// standalone DCDs and by the components embedded in a DynamoGraphDeployment.
//
// In v1beta1 the ten per-component pod-configuration fields that existed in
// v1alpha1 (resources, envs, envFromSecret, livenessProbe, readinessProbe,
// volumeMounts, annotations, labels, extraPodMetadata, extraPodSpec) are
// replaced with a single `podTemplate` field holding a native
// `corev1.PodTemplateSpec`. The operator injects its defaults into the
// container named `"main"` and merges user overrides using strategic-merge-by-name
// semantics. Users can add sidecars, init containers, and pod-level configuration
// directly in `podTemplate` without any `extraPodSpec`-style escape hatch.
// +kubebuilder:validation:XValidation:rule="!has(self.eppConfig) || (has(self.type) && self.type == 'epp')",message="eppConfig may only be set when type is epp"
type DynamoComponentDeploymentSharedSpec struct {
	// name is the stable logical identifier for this component within its
	// DynamoGraphDeployment. It must be unique within the parent's
	// `spec.components` list.
	//
	// For standalone DynamoComponentDeployment objects, the defaulting webhook
	// populates `name` from `metadata.name` on admission, so users
	// typically do not need to set it explicitly.
	//
	// `name` is decoupled from the underlying Kubernetes resource name so that
	// the operator can rename child workloads (e.g. suffixing worker DCDs with
	// a hash during rolling updates) without losing the stable identity that
	// downstream consumers (labels, status maps, DGDSA references, planner
	// RBAC, EPP filters) depend on.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`
	ComponentName string `json:"name"`

	// type indicates the role of this component within a Dynamo graph. Drives
	// port mapping, frontend detection, planner RBAC, and the pod label
	// `nvidia.com/dynamo-component-type`. Because `prefill` and `decode` are
	// first-class values, users can set them directly.
	// +optional
	ComponentType ComponentType `json:"type,omitempty"`

	// globalDynamoNamespace places the component in the global Dynamo
	// namespace rather than the per-deployment namespace derived from the
	// DGD name.
	// +optional
	GlobalDynamoNamespace bool `json:"globalDynamoNamespace,omitempty"`

	// podTemplate is the pod template used to create the component's pods.
	// The operator injects its defaults (image, command, env, ports, probes,
	// resources, volume mounts) into the container named `"main"` inside
	// `podTemplate.spec.containers`, merging user overrides by name. If no
	// container named `"main"` is present, the operator auto-generates it
	// with standard defaults. All other containers in `podTemplate.spec.containers`
	// are treated as user-managed sidecars: the operator does not inject
	// defaults into them, so sidecars must specify required fields (e.g. `image`)
	// themselves. The validation webhook rejects pod templates where a
	// non-`"main"` container is missing a required field such as `image`.
	// +optional
	PodTemplate *corev1.PodTemplateSpec `json:"podTemplate,omitempty"`

	// replicas is the desired number of Pods for this component. When
	// `scalingAdapter` is set on this component, this field is managed by
	// the DynamoGraphDeploymentScalingAdapter and should not be modified
	// directly.
	// +kubebuilder:validation:Minimum=0
	// +optional
	Replicas *int32 `json:"replicas,omitempty"`

	// multinode configures multinode components.
	// +optional
	Multinode *MultinodeSpec `json:"multinode,omitempty"`

	// sharedMemorySize controls the size of the tmpfs mounted at `/dev/shm`.
	// `nil` selects the operator default (8Gi), a positive quantity sets a
	// custom size, and `"0"` disables the shared-memory volume entirely.
	// Simpler replacement for v1alpha1's `SharedMemorySpec` struct with its
	// `disabled bool` + `size Quantity` pattern.
	// +optional
	SharedMemorySize *resource.Quantity `json:"sharedMemorySize,omitempty"`

	// modelRef references a model served by this component. When specified,
	// a headless service is created for endpoint discovery.
	// +optional
	ModelRef *ModelReference `json:"modelRef,omitempty"`

	// scalingAdapter opts this component into using the
	// DynamoGraphDeploymentScalingAdapter. When set (even as an empty object,
	// `scalingAdapter: {}`), a DGDSA is created and owns the `replicas` field
	// so that external autoscalers (HPA/KEDA/Planner) can drive scaling via
	// the Scale subresource. Omit the field to opt out.
	// +optional
	ScalingAdapter *ScalingAdapter `json:"scalingAdapter,omitempty"`

	// eppConfig holds EPP-specific configuration for Endpoint Picker Plugin
	// components. Only meaningful when `type` is `epp`.
	// +optional
	EPPConfig *EPPConfig `json:"eppConfig,omitempty"`

	// frontendSidecar optionally designates a container in
	// `podTemplate.spec.containers` as the frontend sidecar. The value must
	// match the `name` of a container in that list; the operator merges its
	// frontend-sidecar defaults (auto-generated Dynamo env vars, ports,
	// health probes) into that container the same way it merges into `"main"`.
	// The full container definition (image, args, envFrom, env) lives in
	// `podTemplate` -- this eliminates the redundant `image`, `args`,
	// `envFromSecret`, and `envs` fields from v1alpha1's `FrontendSidecarSpec`.
	// The validation webhook rejects values that do not match any container
	// name in `podTemplate.spec.containers`.
	// +optional
	FrontendSidecar *string `json:"frontendSidecar,omitempty"`

	// compilationCache configures a PVC-backed compilation cache. The operator
	// handles backend-specific mount paths and environment variables, so
	// users do not need to hand-wire them into `podTemplate`. Extracted from
	// v1alpha1's `volumeMount.useAsCompilationCache` flag.
	// +optional
	CompilationCache *CompilationCacheConfig `json:"compilationCache,omitempty"`

	// topologyConstraint applies to this component.
	// `topologyConstraint.packDomain` is required. When both this and
	// `spec.topologyConstraint.packDomain` are set, this field's `packDomain`
	// must be narrower than or equal to the spec-level value.
	// +optional
	TopologyConstraint *TopologyConstraint `json:"topologyConstraint,omitempty"`

	// experimental groups opt-in preview features whose API shape and
	// behavior may change in breaking ways between v1beta1 releases,
	// including disappearing without a name-preserving graduation path.
	// In v1beta1 this block holds `gpuMemoryService` and `failover` (which
	// remain tightly coupled -- failover requires GMS -- and are expected to
	// evolve together as the DRA-based GPU sharing story matures), and
	// `checkpoint` (whose interaction with the standalone DynamoCheckpoint
	// resource and identity-hash computation is still settling). Fields here
	// are explicitly NOT covered by the normal v1beta1 deprecation policy;
	// do not depend on them for production workloads.
	// +optional
	Experimental *ExperimentalSpec `json:"experimental,omitempty"`
}

// DynamoComponentDeploymentStatus defines the observed state of a DynamoComponentDeployment.
//
// Unchanged from v1alpha1 except that v1alpha1's unused `podSelector` field is
// removed. `podSelector` was never written or read by the operator controller.
type DynamoComponentDeploymentStatus struct {
	// observedGeneration is the most recent generation observed by the controller.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// conditions captures the latest observed state of the component using
	// standard Kubernetes condition types (including `Available` and
	// `DynamoComponentReady`).
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// component contains replica status information for this component.
	// +optional
	Component *ComponentReplicaStatus `json:"component,omitempty"`
}

// +genclient
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dcd
// +kubebuilder:printcolumn:name="Available",type="string",JSONPath=".status.conditions[?(@.type=='Available')].status",description="Available"
// +kubebuilder:printcolumn:name="Backend",type="string",JSONPath=`.spec.backendFramework`,description="Backend framework (sglang, vllm, trtllm)"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// DynamoComponentDeployment is the Schema for the dynamocomponentdeployments API.
//
// v1beta1 is a served version: the API server accepts reads and writes
// against it, and transparently converts to/from v1alpha1 (still the
// storage version until a later MR flips it). Conversion goes through the
// operator's conversion webhook; see api/v1alpha1/*_conversion.go.
type DynamoComponentDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// spec defines the desired state for this Dynamo component deployment.
	Spec DynamoComponentDeploymentSpec `json:"spec,omitempty"`
	// status reflects the current observed state of the component deployment.
	Status DynamoComponentDeploymentStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoComponentDeploymentList contains a list of DynamoComponentDeployment.
type DynamoComponentDeploymentList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoComponentDeployment `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoComponentDeployment{}, &DynamoComponentDeploymentList{})
}

// IsReady returns true if the component is `Available`.
func (s *DynamoComponentDeployment) IsReady() (bool, string) {
	return s.Status.IsReady()
}

// IsReady reports whether the component status signals `Available=True`.
func (s *DynamoComponentDeploymentStatus) IsReady() (bool, string) {
	for _, condition := range s.Conditions {
		if condition.Type == DynamoComponentDeploymentConditionTypeAvailable && condition.Status == metav1.ConditionTrue {
			return true, ""
		}
	}
	return false, "Component deployment not ready - Available condition not true"
}

// IsMultinode reports whether the component is configured with more than one node.
func (s *DynamoComponentDeployment) IsMultinode() bool {
	return s.GetNumberOfNodes() > 1
}

// GetNumberOfNodes returns the configured node count, defaulting to 1.
func (s *DynamoComponentDeployment) GetNumberOfNodes() int32 {
	return s.Spec.GetNumberOfNodes()
}

// IsMultinode reports whether this shared spec is configured with more than one node.
func (s *DynamoComponentDeploymentSharedSpec) IsMultinode() bool {
	return s.GetNumberOfNodes() > 1
}

// GetNumberOfNodes returns the configured node count, defaulting to 1.
func (s *DynamoComponentDeploymentSharedSpec) GetNumberOfNodes() int32 {
	if s.Multinode != nil {
		return s.Multinode.NodeCount
	}
	return 1
}

// IsFrontendComponent reports whether this DCD is the Dynamo frontend component.
func (s *DynamoComponentDeployment) IsFrontendComponent() bool {
	return s.Spec.ComponentType == ComponentTypeFrontend
}

// GetParentGraphDeploymentName returns the name of the owning DynamoGraphDeployment,
// or "" if this DCD is standalone.
func (s *DynamoComponentDeployment) GetParentGraphDeploymentName() string {
	for _, ownerRef := range s.ObjectMeta.OwnerReferences {
		if ownerRef.Kind == "DynamoGraphDeployment" {
			return ownerRef.Name
		}
	}
	return ""
}

// GetParentGraphDeploymentNamespace returns the namespace of the owning DGD,
// which is always the DCD's own namespace.
func (s *DynamoComponentDeployment) GetParentGraphDeploymentNamespace() string {
	return s.GetNamespace()
}

// GetDynamoNamespace returns the Dynamo namespace for this component.
func (s *DynamoComponentDeployment) GetDynamoNamespace() string {
	return ComputeDynamoNamespace(s.Spec.GlobalDynamoNamespace, s.GetNamespace(), s.GetParentGraphDeploymentName())
}

// ComputeDynamoNamespace is the single source of truth for computing the Dynamo namespace.
// When `globalDynamoNamespace` is true, returns the global namespace constant.
// Otherwise returns `{k8sNamespace}-{dgdName}` with dots in `dgdName` replaced
// by hyphens so that the namespace can safely form the first segment of
// endpoint paths.
func ComputeDynamoNamespace(globalDynamoNamespace bool, k8sNamespace, dgdName string) string {
	if globalDynamoNamespace {
		return commonconsts.GlobalDynamoNamespace
	}
	sanitized := strings.ReplaceAll(dgdName, ".", "-")
	return fmt.Sprintf("%s-%s", k8sNamespace, sanitized)
}

// GetSpec returns the spec as an interface, used by generic resource helpers.
func (s *DynamoComponentDeployment) GetSpec() any {
	return s.Spec
}

// SetSpec assigns the spec from an interface value.
func (s *DynamoComponentDeployment) SetSpec(spec any) {
	s.Spec = spec.(DynamoComponentDeploymentSpec)
}

// GetComponentStatuses returns per-component replica status. In the standalone
// DCD case the map contains a single entry keyed by the DCD's own name,
// mirroring the DGD behaviour and letting generic callers treat both the same
// way.
func (s *DynamoComponentDeployment) GetComponentStatuses() map[string]ComponentReplicaStatus {
	if s.Status.Component == nil {
		return map[string]ComponentReplicaStatus{}
	}
	return map[string]ComponentReplicaStatus{s.GetName(): *s.Status.Component}
}

// GetState returns "ready" or "not_ready" based on status conditions.
func (s *DynamoComponentDeployment) GetState() string {
	ready, _ := s.IsReady()
	if ready {
		return commonconsts.ResourceStateReady
	}
	return commonconsts.ResourceStateNotReady
}
