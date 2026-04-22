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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

// ComponentType identifies the role of a Dynamo service within a graph.
// In v1beta1 this is a strict enum. Unlike v1alpha1 (where `subComponentType` was
// used as a workaround for disaggregated serving), `prefill` and `decode` are
// first-class values: users can set them directly and downstream consumers (e.g.,
// the EPP) can filter on the pod label `nvidia.com/dynamo-component-type`.
// +kubebuilder:validation:Enum=frontend;worker;prefill;decode;planner;epp
type ComponentType string

const (
	ComponentTypeFrontend ComponentType = "frontend"
	ComponentTypeWorker   ComponentType = "worker"
	ComponentTypePrefill  ComponentType = "prefill"
	ComponentTypeDecode   ComponentType = "decode"
	ComponentTypePlanner  ComponentType = "planner"
	ComponentTypeEPP      ComponentType = "epp"
)

// CompilationCacheConfig configures a PVC-backed compilation cache for a service.
// The operator handles backend-specific mount paths and environment variables so
// users do not need to hand-wire them into the pod template.
type CompilationCacheConfig struct {
	// PVCName references a user-created PVC by name. The PVC must exist in the
	// same namespace as the DynamoGraphDeployment.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	PVCName string `json:"pvcName"`

	// MountPath overrides the backend-specific default mount path.
	// When empty, the operator selects a default appropriate for the backend framework.
	// +optional
	MountPath string `json:"mountPath,omitempty"`
}

// MultinodeSpec configures a multinode component.
type MultinodeSpec struct {
	// NodeCount is the number of nodes to deploy for the multinode component.
	// Total GPUs used is NodeCount * container GPU request.
	// +kubebuilder:default=2
	// +kubebuilder:validation:Minimum=2
	NodeCount int32 `json:"nodeCount"`
}

// ModelReference identifies a model served by a component.
// When specified, a headless service is created for endpoint discovery.
type ModelReference struct {
	// Name is the base model identifier (e.g. "llama-3-70b-instruct-v1").
	// +kubebuilder:validation:Required
	Name string `json:"name"`

	// Revision is the model revision/version.
	// +optional
	Revision string `json:"revision,omitempty"`
}

// Restart specifies the restart policy for a graph deployment.
type Restart struct {
	// ID is an arbitrary string that triggers a restart when changed.
	// Any modification to this value initiates a restart of the graph deployment
	// according to the configured strategy.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	ID string `json:"id"`

	// Strategy specifies the restart strategy for the graph deployment.
	// +optional
	Strategy *RestartStrategy `json:"strategy,omitempty"`
}

// RestartStrategyType enumerates restart strategies.
type RestartStrategyType string

const (
	RestartStrategyTypeSequential RestartStrategyType = "Sequential"
	RestartStrategyTypeParallel   RestartStrategyType = "Parallel"
)

// RestartStrategy defines how services are restarted.
type RestartStrategy struct {
	// Type specifies the restart strategy type.
	// +kubebuilder:validation:Enum=Sequential;Parallel
	// +kubebuilder:default=Sequential
	Type RestartStrategyType `json:"type,omitempty"`

	// Order specifies the order in which services should be restarted.
	// +optional
	Order []string `json:"order,omitempty"`
}

// ScalingAdapter configures whether a service uses the DynamoGraphDeploymentScalingAdapter.
// When enabled, the DGDSA owns the `replicas` field and external autoscalers
// (HPA/KEDA/Planner) can control scaling via the Scale subresource.
type ScalingAdapter struct {
	// Enabled indicates whether the ScalingAdapter should be enabled for this service.
	// +optional
	// +kubebuilder:default=false
	Enabled bool `json:"enabled,omitempty"`
}

// EPPConfig contains configuration for EPP (Endpoint Picker Plugin) components.
type EPPConfig struct {
	// ConfigMapRef references a user-provided ConfigMap containing EPP configuration.
	// Mutually exclusive with Config.
	// +optional
	ConfigMapRef *corev1.ConfigMapKeySelector `json:"configMapRef,omitempty"`

	// Config allows specifying EPP EndpointPickerConfig directly as a structured object.
	// The operator marshals this to YAML and creates a ConfigMap automatically.
	// Mutually exclusive with ConfigMapRef. One of ConfigMapRef or Config must be specified.
	// +optional
	// +kubebuilder:validation:Type=object
	// +kubebuilder:pruning:PreserveUnknownFields
	Config *apixv1alpha1.EndpointPickerConfig `json:"config,omitempty"`
}

// GPUMemoryServiceMode selects the GMS deployment topology.
type GPUMemoryServiceMode string

const (
	// GMSModeIntraPod runs GMS as a sidecar within the same pod.
	GMSModeIntraPod GPUMemoryServiceMode = "intraPod"
	// GMSModeInterPod runs GMS as a separate pod (not yet supported).
	GMSModeInterPod GPUMemoryServiceMode = "interPod"
)

// ExperimentalSpec groups opt-in preview features whose API shape and behavior
// may change in breaking ways between v1beta1 releases (including disappearing
// without a name-preserving graduation path). Fields placed under
// `experimental` are explicitly NOT covered by the normal v1beta1 deprecation
// policy and should not be relied on for production workloads. Features
// graduate out of this block (and become first-class fields on the shared
// spec) once their API is considered stable.
type ExperimentalSpec struct {
	// GPUMemoryService configures the GPU Memory Service (GMS) sidecar.
	// When enabled, a GMS sidecar is injected and GPU access is managed via DRA.
	// +optional
	GPUMemoryService *GPUMemoryServiceSpec `json:"gpuMemoryService,omitempty"`

	// Failover configures active-passive GPU failover for this service.
	// When enabled, the main container is cloned into two engine containers
	// (active + standby) sharing GPUs via DRA. Requires
	// `experimental.gpuMemoryService.enabled`, and its `mode` must match
	// `experimental.gpuMemoryService.mode` (enforced by the validation webhook).
	// +optional
	Failover *FailoverSpec `json:"failover,omitempty"`
}

// GPUMemoryServiceSpec configures the GPU Memory Service (GMS) sidecar for a worker component.
// When enabled, the operator injects a GMS sidecar that provides shared GPU memory access
// via DRA (Dynamic Resource Allocation).
//
// Exposed under `DynamoComponentDeploymentSharedSpec.Experimental.GPUMemoryService`
// in v1beta1 -- see ExperimentalSpec for the stability caveat.
type GPUMemoryServiceSpec struct {
	// Enabled activates the GMS sidecar. GPU resources on the main container
	// are replaced with a DRA ResourceClaim for shared GPU access.
	Enabled bool `json:"enabled"`
	// Mode selects the GMS deployment topology.
	// +kubebuilder:default=intraPod
	// +kubebuilder:validation:Enum=intraPod;interPod
	// +optional
	Mode GPUMemoryServiceMode `json:"mode,omitempty"`
	// DeviceClassName is the DRA DeviceClass to request GPUs from.
	// +kubebuilder:default="gpu.nvidia.com"
	// +optional
	DeviceClassName string `json:"deviceClassName,omitempty"`
}

// FailoverSpec configures active-passive failover for a worker component.
// Requires `experimental.gpuMemoryService.enabled` and the
// `nvidia.com/dynamo-kube-discovery-mode: container` annotation on the DGD.
//
// Exposed under `DynamoComponentDeploymentSharedSpec.Experimental.Failover`
// in v1beta1 -- see ExperimentalSpec for the stability caveat.
type FailoverSpec struct {
	// Enabled activates failover mode. The main container is cloned into two
	// engine containers (active + standby) sharing GPUs via DRA. The standby
	// acquires the flock when the active engine fails.
	Enabled bool `json:"enabled"`
	// Mode selects the failover deployment topology. Must match gpuMemoryService.mode.
	// +kubebuilder:default=intraPod
	// +kubebuilder:validation:Enum=intraPod;interPod
	// +optional
	Mode GPUMemoryServiceMode `json:"mode,omitempty"`
	// NumShadows is the number of shadow (standby) engine containers per rank.
	// Reserved for future use; the operator currently creates exactly one shadow.
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=1
	// +optional
	NumShadows int32 `json:"numShadows,omitempty"`
}

// CheckpointMode defines how checkpoint creation is handled.
// +kubebuilder:validation:Enum=Auto;Manual
type CheckpointMode string

const (
	// CheckpointModeAuto means the DGD controller creates the DynamoCheckpoint CR automatically.
	CheckpointModeAuto CheckpointMode = "Auto"
	// CheckpointModeManual means the user creates the DynamoCheckpoint CR themselves.
	CheckpointModeManual CheckpointMode = "Manual"
)

// ServiceCheckpointConfig configures checkpointing for a DGD service.
// +kubebuilder:validation:XValidation:rule="!self.enabled || (has(self.checkpointRef) && size(self.checkpointRef) > 0) || (has(self.identity) && has(self.identity.model) && has(self.identity.backendFramework))",message="When enabled, either checkpointRef or both identity.model and identity.backendFramework must be specified"
type ServiceCheckpointConfig struct {
	// Enabled indicates whether checkpointing is enabled for this service.
	// +optional
	// +kubebuilder:default=false
	Enabled bool `json:"enabled,omitempty"`

	// Mode defines how checkpoint creation is handled.
	// Auto: DGD controller creates the DynamoCheckpoint CR automatically.
	// Manual: user must create the DynamoCheckpoint CR.
	// +optional
	// +kubebuilder:default=Auto
	Mode CheckpointMode `json:"mode,omitempty"`

	// CheckpointRef references an existing DynamoCheckpoint CR by metadata.name.
	// When set, this service's Identity is ignored and the referenced checkpoint is used directly.
	// +optional
	CheckpointRef *string `json:"checkpointRef,omitempty"`

	// Identity defines the checkpoint identity for hash computation.
	// Used when Mode is Auto or when looking up existing checkpoints.
	// Required when checkpointRef is not specified.
	// +optional
	Identity *DynamoCheckpointIdentity `json:"identity,omitempty"`
}

// DynamoCheckpointIdentity defines the inputs that determine checkpoint equivalence.
// Two checkpoints with the same identity hash are considered equivalent.
// Duplicated from v1alpha1 to keep the v1beta1 type graph self-contained. The
// DynamoCheckpoint resource itself is not graduating in this MR; this type is
// only used as a sub-field of ServiceCheckpointConfig.
type DynamoCheckpointIdentity struct {
	// Model is the model identifier (e.g. "meta-llama/Llama-3-70B").
	// +kubebuilder:validation:Required
	Model string `json:"model"`

	// BackendFramework is the runtime framework (vllm, sglang, trtllm).
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=vllm;sglang;trtllm
	BackendFramework string `json:"backendFramework"`

	// DynamoVersion is the Dynamo platform version. If empty, the version is not
	// included in the identity hash, so checkpoints remain compatible across releases.
	// +optional
	DynamoVersion string `json:"dynamoVersion,omitempty"`

	// TensorParallelSize is the tensor parallel configuration.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	TensorParallelSize int32 `json:"tensorParallelSize,omitempty"`

	// PipelineParallelSize is the pipeline parallel configuration.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	PipelineParallelSize int32 `json:"pipelineParallelSize,omitempty"`

	// Dtype is the data type (fp16, bf16, fp8, etc.).
	// +optional
	Dtype string `json:"dtype,omitempty"`

	// MaxModelLen is the maximum sequence length.
	// +optional
	// +kubebuilder:validation:Minimum=1
	MaxModelLen int32 `json:"maxModelLen,omitempty"`

	// ExtraParameters are additional parameters that affect the checkpoint hash.
	// +optional
	ExtraParameters map[string]string `json:"extraParameters,omitempty"`
}

// SpecTopologyConstraint defines deployment-level topology placement requirements.
type SpecTopologyConstraint struct {
	// TopologyProfile is the name of the ClusterTopology CR that defines the
	// topology hierarchy for this deployment.
	// +kubebuilder:validation:MinLength=1
	TopologyProfile string `json:"topologyProfile"`

	// PackDomain is the default topology domain to pack pods within.
	// Optional; omit when only services carry constraints.
	// +optional
	PackDomain TopologyDomain `json:"packDomain,omitempty"`
}

// TopologyConstraint defines service-level topology placement requirements.
// The topology profile is inherited from the deployment-level SpecTopologyConstraint.
type TopologyConstraint struct {
	// PackDomain is the topology domain to pack pods within. Must match a
	// domain defined in the referenced ClusterTopology CR.
	PackDomain TopologyDomain `json:"packDomain"`
}

// TopologyDomain is a free-form topology level identifier.
// Domain names are defined by the cluster admin in the ClusterTopology CR.
// Common examples: "region", "zone", "datacenter", "block", "rack", "host", "numa".
// +kubebuilder:validation:Pattern=`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`
type TopologyDomain string

// ComponentKind represents the type of underlying Kubernetes resource backing a DGD service.
// +kubebuilder:validation:Enum=PodClique;PodCliqueScalingGroup;Deployment;LeaderWorkerSet
type ComponentKind string

const (
	ComponentKindPodClique             ComponentKind = "PodClique"
	ComponentKindPodCliqueScalingGroup ComponentKind = "PodCliqueScalingGroup"
	ComponentKindDeployment            ComponentKind = "Deployment"
	ComponentKindLeaderWorkerSet       ComponentKind = "LeaderWorkerSet"
)

// DGDState is the high-level lifecycle state of a DynamoGraphDeployment.
// +kubebuilder:validation:Enum=initializing;pending;successful;failed
type DGDState string

const (
	DGDStateInitializing DGDState = "initializing"
	DGDStatePending      DGDState = "pending"
	DGDStateSuccessful   DGDState = "successful"
	DGDStateFailed       DGDState = "failed"
)

// RestartPhase enumerates phases of a graph-level restart.
type RestartPhase string

const (
	RestartPhasePending    RestartPhase = "Pending"
	RestartPhaseRestarting RestartPhase = "Restarting"
	RestartPhaseCompleted  RestartPhase = "Completed"
	RestartPhaseFailed     RestartPhase = "Failed"
	RestartPhaseSuperseded RestartPhase = "Superseded"
)

// RestartStatus contains the status of a graph-level restart.
type RestartStatus struct {
	// ObservedID is the restart ID currently being processed. Matches Restart.ID in the spec.
	ObservedID string `json:"observedID,omitempty"`
	// Phase is the phase of the restart.
	Phase RestartPhase `json:"phase,omitempty"`
	// InProgress contains the names of the services currently being restarted.
	// +optional
	InProgress []string `json:"inProgress,omitempty"`
}

// RollingUpdatePhase represents the current phase of a rolling update.
// +kubebuilder:validation:Enum=Pending;InProgress;Completed;Failed;""
type RollingUpdatePhase string

const (
	RollingUpdatePhasePending    RollingUpdatePhase = "Pending"
	RollingUpdatePhaseInProgress RollingUpdatePhase = "InProgress"
	RollingUpdatePhaseCompleted  RollingUpdatePhase = "Completed"
	RollingUpdatePhaseNone       RollingUpdatePhase = ""
)

// RollingUpdateStatus tracks the progress of an operator-managed rolling update.
type RollingUpdateStatus struct {
	// Phase indicates the current phase of the rolling update.
	// +optional
	Phase RollingUpdatePhase `json:"phase,omitempty"`

	// StartTime is when the rolling update began.
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// EndTime is when the rolling update completed (successfully or failed).
	// +optional
	EndTime *metav1.Time `json:"endTime,omitempty"`

	// UpdatedServices is the list of services that have completed the rolling update.
	// +optional
	UpdatedServices []string `json:"updatedServices,omitempty"`
}

// ServiceCheckpointStatus contains checkpoint information for a single service.
type ServiceCheckpointStatus struct {
	// CheckpointName is the name of the associated Checkpoint CR.
	// +optional
	CheckpointName string `json:"checkpointName,omitempty"`
	// IdentityHash is the computed hash of the checkpoint identity.
	// +optional
	IdentityHash string `json:"identityHash,omitempty"`
	// Ready indicates if the checkpoint was visible to the worker at startup.
	// +optional
	Ready bool `json:"ready,omitempty"`
}

// ServiceReplicaStatus contains replica information for a single service.
type ServiceReplicaStatus struct {
	// ComponentKind is the underlying resource kind (e.g. "PodClique", "Deployment", "LeaderWorkerSet").
	ComponentKind ComponentKind `json:"componentKind"`

	// ComponentName is the name of the primary underlying resource.
	// Deprecated: use ComponentNames. During rolling updates this reflects the new (target) component name.
	// +kubebuilder:deprecatedversion:warning="ComponentName is deprecated, view ComponentNames instead"
	ComponentName string `json:"componentName"`

	// ComponentNames is the list of underlying resource names for this service.
	// During normal operation this contains a single name; during rolling updates it
	// contains both old and new component names.
	// +optional
	ComponentNames []string `json:"componentNames,omitempty"`

	// Replicas is the total number of non-terminated replicas.
	// +kubebuilder:validation:Minimum=0
	Replicas int32 `json:"replicas"`

	// UpdatedReplicas is the number of replicas at the current/desired revision.
	// +kubebuilder:validation:Minimum=0
	UpdatedReplicas int32 `json:"updatedReplicas"`

	// ReadyReplicas is the number of ready replicas. Populated for PodClique,
	// Deployment, and LeaderWorkerSet; not available for PodCliqueScalingGroup.
	// +optional
	// +kubebuilder:validation:Minimum=0
	ReadyReplicas *int32 `json:"readyReplicas,omitempty"`

	// AvailableReplicas is the number of available replicas. Populated for Deployment
	// and PodCliqueScalingGroup; not available for PodClique or LeaderWorkerSet.
	// +optional
	// +kubebuilder:validation:Minimum=0
	AvailableReplicas *int32 `json:"availableReplicas,omitempty"`
}
