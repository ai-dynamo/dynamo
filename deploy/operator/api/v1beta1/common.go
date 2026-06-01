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

// ComponentType identifies the role of a Dynamo component within a graph.
// In v1beta1 this is a strict enum. Unlike v1alpha1 (where `subComponentType`
// was used as a workaround for disaggregated serving), `prefill` and `decode`
// are first-class values: users can set them directly and downstream consumers
// (e.g., the EPP) can filter on the pod label `nvidia.com/dynamo-component-type`.
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

const (
	DynamoGraphDeploymentConditionTypeAvailable            = "Available"
	DynamoGraphDeploymentConditionTypeDynamoComponentReady = "DynamoComponentReady"

	ConditionTypeTopologyLevelsAvailable      = "TopologyLevelsAvailable"
	ConditionReasonAllTopologyLevelsAvailable = "AllTopologyLevelsAvailable"
	ConditionReasonTopologyLevelsUnavailable  = "TopologyLevelsUnavailable"
	ConditionReasonTopologyDefinitionNotFound = "TopologyDefinitionNotFound"
	ConditionReasonTopologyConditionPending   = "TopologyConditionPending"
)

// CompilationCacheConfig configures a PVC-backed compilation cache for a component.
// The operator handles backend-specific mount paths and environment variables so
// users do not need to hand-wire them into the pod template.
type CompilationCacheConfig struct {
	// pvcName references a user-created PVC by name. The PVC must exist in
	// the same namespace as the DynamoGraphDeployment.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	PVCName string `json:"pvcName"`

	// mountPath overrides the backend-specific default mount path. When
	// empty, the operator selects a default appropriate for the backend
	// framework.
	// +optional
	MountPath string `json:"mountPath,omitempty"`
}

// MultinodeSpec configures a multinode component.
type MultinodeSpec struct {
	// nodeCount is the number of nodes to deploy for the multinode component.
	// Total GPUs used is `nodeCount * container GPU request`.
	// +optional
	// +kubebuilder:default=2
	// +kubebuilder:validation:Minimum=2
	NodeCount int32 `json:"nodeCount"`
}

// ModelReference identifies a model served by a component.
// When specified, a headless service is created for endpoint discovery.
type ModelReference struct {
	// name is the base model identifier (e.g. "llama-3-70b-instruct-v1").
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// revision is the model revision/version.
	// +optional
	Revision string `json:"revision,omitempty"`
}

// Restart specifies the restart policy for a graph deployment.
type Restart struct {
	// id is an arbitrary string that triggers a restart when changed. Any
	// modification to this value initiates a restart of the graph deployment
	// according to the configured strategy.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	ID string `json:"id"`

	// strategy specifies the restart strategy for the graph deployment.
	// +optional
	Strategy *RestartStrategy `json:"strategy,omitempty"`
}

// RestartStrategyType enumerates restart strategies.
type RestartStrategyType string

const (
	RestartStrategyTypeSequential RestartStrategyType = "Sequential"
	RestartStrategyTypeParallel   RestartStrategyType = "Parallel"
)

// RestartStrategy defines how components are restarted.
type RestartStrategy struct {
	// type specifies the restart strategy type.
	// +optional
	// +kubebuilder:validation:Enum=Sequential;Parallel
	// +kubebuilder:default=Sequential
	Type RestartStrategyType `json:"type,omitempty"`

	// order is the complete ordered set of component names for sequential
	// restarts. Omit or leave empty to use the controller's default order.
	// This field must not be set for parallel restarts.
	// +optional
	Order []string `json:"order,omitempty"`
}

// ScalingAdapter opts a component into using the DynamoGraphDeploymentScalingAdapter
// (DGDSA). When `scalingAdapter` is set on a component (even as an empty
// object, `scalingAdapter: {}`), the DGDSA is created and owns the `replicas`
// field so that external autoscalers (HPA/KEDA/Planner) can drive scaling via
// the Scale subresource. Omitting the field opts the component out.
type ScalingAdapter struct{}

// EPPConfig contains configuration for EPP (Endpoint Picker Plugin) components.
// +kubebuilder:validation:XValidation:rule="has(self.configMapRef) != has(self.config)",message="exactly one of configMapRef or config must be specified"
type EPPConfig struct {
	// configMapRef references a user-provided ConfigMap containing EPP
	// configuration. Mutually exclusive with `config`.
	// +optional
	ConfigMapRef *corev1.ConfigMapKeySelector `json:"configMapRef,omitempty"`

	// config allows specifying EPP `EndpointPickerConfig` directly as a
	// structured object. The operator marshals this to YAML and creates a
	// ConfigMap automatically. Mutually exclusive with `configMapRef`. One of
	// `configMapRef` or `config` must be specified.
	// +optional
	// +kubebuilder:validation:Type=object
	// +kubebuilder:pruning:PreserveUnknownFields
	Config *apixv1alpha1.EndpointPickerConfig `json:"config,omitempty"`
}

// GPUMemoryServiceMode selects the GMS deployment topology.
type GPUMemoryServiceMode string

const (
	// GMSModeIntraPod runs GMS as a sidecar within the same pod.
	GMSModeIntraPod GPUMemoryServiceMode = "IntraPod"
	// GMSModeInterPod runs GMS as rank-local pods that share GPUs through DRA.
	// Extra client pod rendering is reserved for a follow-up change.
	GMSModeInterPod GPUMemoryServiceMode = "InterPod"
)

// ExperimentalSpec groups opt-in preview features whose API shape and behavior
// may change in breaking ways between v1beta1 releases (including disappearing
// without a name-preserving graduation path). Fields placed under
// `experimental` are explicitly NOT covered by the normal v1beta1 deprecation
// policy and should not be relied on for production workloads. Features
// graduate out of this block (and become first-class fields on the shared
// spec) once their API is considered stable.
type ExperimentalSpec struct {
	// gpuMemoryService configures the GPU Memory Service (GMS). When set, GPU
	// access for GMS clients is managed via DRA.
	// +optional
	GPUMemoryService *GPUMemoryServiceSpec `json:"gpuMemoryService,omitempty"`

	// failover configures active-passive GPU failover for this component.
	// Requires `gpuMemoryService` to also be set, and `failover.mode` must
	// match `gpuMemoryService.mode` (enforced by the validation webhook).
	// +optional
	Failover *FailoverSpec `json:"failover,omitempty"`

	// checkpoint configures container-image snapshotting and restore for
	// this component. When set, the DGD controller can produce a DGD-scoped
	// DynamoCheckpoint CR and later restore pods in the same DGD generation
	// from that checkpoint for faster cold start. The user-facing shape of
	// this field is still settling, which is why it lives under `experimental`
	// in v1beta1 instead of at the top level.
	// +optional
	Checkpoint *ComponentCheckpointConfig `json:"checkpoint,omitempty"`
}

// GPUMemoryServiceSpec configures the GPU Memory Service (GMS) for a
// worker component. The operator injects GMS wiring and replaces the main
// container's GPU resources with a DRA `ResourceClaim` for shared GPU access.
// See ExperimentalSpec for the stability caveat.
//
// +kubebuilder:validation:XValidation:rule="!has(self.extraClientContainers) || size(self.extraClientContainers) == 0 || self.mode == 'IntraPod'",message="extraClientContainers is only supported with mode=IntraPod"
// +kubebuilder:validation:XValidation:rule="!has(self.extraClientPods) || size(self.extraClientPods) == 0 || self.mode == 'InterPod'",message="extraClientPods is only supported with mode=InterPod"
// +kubebuilder:validation:XValidation:rule="!has(self.extraClientPods) || size(self.extraClientPods) == 0",message="extraClientPods is reserved for inter-pod GMS and is not implemented yet"
type GPUMemoryServiceSpec struct {
	// mode selects the GMS deployment topology.
	// +optional
	// +kubebuilder:default=IntraPod
	// +kubebuilder:validation:Enum=IntraPod;InterPod
	Mode GPUMemoryServiceMode `json:"mode,omitempty"`
	// deviceClassName is the DRA `DeviceClass` to request GPUs from.
	// +optional
	// +kubebuilder:default="gpu.nvidia.com"
	DeviceClassName string `json:"deviceClassName,omitempty"`

	// extraClientContainers lists additional user-declared containers that should
	// be wired as GMS clients in service pods. Checkpoint Job clients are declared
	// under checkpoint.job.gmsClientContainers. In each rendered pod, only
	// matching container names are wired; absent names are ignored.
	// +optional
	// +listType=set
	// +kubebuilder:validation:items:MinLength=1
	// +kubebuilder:validation:items:MaxLength=63
	// +kubebuilder:validation:items:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`
	ExtraClientContainers []string `json:"extraClientContainers,omitempty"`

	// extraClientPods declares additional GMS client pods for inter-pod GMS. This field is
	// reserved for future use and is rejected until inter-pod client orchestration is wired.
	// +optional
	// +listType=map
	// +listMapKey=name
	ExtraClientPods []GMSClientPodSpec `json:"extraClientPods,omitempty"`
}

// GMSClientPodSpec declares an additional GMS client pod for inter-pod GMS.
type GMSClientPodSpec struct {
	// name identifies this client pod.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`
	Name string `json:"name"`

	// podTemplate configures the pod to run as a GMS client.
	// +kubebuilder:validation:Schemaless
	// +kubebuilder:validation:Type=object
	// +kubebuilder:pruning:PreserveUnknownFields
	PodTemplate corev1.PodTemplateSpec `json:"podTemplate"`
}

// FailoverSpec configures active-passive failover for a worker component.
// The main container is cloned into two engine containers (active + standby)
// sharing GPUs via DRA, and the standby acquires the flock when the active
// engine fails. Failover requires that gpuMemoryService is also set, and that
// failover.mode matches gpuMemoryService.mode. Also requires the
// `nvidia.com/dynamo-kube-discovery-mode: container` annotation on the DGD.
// See ExperimentalSpec for the stability caveat.
type FailoverSpec struct {
	// mode selects the failover deployment topology. Must match
	// `spec.experimental.gpuMemoryService.mode` (or
	// `spec.components[*].experimental.gpuMemoryService.mode` inside a
	// DynamoGraphDeployment).
	// +optional
	// +kubebuilder:default=IntraPod
	// +kubebuilder:validation:Enum=IntraPod;InterPod
	Mode GPUMemoryServiceMode `json:"mode,omitempty"`
	// numShadows is the number of shadow (standby) engine containers per
	// rank. Reserved for future use; the operator currently creates exactly
	// one shadow.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=1
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

// CheckpointStartupPolicy defines when worker pods should wait for a checkpoint.
// +kubebuilder:validation:Enum=Immediate;WaitForCheckpoint
type CheckpointStartupPolicy string

const (
	// CheckpointStartupPolicyImmediate starts workers immediately. The checkpoint
	// job runs in the background, and only pods created after the checkpoint is
	// Ready are restore-shaped by the pod-create mutating webhook.
	CheckpointStartupPolicyImmediate CheckpointStartupPolicy = "Immediate"
	// CheckpointStartupPolicyWaitForCheckpoint gates worker replicas until the
	// component's checkpoint is Ready, then starts them from the checkpoint.
	CheckpointStartupPolicyWaitForCheckpoint CheckpointStartupPolicy = "WaitForCheckpoint"
)

// ComponentCheckpointConfig configures checkpointing for a DGD component.
// +kubebuilder:validation:XValidation:rule="!has(self.job) || !has(self.checkpointRef) || size(self.checkpointRef) == 0",message="checkpoint.job cannot be set when checkpointRef is specified"
// +kubebuilder:validation:XValidation:rule="!has(self.job) || !has(self.mode) || self.mode == 'Auto'",message="checkpoint.job can only be set in Auto mode"
type ComponentCheckpointConfig struct {
	// mode defines how checkpoint creation is handled.
	// `Auto`: DGD controller creates the DynamoCheckpoint CR automatically.
	// `Manual`: user must create the DynamoCheckpoint CR.
	// +optional
	// +kubebuilder:default=Auto
	Mode CheckpointMode `json:"mode,omitempty"`

	// startupPolicy defines when normal worker replicas are started relative to
	// automatic checkpoint readiness.
	// `Immediate` (default): start workers cold immediately; later Pods restore
	// from the checkpoint once it is Ready.
	// `WaitForCheckpoint`: keep worker replicas at zero until the checkpoint is
	// Ready, then start them from the checkpoint.
	// +optional
	// +kubebuilder:default=Immediate
	StartupPolicy CheckpointStartupPolicy `json:"startupPolicy,omitempty"`

	// checkpointRef references an existing DynamoCheckpoint CR by `metadata.name`.
	// When set, this component's `identity` is ignored and the referenced
	// checkpoint is used directly.
	// +optional
	CheckpointRef *string `json:"checkpointRef,omitempty"`

	// Deprecated: identity is ignored by DGD-managed automatic checkpoints.
	// Automatic checkpoints are scoped to the owning DGD/component generation and
	// are never reused across DGDs.
	// +optional
	Identity *DynamoCheckpointIdentity `json:"identity,omitempty"`

	// targetContainerName is the workload container to snapshot and restore.
	// +optional
	// +kubebuilder:default=main
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`
	TargetContainerName string `json:"targetContainerName,omitempty"`

	// job customizes the checkpoint Job that is created in Auto mode.
	// +optional
	Job *ComponentCheckpointJobConfig `json:"job,omitempty"`
}

// ComponentCheckpointJobConfig customizes the checkpoint Job created for a DGD component.
type ComponentCheckpointJobConfig struct {
	// gmsClientContainers lists checkpoint Job containers that should receive
	// GMS client wiring. Requires gpuMemoryService on the component.
	// +optional
	// +listType=set
	// +kubebuilder:validation:items:MinLength=1
	// +kubebuilder:validation:items:MaxLength=63
	// +kubebuilder:validation:items:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`
	GMSClientContainers []string `json:"gmsClientContainers,omitempty"`

	// podTemplate customizes the checkpoint Job pod. The operator starts from the
	// selected workload container and merges this template so users can add helper
	// containers such as gms-saver.
	// +optional
	// +kubebuilder:validation:Schemaless
	// +kubebuilder:validation:Type=object
	// +kubebuilder:pruning:PreserveUnknownFields
	PodTemplate *corev1.PodTemplateSpec `json:"podTemplate,omitempty"`
}

// DynamoCheckpointIdentity is legacy compatibility metadata retained for the
// v1alpha1 standalone DynamoCheckpoint shape. DGD-managed automatic checkpoints
// do not use this as a reuse boundary.
// Duplicated from v1alpha1 to keep the v1beta1 type graph self-contained. The
// DynamoCheckpoint resource itself is not graduating in this MR; this type is
// only used as a sub-field of `ComponentCheckpointConfig`.
type DynamoCheckpointIdentity struct {
	// model is the model identifier (e.g. "meta-llama/Llama-3-70B").
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Model string `json:"model"`

	// backendFramework is the runtime framework (`vllm`, `sglang`, `trtllm`).
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=vllm;sglang;trtllm
	BackendFramework string `json:"backendFramework"`

	// dynamoVersion is the Dynamo platform version. Deprecated for DGD-managed
	// automatic checkpoints; it only participates in the legacy identity hash
	// fallback for standalone objects.
	// +optional
	DynamoVersion string `json:"dynamoVersion,omitempty"`

	// tensorParallelSize is the tensor parallel configuration.
	// Deprecated for DGD-managed automatic checkpoints; it only participates in
	// the legacy identity hash fallback for standalone objects.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	TensorParallelSize int32 `json:"tensorParallelSize,omitempty"`

	// pipelineParallelSize is the pipeline parallel configuration.
	// Deprecated for DGD-managed automatic checkpoints; it only participates in
	// the legacy identity hash fallback for standalone objects.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:default=1
	PipelineParallelSize int32 `json:"pipelineParallelSize,omitempty"`

	// dtype is the data type (`fp16`, `bf16`, `fp8`, etc.).
	// Deprecated for DGD-managed automatic checkpoints; it only participates in
	// the legacy identity hash fallback for standalone objects.
	// +optional
	Dtype string `json:"dtype,omitempty"`

	// maxModelLen is the maximum sequence length.
	// Deprecated for DGD-managed automatic checkpoints; it only participates in
	// the legacy identity hash fallback for standalone objects.
	// +optional
	// +kubebuilder:validation:Minimum=1
	MaxModelLen int32 `json:"maxModelLen,omitempty"`

	// extraParameters are additional parameters that affect the checkpoint hash.
	// Deprecated for DGD-managed automatic checkpoints; it only participates in
	// the legacy identity hash fallback for standalone objects.
	// +optional
	ExtraParameters map[string]string `json:"extraParameters,omitempty"`
}

// SpecTopologyConstraint defines deployment-level topology placement requirements.
type SpecTopologyConstraint struct {
	// clusterTopologyName is the name of the ClusterTopology resource that
	// defines the topology hierarchy for this deployment.
	// +kubebuilder:validation:MinLength=1
	ClusterTopologyName string `json:"clusterTopologyName"`

	// packDomain is the default topology domain to pack pods within.
	// Optional; omit when only components carry constraints.
	// +optional
	PackDomain TopologyDomain `json:"packDomain,omitempty"`
}

// TopologyConstraint defines component-level topology placement requirements.
// The topology profile is inherited from the deployment-level
// `SpecTopologyConstraint`.
type TopologyConstraint struct {
	// packDomain is the topology domain to pack pods within. Must match a
	// domain defined in the referenced ClusterTopology CR.
	PackDomain TopologyDomain `json:"packDomain"`
}

// TopologyDomain is a free-form topology level identifier.
// Common examples: "region", "zone", "datacenter", "block", "rack", "host", "numa".
// When used with a ClusterTopology CR, domain names are defined in the CR's
// hierarchy; when used with `spec.experimental.kvTransferPolicy.labelKey`
// alone, the value is a user-chosen logical name for the topology level.
// +kubebuilder:validation:Pattern=`^[a-z0-9]([a-z0-9-]*[a-z0-9])?$`
type TopologyDomain string

// KvTransferEnforcement controls how the selected prefill worker's topology is
// applied to decode routing.
// +kubebuilder:validation:Enum=required;preferred
type KvTransferEnforcement string

const (
	// KvTransferEnforcementRequired enforces same-domain decode worker
	// selection.
	KvTransferEnforcementRequired KvTransferEnforcement = "required"
	// KvTransferEnforcementPreferred biases decode worker selection toward the
	// same domain.
	KvTransferEnforcementPreferred KvTransferEnforcement = "preferred"
)

// KvTransferPolicy configures topology-aware routing for KV-cache transfers
// between prefill and decode workers. This is a graph-wide concern placed
// under `spec.experimental` while the API is incubating.
// +kubebuilder:validation:XValidation:rule="has(self.labelKey)",message="labelKey is required until alternate topology sources are supported"
// +kubebuilder:validation:XValidation:rule="!has(self.enforcement) || self.enforcement != 'preferred' || has(self.preferredWeight)",message="preferredWeight is required when enforcement is preferred"
// +kubebuilder:validation:XValidation:rule="!has(self.preferredWeight) || (has(self.enforcement) && self.enforcement == 'preferred')",message="preferredWeight may only be set when enforcement is preferred"
type KvTransferPolicy struct {
	// labelKey is a Kubernetes node label key (e.g.
	// "topology.kubernetes.io/zone") whose value identifies the topology
	// domain for each worker. The operator copies the node label onto worker
	// pods so the runtime can publish it as worker metadata. The label
	// should correspond to the topology level named in `domain`.
	// +optional
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=317
	// +kubebuilder:validation:Pattern=`^(([a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?)(\.[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?)*/)?([A-Za-z0-9]([-A-Za-z0-9_.]{0,61}[A-Za-z0-9])?)$`
	// +kubebuilder:validation:XValidation:rule="!self.contains('/') || self.split('/')[0].size() <= 253",message="labelKey prefix must be 253 characters or less"
	LabelKey string `json:"labelKey,omitempty"`

	// domain is the logical name for the topology level to enforce
	// (e.g. "zone", "rack"). The router uses this to match workers that
	// share the same value for the label identified by `labelKey`.
	Domain TopologyDomain `json:"domain"`

	// enforcement controls how the selected prefill worker's topology is
	// applied to decode routing. "required" only allows decode workers in the
	// same topology domain as the selected prefill worker. "preferred" keeps
	// all decode workers eligible, but biases selection toward workers in the
	// same topology domain. Defaults to "required".
	// +optional
	// +kubebuilder:default=required
	Enforcement KvTransferEnforcement `json:"enforcement,omitempty"`

	// preferredWeight is required and used only when enforcement is
	// "preferred". Higher values create a stronger same-domain routing
	// preference, but do not guarantee same-domain selection. The value is not
	// a probability; worker selection still depends on load and other routing
	// inputs. A value of 0 disables the topology preference; 1 is the strongest
	// supported preference.
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	PreferredWeight *float32 `json:"preferredWeight,omitempty"`
}

// ComponentKind represents the type of underlying Kubernetes resource backing a DGD component.
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
	// observedID is the restart ID currently being processed. Matches `Restart.id` in the spec.
	ObservedID string `json:"observedID,omitempty"`
	// phase is the phase of the restart.
	Phase RestartPhase `json:"phase,omitempty"`
	// inProgress contains the names of the components currently being restarted.
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
	RollingUpdatePhaseFailed     RollingUpdatePhase = "Failed"
	RollingUpdatePhaseNone       RollingUpdatePhase = ""
)

// RollingUpdateStatus tracks the progress of an operator-managed rolling update.
type RollingUpdateStatus struct {
	// phase indicates the current phase of the rolling update.
	// +optional
	Phase RollingUpdatePhase `json:"phase,omitempty"`

	// startTime is when the rolling update began.
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// endTime is when the rolling update completed (successfully or failed).
	// +optional
	EndTime *metav1.Time `json:"endTime,omitempty"`

	// updatedComponents is the list of components that have completed the
	// rolling update.
	// +optional
	UpdatedComponents []string `json:"updatedComponents,omitempty"`
}

// ComponentCheckpointStatus contains checkpoint information for a single component.
type ComponentCheckpointStatus struct {
	// checkpointName is the name of the associated DynamoCheckpoint CR.
	// +optional
	CheckpointName string `json:"checkpointName,omitempty"`
	// checkpointID is the artifact ID used by the snapshot protocol.
	// +optional
	CheckpointID string `json:"checkpointID,omitempty"`
	// identityHash is the computed hash of the checkpoint identity.
	// Deprecated: automatic checkpoints use checkpointID. This field is retained
	// for older status consumers.
	// +optional
	IdentityHash string `json:"identityHash,omitempty"`
	// ready indicates the checkpoint artifact is ready for future pods to restore.
	// +optional
	Ready bool `json:"ready,omitempty"`
}

// ComponentReplicaStatus contains replica information for a single component.
type ComponentReplicaStatus struct {
	// componentKind is the underlying resource kind (e.g. `PodClique`,
	// `Deployment`, `LeaderWorkerSet`).
	ComponentKind ComponentKind `json:"componentKind"`

	// componentNames is the list of underlying Kubernetes resource names for
	// this Dynamo component. During normal operation this contains a single
	// name; during rolling updates it contains both old and new resource names.
	// +optional
	ComponentNames []string `json:"componentNames,omitempty"`

	// replicas is the total number of non-terminated replicas.
	// +kubebuilder:validation:Minimum=0
	Replicas int32 `json:"replicas"`

	// updatedReplicas is the number of replicas at the current/desired revision.
	// +kubebuilder:validation:Minimum=0
	UpdatedReplicas int32 `json:"updatedReplicas"`

	// readyReplicas is the number of ready replicas. Populated for
	// `PodClique`, `Deployment`, and `LeaderWorkerSet`; not available for
	// `PodCliqueScalingGroup`.
	// +optional
	// +kubebuilder:validation:Minimum=0
	ReadyReplicas *int32 `json:"readyReplicas,omitempty"`

	// availableReplicas is the number of available replicas. Populated for
	// `Deployment` and `PodCliqueScalingGroup`; not available for
	// `PodClique` or `LeaderWorkerSet`.
	// +optional
	// +kubebuilder:validation:Minimum=0
	AvailableReplicas *int32 `json:"availableReplicas,omitempty"`
}
