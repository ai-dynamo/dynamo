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
	batchv1 "k8s.io/api/batch/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtime "k8s.io/apimachinery/pkg/runtime"
)

// DGDRPhase represents the lifecycle phase of a DynamoGraphDeploymentRequest.
// +kubebuilder:validation:Enum=Pending;Profiling;Ready;Deploying;Deployed;Failed
type DGDRPhase string

const (
	DGDRPhasePending   DGDRPhase = "Pending"
	DGDRPhaseProfiling DGDRPhase = "Profiling"
	DGDRPhaseReady     DGDRPhase = "Ready"
	DGDRPhaseDeploying DGDRPhase = "Deploying"
	DGDRPhaseDeployed  DGDRPhase = "Deployed"
	DGDRPhaseFailed    DGDRPhase = "Failed"

	// Condition types
	// ConditionTypeSucceeded is the aggregate condition for the DGDR lifecycle.
	// True = pipeline completed successfully; False = in progress or failed.
	// Reason and Message reflect the current stage or error.
	ConditionTypeSucceeded = "Succeeded"

	ConditionTypeValidation      = "Validation"
	ConditionTypeProfiling       = "Profiling"
	ConditionTypeSpecGenerated   = "SpecGenerated"
	ConditionTypeDeploymentReady = "DeploymentReady"

	// Event reasons
	EventReasonInitialized          = "Initialized"
	EventReasonValidationFailed     = "ValidationFailed"
	EventReasonProfilingJobCreated  = "ProfilingJobCreated"
	EventReasonProfilingJobFailed   = "ProfilingJobFailed"
	EventReasonAIConfiguratorFailed = "AIConfiguratorFailed"
	EventReasonSpecGenerated        = "SpecGenerated"
	EventReasonSpecChangeRejected   = "SpecChangeRejected"
	EventReasonDeploymentCreated    = "DeploymentCreated"
	EventReasonDeploymentReady      = "DeploymentReady"
	EventReasonDeploymentDegraded   = "DeploymentDegraded"
	EventReasonDeploymentDeleted    = "DeploymentDeleted"

	// Label keys
	LabelApp           = "app"
	LabelDGDR          = "dgdr"
	LabelDGDRName      = "dgdr.nvidia.com/name"
	LabelDGDRNamespace = "dgdr.nvidia.com/namespace"
	LabelManagedBy     = "nvidia.com/managed-by"

	// Label values
	LabelValueDynamoProfiler = "dynamo-profiler"
	LabelValueAICProfiler    = "aic-profiler"
	LabelValueDynamoOperator = "dynamo-operator"
)

// ProfilingPhase represents a sub-phase within the profiling pipeline.
// When the DGDR Phase is "Profiling", this value indicates which step
// of the profiling pipeline is currently executing.
// +kubebuilder:validation:Enum=Initializing;SweepingPrefill;SweepingDecode;SelectingConfig;BuildingCurves;GeneratingDGD;Done
type ProfilingPhase string

const (
	// Profiler is loading the DGD template, detecting GPU hardware,
	// and resolving the model architecture from HuggingFace.
	ProfilingPhaseInitializing ProfilingPhase = "Initializing"

	// Sweeping parallelization strategies (TP/TEP/DEP) across GPU counts
	// for prefill, measuring TTFT at each configuration.
	ProfilingPhaseSweepingPrefill ProfilingPhase = "SweepingPrefill"

	// Sweeping parallelization strategies and concurrency levels
	// for decode, measuring ITL at each configuration.
	ProfilingPhaseSweepingDecode ProfilingPhase = "SweepingDecode"

	// Filtering results against SLA targets and selecting the most
	// cost-efficient configuration that meets TTFT/ITL requirements.
	ProfilingPhaseSelectingConfig ProfilingPhase = "SelectingConfig"

	// Building detailed interpolation curves (ISL→TTFT for prefill,
	// KV-usage×context-length→ITL for decode) using the selected configs.
	ProfilingPhaseBuildingCurves ProfilingPhase = "BuildingCurves"

	// Packaging profiling data into a ConfigMap and generating
	// the final DGD YAML with planner integration.
	ProfilingPhaseGeneratingDGD ProfilingPhase = "GeneratingDGD"

	// Profiling pipeline finished successfully.
	ProfilingPhaseDone ProfilingPhase = "Done"
)

// Profiling condition Reasons.
//
// Hybrid A+D approach: the status.profilingPhase field is the canonical source
// of the current profiling sub-phase, while the Profiling condition's Reason
// mirrors the phase for kubectl-describe readability. On failure, the Reason
// is set to "<Phase>Failed" to encode both the phase and the error in one field.
const (
	// ProfilingReasonInitializing indicates the profiler is loading the DGD template,
	// detecting GPU hardware, and resolving the model architecture.
	ProfilingReasonInitializing = "Initializing"

	// ProfilingReasonSweepingPrefill indicates the profiler is sweeping parallelization
	// strategies (TP/TEP/DEP) across GPU counts for prefill, measuring TTFT.
	ProfilingReasonSweepingPrefill = "SweepingPrefill"

	// ProfilingReasonSweepingDecode indicates the profiler is sweeping parallelization
	// strategies and concurrency levels for decode, measuring ITL.
	ProfilingReasonSweepingDecode = "SweepingDecode"

	// ProfilingReasonSelectingConfig indicates the profiler is filtering results against
	// SLA targets and selecting the most cost-efficient configuration.
	ProfilingReasonSelectingConfig = "SelectingConfig"

	// ProfilingReasonBuildingCurves indicates the profiler is building interpolation
	// curves (ISL→TTFT, KV-usage×context-length→ITL) for planner integration.
	ProfilingReasonBuildingCurves = "BuildingCurves"

	// ProfilingReasonGeneratingDGD indicates the profiler is packaging data into a
	// ConfigMap and generating the final DGD YAML.
	ProfilingReasonGeneratingDGD = "GeneratingDGD"

	// ProfilingReasonInitializingFailed indicates the initialization phase failed.
	ProfilingReasonInitializingFailed = "InitializingFailed"

	// ProfilingReasonSweepingPrefillFailed indicates the prefill sweep phase failed.
	ProfilingReasonSweepingPrefillFailed = "SweepingPrefillFailed"

	// ProfilingReasonSweepingDecodeFailed indicates the decode sweep phase failed.
	ProfilingReasonSweepingDecodeFailed = "SweepingDecodeFailed"

	// ProfilingReasonSelectingConfigFailed indicates the config selection phase failed.
	ProfilingReasonSelectingConfigFailed = "SelectingConfigFailed"

	// ProfilingReasonBuildingCurvesFailed indicates the curve-building phase failed.
	ProfilingReasonBuildingCurvesFailed = "BuildingCurvesFailed"

	// ProfilingReasonGeneratingDGDFailed indicates the DGD generation phase failed.
	ProfilingReasonGeneratingDGDFailed = "GeneratingDGDFailed"

	// ProfilingReasonCompleted indicates the profiling pipeline finished successfully.
	ProfilingReasonCompleted = "Completed"

	// ProfilingReasonJobCreationFailed indicates the Kubernetes Job for profiling
	// could not be created.
	ProfilingReasonJobCreationFailed = "JobCreationFailed"
)

// OptimizationType specifies the profiling optimization strategy.
// +kubebuilder:validation:Enum=latency;throughput
type OptimizationType string

const (
	OptimizationTypeLatency    OptimizationType = "latency"
	OptimizationTypeThroughput OptimizationType = "throughput"
)

// SearchStrategy controls the profiling search depth.
// +kubebuilder:validation:Enum=rapid;thorough
type SearchStrategy string

const (
	SearchStrategyRapid    SearchStrategy = "rapid"
	SearchStrategyThorough SearchStrategy = "thorough"
)

// BackendType specifies the inference backend.
// +kubebuilder:validation:Enum=auto;sglang;trtllm;vllm
type BackendType string

const (
	BackendTypeAuto   BackendType = "auto"
	BackendTypeSglang BackendType = "sglang"
	BackendTypeTrtllm BackendType = "trtllm"
	BackendTypeVllm   BackendType = "vllm"
)

// WorkloadSpec defines the workload characteristics for SLA-based profiling.
type WorkloadSpec struct {
	// ISL is the Input Sequence Length (number of tokens).
	// +optional
	// +kubebuilder:default=4000
	ISL *int32 `json:"isl,omitempty"`

	// OSL is the Output Sequence Length (number of tokens).
	// +optional
	// +kubebuilder:default=1000
	OSL *int32 `json:"osl,omitempty"`

	// Concurrency is the target concurrency level.
	// Required (or RequestRate) when the planner is disabled.
	// +optional
	Concurrency *float64 `json:"concurrency,omitempty"`

	// RequestRate is the target request rate (req/s).
	// Required (or Concurrency) when the planner is disabled.
	// +optional
	RequestRate *float64 `json:"requestRate,omitempty"`
}

// SLASpec defines the service-level agreement targets for profiling optimization.
// Exactly one mode should be active: ttft+itl (default), e2eLatency, or optimizationType.
type SLASpec struct {
	// OptimizationType controls the profiling optimization strategy.
	// Use when explicit SLA targets (ttft+itl or e2eLatency) are not known.
	// +optional
	// +kubebuilder:validation:Enum=latency;throughput
	OptimizationType OptimizationType `json:"optimizationType,omitempty"`

	// TTFT is the Time To First Token target in milliseconds.
	// +optional
	// +python-default=2000
	TTFT *float64 `json:"ttft,omitempty"`

	// ITL is the Inter-Token Latency target in milliseconds.
	// +optional
	// +python-default=30
	ITL *float64 `json:"itl,omitempty"`

	// E2ELatency is the target end-to-end request latency in milliseconds.
	// Alternative to specifying TTFT + ITL.
	// +optional
	E2ELatency *float64 `json:"e2eLatency,omitempty"`
}

// ModelCacheSpec references a PVC containing pre-downloaded model weights.
type ModelCacheSpec struct {
	// PVCName is the name of the PersistentVolumeClaim containing model weights.
	// The PVC must exist in the same namespace as the DGDR.
	// +optional
	PVCName string `json:"pvcName,omitempty"`

	// PVCModelPath is the path to the model checkpoint directory within the PVC
	// (e.g. "deepseek-r1" or "models/Llama-3.1-405B-FP8").
	// +optional
	PVCModelPath string `json:"pvcModelPath,omitempty"`

	// PVCMountPath is the mount path for the PVC inside the container.
	// +optional
	// +kubebuilder:default="/opt/model-cache"
	PVCMountPath string `json:"pvcMountPath,omitempty"`
}

// OverridesSpec allows customizing the profiling job and the generated DynamoGraphDeployment.
type OverridesSpec struct {
	// ProfilingJob allows overriding the profiling Job specification.
	// Fields set here are merged into the controller-generated Job spec.
	// +optional
	ProfilingJob *batchv1.JobSpec `json:"profilingJob,omitempty"`

	// DGD allows providing a full or partial nvidia.com/v1alpha1 DynamoGraphDeployment
	// to use as the base for the generated deployment. Fields from profiling results
	// are merged on top. Use this to override backend worker images.
	//
	// The field is stored as a raw embedded resource rather than a typed
	// *v1alpha1.DynamoGraphDeployment to avoid a circular import: v1alpha1 already
	// imports v1beta1 as the conversion hub and Go does not allow import cycles.
	//
	// The EmbeddedResource marker tells the API server to validate that the value is a
	// well-formed Kubernetes object (has apiVersion/kind), but does not enforce that it
	// is specifically a DynamoGraphDeployment. Full type validation (correct apiVersion,
	// kind, and field schema) is performed by the controller during reconciliation.
	// TODO(future MR): add webhook admission validation for the DGD field type.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:EmbeddedResource
	DGD *runtime.RawExtension `json:"dgd,omitempty"`
}

// PlannerPreDeploymentSweepMode controls pre-deployment sweeping thoroughness for planner profiling.
// +kubebuilder:validation:Enum=none;rapid;thorough
type PlannerPreDeploymentSweepMode string

const (
	PlannerPreDeploymentSweepModeNone     PlannerPreDeploymentSweepMode = "none"
	PlannerPreDeploymentSweepModeRapid    PlannerPreDeploymentSweepMode = "rapid"
	PlannerPreDeploymentSweepModeThorough PlannerPreDeploymentSweepMode = "thorough"
)

// PlannerSpec configures the SLA planner for autoscaling in the generated DGD.
type PlannerSpec struct {
	// Enabled indicates whether the planner is enabled.
	// +optional
	Enabled bool `json:"enabled,omitempty"`

	// PlannerPreDeploymentSweeping controls pre-deployment sweeping mode for planner in-depth profiling.
	// "none" means no pre-deployment sweep (only load-based scaling).
	// "rapid" uses AI Configurator to simulate engine performance.
	// "thorough" uses real GPUs to measure engine performance (takes several hours).
	// +optional
	// +kubebuilder:validation:Enum=none;rapid;thorough
	PlannerPreDeploymentSweeping *PlannerPreDeploymentSweepMode `json:"plannerPreDeploymentSweeping,omitempty"`

	// PlannerArgsList is a list of additional planner arguments.
	// +optional
	PlannerArgsList []string `json:"plannerArgsList,omitempty"`
}

// MockerSpec configures the simulated (mocker) backend.
type MockerSpec struct {
	// Enabled indicates whether to deploy mocker workers instead of real inference workers.
	// Useful for large-scale testing without GPUs.
	// +optional
	Enabled bool `json:"enabled,omitempty"`
}

// KVRouterSpec configures KV-cache-aware routing.
type KVRouterSpec struct {
	// Enabled indicates whether to enable KV-cache-aware routing in the generated DGD.
	// KV routing optimizes request scheduling based on KV cache locality.
	// +optional
	Enabled bool `json:"enabled,omitempty"`
}

// FeaturesSpec controls optional Dynamo platform features in the generated deployment.
type FeaturesSpec struct {
	// Planner configures the SLA planner for autoscaling in the generated DGD.
	// +optional
	Planner *PlannerSpec `json:"planner,omitempty"`

	// TODO: KVRouter support is not yet implemented in the operator.
	// KVRouter *KVRouterSpec `json:"kvRouter,omitempty"`

	// Mocker configures the simulated (mocker) backend for testing without GPUs.
	// +optional
	Mocker *MockerSpec `json:"mocker,omitempty"`
}

// HardwareSpec describes the hardware resources available for profiling and deployment.
// These fields are typically auto-filled by the operator from cluster discovery.
type HardwareSpec struct {
	// GPUSKU is the GPU SKU identifier (e.g., "H100_SXM", "A100_80GB").
	// +optional
	GPUSKU string `json:"gpuSku,omitempty"`

	// VRAMMB is the VRAM per GPU in MiB.
	// +optional
	VRAMMB *float64 `json:"vramMb,omitempty"`

	// TotalGPUs is the total number of GPUs available in the cluster.
	// +optional
	TotalGPUs *int32 `json:"totalGpus,omitempty"`

	// NumGPUsPerNode is the number of GPUs per node.
	// +optional
	NumGPUsPerNode *int32 `json:"numGpusPerNode,omitempty"`
}

// DynamoGraphDeploymentRequestSpec defines the desired state of a DynamoGraphDeploymentRequest.
// Only the Model field is required; all other fields are optional and have sensible defaults.
type DynamoGraphDeploymentRequestSpec struct {
	// Model specifies the model to deploy (e.g., "Qwen/Qwen3-0.6B", "meta-llama/Llama-3-70b").
	// Can be a HuggingFace ID or a private model name.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Model string `json:"model"`

	// Backend specifies the inference backend to use for profiling and deployment.
	// +optional
	// +kubebuilder:default=auto
	// +kubebuilder:validation:Enum=auto;sglang;trtllm;vllm
	Backend BackendType `json:"backend,omitempty"`

	// Image is the container image reference for the profiling job (frontend image).
	// Example: "nvcr.io/nvidia/dynamo-runtime:latest"
	// TODO: In a future MR, the operator will derive the backend inference image from the
	// backend type automatically; backend images can be overridden via overrides.dgd.
	// +optional
	Image string `json:"image,omitempty"`

	// ModelCache provides optional PVC configuration for pre-downloaded model weights.
	// When provided, weights are loaded from the PVC instead of downloading from HuggingFace.
	// +optional
	ModelCache *ModelCacheSpec `json:"modelCache,omitempty"`

	// Hardware describes the hardware resources available for profiling and deployment.
	// Typically auto-filled by the operator from cluster discovery.
	// +optional
	Hardware *HardwareSpec `json:"hardware,omitempty"`

	// Workload defines the expected workload characteristics for SLA-based profiling.
	// +optional
	Workload *WorkloadSpec `json:"workload,omitempty"`

	// SLA defines service-level agreement targets that drive profiling optimization.
	// +optional
	SLA *SLASpec `json:"sla,omitempty"`

	// Overrides allows customizing the profiling job and the generated DynamoGraphDeployment.
	// +optional
	Overrides *OverridesSpec `json:"overrides,omitempty"`

	// Features controls optional Dynamo platform features in the generated deployment.
	// +optional
	Features *FeaturesSpec `json:"features,omitempty"`

	// SearchStrategy controls the profiling search depth.
	// "rapid" performs a fast sweep; "thorough" explores more configurations.
	// +optional
	// +kubebuilder:default=rapid
	// +kubebuilder:validation:Enum=rapid;thorough
	SearchStrategy SearchStrategy `json:"searchStrategy,omitempty"`

	// AutoApply indicates whether to automatically create a DynamoGraphDeployment
	// after profiling completes. If false, the generated spec is stored in status
	// for manual review and application.
	// +optional
	// +kubebuilder:default=true
	AutoApply bool `json:"autoApply,omitempty"`
}

// ParetoConfig represents a single Pareto-optimal deployment configuration
// discovered during profiling.
type ParetoConfig struct {
	// Config is the full deployment configuration for this Pareto point.
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	Config runtime.RawExtension `json:"config"`
}

// ProfilingResultsStatus contains the output of the profiling process.
type ProfilingResultsStatus struct {
	// Pareto is the list of Pareto-optimal deployment configurations discovered during profiling.
	// Each entry represents a different cost/performance trade-off.
	// +optional
	Pareto []ParetoConfig `json:"pareto,omitempty"`

	// SelectedConfig is the recommended configuration chosen by the profiler
	// based on the SLA targets. This is the configuration used for deployment
	// when autoApply is true.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	SelectedConfig *runtime.RawExtension `json:"selectedConfig,omitempty"`
}

// DeploymentInfoStatus tracks the state of the deployed DynamoGraphDeployment.
type DeploymentInfoStatus struct {
	// Replicas is the desired number of replicas.
	// +optional
	Replicas *int32 `json:"replicas,omitempty"`

	// AvailableReplicas is the number of replicas that are available and ready.
	// +optional
	AvailableReplicas *int32 `json:"availableReplicas,omitempty"`
}

// DynamoGraphDeploymentRequestStatus represents the observed state of a DynamoGraphDeploymentRequest.
type DynamoGraphDeploymentRequestStatus struct {
	// Phase is the high-level lifecycle phase of the deployment request.
	// +optional
	Phase DGDRPhase `json:"phase,omitempty"`

	// ProfilingPhase indicates the current sub-phase of the profiling pipeline.
	// Only meaningful when Phase is "Profiling". Cleared when profiling completes or fails.
	// +optional
	ProfilingPhase ProfilingPhase `json:"profilingPhase,omitempty"`

	// DGDName is the name of the generated or created DynamoGraphDeployment.
	// +optional
	DGDName string `json:"dgdName,omitempty"`

	// ProfilingJobName is the name of the Kubernetes Job running the profiler.
	// +optional
	ProfilingJobName string `json:"profilingJobName,omitempty"`

	// Conditions contains the latest observed conditions of the deployment request.
	// Standard condition types include: Validated, ProfilingComplete, DeploymentReady.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// ProfilingResults contains the output of the profiling process including
	// Pareto-optimal configurations and the selected deployment configuration.
	// +optional
	ProfilingResults *ProfilingResultsStatus `json:"profilingResults,omitempty"`

	// DeploymentInfo tracks the state of the deployed DynamoGraphDeployment.
	// Populated when a DGD has been created (either via autoApply or manually).
	// +optional
	DeploymentInfo *DeploymentInfoStatus `json:"deploymentInfo,omitempty"`

	// ObservedGeneration is the most recent generation observed by the controller.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// DynamoGraphDeploymentRequest is the Schema for the dynamographdeploymentrequests API.
// It provides a simplified, SLA-driven interface for deploying inference models on Dynamo.
// Users specify a model and optional performance targets; the controller handles profiling,
// configuration selection, and deployment.
//
// Lifecycle:
//  1. Pending: Spec validated, preparing for profiling
//  2. Profiling: Profiling job is running to discover optimal configurations
//  3. Ready: Profiling complete, generated DGD spec available in status
//  4. Deploying: DGD is being created and rolled out (when autoApply=true)
//  5. Deployed: DGD is running and healthy
//  6. Failed: An unrecoverable error occurred
//
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dgdr
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.model`
// +kubebuilder:printcolumn:name="Backend",type=string,JSONPath=`.spec.backend`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Profiling",type=string,JSONPath=`.status.profilingPhase`
// +kubebuilder:printcolumn:name="Reason",type=string,JSONPath=`.status.conditions[?(@.type=="Succeeded")].reason`,priority=1
// +kubebuilder:printcolumn:name="Message",type=string,JSONPath=`.status.conditions[?(@.type=="Succeeded")].message`,priority=1
// +kubebuilder:printcolumn:name="DGD",type=string,JSONPath=`.status.dgdName`
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type DynamoGraphDeploymentRequest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired state for this deployment request.
	Spec DynamoGraphDeploymentRequestSpec `json:"spec,omitempty"`

	// Status reflects the current observed state of this deployment request.
	Status DynamoGraphDeploymentRequestStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoGraphDeploymentRequestList contains a list of DynamoGraphDeploymentRequest resources.
type DynamoGraphDeploymentRequestList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentRequest `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DynamoGraphDeploymentRequest{}, &DynamoGraphDeploymentRequestList{})
}

// SetPhase updates the Phase field in the DGDR status.
func (d *DynamoGraphDeploymentRequest) SetPhase(phase DGDRPhase) {
	d.Status.Phase = phase
}

// GetPhase returns the current lifecycle phase.
func (d *DynamoGraphDeploymentRequest) GetPhase() DGDRPhase {
	return d.Status.Phase
}

// SetProfilingPhase updates the profiling sub-phase.
func (d *DynamoGraphDeploymentRequest) SetProfilingPhase(phase ProfilingPhase) {
	d.Status.ProfilingPhase = phase
}

// ClearProfilingPhase resets the profiling sub-phase (e.g., on completion or failure).
func (d *DynamoGraphDeploymentRequest) ClearProfilingPhase() {
	d.Status.ProfilingPhase = ""
}

// AddStatusCondition adds or updates a condition in the status.
// Uses apimeta.SetStatusCondition to correctly preserve LastTransitionTime.
func (d *DynamoGraphDeploymentRequest) AddStatusCondition(condition metav1.Condition) {
	apimeta.SetStatusCondition(&d.Status.Conditions, condition)
}
