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
// +kubebuilder:validation:Enum=hybrid;latency;throughput
type OptimizationType string

const (
	OptimizationTypeHybrid     OptimizationType = "hybrid"
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
	BackendTypeSGLang BackendType = "sglang"
	BackendTypeTRTLLM BackendType = "trtllm"
	BackendTypeVLLM   BackendType = "vllm"
)

// WorkloadSpec defines the workload characteristics for SLA-based profiling.
type WorkloadSpec struct {
	// ISL is the Input Sequence Length (number of tokens).
	// +optional
	ISL *int32 `json:"isl,omitempty"`

	// OSL is the Output Sequence Length (number of tokens).
	// +optional
	OSL *int32 `json:"osl,omitempty"`
}

// SLASpec defines the service-level agreement targets.
type SLASpec struct {
	// OptimizationType controls the profiling optimization strategy.
	// +optional
	// +kubebuilder:validation:Enum=hybrid;latency;throughput
	OptimizationType OptimizationType `json:"optimization_type,omitempty"`

	// TTFT is the Time To First Token target in milliseconds.
	// +optional
	TTFT *float64 `json:"ttft,omitempty"`

	// ITL is the Inter-Token Latency target in milliseconds.
	// +optional
	ITL *float64 `json:"itl,omitempty"`
}

// ModelCacheSpec references a PVC containing pre-downloaded model weights.
type ModelCacheSpec struct {
	// PVCName is the name of the PersistentVolumeClaim containing model weights.
	// The PVC must exist in the same namespace as the DGDR.
	// +optional
	PVCName string `json:"pvcName,omitempty"`

	// PVCPath is the subpath within the PVC where the model is stored.
	// +optional
	PVCPath string `json:"pvcPath,omitempty"`
}

// OverridesSpec allows customizing the profiling job and the generated DynamoGraphDeployment.
type OverridesSpec struct {
	// ProfilingJob allows overriding the profiling Job specification.
	// Fields set here are merged into the controller-generated Job spec.
	// +optional
	ProfilingJob *batchv1.JobSpec `json:"profilingJob,omitempty"`

	// DGD allows providing a full or partial DynamoGraphDeployment to use as the base
	// for the generated deployment. Fields from profiling results are merged on top.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:EmbeddedResource
	DGD *runtime.RawExtension `json:"dgd,omitempty"`
}

// FeaturesSpec controls optional Dynamo platform features in the generated deployment.
type FeaturesSpec struct {
	// Planner enables the SLA planner for autoscaling in the generated DGD.
	// +optional
	Planner *bool `json:"planner,omitempty"`

	// KVRouter enables KV-cache-aware routing in the generated DGD.
	// +optional
	KVRouter *bool `json:"kvRouter,omitempty"`

	// Mocker deploys a simulated (mocker) backend instead of a real inference engine.
	// Useful for large-scale testing without GPUs.
	// +optional
	Mocker *bool `json:"mocker,omitempty"`
}

// DynamoGraphDeploymentRequestSpec defines the desired state of a DynamoGraphDeploymentRequest.
// Only the Model field is required; all other fields are optional and have sensible defaults.
type DynamoGraphDeploymentRequestSpec struct {
	// Model specifies the model to deploy (e.g., "meta-llama/Llama-3-70b", "Qwen/Qwen3-0.6B").
	// This can be a HuggingFace model ID or a served model name.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Model string `json:"model"`

	// Backend specifies the inference backend to use for profiling and deployment.
	// +optional
	// +kubebuilder:default=auto
	// +kubebuilder:validation:Enum=auto;sglang;trtllm;vllm
	Backend BackendType `json:"backend,omitempty"`

	// Image specifies the container image for profiling and deployment workers.
	// +optional
	Image string `json:"image,omitempty"`

	// Workload defines the expected workload characteristics for SLA-based profiling.
	// +optional
	Workload *WorkloadSpec `json:"workload,omitempty"`

	// SLA defines service-level agreement targets that drive profiling optimization.
	// +optional
	SLA *SLASpec `json:"sla,omitempty"`

	// ModelCache references a PVC containing pre-downloaded model weights.
	// Use this to avoid re-downloading large models during profiling and deployment.
	// +optional
	ModelCache *ModelCacheSpec `json:"modelCache,omitempty"`

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
// +kubebuilder:storageversion
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

const hello = batchv1.JobComplete

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
func (s *DynamoGraphDeploymentRequest) SetPhase(phase DGDRPhase) {
	s.Status.Phase = phase
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
// If a condition with the same type already exists, it replaces it.
func (s *DynamoGraphDeploymentRequest) AddStatusCondition(condition metav1.Condition) {
	if s.Status.Conditions == nil {
		s.Status.Conditions = []metav1.Condition{}
	}
	for i, existing := range s.Status.Conditions {
		if existing.Type == condition.Type {
			s.Status.Conditions[i] = condition
			return
		}
	}
	s.Status.Conditions = append(s.Status.Conditions, condition)
}
