/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package v1beta2

import (
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// RemoteCodePolicy controls whether generated backend commands may execute
// Python code from the model repository.
// +kubebuilder:validation:Enum=Never;TrustCacheAndRevision;AlwaysTrust
type RemoteCodePolicy string

const (
	RemoteCodeNever                 RemoteCodePolicy = "Never"
	RemoteCodeTrustCacheAndRevision RemoteCodePolicy = "TrustCacheAndRevision"
	RemoteCodeAlwaysTrust           RemoteCodePolicy = "AlwaysTrust"
)

// ModelPVCSpec identifies model weights on a PersistentVolumeClaim in the DGDR namespace.
type ModelPVCSpec struct {
	// Name is the PersistentVolumeClaim containing the model weights.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// ModelPath is the model directory relative to the root of the claim.
	// +kubebuilder:validation:MinLength=1
	ModelPath string `json:"modelPath"`

	// MountPath is the absolute mount path in generated containers.
	// +kubebuilder:validation:Pattern=`^/`
	MountPath string `json:"mountPath"`
}

// ModelCacheSpec identifies model weights already available to the search Job
// and generated DGD.
type ModelCacheSpec struct {
	// PVC mounts model weights from a PersistentVolumeClaim in the DGDR namespace.
	PVC *ModelPVCSpec `json:"pvc,omitempty"`
}

// ModelReference identifies the model whose deployment configurations are evaluated.
// +kubebuilder:validation:XValidation:rule="self.remoteCode != 'TrustCacheAndRevision' || (has(self.revision) && has(self.cache) && has(self.cache.pvc))",message="remoteCode TrustCacheAndRevision requires revision and cache.pvc"
type ModelReference struct {
	// Name identifies the model in the syntax accepted by the selected backend.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Revision pins repository contents. Backends that do not support revisions reject this field.
	// +optional
	Revision string `json:"revision,omitempty"`

	// RemoteCode controls whether generated backend commands may execute model-repository Python code.
	// +optional
	// +kubebuilder:default=Never
	RemoteCode RemoteCodePolicy `json:"remoteCode,omitempty"`

	// Cache identifies model weights already available to the search Job and generated DGD.
	// +optional
	Cache *ModelCacheSpec `json:"cache,omitempty"`
}

// Backend identifies an inference backend searched by Sweeper and used by generated candidates.
// +kubebuilder:validation:Enum=vllm;sglang;trtllm
type Backend string

const (
	BackendVLLM   Backend = "vllm"
	BackendSGLang Backend = "sglang"
	BackendTRTLLM Backend = "trtllm"
)

// GPUHardwareSpec describes accelerator identity and the maximum allocation per candidate.
type GPUHardwareSpec struct {
	// SKU identifies the accelerator using the Sweeper hardware catalog name.
	// +kubebuilder:validation:MinLength=1
	SKU string `json:"sku"`

	// Budget is the maximum total number of GPUs one candidate may use.
	// +kubebuilder:validation:Minimum=1
	Budget int32 `json:"budget"`
}

// HardwareSpec bounds the accelerator configurations evaluated by the search.
type HardwareSpec struct {
	// GPU describes accelerator identity and the maximum allocation per candidate.
	GPU GPUHardwareSpec `json:"gpu"`
}

// TracePVCSource identifies a trace artifact on a PersistentVolumeClaim in the DGDR namespace.
type TracePVCSource struct {
	// ClaimName is the PersistentVolumeClaim containing the trace.
	// +kubebuilder:validation:MinLength=1
	ClaimName string `json:"claimName"`

	// Path is the trace path relative to the root of the claim.
	// +kubebuilder:validation:MinLength=1
	Path string `json:"path"`
}

// TraceSource identifies exactly one trace artifact.
// +kubebuilder:validation:XValidation:rule="has(self.pvc) != has(self.uri)",message="exactly one of pvc or uri must be specified"
type TraceSource struct {
	// PVC reads the artifact from a PersistentVolumeClaim in the DGDR namespace.
	// +optional
	PVC *TracePVCSource `json:"pvc,omitempty"`

	// URI identifies an object-storage location supported by the DGDR integration.
	// +optional
	// +kubebuilder:validation:MinLength=1
	URI string `json:"uri,omitempty"`
}

// TraceReplaySpec configures current Sweeper controls for a decoded trace.
type TraceReplaySpec struct {
	// ArrivalSpeedupRatio multiplies the recorded request-arrival rate. 1.0 preserves it.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:ExclusiveMinimum=true
	// +kubebuilder:validation:Minimum=0
	ArrivalSpeedupRatio *float64 `json:"arrivalSpeedupRatio,omitempty"`

	// Concurrency changes trace replay to closed loop and ignores recorded timestamps.
	// +optional
	// +kubebuilder:validation:Minimum=1
	Concurrency *int32 `json:"concurrency,omitempty"`
}

// TraceWorkloadSpec replays requests from an artifact.
type TraceWorkloadSpec struct {
	// Format identifies the trace decoder.
	// +optional
	// +kubebuilder:default=mooncake
	// +kubebuilder:validation:MinLength=1
	Format string `json:"format,omitempty"`

	// Source identifies exactly one trace artifact.
	Source TraceSource `json:"source"`

	// Replay configures current Sweeper controls for a decoded trace.
	// +optional
	Replay *TraceReplaySpec `json:"replay,omitempty"`
}

// KVLoadRatio is either a pinned load ratio or a minimum/maximum range.
// The installed Sweeper performs mode-specific validation.
// +kubebuilder:validation:MinItems=1
// +kubebuilder:validation:MaxItems=2
type KVLoadRatio []float64

// StaticWorkloadSpec defines one of Sweeper's synthetic workload shapes.
// +kubebuilder:validation:XValidation:rule="(has(self.requestRate) ? 1 : 0) + (has(self.concurrency) ? 1 : 0) + (has(self.kvLoadRatio) ? 1 : 0) == 1",message="exactly one of requestRate, concurrency, or kvLoadRatio must be specified"
type StaticWorkloadSpec struct {
	// ISL is the input sequence length in tokens.
	// +kubebuilder:validation:Minimum=1
	ISL int32 `json:"isl"`

	// OSL is the output sequence length in tokens.
	// +kubebuilder:validation:Minimum=1
	OSL int32 `json:"osl"`

	// NumRequestRatio sets request count relative to the selected load.
	// +kubebuilder:validation:Minimum=1
	NumRequestRatio float64 `json:"numRequestRatio"`

	// RequestRate is a fixed open-loop arrival rate in requests per second.
	// +optional
	// +kubebuilder:validation:ExclusiveMinimum=true
	// +kubebuilder:validation:Minimum=0
	RequestRate *float64 `json:"requestRate,omitempty"`

	// Concurrency is a fixed closed-loop in-flight request cap.
	// +optional
	// +kubebuilder:validation:Minimum=1
	Concurrency *int32 `json:"concurrency,omitempty"`

	// KVLoadRatio derives concurrency from candidate KV capacity. A two-value range
	// is supported only for a Pareto objective.
	// +optional
	KVLoadRatio KVLoadRatio `json:"kvLoadRatio,omitempty"`
}

// WorkloadSpec defines exactly one traffic model used for every candidate evaluation.
// +kubebuilder:validation:XValidation:rule="has(self.trace) != has(self.static)",message="exactly one of trace or static must be specified"
type WorkloadSpec struct {
	// Trace replays requests from an artifact.
	// +optional
	Trace *TraceWorkloadSpec `json:"trace,omitempty"`

	// Static defines one of Sweeper's synthetic workload shapes.
	// +optional
	Static *StaticWorkloadSpec `json:"static,omitempty"`
}

// ObjectiveMode selects scalar optimization or a multi-objective Pareto search.
// +kubebuilder:validation:Enum=optimize;pareto
type ObjectiveMode string

const (
	ObjectiveModeOptimize ObjectiveMode = "optimize"
	ObjectiveModePareto   ObjectiveMode = "pareto"
)

// ObjectiveMetric is a metric optimized by Sweeper.
// +kubebuilder:validation:Enum=throughput;throughputPerGpu;throughputPerUser;e2eLatency;goodput;goodputPerGpu
type ObjectiveMetric string

// ObjectiveSLASpec defines per-request latency bounds used to calculate goodput.
// +kubebuilder:validation:XValidation:rule="has(self.e2eMs) != (has(self.ttftMs) && has(self.itlMs))",message="specify either e2eMs or both ttftMs and itlMs"
type ObjectiveSLASpec struct {
	// TTFTMs is the Time To First Token bound in milliseconds.
	// +optional
	// +kubebuilder:validation:ExclusiveMinimum=true
	// +kubebuilder:validation:Minimum=0
	TTFTMs *float64 `json:"ttftMs,omitempty"`

	// ITLMs is the Inter-Token Latency bound in milliseconds.
	// +optional
	// +kubebuilder:validation:ExclusiveMinimum=true
	// +kubebuilder:validation:Minimum=0
	ITLMs *float64 `json:"itlMs,omitempty"`

	// E2EMs is an alternative end-to-end bound in milliseconds.
	// +optional
	// +kubebuilder:validation:ExclusiveMinimum=true
	// +kubebuilder:validation:Minimum=0
	E2EMs *float64 `json:"e2eMs,omitempty"`
}

// ObjectiveSpec defines scalar optimization or a multi-objective Pareto search.
// +kubebuilder:validation:XValidation:rule="(self.mode == 'optimize' && has(self.metric) && !has(self.metrics)) || (self.mode == 'pareto' && !has(self.metric))",message="optimize requires metric; pareto accepts only metrics"
type ObjectiveSpec struct {
	// Mode is optimize or pareto.
	Mode ObjectiveMode `json:"mode"`

	// Metric is the scalar target when mode is optimize.
	// +optional
	Metric *ObjectiveMetric `json:"metric,omitempty"`

	// Metrics lists two or more scalar targets when mode is pareto. Sweeper uses
	// its default pair when omitted.
	// +optional
	// +kubebuilder:validation:MinItems=2
	Metrics []ObjectiveMetric `json:"metrics,omitempty"`

	// SLA defines per-request latency bounds used to calculate goodput.
	// +optional
	SLA *ObjectiveSLASpec `json:"sla,omitempty"`
}

// SearchBudgetSpec maps directly to Sweeper's current SweepConfig.
type SearchBudgetSpec struct {
	// MaxRounds is the maximum number of optimizer barrier rounds per resolved deployment-mode branch.
	// +kubebuilder:validation:Minimum=1
	MaxRounds int32 `json:"maxRounds"`

	// CandidatesPerRound is the target number of successful unique replay configurations
	// in one branch round. When omitted, Sweeper uses ParallelEvaluations.
	// +optional
	// +kubebuilder:validation:Minimum=1
	CandidatesPerRound *int32 `json:"candidatesPerRound,omitempty"`

	// ParallelEvaluations is the replay worker-process fan-out.
	// +kubebuilder:validation:Minimum=1
	ParallelEvaluations int32 `json:"parallelEvaluations"`

	// MaxEvaluationDuration is the timeout for one candidate replay.
	// +optional
	MaxEvaluationDuration *metav1.Duration `json:"maxEvaluationDuration,omitempty"`
}

// SearchSpec configures current Sweeper run control and implementation-owned dimensions.
type SearchSpec struct {
	// Budget maps directly to Sweeper's current SweepConfig.
	Budget SearchBudgetSpec `json:"budget"`

	// Parameters contains current Sweeper fields whose nested schema is not copied into the CRD.
	// The installed Sweeper validates it before evaluating candidates.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	Parameters *runtime.RawExtension `json:"parameters,omitempty"`
}

// RecommendationSpec controls bounded projection into DGDC resources.
type RecommendationSpec struct {
	// MaxCandidates is the maximum number of DGDCs owned by one run. It does not limit internal trials.
	// +optional
	// +kubebuilder:default=5
	// +kubebuilder:validation:Minimum=1
	MaxCandidates int32 `json:"maxCandidates,omitempty"`
}

// RerunSpec intentionally changes the DGDR spec when search inputs otherwise remain unchanged.
type RerunSpec struct {
	// Reason records why the user requested another run. Any new value creates a new generation.
	// +kubebuilder:validation:MinLength=1
	Reason string `json:"reason"`
}

// OverridesSpec customizes the generated Job and DGD without changing modeled search semantics.
type OverridesSpec struct {
	// ProfilingJob is a partial batch/v1 JobSpec merged into the controller-generated Job.
	// +optional
	ProfilingJob *batchv1.JobSpec `json:"profilingJob,omitempty"`

	// DGD is a partial versioned DGD merged after candidate materialization and before hashing.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:EmbeddedResource
	DGD *runtime.RawExtension `json:"dgd,omitempty"`
}

// DynamoGraphDeploymentRequestSpec defines persistent desired search intent.
type DynamoGraphDeploymentRequestSpec struct {
	// ModelRef identifies the model whose deployment configurations are evaluated.
	ModelRef ModelReference `json:"modelRef"`

	// Backends lists one or more inference backends searched by Sweeper and used by generated candidates.
	// +kubebuilder:validation:MinItems=1
	Backends []Backend `json:"backends"`

	// Image is the versioned container image used by the controller-generated search Job.
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image"`

	// Hardware bounds the accelerator configurations evaluated by the search.
	Hardware HardwareSpec `json:"hardware"`

	// Workload defines exactly one traffic model used for every candidate evaluation.
	Workload WorkloadSpec `json:"workload"`

	// Objective defines scalar optimization or a multi-objective Pareto search.
	Objective ObjectiveSpec `json:"objective"`

	// Search configures current Sweeper run control and implementation-owned dimensions.
	Search SearchSpec `json:"search"`

	// Recommendation controls bounded projection into DGDC resources.
	// +optional
	Recommendation RecommendationSpec `json:"recommendation,omitempty"`

	// Rerun intentionally changes the DGDR spec when search inputs otherwise remain unchanged.
	// +optional
	Rerun *RerunSpec `json:"rerun,omitempty"`

	// Overrides customizes the generated Job and DGD without changing modeled search semantics.
	// +optional
	Overrides *OverridesSpec `json:"overrides,omitempty"`
}

// DynamoGraphDeploymentRequestStatus represents reconciliation of persistent search intent.
type DynamoGraphDeploymentRequestStatus struct {
	// ObservedGeneration is the most recent DGDR generation accepted by the controller.
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// ActiveRunRef identifies the one non-terminal run.
	// +optional
	ActiveRunRef *corev1.LocalObjectReference `json:"activeRunRef,omitempty"`

	// RecentRunRefs identifies retained terminal runs in newest-first order.
	// +optional
	RecentRunRefs []corev1.LocalObjectReference `json:"recentRunRefs,omitempty"`

	// Conditions reports reconciliation of the persistent request, not run execution outcomes.
	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:unservedversion
// +kubebuilder:resource:shortName=dgdr
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.modelRef.name`
// +kubebuilder:printcolumn:name="Active Run",type=string,JSONPath=`.status.activeRunRef.name`
// +kubebuilder:printcolumn:name="Accepted",type=string,JSONPath=`.status.conditions[?(@.type=="Accepted")].status`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// DynamoGraphDeploymentRequest is the Schema for replay-backed deployment searches.
type DynamoGraphDeploymentRequest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DynamoGraphDeploymentRequestSpec   `json:"spec"`
	Status DynamoGraphDeploymentRequestStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DynamoGraphDeploymentRequestList contains a list of DynamoGraphDeploymentRequest resources.
type DynamoGraphDeploymentRequestList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DynamoGraphDeploymentRequest `json:"items"`
}
