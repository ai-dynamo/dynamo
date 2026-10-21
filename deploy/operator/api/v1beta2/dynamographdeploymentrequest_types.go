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
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// ModelSpec identifies the model whose deployment configurations are evaluated.
type ModelSpec struct {
	// Name identifies the model in the syntax accepted by the selected backend.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`
}

// Backend identifies an inference backend searched by AISimulate and used by generated candidates.
// +kubebuilder:validation:Enum=vllm;sglang;trtllm
type Backend string

const (
	BackendVLLM   Backend = "vllm"
	BackendSGLang Backend = "sglang"
	BackendTRTLLM Backend = "trtllm"
)

// GPUHardwareSpec describes accelerator identity and the maximum allocation per candidate.
type GPUHardwareSpec struct {
	// SKU identifies the accelerator using the AISimulate hardware catalog name.
	// When omitted, admission discovers the accelerator from the cluster.
	// +optional
	// +kubebuilder:validation:MinLength=1
	SKU *string `json:"sku,omitempty"`

	// Budget is the maximum total number of GPUs one candidate may use.
	// When omitted, admission derives the available GPU budget from the cluster.
	// +optional
	// +kubebuilder:validation:Minimum=1
	Budget *int32 `json:"budget,omitempty"`
}

// HardwareSpec bounds the accelerator configurations evaluated by the search.
type HardwareSpec struct {
	// GPU describes accelerator identity and the maximum allocation per candidate.
	// Omitted fields are resolved independently by admission.
	// +optional
	GPU *GPUHardwareSpec `json:"gpu,omitempty"`
}

// KVLoadRatio is either a pinned load ratio or a minimum/maximum range.
// AISimulate performs mode-specific validation.
// +kubebuilder:validation:MinItems=1
// +kubebuilder:validation:MaxItems=2
// +kubebuilder:validation:items:Minimum=0
// +kubebuilder:validation:items:Maximum=1
type KVLoadRatio []float64

// StaticWorkloadSpec defines one of AISimulate's synthetic workload shapes.
// +kubebuilder:validation:XValidation:rule="(has(self.requestRate) ? 1 : 0) + (has(self.concurrency) ? 1 : 0) + (has(self.kvLoadRatio) ? 1 : 0) <= 1",message="at most one of requestRate, concurrency, or kvLoadRatio may be specified"
type StaticWorkloadSpec struct {
	// ISL is the input sequence length in tokens.
	// +optional
	// +kubebuilder:validation:Minimum=1
	ISL *int32 `json:"isl,omitempty"`

	// OSL is the output sequence length in tokens.
	// +optional
	// +kubebuilder:validation:Minimum=1
	OSL *int32 `json:"osl,omitempty"`

	// NumRequestRatio sets request count relative to the selected load.
	// +optional
	// +kubebuilder:validation:ExclusiveMinimum=true
	// +kubebuilder:validation:Minimum=0
	NumRequestRatio *float64 `json:"numRequestRatio,omitempty"`

	// RequestRate is a fixed open-loop arrival rate in requests per second.
	// +optional
	// +kubebuilder:validation:ExclusiveMinimum=true
	// +kubebuilder:validation:Minimum=0
	RequestRate *float64 `json:"requestRate,omitempty"`

	// Concurrency is a fixed closed-loop in-flight request cap.
	// +optional
	// +kubebuilder:validation:Minimum=1
	Concurrency *int32 `json:"concurrency,omitempty"`

	// KVLoadRatio derives concurrency from candidate KV capacity. A two-value
	// range requires a bounded AISimulate sweep.
	// +optional
	KVLoadRatio KVLoadRatio `json:"kvLoadRatio,omitempty"`
}

// WorkloadSpec defines the synthetic traffic model used for every candidate evaluation.
type WorkloadSpec struct {
	// Static defines one of AISimulate's synthetic workload shapes. When omitted,
	// it searches the full KV-load ratio range at ISL 4000 and OSL 1000.
	// An explicit empty object opts out of this underlay.
	// +optional
	// +kubebuilder:default={isl: 4000, osl: 1000, kvLoadRatio: {0.0, 1.0}}
	Static *StaticWorkloadSpec `json:"static,omitempty"`
}

// ObjectiveMode selects scalar optimization or a multi-objective Pareto search.
// +kubebuilder:validation:Enum=Optimize;Pareto
type ObjectiveMode string

const (
	ObjectiveModeOptimize ObjectiveMode = "Optimize"
	ObjectiveModePareto   ObjectiveMode = "Pareto"
)

// ObjectiveMetric is a metric optimized by AISimulate.
// +kubebuilder:validation:Enum=Throughput;ThroughputPerGPU;ThroughputPerUser;TTFT;E2ELatency;Goodput;GoodputPerGPU
type ObjectiveMetric string

// ObjectiveSLASpec defines per-request latency bounds used to calculate goodput.
// An empty object disables SLA filtering.
// +kubebuilder:validation:XValidation:rule="!has(self.e2eMs) || (!has(self.ttftMs) && !has(self.itlMs))",message="e2eMs cannot be combined with ttftMs or itlMs"
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
// +kubebuilder:validation:XValidation:rule="!has(self.mode) || (self.mode == 'Optimize' && !has(self.metrics)) || (self.mode == 'Pareto' && !has(self.metric) && has(self.metrics))",message="Optimize accepts only metric; Pareto requires metrics"
// +kubebuilder:validation:XValidation:rule="!(has(self.metric) && self.metric in ['Goodput', 'GoodputPerGPU']) || (has(self.sla) && (has(self.sla.ttftMs) || has(self.sla.itlMs) || has(self.sla.e2eMs)))",message="Goodput and GoodputPerGPU require at least one SLA bound"
// +kubebuilder:validation:XValidation:rule="!(has(self.metrics) && self.metrics.exists(m, m in ['Goodput', 'GoodputPerGPU'])) || (has(self.sla) && (has(self.sla.ttftMs) || has(self.sla.itlMs) || has(self.sla.e2eMs)))",message="Pareto metrics containing Goodput or GoodputPerGPU require at least one SLA bound"
type ObjectiveSpec struct {
	// Mode selects scalar optimization or a Pareto search.
	// +optional
	// +kubebuilder:default=Optimize
	Mode ObjectiveMode `json:"mode,omitempty"`

	// Metric is the scalar target when mode is Optimize.
	// When omitted in Optimize mode, admission defaults it to ThroughputPerGPU.
	// +optional
	Metric *ObjectiveMetric `json:"metric,omitempty"`

	// Metrics lists two or more distinct scalar targets when mode is Pareto.
	// +optional
	// +kubebuilder:validation:MinItems=2
	// +kubebuilder:validation:MaxItems=7
	// +listType=set
	Metrics []ObjectiveMetric `json:"metrics,omitempty"`

	// SLA defines per-request latency bounds used to calculate goodput.
	// +optional
	// +kubebuilder:default={ttftMs: 2000, itlMs: 30}
	SLA *ObjectiveSLASpec `json:"sla,omitempty"`
}

// AISimulateSpec selects the AISimulate execution artifact and supplies native input.
type AISimulateSpec struct {
	// Image selects a Dynamo-compatible AISimulate runner. When omitted, admission
	// and execution defaults are resolved outside this API.
	// +optional
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image,omitempty"`

	// VersionOverride declares the AISimulate compatibility version when it
	// cannot be inferred from the effective image.
	// +optional
	// +kubebuilder:validation:MinLength=1
	VersionOverride string `json:"versionOverride,omitempty"`

	// Parameters contains native AISimulate input. Field names and nesting use
	// the upstream AISimulate JSON contract, including snake_case names.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	Parameters *runtime.RawExtension `json:"parameters,omitempty"`
}

// RecommendationSpec controls bounded projection into DGDC resources.
type RecommendationSpec struct {
	// MaxCandidates is the maximum number of DGDCs owned by one run. Scalar
	// candidates are ordered best first; Pareto candidates have no scalar rank.
	// This field does not limit internal trials.
	// +optional
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=100
	MaxCandidates int32 `json:"maxCandidates,omitempty"`
}

// NativeSearchInput defines the v1beta2 search input shared by Requests and
// immutable Runs.
type NativeSearchInput struct {
	// Model identifies the model whose deployment configurations are evaluated.
	// +optional
	Model *ModelSpec `json:"model,omitempty"`

	// Backends lists one or more inference backends searched by AISimulate and used by generated candidates.
	// +optional
	// +kubebuilder:validation:MinItems=1
	Backends []Backend `json:"backends,omitempty"`

	// Hardware bounds the accelerator configurations evaluated by the search.
	// Admission discovers any omitted hardware values from the cluster.
	// +optional
	Hardware *HardwareSpec `json:"hardware,omitempty"`

	// Workload defines the synthetic traffic model used for every candidate evaluation.
	// +optional
	Workload *WorkloadSpec `json:"workload,omitempty"`

	// Objective defines scalar optimization or a multi-objective Pareto search.
	// When omitted, the schema supplies the default Optimize mode and SLA;
	// admission supplies the conditional scalar metric.
	// +optional
	Objective *ObjectiveSpec `json:"objective,omitempty"`

	// AISimulate selects the execution artifact and supplies native AISimulate input.
	// +optional
	AISimulate *AISimulateSpec `json:"aisimulate,omitempty"`

	// Recommendation controls bounded projection into DGDC resources.
	// +optional
	Recommendation *RecommendationSpec `json:"recommendation,omitempty"`
}

// DynamoGraphDeploymentRequestNativeSpec contains the native v1beta2 request fields.
type DynamoGraphDeploymentRequestNativeSpec struct {
	NativeSearchInput `json:",inline"`
}

// DynamoGraphDeploymentRequestSpec defines persistent desired search intent.
// Native v1beta2 fields and the V1Beta1 conversion envelope are mutually exclusive.
// +kubebuilder:validation:XValidation:rule="has(self.v1beta1) != has(self.model)",message="exactly one of native v1beta2 fields or v1beta1 must be specified"
// +kubebuilder:validation:XValidation:rule="!has(self.v1beta1) || !(has(self.backends) || has(self.hardware) || has(self.workload) || has(self.objective) || has(self.aisimulate) || has(self.recommendation))",message="v1beta1 is mutually exclusive with native v1beta2 fields"
// +kubebuilder:validation:XValidation:rule="has(self.v1beta1) == has(oldSelf.v1beta1) && (!has(self.v1beta1) || self.v1beta1 == oldSelf.v1beta1)",message="the conversion carrier is immutable"
type DynamoGraphDeploymentRequestSpec struct {
	DynamoGraphDeploymentRequestNativeSpec `json:",inline"`

	// V1Beta1 losslessly preserves a native v1beta1 spec while this object is
	// represented through v1beta2. It is mutually exclusive with native fields.
	// This field is a read-only conversion view; modify the request through the
	// native v1beta1 API.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	V1Beta1 *apiextensionsv1.JSON `json:"v1beta1,omitempty"`
}

// DynamoGraphDeploymentRequestNativeStatus contains the native v1beta2 status fields.
type DynamoGraphDeploymentRequestNativeStatus struct {
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

// DynamoGraphDeploymentRequestStatus represents reconciliation of persistent search intent.
// +kubebuilder:validation:XValidation:rule="!has(self.v1beta1) || !(has(self.observedGeneration) || has(self.activeRunRef) || has(self.recentRunRefs) || has(self.conditions))",message="v1beta1 is mutually exclusive with native v1beta2 status fields"
type DynamoGraphDeploymentRequestStatus struct {
	DynamoGraphDeploymentRequestNativeStatus `json:",inline"`

	// V1Beta1 losslessly preserves a native v1beta1 status while this object is
	// represented through v1beta2.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	V1Beta1 *apiextensionsv1.JSON `json:"v1beta1,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=dgdr
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.spec.model.name`
// +kubebuilder:printcolumn:name="Active Run",type=string,JSONPath=`.status.activeRunRef.name`
// +kubebuilder:printcolumn:name="Accepted",type=string,JSONPath=`.status.conditions[?(@.type=="Accepted")].status`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
// +kubebuilder:validation:XValidation:rule="!has(self.status) || !(has(self.status.v1beta1) || has(self.status.observedGeneration) || has(self.status.activeRunRef) || has(self.status.recentRunRefs) || has(self.status.conditions)) || has(self.spec.v1beta1) == has(self.status.v1beta1)",message="nonempty status must use the representation selected by spec"

// DynamoGraphDeploymentRequest is the Schema for simulation-backed deployment searches.
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
