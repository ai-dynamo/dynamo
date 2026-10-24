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
	// For a Request, admission discovers an omitted SKU from the cluster. Discovery
	// rejects an unknown accelerator and an ambiguous cluster with multiple GPU types.
	// +optional
	// +kubebuilder:validation:MinLength=1
	SKU *string `json:"sku,omitempty"`

	// Budget is the maximum total number of GPUs one candidate may use.
	// For a Request, admission defaults an omitted budget to the number of matching
	// cluster GPUs, capped at 32. An explicit budget is authoritative and is not capped.
	// +optional
	// +kubebuilder:validation:Minimum=1
	Budget *int32 `json:"budget,omitempty"`
}

// HardwareSpec bounds the accelerator configurations evaluated by the search.
type HardwareSpec struct {
	// GPU describes accelerator identity and the maximum allocation per candidate.
	// For a Request, admission resolves omitted fields independently. It reads the
	// cluster on create and on an update that changes hardware, but not on an update
	// that leaves hardware unchanged. Supplying both fields avoids cluster discovery.
	// +optional
	GPU *GPUHardwareSpec `json:"gpu,omitempty"`
}

// KVLoadRatio is either a pinned load ratio or a minimum/maximum range.
// AISimulate performs mode-specific validation.
// +kubebuilder:validation:MinItems=1
// +kubebuilder:validation:MaxItems=2
// +kubebuilder:validation:items:Minimum=0
// +kubebuilder:validation:items:Maximum=1
// +kubebuilder:validation:XValidation:rule="size(self) != 2 || self[0] < self[1]",message="a range needs min < max"
type KVLoadRatio []float64

// StaticWorkloadSpec defines one of AISimulate's synthetic workload shapes. Admission
// does not default its individual fields when the object already exists.
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
	// Static defines one of AISimulate's synthetic workload shapes. For a Request whose
	// workload is omitted or empty, admission creates this object with ISL 1024, OSL 128,
	// and concurrency 10. An existing static object is preserved exactly as written, so
	// static: {} opts out of the model-only workload default.
	// +optional
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

// ObjectiveSLASpec defines per-request latency bounds used to calculate goodput. Admission
// does not default its individual fields when the object already exists, so an empty object
// disables the model-only SLA default and SLA filtering.
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
	// For a Request, admission defaults an omitted mode to Optimize.
	// +optional
	Mode ObjectiveMode `json:"mode,omitempty"`

	// Metric is the scalar target when mode is Optimize.
	// For a Request in Optimize mode, admission defaults an omitted metric to
	// ThroughputPerGPU. Pareto mode has no scalar metric default.
	// +optional
	Metric *ObjectiveMetric `json:"metric,omitempty"`

	// Metrics lists two or more distinct scalar targets when mode is Pareto.
	// +optional
	// +kubebuilder:validation:MinItems=2
	// +kubebuilder:validation:MaxItems=7
	// +listType=set
	Metrics []ObjectiveMetric `json:"metrics,omitempty"`

	// SLA defines per-request latency bounds used to calculate goodput. For a Request,
	// admission defaults an omitted SLA to TTFT 2000 ms and ITL 30 ms. An existing SLA
	// object is preserved exactly as written, so sla: {} opts out of these defaults.
	// +optional
	SLA *ObjectiveSLASpec `json:"sla,omitempty"`
}

// AISimulateSpec selects the AISimulate execution artifact and supplies native input.
type AISimulateSpec struct {
	// Image selects a Dynamo-compatible AISimulate runner. When omitted, admission
	// persists the operator's configured image or a version-matched Dynamo planner image.
	// +optional
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image,omitempty"`

	// VersionOverride declares the AISimulate compatibility version as three numeric parts,
	// for example 0.13.0. It is a numeric version without a v prefix, pre-release, or build
	// suffix. It is required when the image tag is not a version and wins over a version in
	// the tag. An operator default override is persisted only with its operator default image;
	// a caller-provided image must carry or imply its own version.
	// +optional
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})$`
	VersionOverride string `json:"versionOverride,omitempty"`

	// Parameters contains native AISimulate input. Field names and nesting use the upstream
	// AISimulate JSON contract, including snake_case names. For a Request, the controller
	// builds a typed underlay, adds derived defaults, and applies Parameters last as an RFC
	// 7386 JSON Merge Patch. Parameters therefore overrides typed values, replaces arrays and
	// scalars, and deletes fields with null. Admission preserves it without interpreting it.
	// In a Run, Parameters contains the resulting complete native input rather than a patch.
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Type=object
	Parameters *runtime.RawExtension `json:"parameters,omitempty"`
}

// RecommendationSpec controls bounded projection into DGDC resources.
type RecommendationSpec struct {
	// MaxCandidates is the maximum number of DGDCs owned by one run. Scalar
	// candidates are ordered best first; Pareto candidates have no scalar rank.
	// This field does not limit internal trials. For a Request, admission defaults
	// the zero value, including omission, to 1.
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=100
	MaxCandidates int32 `json:"maxCandidates,omitempty"`
}

// NativeSearchInput defines the v1beta2 search input shared by Requests and immutable Runs.
// The admission defaults documented below apply to native Requests. A Run is a fully
// resolved snapshot and does not receive Request defaults.
type NativeSearchInput struct {
	// Model identifies the model whose deployment configurations are evaluated.
	// +optional
	Model *ModelSpec `json:"model,omitempty"`

	// Backends lists one or more inference backends searched by AISimulate and used by
	// generated candidates. For a Request, admission defaults an omitted list to vllm,
	// sglang, and trtllm. An explicit list is preserved.
	// +optional
	// +kubebuilder:validation:MinItems=1
	// +listType=set
	Backends []Backend `json:"backends,omitempty"`

	// Hardware bounds the accelerator configurations evaluated by the search.
	// For a Request, omitting hardware, using hardware: {}, or using gpu: {} requests
	// automatic discovery. Admission persists a complete typed SKU and budget or rejects
	// the write. Parameters remains opaque to admission and may override either value later.
	// +optional
	Hardware *HardwareSpec `json:"hardware,omitempty"`

	// Workload defines the synthetic traffic model used for every candidate evaluation.
	// For a Request, an omitted or empty workload receives a static shape of ISL 1024,
	// OSL 128, and concurrency 10. workload.static: {} opts out and contributes no typed
	// workload fields. Parameters may supply or override the native workload later.
	// The default evaluates a fixed concurrency, so it creates neither sweep.max_trials nor a
	// KV-load ratio search.
	// +optional
	Workload *WorkloadSpec `json:"workload,omitempty"`

	// Objective defines scalar optimization or a multi-objective Pareto search.
	// For a Request, an omitted or empty objective receives Optimize, ThroughputPerGPU,
	// and SLA bounds of TTFT 2000 ms and ITL 30 ms. objective.sla: {} opts out only
	// of the SLA defaults; mode and the conditional metric default still apply.
	// +optional
	Objective *ObjectiveSpec `json:"objective,omitempty"`

	// AISimulate selects the execution artifact and supplies native AISimulate input.
	// For a Request, admission persists the effective image and any required version override.
	// +optional
	AISimulate *AISimulateSpec `json:"aisimulate,omitempty"`

	// Recommendation controls bounded projection into DGDC resources. For a Request,
	// admission creates an omitted object and defaults maxCandidates to 1.
	// +optional
	Recommendation *RecommendationSpec `json:"recommendation,omitempty"`
}

// DynamoGraphDeploymentRequestNativeSpec contains the native v1beta2 request fields.
// Admission persists Request defaults before the controller snapshots this input into a Run.
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
