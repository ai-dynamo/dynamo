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

// Fuzz-based round-trip tests for the v1alpha1 / v1beta1 conversion code.
//
// For every randomly generated object the test asserts both directions of
// the round-trip:
//
//   - hub -> spoke -> hub (start from a v1beta1 hub object); the spoke must
//     losslessly carry every hub shape, with no-v1alpha1-equivalent fields
//     stashed via reserved "nvidia.com/{dgd,dcd,dgdr,dgdsa}-*" annotations.
//   - spoke -> hub -> spoke (start from a v1alpha1 spoke object); the hub
//     must be a strict superset of the spoke.
//
// Follows the upstream Kubernetes round-trip-fuzz pattern built on
// k8s.io/apimachinery/pkg/api/apitesting/fuzzer and
// k8s.io/apimachinery/pkg/apis/meta/fuzzer (with sigs.k8s.io/randfill as the
// underlying fuzzing library), but driven through controller-runtime's
// Convertible interface instead of apimachinery scheme-based conversion.
// Filler funcs are layered on top of metafuzzer (ObjectMeta, Time, ListMeta,
// ...) so we only hand-write fillers for our own types and the few corev1
// types where the default randfill output is not legal (Quantity,
// IntOrString, RawExtension).

package api

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apitestingfuzzer "k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/resource"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/conversion"
	"sigs.k8s.io/randfill"
	"sigs.k8s.io/yaml"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
)

var (
	fuzzIters = flag.Int("roundtrip-fuzz-iters", 1000, "iterations per direction per type for the fuzz round-trip tests")
	fuzzSeed  = flag.Int64("roundtrip-fuzz-seed", 1, "rand seed for the fuzz round-trip tests; bump to randomize CI runs")
)

// reservedAnnotationPrefixes are the annotation key namespaces the operator's
// conversion code uses to stash data without a v1alpha1 representation. A
// user-set annotation under one of these prefixes would be eaten on
// ConvertFrom and break the round-trip, so the filler scrubs them from
// generated ObjectMeta annotations.
var reservedAnnotationPrefixes = []string{
	"nvidia.com/dgd-",
	"nvidia.com/dcd-",
	"nvidia.com/dgdr-",
	"nvidia.com/dgdsa-",
}

func scrubReservedAnnotations(m map[string]string) map[string]string {
	if len(m) == 0 {
		return m
	}
	for k := range m {
		for _, p := range reservedAnnotationPrefixes {
			if strings.HasPrefix(k, p) {
				delete(m, k)
				break
			}
		}
	}
	if len(m) == 0 {
		return nil
	}
	return m
}

// dynamoFuzzerFuncs constrains generated values so that random objects on
// either side represent shapes the conversion is expected to round-trip
// losslessly.
//
// apitestingfuzzer.MergeFuzzerFuncs picks the *last* filler for a given
// first-arg type, so this set wins over metafuzzer.Funcs for any overlapping
// types (RawExtension, ObjectMeta).
func dynamoFuzzerFuncs(_ runtimeserializer.CodecFactory) []any {
	return []any{
		// ObjectMeta: keep randfill defaults but scrub the operator-owned
		// annotation namespace and drop ManagedFields (the typed conversion
		// does not preserve it).
		func(m *metav1.ObjectMeta, c randfill.Continue) {
			c.FillNoCustom(m)
			m.Annotations = scrubReservedAnnotations(m.Annotations)
			m.ManagedFields = nil
		},
		// RawExtension: emit a small valid JSON object so paths that stash
		// the value through annotations or JSON marshalling are stable
		// across a round-trip.
		func(r *runtime.RawExtension, c randfill.Continue) {
			obj := map[string]string{
				fmt.Sprintf("k%d", c.Uint32()%32): fmt.Sprintf("v%d", c.Uint32()%32),
			}
			raw, err := json.Marshal(obj)
			if err != nil {
				panic(err)
			}
			r.Raw = raw
			r.Object = nil
			apitestingfuzzer.NormalizeJSONRawExtension(r)
		},
		// json.RawMessage: generate valid JSON so marshal-based snapshots can
		// detect in-memory mutations instead of tripping over random bytes.
		func(raw *json.RawMessage, c randfill.Continue) {
			if c.Bool() {
				*raw = nil
				return
			}
			data, err := json.Marshal(fuzzJSONValue(c, 0))
			if err != nil {
				panic(err)
			}
			*raw = append((*raw)[:0], data...)
		},
		// apiextensionsv1.JSON wraps raw JSON bytes with custom marshal logic.
		// These fields are admitted as JSON objects, not arbitrary JSON values.
		func(v *apiextensionsv1.JSON, c randfill.Continue) {
			data, err := json.Marshal(fuzzJSONObject(c, 0))
			if err != nil {
				panic(err)
			}
			v.Raw = append(v.Raw[:0], data...)
		},
		// resource.Quantity: parseable suffix; randfill's default produces
		// inconsistent Format/Value combinations.
		func(q *resource.Quantity, c randfill.Continue) {
			n := c.Int63() % 65536
			*q = resource.MustParse(strconv.FormatInt(n, 10) + "Mi")
		},
		// intstr.IntOrString: alternate int / string form; randfill leaves
		// the discriminator and value out of sync by default.
		func(v *intstr.IntOrString, c randfill.Continue) {
			if c.Bool() {
				*v = intstr.FromInt32(c.Int31() % 65535)
			} else {
				*v = intstr.FromString(fmt.Sprintf("p%d", c.Uint32()%65535))
			}
		},
		// v1beta1 OptimizationType is an enum. randfill's default arbitrary
		// string output is not admitted by the CRD schema and would be dropped
		// when projecting through v1alpha1's profiling JSON blob.
		func(v *v1beta1.OptimizationType, c randfill.Continue) {
			*v = oneOf(c, v1beta1.OptimizationTypeLatency, v1beta1.OptimizationTypeThroughput)
		},
		fuzzAlphaDGDR,
		fuzzAlphaDGDRNativeSpec,
		fuzzAlphaDGDRNativeStatus,
		fuzzAlphaDGDRSpec,
		fuzzAlphaDGDRStatus,
		fuzzBetaDGDR,
		fuzzBetaDGDRNativeSpec,
		fuzzBetaDGDRNativeStatus,
		fuzzBetaDGDRSpec,
		fuzzBetaDGDRStatus,
		fuzzBeta2DGDR,
		fuzzBeta2DGDRNativeSpec,
		fuzzBeta2DGDRNativeStatus,
		fuzzBeta2DGDRSpec,
		fuzzBeta2DGDRStatus,
		fuzzAISimulateSpec,
		fuzzBetaDGDROverrides,
		// PlacementStatus (v1alpha1 + v1beta1): pick admissible values so the
		// round-trip fuzzer exercises Placement without producing shapes the CRD
		// schema rejects. State draws from the enum; Score draws either nil or a
		// value inside the [0, 1] bounds enforced by the CRD.
		func(p *v1beta1.PlacementStatus, c randfill.Continue) {
			p.State = oneOf(c,
				v1beta1.PlacementScoreStateReported,
				v1beta1.PlacementScoreStatePartial,
				v1beta1.PlacementScoreStateUnsupported,
				v1beta1.PlacementScoreStateUnknown,
			)
			p.Score = oneOfPtr(c, 0.0, 0.25, 0.5, 0.75, 1.0)
		},
		func(p *v1alpha1.PlacementStatus, c randfill.Continue) {
			p.State = oneOf(c,
				v1alpha1.PlacementScoreStateReported,
				v1alpha1.PlacementScoreStatePartial,
				v1alpha1.PlacementScoreStateUnsupported,
				v1alpha1.PlacementScoreStateUnknown,
			)
			p.Score = oneOfPtr(c, 0.0, 0.25, 0.5, 0.75, 1.0)
		},
		// v1beta1 Components: the listMapKey marker requires name
		// to be non-empty and unique; MaxItems caps the length at 25.
		// Enforce both so the input is admissible.
		func(s *[]v1beta1.DynamoComponentDeploymentSharedSpec, c randfill.Continue) {
			c.FillNoCustom(s)
			if len(*s) > 25 {
				*s = (*s)[:25]
			}
			for i := range *s {
				(*s)[i].ComponentName = fmt.Sprintf("c%d", i)
			}
		},
	}
}

func fuzzAlphaDGDR(r *v1alpha1.DynamoGraphDeploymentRequest, c randfill.Continue) {
	c.FillNoCustom(r)
	if c.Bool() {
		return
	}

	var spec v1beta2.DynamoGraphDeploymentRequestNativeSpec
	c.Fill(&spec)
	r.Spec = v1alpha1.DynamoGraphDeploymentRequestSpec{
		V1Beta2: marshalFuzzEnvelope(spec),
	}
	r.Status = v1alpha1.DynamoGraphDeploymentRequestStatus{}
	if c.Bool() {
		return
	}

	var status v1beta2.DynamoGraphDeploymentRequestNativeStatus
	c.Fill(&status)
	r.Status.V1Beta2 = marshalFuzzEnvelope(status)
}

func fuzzAlphaDGDRSpec(s *v1alpha1.DynamoGraphDeploymentRequestSpec, c randfill.Continue) {
	c.Fill(&s.DynamoGraphDeploymentRequestNativeSpec)
	s.V1Beta2 = nil
}

func fuzzAlphaDGDRNativeSpec(s *v1alpha1.DynamoGraphDeploymentRequestNativeSpec, c randfill.Continue) {
	c.FillNoCustom(s)
	if s.ProfilingConfig == nil {
		s.ProfilingConfig = &v1alpha1.ProfilingConfigSpec{}
	}

	s.Model = fmt.Sprintf("model-%d", c.Uint32())
	s.Backend = oneOf(c, "auto", "vllm", "sglang", "trtllm")

	// Empty resources do not survive the alpha resources to beta profiling job projection.
	if s.ProfilingConfig != nil &&
		s.ProfilingConfig.Resources != nil &&
		len(s.ProfilingConfig.Resources.Requests) == 0 &&
		len(s.ProfilingConfig.Resources.Limits) == 0 &&
		len(s.ProfilingConfig.Resources.Claims) == 0 {
		s.ProfilingConfig.Resources = nil
	}

	// Only true EnableGPUDiscovery is annotation-preserved; false is dropped.
	if s.EnableGPUDiscovery != nil {
		*s.EnableGPUDiscovery = true
	}

	// WorkersImage maps to Image and is not restored as a deployment override.
	if s.DeploymentOverrides != nil {
		s.DeploymentOverrides.WorkersImage = ""

		// Empty deployment overrides are not annotation-preserved.
		if s.DeploymentOverrides.Name == "" &&
			s.DeploymentOverrides.Namespace == "" &&
			len(s.DeploymentOverrides.Labels) == 0 &&
			len(s.DeploymentOverrides.Annotations) == 0 {
			s.DeploymentOverrides = nil
		}
	}
}

func fuzzAlphaDGDRStatus(s *v1alpha1.DynamoGraphDeploymentRequestStatus, c randfill.Continue) {
	c.Fill(&s.DynamoGraphDeploymentRequestNativeStatus)
	s.V1Beta2 = nil
}

func fuzzAlphaDGDRNativeStatus(s *v1alpha1.DynamoGraphDeploymentRequestNativeStatus, c randfill.Continue) {
	c.FillNoCustom(s)

	s.State = oneOf(c,
		v1alpha1.DGDRStateInitializing,
		v1alpha1.DGDRStatePending,
		v1alpha1.DGDRStateProfiling,
		v1alpha1.DGDRStateReady,
		v1alpha1.DGDRStateDeploying,
		v1alpha1.DGDRStateDeploymentDeleted,
		v1alpha1.DGDRStateFailed,
	)
}

func fuzzBetaDGDR(r *v1beta1.DynamoGraphDeploymentRequest, c randfill.Continue) {
	c.FillNoCustom(r)
	if c.Bool() {
		return
	}

	var spec v1beta2.DynamoGraphDeploymentRequestNativeSpec
	c.Fill(&spec)
	r.Spec = v1beta1.DynamoGraphDeploymentRequestSpec{
		V1Beta2: marshalFuzzEnvelope(spec),
	}
	r.Status = v1beta1.DynamoGraphDeploymentRequestStatus{}
	if c.Bool() {
		return
	}

	var status v1beta2.DynamoGraphDeploymentRequestNativeStatus
	c.Fill(&status)
	r.Status.V1Beta2 = marshalFuzzEnvelope(status)
}

func fuzzBetaDGDRSpec(s *v1beta1.DynamoGraphDeploymentRequestSpec, c randfill.Continue) {
	c.Fill(&s.DynamoGraphDeploymentRequestNativeSpec)
	s.V1Beta2 = nil
}

func fuzzBetaDGDRNativeSpec(s *v1beta1.DynamoGraphDeploymentRequestNativeSpec, c randfill.Continue) {
	c.FillNoCustom(s)

	s.Model = fmt.Sprintf("model-%d", c.Uint32())
	s.Backend = oneOf(c, v1beta1.BackendTypeAuto, v1beta1.BackendTypeVllm, v1beta1.BackendTypeSglang, v1beta1.BackendTypeTrtllm)

	if s.Workload != nil && s.SLA == nil && (s.Workload.ISL != nil || s.Workload.OSL != nil) {
		// ISL/OSL live under the alpha profiling config's "sla" object, whose
		// reconstruction creates an empty v1beta1 SLA shell.
		s.SLA = &v1beta1.SLASpec{}
	}
	if s.Workload != nil &&
		s.Workload.ISL == nil &&
		s.Workload.OSL == nil &&
		s.Workload.Concurrency == nil &&
		s.Workload.RequestRate == nil {
		s.Workload = nil
	}

	// Empty ModelCache is not reconstructed from the alpha profiling config blob.
	if s.ModelCache != nil &&
		s.ModelCache.PVCName == "" &&
		s.ModelCache.PVCModelPath == "" &&
		s.ModelCache.PVCMountPath == "" {
		s.ModelCache = nil
	}

	if s.Features != nil {
		// Empty Features is not reconstructed without Planner or Mocker.
		if s.Features.Planner == nil && s.Features.Mocker == nil {
			s.Features = nil
		}
	}

	// Nil AutoApply round-trips as the v1beta1 default true.
	if s.AutoApply == nil {
		autoApply := true
		s.AutoApply = &autoApply
	}
}

func fuzzBetaDGDROverrides(overrides *v1beta1.OverridesSpec, c randfill.Continue) {
	// Fill the complete override normally first. This also produces the
	// ProfilingJob == nil case without special alpha/beta logic.
	c.FillNoCustom(overrides)

	if overrides.ProfilingJob != nil {
		fuzzDGDRProfilingJob(overrides.ProfilingJob, c)
	}
}

func fuzzDGDRProfilingJob(job *batchv1.JobSpec, c randfill.Continue) {
	fuzzDGDRProfilingPodTemplate(&job.Template, c)
}

func fuzzDGDRProfilingPodTemplate(template *corev1.PodTemplateSpec, c randfill.Continue) {
	fuzzDGDRProfilingPodSpec(&template.Spec, c)
}

func fuzzDGDRProfilingPodSpec(podSpec *corev1.PodSpec, c randfill.Continue) {
	names := fuzzDGDRProfilingContainerNames(c)
	if names == nil {
		podSpec.Containers = nil
		return
	}

	podSpec.Containers = make([]corev1.Container, len(names))
	for i, name := range names {
		c.Fill(&podSpec.Containers[i])
		podSpec.Containers[i].Name = name
	}
}

const (
	fuzzDGDRProfilerContainerName     = "profiler"
	fuzzDGDROutputCopierContainerName = "output-copier"
)

func fuzzDGDRProfilingExtraContainerNames(c randfill.Continue) []string {
	names := make([]string, 1+c.Intn(3))
	for i := range names {
		names[i] = fmt.Sprintf("container-%d", i)
	}
	return names
}

func fuzzDGDRProfilingContainerNames(c randfill.Continue) []string {
	switch c.Intn(10) {
	case 0:
		return nil
	case 1:
		return []string{}
	}

	var names []string
	switch c.Intn(7) {
	case 0:
		names = []string{fuzzDGDRProfilerContainerName}
	case 1:
		names = []string{fuzzDGDROutputCopierContainerName}
	case 2:
		names = fuzzDGDRProfilingExtraContainerNames(c)
	case 3:
		names = []string{fuzzDGDRProfilerContainerName, fuzzDGDROutputCopierContainerName}
	case 4:
		names = append([]string{fuzzDGDROutputCopierContainerName}, fuzzDGDRProfilingExtraContainerNames(c)...)
	case 5:
		names = append([]string{fuzzDGDRProfilerContainerName}, fuzzDGDRProfilingExtraContainerNames(c)...)
	default:
		names = append(
			[]string{fuzzDGDRProfilerContainerName, fuzzDGDROutputCopierContainerName},
			fuzzDGDRProfilingExtraContainerNames(c)...,
		)
	}

	for i := len(names) - 1; i > 0; i-- {
		j := c.Intn(i + 1)
		names[i], names[j] = names[j], names[i]
	}

	return names
}

func fuzzBetaDGDRStatus(s *v1beta1.DynamoGraphDeploymentRequestStatus, c randfill.Continue) {
	c.Fill(&s.DynamoGraphDeploymentRequestNativeStatus)
	s.V1Beta2 = nil
}

func fuzzBetaDGDRNativeStatus(s *v1beta1.DynamoGraphDeploymentRequestNativeStatus, c randfill.Continue) {
	c.FillNoCustom(s)

	s.Phase = oneOf(c, v1beta1.DGDRPhasePending, v1beta1.DGDRPhaseProfiling, v1beta1.DGDRPhaseReady, v1beta1.DGDRPhaseDeploying, v1beta1.DGDRPhaseDeployed, v1beta1.DGDRPhaseFailed)

	// Profiling substatus is only valid while the request is profiling.
	if s.Phase != v1beta1.DGDRPhaseProfiling {
		s.ProfilingPhase = ""
		s.ProfilingJobName = ""
	}

	if s.ProfilingResults != nil {
		// Empty ProfilingResults is not reconstructed without SelectedConfig or Pareto.
		if s.ProfilingResults.SelectedConfig == nil && len(s.ProfilingResults.Pareto) == 0 {
			s.ProfilingResults = nil
		}
	}
}

func fuzzBeta2DGDR(r *v1beta2.DynamoGraphDeploymentRequest, c randfill.Continue) {
	c.FillNoCustom(r)
	if c.Bool() {
		return
	}

	var spec v1beta1.DynamoGraphDeploymentRequestNativeSpec
	c.Fill(&spec)
	r.Spec = v1beta2.DynamoGraphDeploymentRequestSpec{
		V1Beta1: marshalFuzzEnvelope(spec),
	}
	r.Status = v1beta2.DynamoGraphDeploymentRequestStatus{}
	if c.Bool() {
		return
	}

	var status v1beta1.DynamoGraphDeploymentRequestNativeStatus
	c.Fill(&status)
	r.Status.V1Beta1 = marshalFuzzEnvelope(status)
}

func fuzzBeta2DGDRSpec(s *v1beta2.DynamoGraphDeploymentRequestSpec, c randfill.Continue) {
	c.Fill(&s.DynamoGraphDeploymentRequestNativeSpec)
	s.V1Beta1 = nil
}

func fuzzBeta2DGDRStatus(s *v1beta2.DynamoGraphDeploymentRequestStatus, c randfill.Continue) {
	c.Fill(&s.DynamoGraphDeploymentRequestNativeStatus)
	s.V1Beta1 = nil
}

func fuzzBeta2DGDRNativeSpec(s *v1beta2.DynamoGraphDeploymentRequestNativeSpec, c randfill.Continue) {
	c.FillNoCustom(&s.NativeSearchInput)

	// Keep the required model and all enum-backed values inside their admitted domains.
	s.Model = &v1beta2.ModelSpec{Name: fmt.Sprintf("model-%d", c.Uint32())}
	if c.Bool() {
		backendCount := 1 + c.Intn(3)
		s.Backends = []v1beta2.Backend{v1beta2.BackendVLLM, v1beta2.BackendSGLang, v1beta2.BackendTRTLLM}[:backendCount]
	} else {
		s.Backends = nil
	}
	fuzzBeta2Hardware(s.Hardware, c)
	fuzzBeta2Workload(s.Workload, c)
	fuzzBeta2Objective(s.Objective, c)

	// Keep the optional candidate bound inside its admitted range.
	if s.Recommendation != nil {
		s.Recommendation.MaxCandidates = int32(1 + c.Intn(100))
	}
}

func fuzzBeta2DGDRNativeStatus(s *v1beta2.DynamoGraphDeploymentRequestNativeStatus, c randfill.Continue) {
	c.FillNoCustom(s)
	for i := range s.Conditions {
		s.Conditions[i].Type = fmt.Sprintf("Condition%d", i)
		s.Conditions[i].Status = oneOf(c, metav1.ConditionTrue, metav1.ConditionFalse, metav1.ConditionUnknown)
		s.Conditions[i].ObservedGeneration = c.Int63()
		s.Conditions[i].Reason = fmt.Sprintf("Reason%d", i)
	}
}

func fuzzBeta2Hardware(hardware *v1beta2.HardwareSpec, c randfill.Continue) {
	if hardware == nil || hardware.GPU == nil {
		return
	}
	if hardware.GPU.SKU != nil {
		*hardware.GPU.SKU = fmt.Sprintf("gpu-%d", c.Uint32())
	}
	if hardware.GPU.Budget != nil {
		*hardware.GPU.Budget = int32(1 + c.Intn(1024))
	}
}

func fuzzBeta2Workload(workload *v1beta2.WorkloadSpec, c randfill.Continue) {
	if workload == nil || workload.Static == nil {
		return
	}

	static := workload.Static
	static.ISL = oneOfPtr(c, int32(1), int32(4000), int32(131072))
	static.OSL = oneOfPtr(c, int32(1), int32(1000), int32(16384))
	static.NumRequestRatio = oneOfPtr(c, 0.5, 1.0, 1000.0)
	static.RequestRate = nil
	static.Concurrency = nil
	static.KVLoadRatio = nil
	switch c.Intn(4) {
	case 1:
		static.RequestRate = oneOfPtr(c, 0.1, 1.0, 100.0)
	case 2:
		static.Concurrency = oneOfPtr(c, int32(1), int32(32), int32(1024))
	case 3:
		if c.Bool() {
			static.KVLoadRatio = v1beta2.KVLoadRatio{0.5}
		} else {
			static.KVLoadRatio = v1beta2.KVLoadRatio{0.0, 1.0}
		}
	}
}

func fuzzBeta2Objective(objective *v1beta2.ObjectiveSpec, c randfill.Continue) {
	if objective == nil {
		return
	}

	metrics := []v1beta2.ObjectiveMetric{
		"Throughput",
		"ThroughputPerGPU",
		"ThroughputPerUser",
		"TTFT",
		"E2ELatency",
		"Goodput",
		"GoodputPerGPU",
	}
	fuzzBeta2SLA(objective.SLA, c)
	switch c.Intn(3) {
	case 0:
		*objective = v1beta2.ObjectiveSpec{}
	case 1:
		objective.Mode = v1beta2.ObjectiveModeOptimize
		objective.Metrics = nil
		objective.Metric = oneOfPtr(c, metrics...)
		if objective.Metric != nil && isGoodputMetric(*objective.Metric) {
			ensureBeta2SLA(objective, c)
		}
	case 2:
		objective.Mode = v1beta2.ObjectiveModePareto
		objective.Metric = nil
		metricCount := 2 + c.Intn(len(metrics)-1)
		objective.Metrics = append(objective.Metrics[:0], metrics[:metricCount]...)
		if slices.ContainsFunc(objective.Metrics, isGoodputMetric) {
			ensureBeta2SLA(objective, c)
		}
	}
}

func fuzzBeta2SLA(sla *v1beta2.ObjectiveSLASpec, c randfill.Continue) {
	if sla == nil {
		return
	}

	*sla = v1beta2.ObjectiveSLASpec{}
	switch c.Intn(3) {
	case 1:
		sla.TTFTMs = oneOfPtr(c, 1.0, 2000.0)
		sla.ITLMs = oneOfPtr(c, 1.0, 30.0)
	case 2:
		sla.E2EMs = oneOfPtr(c, 1.0, 30000.0)
	}
}

func ensureBeta2SLA(objective *v1beta2.ObjectiveSpec, c randfill.Continue) {
	if objective.SLA == nil {
		objective.SLA = &v1beta2.ObjectiveSLASpec{}
	}
	if objective.SLA.TTFTMs == nil && objective.SLA.ITLMs == nil && objective.SLA.E2EMs == nil {
		ttft := oneOf(c, 1.0, 2000.0)
		objective.SLA.TTFTMs = &ttft
	}
}

func isGoodputMetric(metric v1beta2.ObjectiveMetric) bool {
	return metric == "Goodput" || metric == "GoodputPerGPU"
}

type fuzzAISimulateParameters struct {
	Search   *fuzzAISimulateSearch   `json:"search,omitempty"`
	Workload *fuzzAISimulateWorkload `json:"workload,omitempty"`
}

type fuzzAISimulateSearch struct {
	MaxTrials int32 `json:"max_trials,omitempty"`
	StrictSLA bool  `json:"strict_sla,omitempty"`
}

type fuzzAISimulateWorkload struct {
	RequestCount int32             `json:"request_count,omitempty"`
	Custom       map[string]string `json:"custom,omitempty"`
}

func fuzzAISimulateSpec(s *v1beta2.AISimulateSpec, c randfill.Continue) {
	c.FillNoCustom(s)

	// Exercise both absent and typed object parameter payloads.
	if c.Bool() {
		s.Parameters = nil
		return
	}

	// Marshal a typed test input instead of generating arbitrary RawExtension bytes.
	parameters := &fuzzAISimulateParameters{}
	c.Fill(parameters)
	raw, err := json.Marshal(parameters)
	if err != nil {
		panic(err)
	}
	s.Parameters = &runtime.RawExtension{Raw: raw}
}

func marshalFuzzEnvelope(value any) *apiextensionsv1.JSON {
	raw, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return &apiextensionsv1.JSON{Raw: raw}
}

func newRoundTripFiller(seed int64) *randfill.Filler {
	funcs := apitestingfuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, dynamoFuzzerFuncs)
	return apitestingfuzzer.FuzzerFor(funcs, rand.NewSource(seed), runtimeserializer.NewCodecFactory(runtime.NewScheme()))
}

func oneOf[T any](c randfill.Continue, values ...T) T {
	return values[c.Intn(len(values))]
}

// oneOfPtr returns nil roughly half the time; otherwise a pointer to one of
// values. Useful for optional API fields that must either be unset or draw
// from a constrained value set.
func oneOfPtr[T any](c randfill.Continue, values ...T) *T {
	if c.Bool() {
		return nil
	}
	v := oneOf(c, values...)
	return &v
}

func fuzzJSONValue(c randfill.Continue, depth int) any {
	if depth >= 2 {
		switch c.Uint32() % 5 {
		case 0:
			return nil
		case 1:
			return c.Bool()
		case 2:
			return c.Int63() % 1024
		case 3:
			return fmt.Sprintf("s%d", c.Uint32()%1024)
		default:
			return float64(c.Uint32()%1000) / 10
		}
	}

	switch c.Uint32() % 7 {
	case 0:
		return nil
	case 1:
		return c.Bool()
	case 2:
		return c.Int63() % 1024
	case 3:
		return fmt.Sprintf("s%d", c.Uint32()%1024)
	case 4:
		out := make([]any, int(c.Uint32()%3))
		for i := range out {
			out[i] = fuzzJSONValue(c, depth+1)
		}
		return out
	case 5:
		n := int(c.Uint32() % 3)
		out := make(map[string]any, n)
		for i := range n {
			out[fmt.Sprintf("k%d", i)] = fuzzJSONValue(c, depth+1)
		}
		return out
	default:
		return float64(c.Uint32()%1000) / 10
	}
}

func fuzzJSONObject(c randfill.Continue, depth int) map[string]any {
	n := int(c.Uint32() % 3)
	out := make(map[string]any, n)
	for i := range n {
		out[fmt.Sprintf("k%d", i)] = fuzzJSONValue(c, depth+1)
	}
	return out
}

func mustJSON(v any) string {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("(marshal err: %v)", err)
	}
	return string(b)
}

func toYAML(t *testing.T, v any) string {
	t.Helper()
	b, err := yaml.Marshal(v)
	if err != nil {
		t.Fatalf("toYAML: %v", err)
	}
	return string(b)
}

// fuzzHubSpokeHub runs the hub -> spoke -> hub round-trip *fuzzIters times
// for the given type pair.
//
//   - H must implement conversion.Hub (e.g. *v1beta1.DynamoGraphDeployment).
//   - PS must implement conversion.Convertible and have underlying type *S
//     (e.g. *v1alpha1.DynamoGraphDeployment with S = v1alpha1.DynamoGraphDeployment).
func fuzzHubSpokeHub[
	H conversion.Hub,
	S any,
	PS interface {
		*S
		conversion.Convertible
	},
](t *testing.T, name string, newHub func() H, diffOpts ...cmp.Option) {
	t.Helper()
	t.Logf("hub->spoke->hub %s seed=%d iters=%d", name, *fuzzSeed, *fuzzIters)
	f := newRoundTripFiller(*fuzzSeed)
	diffOpts = append([]cmp.Option{cmpopts.EquateEmpty()}, diffOpts...)
	for i := 0; i < *fuzzIters; i++ {
		in := newHub()
		f.Fill(in)
		inBeforeYAML := toYAML(t, in)
		spoke := PS(new(S))
		if err := spoke.ConvertFrom(in); err != nil {
			t.Fatalf("%s iter %d ConvertFrom: %v\ninput=%s", name, i, err, mustJSON(in))
		}
		if diff := cmp.Diff(inBeforeYAML, toYAML(t, in)); diff != "" {
			t.Fatalf("%s iter %d ConvertFrom mutated input (-before +after):\n%s\ninput=%s", name, i, diff, mustJSON(in))
		}
		out := newHub()
		if err := spoke.ConvertTo(out); err != nil {
			t.Fatalf("%s iter %d ConvertTo: %v\ninput=%s", name, i, err, mustJSON(in))
		}
		if diff := cmp.Diff(in, out, diffOpts...); diff != "" {
			t.Fatalf("%s iter %d hub->spoke->hub mismatch (-want +got):\n%s\ninput=%s", name, i, diff, mustJSON(in))
		}
	}
}

// fuzzSpokeHubSpoke runs the spoke -> hub -> spoke round-trip *fuzzIters times.
// Symmetric to fuzzHubSpokeHub.
func fuzzSpokeHubSpoke[
	H conversion.Hub,
	S any,
	PS interface {
		*S
		conversion.Convertible
	},
](t *testing.T, name string, newHub func() H, diffOpts ...cmp.Option) {
	t.Helper()
	t.Logf("spoke->hub->spoke %s seed=%d iters=%d", name, *fuzzSeed, *fuzzIters)
	f := newRoundTripFiller(*fuzzSeed)
	diffOpts = append([]cmp.Option{cmpopts.EquateEmpty()}, diffOpts...)
	for i := 0; i < *fuzzIters; i++ {
		in := PS(new(S))
		f.Fill(in)
		inBeforeYAML := toYAML(t, in)
		hub := newHub()
		if err := in.ConvertTo(hub); err != nil {
			t.Fatalf("%s iter %d ConvertTo: %v\ninput=%s", name, i, err, mustJSON(in))
		}
		if diff := cmp.Diff(inBeforeYAML, toYAML(t, in)); diff != "" {
			t.Fatalf("%s iter %d ConvertTo mutated input (-before +after):\n%s\ninput=%s", name, i, diff, mustJSON(in))
		}
		out := PS(new(S))
		if err := out.ConvertFrom(hub); err != nil {
			t.Fatalf("%s iter %d ConvertFrom: %v\ninput=%s", name, i, err, mustJSON(in))
		}
		if diff := cmp.Diff(in, out, diffOpts...); diff != "" {
			t.Fatalf("%s iter %d spoke->hub->spoke mismatch (-want +got):\n%s\ninput=%s", name, i, diff, mustJSON(in))
		}
	}
}

func TestFuzzRoundTrip_DGD_HubSpokeHub(t *testing.T) {
	fuzzHubSpokeHub[*v1beta1.DynamoGraphDeployment, v1alpha1.DynamoGraphDeployment](t, "DGD",
		func() *v1beta1.DynamoGraphDeployment { return &v1beta1.DynamoGraphDeployment{} },
	)
}

func TestFuzzRoundTrip_DGD_SpokeHubSpoke(t *testing.T) {
	fuzzSpokeHubSpoke[*v1beta1.DynamoGraphDeployment, v1alpha1.DynamoGraphDeployment](t, "DGD",
		func() *v1beta1.DynamoGraphDeployment { return &v1beta1.DynamoGraphDeployment{} },
	)
}

func TestFuzzRoundTrip_DCD_HubSpokeHub(t *testing.T) {
	fuzzHubSpokeHub[*v1beta1.DynamoComponentDeployment, v1alpha1.DynamoComponentDeployment](t, "DCD",
		func() *v1beta1.DynamoComponentDeployment { return &v1beta1.DynamoComponentDeployment{} },
	)
}

func TestFuzzRoundTrip_DCD_SpokeHubSpoke(t *testing.T) {
	fuzzSpokeHubSpoke[*v1beta1.DynamoComponentDeployment, v1alpha1.DynamoComponentDeployment](t, "DCD",
		func() *v1beta1.DynamoComponentDeployment { return &v1beta1.DynamoComponentDeployment{} },
	)
}

func TestFuzzRoundTrip_DGDR_HubSpokeHub(t *testing.T) {
	fuzzHubSpokeHub[*v1beta1.DynamoGraphDeploymentRequest, v1alpha1.DynamoGraphDeploymentRequest](t, "DGDR",
		func() *v1beta1.DynamoGraphDeploymentRequest { return &v1beta1.DynamoGraphDeploymentRequest{} },
	)
}

func TestFuzzRoundTrip_DGDR_SpokeHubSpoke(t *testing.T) {
	fuzzSpokeHubSpoke[*v1beta1.DynamoGraphDeploymentRequest, v1alpha1.DynamoGraphDeploymentRequest](t, "DGDR",
		func() *v1beta1.DynamoGraphDeploymentRequest { return &v1beta1.DynamoGraphDeploymentRequest{} },
	)
}

func TestFuzzRoundTrip_DGDRV1Beta2_HubSpokeHub(t *testing.T) {
	fuzzHubSpokeHub[*v1beta1.DynamoGraphDeploymentRequest, v1beta2.DynamoGraphDeploymentRequest](t, "DGDR v1beta2",
		func() *v1beta1.DynamoGraphDeploymentRequest { return &v1beta1.DynamoGraphDeploymentRequest{} },
	)
}

func TestFuzzRoundTrip_DGDRV1Beta2_SpokeHubSpoke(t *testing.T) {
	fuzzSpokeHubSpoke[*v1beta1.DynamoGraphDeploymentRequest, v1beta2.DynamoGraphDeploymentRequest](t, "DGDR v1beta2",
		func() *v1beta1.DynamoGraphDeploymentRequest { return &v1beta1.DynamoGraphDeploymentRequest{} },
	)
}

func TestFuzzRoundTrip_DGDSA_HubSpokeHub(t *testing.T) {
	fuzzHubSpokeHub[*v1beta1.DynamoGraphDeploymentScalingAdapter, v1alpha1.DynamoGraphDeploymentScalingAdapter](t, "DGDSA",
		func() *v1beta1.DynamoGraphDeploymentScalingAdapter {
			return &v1beta1.DynamoGraphDeploymentScalingAdapter{}
		},
	)
}

func TestFuzzRoundTrip_DGDSA_SpokeHubSpoke(t *testing.T) {
	fuzzSpokeHubSpoke[*v1beta1.DynamoGraphDeploymentScalingAdapter, v1alpha1.DynamoGraphDeploymentScalingAdapter](t, "DGDSA",
		func() *v1beta1.DynamoGraphDeploymentScalingAdapter {
			return &v1beta1.DynamoGraphDeploymentScalingAdapter{}
		},
	)
}
