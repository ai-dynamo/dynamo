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

package v1alpha1

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apitestingfuzzer "k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/resource"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/randfill"
)

var (
	dgdrLegacyFuzzIters = flag.Int("dgdr-legacy-fuzz-iters", 1000, "iterations per direction for DGDR legacy/structural conversion equivalence tests")
	dgdrLegacyFuzzSeed  = flag.Int64("dgdr-legacy-fuzz-seed", 1, "rand seed for DGDR legacy/structural conversion equivalence tests")
)

func TestDGDRFuzzLegacyAndStructuralConvertToHubEquivalent(t *testing.T) {
	f := newDGDRLegacyComparisonFiller(*dgdrLegacyFuzzSeed)
	for i := 0; i < *dgdrLegacyFuzzIters; i++ {
		src := &DynamoGraphDeploymentRequest{}
		f.Fill(src)
		normalizeAlphaDGDRForLegacyComparison(src)

		legacy, err := legacyDGDRConvertToHubForTest(src)
		if err != nil {
			t.Fatalf("iter %d legacy convert to hub: %v\ninput=%s", i, err, mustDGDRLegacyJSON(src))
		}
		structural := &v1beta1.DynamoGraphDeploymentRequest{}
		if err := src.ConvertTo(structural); err != nil {
			t.Fatalf("iter %d structural ConvertTo: %v\ninput=%s", i, err, mustDGDRLegacyJSON(src))
		}
		stripDGDRSparseAnnotationsForLegacyComparison(&structural.ObjectMeta)

		if diff := cmp.Diff(legacy, structural, cmpopts.EquateEmpty()); diff != "" {
			t.Fatalf("iter %d legacy/structural alpha->hub mismatch (-legacy +structural):\n%s\ninput=%s", i, diff, mustDGDRLegacyJSON(src))
		}
	}
}

func TestDGDRFuzzLegacyAndStructuralConvertFromHubEquivalent(t *testing.T) {
	f := newDGDRLegacyComparisonFiller(*dgdrLegacyFuzzSeed)
	for i := 0; i < *dgdrLegacyFuzzIters; i++ {
		src := &v1beta1.DynamoGraphDeploymentRequest{}
		f.Fill(src)
		normalizeHubDGDRForLegacyComparison(src)

		legacy := legacyDGDRConvertFromHubForTest(src)
		structural := &DynamoGraphDeploymentRequest{}
		if err := structural.ConvertFrom(src); err != nil {
			t.Fatalf("iter %d structural ConvertFrom: %v\ninput=%s", i, err, mustDGDRLegacyJSON(src))
		}
		stripDGDRSparseAnnotationsForLegacyComparison(&structural.ObjectMeta)

		if diff := cmp.Diff(legacy, structural, cmpopts.EquateEmpty()); diff != "" {
			t.Fatalf("iter %d legacy/structural hub->alpha mismatch (-legacy +structural):\n%s\ninput=%s", i, diff, mustDGDRLegacyJSON(src))
		}
	}
}

func TestDGDRFuzzLegacyHubStorageRoundTripsThroughStructural(t *testing.T) {
	f := newDGDRLegacyComparisonFiller(*dgdrLegacyFuzzSeed)
	for i := 0; i < *dgdrLegacyFuzzIters; i++ {
		src := &DynamoGraphDeploymentRequest{}
		f.Fill(src)
		normalizeAlphaDGDRForLegacyComparison(src)

		oldStored, err := legacyDGDRConvertToHubForTest(src)
		if err != nil {
			t.Fatalf("iter %d legacy convert to hub: %v\ninput=%s", i, err, mustDGDRLegacyJSON(src))
		}
		spoke := &DynamoGraphDeploymentRequest{}
		if err := spoke.ConvertFrom(oldStored); err != nil {
			t.Fatalf("iter %d structural ConvertFrom old hub: %v\ninput=%s", i, err, mustDGDRLegacyJSON(oldStored))
		}
		got := &v1beta1.DynamoGraphDeploymentRequest{}
		if err := spoke.ConvertTo(got); err != nil {
			t.Fatalf("iter %d structural ConvertTo old hub roundtrip: %v\ninput=%s", i, err, mustDGDRLegacyJSON(oldStored))
		}
		stripDGDRSparseAnnotationsForLegacyComparison(&got.ObjectMeta)

		if diff := cmp.Diff(oldStored, got, cmpopts.EquateEmpty()); diff != "" {
			t.Fatalf("iter %d legacy hub storage roundtrip mismatch (-old +got):\n%s\ninput=%s", i, diff, mustDGDRLegacyJSON(oldStored))
		}
	}
}

func TestDGDRFuzzLegacySpokeStorageRoundTripsThroughStructural(t *testing.T) {
	f := newDGDRLegacyComparisonFiller(*dgdrLegacyFuzzSeed)
	for i := 0; i < *dgdrLegacyFuzzIters; i++ {
		src := &v1beta1.DynamoGraphDeploymentRequest{}
		f.Fill(src)
		normalizeHubDGDRForLegacyComparison(src)

		oldStored := legacyDGDRConvertFromHubForTest(src)
		hub := &v1beta1.DynamoGraphDeploymentRequest{}
		if err := oldStored.ConvertTo(hub); err != nil {
			t.Fatalf("iter %d structural ConvertTo old spoke: %v\ninput=%s", i, err, mustDGDRLegacyJSON(oldStored))
		}
		got := &DynamoGraphDeploymentRequest{}
		if err := got.ConvertFrom(hub); err != nil {
			t.Fatalf("iter %d structural ConvertFrom old spoke roundtrip: %v\ninput=%s", i, err, mustDGDRLegacyJSON(oldStored))
		}
		stripDGDRSparseAnnotationsForLegacyComparison(&got.ObjectMeta)

		if diff := cmp.Diff(oldStored, got, cmpopts.EquateEmpty()); diff != "" {
			t.Fatalf("iter %d legacy spoke storage roundtrip mismatch (-old +got):\n%s\ninput=%s", i, diff, mustDGDRLegacyJSON(oldStored))
		}
	}
}

func newDGDRLegacyComparisonFiller(seed int64) *randfill.Filler {
	dgdrLegacyFuzzerFuncs := func(_ runtimeserializer.CodecFactory) []interface{} {
		return []interface{}{
			func(m *metav1.ObjectMeta, c randfill.Continue) {
				c.FillNoCustom(m)
				m.Annotations = scrubDGDRConversionAnnotations(m.Annotations)
				m.ManagedFields = nil
			},
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
			func(raw *json.RawMessage, c randfill.Continue) {
				if c.Bool() {
					*raw = nil
					return
				}
				data, err := json.Marshal(fuzzDGDRLegacyJSONValue(c, 0))
				if err != nil {
					panic(err)
				}
				*raw = append((*raw)[:0], data...)
			},
			func(v *apiextensionsv1.JSON, c randfill.Continue) {
				data, err := json.Marshal(fuzzDGDRLegacyJSONObject(c, 0))
				if err != nil {
					panic(err)
				}
				v.Raw = append(v.Raw[:0], data...)
			},
			func(q *resource.Quantity, c randfill.Continue) {
				n := c.Int63() % 65536
				*q = resource.MustParse(strconv.FormatInt(n, 10) + "Mi")
			},
			func(v *intstr.IntOrString, c randfill.Continue) {
				if c.Bool() {
					*v = intstr.FromInt32(c.Int31() % 65535)
				} else {
					*v = intstr.FromString(fmt.Sprintf("p%d", c.Uint32()%65535))
				}
			},
			func(v *v1beta1.OptimizationType, c randfill.Continue) {
				*v = oneOfDGDRLegacy(c, v1beta1.OptimizationTypeLatency, v1beta1.OptimizationTypeThroughput)
			},
			func(s *DynamoGraphDeploymentRequestSpec, c randfill.Continue) {
				c.FillNoCustom(s)
				s.Backend = oneOfDGDRLegacy(c, "auto", "vllm", "sglang", "trtllm")
			},
			func(s *DynamoGraphDeploymentRequestStatus, c randfill.Continue) {
				c.FillNoCustom(s)
				s.State = oneOfDGDRLegacy(c,
					DGDRStateInitializing,
					DGDRStatePending,
					DGDRStateProfiling,
					DGDRStateReady,
					DGDRStateDeploying,
					DGDRStateDeploymentDeleted,
					DGDRStateFailed,
				)
			},
			func(s *v1beta1.DynamoGraphDeploymentRequestSpec, c randfill.Continue) {
				c.FillNoCustom(s)
				s.Backend = oneOfDGDRLegacy(c, v1beta1.BackendTypeAuto, v1beta1.BackendTypeVllm, v1beta1.BackendTypeSglang, v1beta1.BackendTypeTrtllm)
			},
			func(s *v1beta1.DynamoGraphDeploymentRequestStatus, c randfill.Continue) {
				c.FillNoCustom(s)
				s.Phase = oneOfDGDRLegacy(c, v1beta1.DGDRPhasePending, v1beta1.DGDRPhaseProfiling, v1beta1.DGDRPhaseReady, v1beta1.DGDRPhaseDeploying, v1beta1.DGDRPhaseDeployed, v1beta1.DGDRPhaseFailed)
			},
		}
	}
	funcs := apitestingfuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, dgdrLegacyFuzzerFuncs)
	return apitestingfuzzer.FuzzerFor(funcs, rand.NewSource(seed), runtimeserializer.NewCodecFactory(runtime.NewScheme()))
}

func normalizeAlphaDGDRForLegacyComparison(obj *DynamoGraphDeploymentRequest) {
	obj.Annotations = scrubDGDRConversionAnnotations(obj.Annotations)

	// NodeSelector is one of the fixed conversion bugs. The regular DGDR
	// round-trip fuzz tests keep it enabled; the legacy equivalence fuzz keeps
	// the visible legacy-compatible shape and compares the extra sparse payload
	// separately by stripping annDGDRSpec/annDGDRStatus from the structural result.
	obj.Spec.ProfilingConfig.NodeSelector = nil
	if obj.Spec.ProfilingConfig.Resources != nil {
		obj.Spec.ProfilingConfig.Resources.Claims = nil
		if len(obj.Spec.ProfilingConfig.Resources.Requests) == 0 &&
			len(obj.Spec.ProfilingConfig.Resources.Limits) == 0 {
			obj.Spec.ProfilingConfig.Resources = nil
		}
	}
	if obj.Status.Deployment != nil && obj.Status.Deployment.Name == "" {
		obj.Status.Deployment = nil
	}
}

func normalizeHubDGDRForLegacyComparison(obj *v1beta1.DynamoGraphDeploymentRequest) {
	obj.Annotations = scrubDGDRConversionAnnotations(obj.Annotations)
	spec := &obj.Spec
	status := &obj.Status

	if spec.Workload != nil {
		spec.Workload.Concurrency = nil
		spec.Workload.RequestRate = nil
		if spec.Workload.ISL == nil && spec.Workload.OSL == nil {
			spec.Workload = nil
		}
	}
	if spec.SLA != nil {
		spec.SLA.E2ELatency = nil
		if spec.SLA.TTFT == nil && spec.SLA.ITL == nil && spec.SLA.OptimizationType == nil {
			spec.SLA = nil
		}
	}
	if spec.ModelCache != nil &&
		spec.ModelCache.PVCName == "" &&
		spec.ModelCache.PVCModelPath == "" &&
		spec.ModelCache.PVCMountPath == "" {
		spec.ModelCache = nil
	}
	if spec.Overrides != nil && spec.Overrides.ProfilingJob != nil {
		podSpec := &spec.Overrides.ProfilingJob.Template.Spec
		podSpec.NodeSelector = nil
		if len(podSpec.Containers) > 0 {
			podSpec.Containers[0].Resources.Claims = nil
		}
	}

	// Profiling substatus is only valid while the request is profiling.
	if status.Phase != v1beta1.DGDRPhaseProfiling {
		status.ProfilingPhase = ""
		status.ProfilingJobName = ""
	}
}

func stripDGDRSparseAnnotationsForLegacyComparison(obj metav1.Object) {
	delAnnFromObj(obj, annDGDRSpec)
	delAnnFromObj(obj, annDGDRStatus)
}

func scrubDGDRConversionAnnotations(annotations map[string]string) map[string]string {
	if len(annotations) == 0 {
		return annotations
	}
	for k := range annotations {
		if strings.HasPrefix(k, "nvidia.com/dgdr-") {
			delete(annotations, k)
		}
	}
	if len(annotations) == 0 {
		return nil
	}
	return annotations
}

func fuzzDGDRLegacyJSONValue(c randfill.Continue, depth int) any {
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
			out[i] = fuzzDGDRLegacyJSONValue(c, depth+1)
		}
		return out
	case 5:
		return fuzzDGDRLegacyJSONObject(c, depth+1)
	default:
		return float64(c.Uint32()%1000) / 10
	}
}

func fuzzDGDRLegacyJSONObject(c randfill.Continue, depth int) map[string]any {
	n := int(c.Uint32() % 3)
	out := make(map[string]any, n)
	for i := 0; i < n; i++ {
		out[fmt.Sprintf("k%d", i)] = fuzzDGDRLegacyJSONValue(c, depth+1)
	}
	return out
}

func oneOfDGDRLegacy[T any](c randfill.Continue, values ...T) T {
	return values[c.Intn(len(values))]
}

func mustDGDRLegacyJSON(v any) string {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return fmt.Sprintf("(marshal err: %v)", err)
	}
	return string(data)
}
