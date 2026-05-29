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
	"strconv"
	"strings"
	"testing"

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
	"sigs.k8s.io/controller-runtime/pkg/conversion"
	"sigs.k8s.io/randfill"
	"sigs.k8s.io/yaml"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
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

// DGDR still writes legacy nvidia.com/dgdr-* annotations for downgrade
// compatibility with Dynamo 1.1. Direct round-trip fuzzing compares live API
// fields and ignores those intentionally re-emitted compatibility annotations.
var legacyDGDRCompatibilityAnnotations = []string{
	"nvidia.com/dgdr-config-map-ref",
	"nvidia.com/dgdr-output-pvc",
	"nvidia.com/dgdr-enable-gpu-discovery",
	"nvidia.com/dgdr-deployment-overrides",
	"nvidia.com/dgdr-profiling-config",
	"nvidia.com/dgdr-status-backend",
	"nvidia.com/dgdr-profiling-results",
	"nvidia.com/dgdr-deployment-status",
	"nvidia.com/dgdr-profiling-job-name",
}

var ignoreDGDRCompatibilityAnnotations = cmpopts.AcyclicTransformer(
	"ignoreDGDRCompatibilityAnnotations",
	func(m metav1.ObjectMeta) metav1.ObjectMeta {
		if len(m.Annotations) == 0 {
			return m
		}
		annotations := make(map[string]string, len(m.Annotations))
		for k, v := range m.Annotations {
			annotations[k] = v
		}
		for _, k := range legacyDGDRCompatibilityAnnotations {
			delete(annotations, k)
		}
		if len(annotations) == 0 {
			m.Annotations = nil
		} else {
			m.Annotations = annotations
		}
		return m
	},
)

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
		fuzzAlphaDGDRSpec,
		fuzzAlphaDGDRStatus,
		fuzzBetaDGDRSpec,
		fuzzBetaDGDRStatus,
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

func fuzzAlphaDGDRSpec(s *v1alpha1.DynamoGraphDeploymentRequestSpec, c randfill.Continue) {
	c.FillNoCustom(s)

	s.Backend = oneOf(c, "auto", "vllm", "sglang", "trtllm")

	// Empty resources do not survive the alpha resources to beta profiling job projection.
	if s.ProfilingConfig.Resources != nil &&
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

func fuzzBetaDGDRSpec(s *v1beta1.DynamoGraphDeploymentRequestSpec, c randfill.Continue) {
	c.FillNoCustom(s)

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

func fuzzBetaDGDRStatus(s *v1beta1.DynamoGraphDeploymentRequestStatus, c randfill.Continue) {
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

func newRoundTripFiller(seed int64) *randfill.Filler {
	funcs := apitestingfuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, dynamoFuzzerFuncs)
	return apitestingfuzzer.FuzzerFor(funcs, rand.NewSource(seed), runtimeserializer.NewCodecFactory(runtime.NewScheme()))
}

func oneOf[T any](c randfill.Continue, values ...T) T {
	return values[c.Intn(len(values))]
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
		for i := 0; i < n; i++ {
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
	for i := 0; i < n; i++ {
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
		ignoreDGDRCompatibilityAnnotations,
	)
}

func TestFuzzRoundTrip_DGDR_SpokeHubSpoke(t *testing.T) {
	fuzzSpokeHubSpoke[*v1beta1.DynamoGraphDeploymentRequest, v1alpha1.DynamoGraphDeploymentRequest](t, "DGDR",
		func() *v1beta1.DynamoGraphDeploymentRequest { return &v1beta1.DynamoGraphDeploymentRequest{} },
		ignoreDGDRCompatibilityAnnotations,
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
