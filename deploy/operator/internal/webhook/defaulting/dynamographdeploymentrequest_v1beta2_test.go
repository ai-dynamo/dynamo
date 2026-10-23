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

package defaulting

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/aisimulate"
	admissionv1 "k8s.io/api/admission/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

func v1beta2Ptr[T any](v T) *T { return &v }

// v1beta2SchemaDefaulted is a model-only Request as the API server hands it to admission: the
// CRD has applied its defaults, and admission has not run.
func v1beta2SchemaDefaulted() *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	return &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
		Model:    &nvidiacomv1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
		Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM, nvidiacomv1beta2.BackendSGLang, nvidiacomv1beta2.BackendTRTLLM},
		Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
			ISL: v1beta2Ptr(int32(1024)), OSL: v1beta2Ptr(int32(128)), Concurrency: v1beta2Ptr(int32(10)),
		}},
		Objective: &nvidiacomv1beta2.ObjectiveSpec{
			Mode: nvidiacomv1beta2.ObjectiveModeOptimize,
		},
		Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 1},
	}
}

func v1beta2JSON(t *testing.T, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) string {
	t.Helper()
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	return string(raw)
}

func TestDefaultSpecV1beta2ModelOnlyRequest(t *testing.T) {
	spec := v1beta2SchemaDefaulted()
	defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, dynamoGraphDeploymentRequestV1beta2DefaultOptions{
		hardware:        hardwareDefaultsV1beta2{sku: "h200_sxm", budget: 8},
		aisimulateImage: "registry.example.com/aisimulate:1.2.0",
	})
	if got := ptrDeref(spec.Objective.Metric); got != "Throughput" {
		t.Fatalf("metric = %q", got)
	}
	gpu := spec.Hardware.GPU
	if *gpu.SKU != "h200_sxm" || *gpu.Budget != 8 {
		t.Fatalf("hardware = %s/%d", *gpu.SKU, *gpu.Budget)
	}
	if spec.AISimulate.Image != "registry.example.com/aisimulate:1.2.0" {
		t.Fatalf("image = %q", spec.AISimulate.Image)
	}
}

func ptrDeref(metric *nvidiacomv1beta2.ObjectiveMetric) nvidiacomv1beta2.ObjectiveMetric {
	if metric == nil {
		return ""
	}
	return *metric
}

func TestDefaultSpecV1beta2IsIdempotent(t *testing.T) {
	opts := dynamoGraphDeploymentRequestV1beta2DefaultOptions{
		hardware:        hardwareDefaultsV1beta2{sku: "h200_sxm", budget: 8},
		aisimulateImage: "registry.example.com/aisimulate:1.2.0",
	}
	spec := v1beta2SchemaDefaulted()
	defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
	first := v1beta2JSON(t, spec)
	defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
	if second := v1beta2JSON(t, spec); second != first {
		t.Fatalf("a second pass changed the spec:\n%s\n%s", first, second)
	}
}

func TestDefaultSpecV1beta2PreservesExplicitValues(t *testing.T) {
	opts := dynamoGraphDeploymentRequestV1beta2DefaultOptions{
		hardware:        hardwareDefaultsV1beta2{sku: "h200_sxm", budget: 8},
		aisimulateImage: "registry.example.com/aisimulate:1.2.0",
	}
	t.Run("hardware", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: v1beta2Ptr("b200_sxm"), Budget: v1beta2Ptr(int32(2))}}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if *spec.Hardware.GPU.SKU != "b200_sxm" || *spec.Hardware.GPU.Budget != 2 {
			t.Fatalf("hardware overwritten: %s/%d", *spec.Hardware.GPU.SKU, *spec.Hardware.GPU.Budget)
		}
	})
	t.Run("a pinned SKU lets admission size the budget", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: v1beta2Ptr("b200_sxm")}}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if *spec.Hardware.GPU.SKU != "b200_sxm" || *spec.Hardware.GPU.Budget != 8 {
			t.Fatalf("got %s/%d", *spec.Hardware.GPU.SKU, *spec.Hardware.GPU.Budget)
		}
	})
	t.Run("an empty hardware object is filled", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.Hardware = &nvidiacomv1beta2.HardwareSpec{}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if spec.Hardware.GPU == nil || *spec.Hardware.GPU.SKU != "h200_sxm" {
			t.Fatal("hardware not filled")
		}
	})
	t.Run("metric", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.Objective.Metric = v1beta2Ptr(nvidiacomv1beta2.ObjectiveMetric("TTFT"))
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if *spec.Objective.Metric != "TTFT" {
			t.Fatalf("metric overwritten: %s", *spec.Objective.Metric)
		}
	})
	t.Run("image", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Image: "registry.example.com/aisimulate:custom", VersionOverride: "1.4.0"}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if spec.AISimulate.Image != "registry.example.com/aisimulate:custom" {
			t.Fatalf("image overwritten: %s", spec.AISimulate.Image)
		}
	})
	t.Run("parameters are untouched", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{Raw: []byte(`{"goal":{"target":"ttft"}}`)}}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if string(spec.AISimulate.Parameters.Raw) != `{"goal":{"target":"ttft"}}` {
			t.Fatal("parameters changed")
		}
		if spec.AISimulate.Image != "registry.example.com/aisimulate:1.2.0" {
			t.Fatal("image not persisted next to parameters")
		}
	})
}

func TestDefaultSpecV1beta2KeepsExplicitValues(t *testing.T) {
	opts := dynamoGraphDeploymentRequestV1beta2DefaultOptions{
		hardware:        hardwareDefaultsV1beta2{sku: "h200_sxm", budget: 8},
		aisimulateImage: "registry.example.com/aisimulate:1.2.0",
	}
	t.Run("an empty workload receives the static default", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.Workload = &nvidiacomv1beta2.WorkloadSpec{}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if !hasModelOnlyStatic(spec.Workload) {
			t.Fatal("workload: {} must receive the static model-only default")
		}
	})
	t.Run("an explicit empty static workload is an opt-out", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.Workload = &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{}}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if spec.Workload.Static.ISL != nil || spec.Workload.Static.OSL != nil || spec.Workload.Static.Concurrency != nil {
			t.Fatal("an explicit workload.static: {} was filled")
		}
	})
	t.Run("omitted workload", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.Workload = nil
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if !hasModelOnlyStatic(spec.Workload) {
			t.Fatal("an omitted workload must become workload: {}, with the static model-only default")
		}
	})
	t.Run("explicit empty SLA", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.Objective.SLA = &nvidiacomv1beta2.ObjectiveSLASpec{}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if spec.Objective.SLA.TTFTMs != nil || spec.Objective.SLA.ITLMs != nil {
			t.Fatal("an explicit sla: {} was filled")
		}
	})
	t.Run("an explicit zero-valued backend list element is kept", func(t *testing.T) {
		spec := v1beta2SchemaDefaulted()
		spec.Backends = []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if len(spec.Backends) != 1 {
			t.Fatal("backends changed")
		}
	})
}

func TestDefaultObjectiveSpecV1beta2(t *testing.T) {
	cases := []struct {
		name      string
		objective nvidiacomv1beta2.ObjectiveSpec
		want      nvidiacomv1beta2.ObjectiveMetric
	}{
		{"optimize without a metric", nvidiacomv1beta2.ObjectiveSpec{Mode: nvidiacomv1beta2.ObjectiveModeOptimize}, "Throughput"},
		{"an explicit metric is kept", nvidiacomv1beta2.ObjectiveSpec{Mode: nvidiacomv1beta2.ObjectiveModeOptimize, Metric: v1beta2Ptr(nvidiacomv1beta2.ObjectiveMetric("TTFT"))}, "TTFT"},
		{"pareto has no scalar metric", nvidiacomv1beta2.ObjectiveSpec{Mode: nvidiacomv1beta2.ObjectiveModePareto, Metrics: []nvidiacomv1beta2.ObjectiveMetric{"TTFT", "Throughput"}}, ""},
		{"an absent mode is Optimize, which admission fills", nvidiacomv1beta2.ObjectiveSpec{}, "Throughput"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defaultObjectiveSpecV1beta2(&tc.objective)
			if got := ptrDeref(tc.objective.Metric); got != tc.want {
				t.Fatalf("metric = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestDefaultSpecV1beta2DoesNotCreateWhatTheEnvironmentCannotFill(t *testing.T) {
	spec := v1beta2SchemaDefaulted()
	defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, dynamoGraphDeploymentRequestV1beta2DefaultOptions{})
	if spec.Hardware != nil || spec.AISimulate != nil {
		t.Fatalf("created sections with nothing to put in them: %s", v1beta2JSON(t, spec))
	}
	spec.Objective = nil
	defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, dynamoGraphDeploymentRequestV1beta2DefaultOptions{})
	if spec.Objective == nil || ptrDeref(spec.Objective.Metric) != "Throughput" {
		t.Fatal("an omitted objective must become objective: {} with the Optimize metric default")
	}
}

func TestNativeTargetOverridesTheDefaultedMetric(t *testing.T) {
	spec := v1beta2SchemaDefaulted()
	spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{Raw: []byte(`{"goal":{"target":"ttft"}}`)}}
	defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, dynamoGraphDeploymentRequestV1beta2DefaultOptions{})
	if *spec.Objective.Metric != "Throughput" {
		t.Fatalf("the typed metric must be defaulted regardless of the parameters, got %q", *spec.Objective.Metric)
	}
	input, err := aisimulate.Lower(spec)
	if err != nil {
		t.Fatal(err)
	}
	if target, _ := input["goal"].(map[string]any)["target"]; target != "ttft" {
		t.Fatalf("the native target must win when lowered, got %v", target)
	}
}

type fakeDiscoverer struct {
	found   DiscoveredHardware
	err     error
	calls   int
	filter  string
	timeout time.Duration
}

func (f *fakeDiscoverer) Discover(ctx context.Context, filterSKU string) (DiscoveredHardware, error) {
	f.calls++
	f.filter = filterSKU
	if deadline, ok := ctx.Deadline(); ok {
		f.timeout = time.Until(deadline)
	}
	return f.found, f.err
}

func TestResolveHardwareV1beta2(t *testing.T) {
	spec := func(sku *string, budget *int32) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
		s := v1beta2SchemaDefaulted()
		if sku != nil || budget != nil {
			s.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: sku, Budget: budget}}
		}
		return s
	}

	t.Run("no discoverer", func(t *testing.T) {
		if got := (DGDRV1beta2Options{}).resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil)); !(got == hardwareDefaultsV1beta2{}) {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("complete hardware never asks the cluster", func(t *testing.T) {
		d := &fakeDiscoverer{found: DiscoveredHardware{SKU: "h100_sxm", GPUs: 4}}
		got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), admissionv1.Create, nil, spec(v1beta2Ptr("b200_sxm"), v1beta2Ptr(int32(2))))
		if d.calls != 0 || !(got == hardwareDefaultsV1beta2{}) {
			t.Fatalf("calls=%d got=%+v", d.calls, got)
		}
	})
	t.Run("everything missing", func(t *testing.T) {
		d := &fakeDiscoverer{found: DiscoveredHardware{SKU: "h200_sxm", GPUs: 16}}
		got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil))
		if got.sku != "h200_sxm" || got.budget != 16 || d.filter != "" {
			t.Fatalf("got %+v, filter %q", got, d.filter)
		}
		if d.timeout <= 0 || d.timeout > defaultHardwareDiscoveryTimeout {
			t.Fatalf("discovery timeout = %v", d.timeout)
		}
	})
	t.Run("a pinned SKU filters discovery and keeps its value", func(t *testing.T) {
		d := &fakeDiscoverer{found: DiscoveredHardware{SKU: "b200_sxm", GPUs: 4}}
		got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), admissionv1.Create, nil, spec(v1beta2Ptr("b200_sxm"), nil))
		if d.filter != "b200_sxm" || got.sku != "" || got.budget != 4 {
			t.Fatalf("got %+v, filter %q", got, d.filter)
		}
	})
	t.Run("a pinned budget keeps its value", func(t *testing.T) {
		d := &fakeDiscoverer{found: DiscoveredHardware{SKU: "h200_sxm", GPUs: 64}}
		got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, v1beta2Ptr(int32(3))))
		if got.sku != "h200_sxm" || got.budget != 0 {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("the budget is capped", func(t *testing.T) {
		d := &fakeDiscoverer{found: DiscoveredHardware{SKU: "h200_sxm", GPUs: 1024}}
		if got := (DGDRV1beta2Options{Hardware: d}).resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil)); got.budget != 32 {
			t.Fatalf("default cap: %+v", got)
		}
		if got := (DGDRV1beta2Options{Hardware: d, MaxHardwareBudget: 8}).resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil)); got.budget != 8 {
			t.Fatalf("explicit cap: %+v", got)
		}
	})
	t.Run("an unknown GPU model yields no SKU", func(t *testing.T) {
		d := &fakeDiscoverer{found: DiscoveredHardware{GPUs: 8}}
		if got := (DGDRV1beta2Options{Hardware: d}).resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil)); got.sku != "" || got.budget != 8 {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("a discovery failure leaves everything unset", func(t *testing.T) {
		d := &fakeDiscoverer{err: errors.New("no DCGM")}
		if got := (DGDRV1beta2Options{Hardware: d}).resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil)); !(got == hardwareDefaultsV1beta2{}) {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("the timeout is configurable", func(t *testing.T) {
		d := &fakeDiscoverer{found: DiscoveredHardware{GPUs: 1}}
		DGDRV1beta2Options{Hardware: d, HardwareDiscoveryTimeout: time.Hour}.resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil))
		if d.timeout < time.Minute {
			t.Fatalf("timeout = %v", d.timeout)
		}
	})
}

func TestResolveDefaultsV1beta2(t *testing.T) {
	d := &fakeDiscoverer{found: DiscoveredHardware{SKU: "h200_sxm", GPUs: 8}}
	got := DGDRV1beta2Options{Hardware: d, AISimulateImage: "registry.example.com/aisimulate:1.2.0"}.
		resolveDefaults(context.Background(), admissionv1.Create, nil, v1beta2SchemaDefaulted())
	if got.aisimulateImage != "registry.example.com/aisimulate:1.2.0" || got.hardware.sku != "h200_sxm" || got.hardware.budget != 8 {
		t.Fatalf("got %+v", got)
	}
}

func v1beta2Envelope(t *testing.T, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) *nvidiacomv1beta1.DynamoGraphDeploymentRequest {
	t.Helper()
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}
	dgdr.Spec.V1Beta2 = &apiextensionsv1.JSON{Raw: raw}
	return dgdr
}

func TestDefaultEnvelopeV1beta2(t *testing.T) {
	options := DGDRV1beta2Options{
		AISimulateImage: "registry.example.com/aisimulate:1.2.0",
		Hardware:        &fakeDiscoverer{found: DiscoveredHardware{SKU: "h200_sxm", GPUs: 8}},
	}
	dgdr := v1beta2Envelope(t, v1beta2SchemaDefaulted())

	if err := defaultDynamoGraphDeploymentRequestV1beta2(context.Background(), dgdr, admissionv1.Create, nil, options); err != nil {
		t.Fatal(err)
	}
	var got nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal(dgdr.Spec.V1Beta2.Raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.Hardware == nil || got.Hardware.GPU == nil || *got.Hardware.GPU.SKU != "h200_sxm" || *got.Hardware.GPU.Budget != 8 {
		t.Fatalf("hardware not defaulted: %+v", got.Hardware)
	}
	if got.AISimulate == nil || got.AISimulate.Image != "registry.example.com/aisimulate:1.2.0" {
		t.Fatalf("image not defaulted: %+v", got.AISimulate)
	}
	if got.Objective == nil || got.Objective.Metric == nil {
		t.Fatalf("objective metric not defaulted: %+v", got.Objective)
	}

	// A second admission of the defaulted object changes nothing, byte for byte.
	first := string(dgdr.Spec.V1Beta2.Raw)
	if err := defaultDynamoGraphDeploymentRequestV1beta2(context.Background(), dgdr, admissionv1.Create, nil, options); err != nil {
		t.Fatal(err)
	}
	if string(dgdr.Spec.V1Beta2.Raw) != first {
		t.Fatalf("not idempotent:\n%s\n%s", first, dgdr.Spec.V1Beta2.Raw)
	}
}

func TestDefaultEnvelopeV1beta2LeavesUnchangedEnvelopeUntouched(t *testing.T) {
	// Nothing for admission to add: the objective already carries its metric and there is no
	// environment to contribute. The envelope bytes stay exactly as sent, even though they are
	// not in canonical form.
	spec := v1beta2SchemaDefaulted()
	spec.Objective.Metric = v1beta2Ptr(nvidiacomv1beta2.ObjectiveMetric("ThroughputPerGPU"))
	raw, _ := json.Marshal(spec)
	dgdr := v1beta2Envelope(t, spec)
	dgdr.Spec.V1Beta2.Raw = append([]byte(" "), raw...)
	original := string(dgdr.Spec.V1Beta2.Raw)
	if err := defaultDynamoGraphDeploymentRequestV1beta2(context.Background(), dgdr, admissionv1.Create, nil, DGDRV1beta2Options{}); err != nil {
		t.Fatal(err)
	}
	if string(dgdr.Spec.V1Beta2.Raw) != original {
		t.Fatalf("envelope was rewritten without a change: %q", dgdr.Spec.V1Beta2.Raw)
	}
}

func TestDefaultEnvelopeV1beta2RejectsMalformedSpec(t *testing.T) {
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}
	dgdr.Spec.V1Beta2 = &apiextensionsv1.JSON{Raw: []byte(`{"model": 3}`)}
	if err := defaultDynamoGraphDeploymentRequestV1beta2(context.Background(), dgdr, admissionv1.Create, nil, DGDRV1beta2Options{}); err == nil {
		t.Fatal("want a decode error")
	}
}

func hardwareSpecV1beta2(sku *string, budget *int32) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	spec := v1beta2SchemaDefaulted()
	if sku != nil || budget != nil {
		spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: sku, Budget: budget}}
	}
	return spec
}

func TestHardwareIsNeverDiscoveredOnOtherOperations(t *testing.T) {
	for _, op := range []admissionv1.Operation{admissionv1.Delete, admissionv1.Connect, ""} {
		t.Run("operation "+string(op), func(t *testing.T) {
			d := &fakeDiscoverer{found: DiscoveredHardware{SKU: "h200_sxm", GPUs: 8}}
			old := hardwareSpecV1beta2(v1beta2Ptr("h100_sxm"), v1beta2Ptr(int32(4)))
			got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), op, old, hardwareSpecV1beta2(nil, nil))
			if got != (hardwareDefaultsV1beta2{}) || d.calls != 0 {
				t.Fatalf("hardware resolved on %q: %+v, calls %d", op, got, d.calls)
			}
		})
	}
}

func TestResolveHardwareByOperationV1beta2(t *testing.T) {
	full := func() *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
		return hardwareSpecV1beta2(v1beta2Ptr("h100_sxm"), v1beta2Ptr(int32(4)))
	}
	tests := []struct {
		name       string
		op         admissionv1.Operation
		old, spec  *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
		want       hardwareDefaultsV1beta2
		wantCalls  int
		wantFilter string
	}{
		{"create discovers everything", admissionv1.Create, nil, hardwareSpecV1beta2(nil, nil),
			hardwareDefaultsV1beta2{sku: "h200_sxm", budget: 8}, 1, ""},
		{"metadata-only update without hardware does not discover", admissionv1.Update,
			hardwareSpecV1beta2(nil, nil), hardwareSpecV1beta2(nil, nil), hardwareDefaultsV1beta2{}, 0, ""},
		{"update with unchanged incomplete hardware does not discover", admissionv1.Update,
			hardwareSpecV1beta2(v1beta2Ptr("h100_sxm"), nil), hardwareSpecV1beta2(v1beta2Ptr("h100_sxm"), nil),
			hardwareDefaultsV1beta2{}, 0, ""},
		{"update without a stored spec does not discover", admissionv1.Update,
			nil, hardwareSpecV1beta2(nil, nil), hardwareDefaultsV1beta2{}, 0, ""},
		{"removing only the budget discovers only the budget", admissionv1.Update,
			full(), hardwareSpecV1beta2(v1beta2Ptr("h100_sxm"), nil),
			hardwareDefaultsV1beta2{budget: 8}, 1, "h100_sxm"},
		{"removing only the SKU discovers only the SKU", admissionv1.Update,
			full(), hardwareSpecV1beta2(nil, v1beta2Ptr(int32(4))),
			hardwareDefaultsV1beta2{sku: "h200_sxm"}, 1, ""},
		{"removing all typed hardware discovers both", admissionv1.Update,
			full(), hardwareSpecV1beta2(nil, nil),
			hardwareDefaultsV1beta2{sku: "h200_sxm", budget: 8}, 1, ""},
		{"changing to complete hardware does not discover", admissionv1.Update,
			full(), hardwareSpecV1beta2(v1beta2Ptr("b200_sxm"), v1beta2Ptr(int32(2))),
			hardwareDefaultsV1beta2{}, 0, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := &fakeDiscoverer{found: DiscoveredHardware{SKU: "h200_sxm", GPUs: 8}}
			got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), tt.op, tt.old, tt.spec)
			if got != tt.want || d.calls != tt.wantCalls || (tt.wantCalls > 0 && d.filter != tt.wantFilter) {
				t.Fatalf("got %+v, calls %d, filter %q; want %+v, calls %d, filter %q",
					got, d.calls, d.filter, tt.want, tt.wantCalls, tt.wantFilter)
			}
		})
	}
}

// A Request created while discovery had nothing to offer keeps its spec through a later
// metadata-only update, even after the cluster recovers.
func TestUpdateKeepsUnchangedHardwareUnresolved(t *testing.T) {
	options := DGDRV1beta2Options{Hardware: &fakeDiscoverer{found: DiscoveredHardware{SKU: "h200_sxm", GPUs: 8}}}
	// The objective metric is still defaulted on update, so the spec already carries it.
	spec := v1beta2SchemaDefaulted()
	metric := defaultMetricV1beta2
	spec.Objective.Metric = &metric
	dgdr := v1beta2Envelope(t, spec)
	first := string(dgdr.Spec.V1Beta2.Raw)
	if err := defaultDynamoGraphDeploymentRequestV1beta2(
		context.Background(), dgdr, admissionv1.Update, hardwareSpecV1beta2(nil, nil), options); err != nil {
		t.Fatal(err)
	}
	if string(dgdr.Spec.V1Beta2.Raw) != first {
		t.Fatalf("update changed the spec:\n%s\n%s", first, dgdr.Spec.V1Beta2.Raw)
	}
}

func TestUpdateRediscoversRemovedHardware(t *testing.T) {
	options := DGDRV1beta2Options{Hardware: &fakeDiscoverer{found: DiscoveredHardware{SKU: "h200_sxm", GPUs: 8}}}
	dgdr := v1beta2Envelope(t, v1beta2SchemaDefaulted())
	old := hardwareSpecV1beta2(v1beta2Ptr("h100_sxm"), v1beta2Ptr(int32(4)))
	if err := defaultDynamoGraphDeploymentRequestV1beta2(context.Background(), dgdr, admissionv1.Update, old, options); err != nil {
		t.Fatal(err)
	}
	var got nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal(dgdr.Spec.V1Beta2.Raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.Hardware == nil || got.Hardware.GPU == nil || *got.Hardware.GPU.SKU != "h200_sxm" || *got.Hardware.GPU.Budget != 8 {
		t.Fatalf("hardware not rediscovered: %+v", got.Hardware)
	}
}

func TestDecodeOldSpecV1beta2(t *testing.T) {
	old, err := json.Marshal(v1beta2Envelope(t, hardwareSpecV1beta2(v1beta2Ptr("h100_sxm"), nil)))
	if err != nil {
		t.Fatal(err)
	}
	got, err := decodeOldSpecV1beta2(admissionv1.Update, old)
	if err != nil || got == nil || got.Hardware == nil || *got.Hardware.GPU.SKU != "h100_sxm" {
		t.Fatalf("got %+v, err %v", got, err)
	}
	if got, err := decodeOldSpecV1beta2(admissionv1.Create, old); got != nil || err != nil {
		t.Fatalf("create has no old spec: %+v, %v", got, err)
	}
	if got, err := decodeOldSpecV1beta2(admissionv1.Update, []byte(`{}`)); got != nil || err != nil {
		t.Fatalf("an old object without the envelope has no old spec: %+v, %v", got, err)
	}
	if _, err := decodeOldSpecV1beta2(admissionv1.Update, []byte(`{`)); err == nil {
		t.Fatal("want a decode error")
	}
}

func TestDefaultSpecV1beta2StructuralDefaults(t *testing.T) {
	opts := dynamoGraphDeploymentRequestV1beta2DefaultOptions{}

	t.Run("an empty native spec receives the four structural parents", func(t *testing.T) {
		spec := &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if len(spec.Backends) != 3 {
			t.Fatalf("backends = %v, want vllm, sglang and trtllm", spec.Backends)
		}
		if !hasModelOnlyStatic(spec.Workload) {
			t.Fatalf("workload = %s, want {} with the static model-only default", v1beta2JSON(t, spec))
		}
		if spec.Objective == nil || spec.Recommendation == nil {
			t.Fatalf("objective and recommendation must exist: %s", v1beta2JSON(t, spec))
		}
	})
	t.Run("explicit values and opt-outs are kept", func(t *testing.T) {
		spec := &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{}
		spec.Backends = []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM}
		spec.Workload = &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{}}
		defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, opts)
		if len(spec.Backends) != 1 {
			t.Fatalf("backends changed: %v", spec.Backends)
		}
		if spec.Workload.Static == nil {
			t.Fatal("workload.static: {} (the opt-out) was removed")
		}
	})
}

func hasModelOnlyStatic(w *nvidiacomv1beta2.WorkloadSpec) bool {
	return w != nil && w.Static != nil &&
		int32Is(w.Static.ISL, 1024) && int32Is(w.Static.OSL, 128) && int32Is(w.Static.Concurrency, 10)
}

func int32Is(p *int32, want int32) bool { return p != nil && *p == want }
