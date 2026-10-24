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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	admissionv1 "k8s.io/api/admission/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
)

func v1beta2Ptr[T any](v T) *T { return &v }

type fakeDiscoverer struct {
	found   gpu.GPUInfo
	err     error
	calls   int
	filter  string
	timeout time.Duration
}

func (f *fakeDiscoverer) Discover(ctx context.Context, filterSKU nvidiacomv1beta1.GPUSKUType) (*gpu.GPUInfo, error) {
	f.calls++
	f.filter = string(filterSKU)
	if deadline, ok := ctx.Deadline(); ok {
		f.timeout = time.Until(deadline)
	}
	found := f.found
	return &found, f.err
}

func TestResolveHardwareV1beta2(t *testing.T) {
	spec := func(sku *string, budget *int32) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
		s := modelOnlySpecV1beta2()
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
		d := &fakeDiscoverer{found: gpu.GPUInfo{System: "h100_sxm", GPUsPerNode: 4, NodesWithGPUs: 1}}
		got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), admissionv1.Create, nil, spec(v1beta2Ptr("b200_sxm"), v1beta2Ptr(int32(2))))
		if d.calls != 0 || !(got == hardwareDefaultsV1beta2{}) {
			t.Fatalf("calls=%d got=%+v", d.calls, got)
		}
	})
	t.Run("everything missing", func(t *testing.T) {
		d := &fakeDiscoverer{found: gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 16, NodesWithGPUs: 1}}
		got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil))
		if got.sku != "h200_sxm" || got.budget != 16 || d.filter != "" {
			t.Fatalf("got %+v, filter %q", got, d.filter)
		}
		if d.timeout <= 0 || d.timeout > defaultHardwareDiscoveryTimeout {
			t.Fatalf("discovery timeout = %v", d.timeout)
		}
	})
	t.Run("a pinned SKU filters discovery and keeps its value", func(t *testing.T) {
		d := &fakeDiscoverer{found: gpu.GPUInfo{System: "b200_sxm", GPUsPerNode: 4, NodesWithGPUs: 1}}
		got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), admissionv1.Create, nil, spec(v1beta2Ptr("b200_sxm"), nil))
		if d.filter != "b200_sxm" || got.sku != "" || got.budget != 4 {
			t.Fatalf("got %+v, filter %q", got, d.filter)
		}
	})
	t.Run("a pinned budget keeps its value", func(t *testing.T) {
		d := &fakeDiscoverer{found: gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 64, NodesWithGPUs: 1}}
		got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, v1beta2Ptr(int32(3))))
		if got.sku != "h200_sxm" || got.budget != 0 {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("the budget is capped", func(t *testing.T) {
		d := &fakeDiscoverer{found: gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 1024, NodesWithGPUs: 1}}
		if got := (DGDRV1beta2Options{Hardware: d}).resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil)); got.budget != 32 {
			t.Fatalf("default cap: %+v", got)
		}
		if got := (DGDRV1beta2Options{Hardware: d, MaxHardwareBudget: 8}).resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil)); got.budget != 8 {
			t.Fatalf("explicit cap: %+v", got)
		}
	})
	t.Run("an unknown GPU model yields no SKU", func(t *testing.T) {
		d := &fakeDiscoverer{found: gpu.GPUInfo{GPUsPerNode: 8, NodesWithGPUs: 1}}
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
		d := &fakeDiscoverer{found: gpu.GPUInfo{GPUsPerNode: 1, NodesWithGPUs: 1}}
		DGDRV1beta2Options{Hardware: d, HardwareDiscoveryTimeout: time.Hour}.resolveHardware(context.Background(), admissionv1.Create, nil, spec(nil, nil))
		if d.timeout < time.Minute {
			t.Fatalf("timeout = %v", d.timeout)
		}
	})
}

func TestResolveDefaultsV1beta2(t *testing.T) {
	d := &fakeDiscoverer{found: gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1}}
	got := DGDRV1beta2Options{Hardware: d, AISimulateImage: "registry.example.com/aisimulate:1.2.0"}.
		resolveDefaults(context.Background(), admissionv1.Create, nil, modelOnlySpecV1beta2())
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
		Hardware:        &fakeDiscoverer{found: gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1}},
	}
	dgdr := v1beta2Envelope(t, modelOnlySpecV1beta2())

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

func TestDefaultEnvelopeV1beta2RejectsMalformedSpec(t *testing.T) {
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}
	dgdr.Spec.V1Beta2 = &apiextensionsv1.JSON{Raw: []byte(`{"model": 3}`)}
	if err := defaultDynamoGraphDeploymentRequestV1beta2(context.Background(), dgdr, admissionv1.Create, nil, DGDRV1beta2Options{}); err == nil {
		t.Fatal("want a decode error")
	}
}

func hardwareSpecV1beta2(sku *string, budget *int32) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	spec := modelOnlySpecV1beta2()
	if sku != nil || budget != nil {
		spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: sku, Budget: budget}}
	}
	return spec
}

func TestHardwareIsNeverDiscoveredOnOtherOperations(t *testing.T) {
	for _, op := range []admissionv1.Operation{admissionv1.Delete, admissionv1.Connect, ""} {
		t.Run("operation "+string(op), func(t *testing.T) {
			d := &fakeDiscoverer{found: gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1}}
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
			d := &fakeDiscoverer{found: gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1}}
			got := DGDRV1beta2Options{Hardware: d}.resolveHardware(context.Background(), tt.op, tt.old, tt.spec)
			if got != tt.want || d.calls != tt.wantCalls || (tt.wantCalls > 0 && d.filter != tt.wantFilter) {
				t.Fatalf("got %+v, calls %d, filter %q; want %+v, calls %d, filter %q",
					got, d.calls, d.filter, tt.want, tt.wantCalls, tt.wantFilter)
			}
		})
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

// modelOnlySpecV1beta2 is a Request as a client sends it: only the model is named.
func modelOnlySpecV1beta2() *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	spec := &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{}
	spec.Model = &nvidiacomv1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"}
	return spec
}

// An envelope that admission has nothing to add to keeps its exact bytes, even when they are
// not in canonical form.
func TestDefaultEnvelopeV1beta2LeavesUnchangedEnvelopeUntouched(t *testing.T) {
	spec := modelOnlySpecV1beta2()
	defaultDynamoGraphDeploymentRequestSpecV1beta2(spec, dynamoGraphDeploymentRequestV1beta2DefaultOptions{})
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	dgdr := v1beta2Envelope(t, spec)
	dgdr.Spec.V1Beta2.Raw = append([]byte(" "), raw...)
	original := string(dgdr.Spec.V1Beta2.Raw)
	if err := defaultDynamoGraphDeploymentRequestV1beta2(
		context.Background(), dgdr, admissionv1.Create, nil, DGDRV1beta2Options{}); err != nil {
		t.Fatal(err)
	}
	if string(dgdr.Spec.V1Beta2.Raw) != original {
		t.Fatalf("envelope was rewritten without a change: %q", dgdr.Spec.V1Beta2.Raw)
	}
}
