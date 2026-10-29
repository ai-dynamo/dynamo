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
	"errors"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrinput"
	"k8s.io/apimachinery/pkg/runtime"
)

type fakeHardware struct {
	found   DiscoveredHardware
	err     error
	calls   int
	filters []string
}

func (f *fakeHardware) Discover(_ context.Context, filterSKU string) (DiscoveredHardware, error) {
	f.calls++
	f.filters = append(f.filters, filterSKU)
	return f.found, f.err
}

func ptrTo[T any](v T) *T { return &v }

// modelOnly is a model-only spec after the CRD's structural defaults.
func modelOnly() *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	return &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{
		Model:    nvidiacomv1beta2.ModelSpec{Name: "meta-llama/Llama-3.1-8B-Instruct"},
		Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM},
		Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
			ISL: ptrTo(int32(4000)), OSL: ptrTo(int32(1000)), KVLoadRatio: nvidiacomv1beta2.KVLoadRatio{0, 1},
		}},
		Objective: &nvidiacomv1beta2.ObjectiveSpec{
			Mode: nvidiacomv1beta2.ObjectiveModeOptimize,
			SLA:  &nvidiacomv1beta2.ObjectiveSLASpec{TTFTMs: ptrTo(2000.0), ITLMs: ptrTo(30.0)},
		},
		Recommendation: &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: 1},
	}
}

func v1beta2WithHardware(sku *string, budget *int32) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	spec := modelOnly()
	spec.Hardware = &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: sku, Budget: budget}}
	return spec
}

func newV1Beta2Defaulter(hardware HardwareDiscoverer) *DGDRV1Beta2Defaulter {
	return NewDGDRV1Beta2Defaulter(DGDRV1Beta2DefaulterOptions{
		Hardware:        hardware,
		AISimulateImage: "registry.example.com/aisimulate-runner:0.13.0",
	})
}

func TestV1Beta2DefaultSpecFillsAModelOnlyRequest(t *testing.T) {
	hardware := &fakeHardware{found: DiscoveredHardware{SKU: "h200_sxm", GPUs: 64}}
	spec := modelOnly()
	newV1Beta2Defaulter(hardware).DefaultSpec(context.Background(), spec)

	if spec.Objective.Metric == nil || *spec.Objective.Metric != "ThroughputPerGPU" {
		t.Fatalf("metric not defaulted: %v", spec.Objective.Metric)
	}
	if spec.AISimulate == nil || spec.AISimulate.Image != "registry.example.com/aisimulate-runner:0.13.0" {
		t.Fatalf("image not defaulted: %+v", spec.AISimulate)
	}
	gpuSpec := spec.Hardware.GPU
	if *gpuSpec.SKU != "h200_sxm" || *gpuSpec.Budget != 32 {
		t.Fatalf("hardware not defaulted (budget capped at 32): %s %d", *gpuSpec.SKU, *gpuSpec.Budget)
	}

	input, err := dgdrinput.Build(spec, dgdrinput.DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	if errs := dgdrinput.Validate(input); len(errs) > 0 {
		t.Fatalf("a defaulted model-only Request must validate: %v", errs)
	}
}

func TestV1Beta2DefaultSpecKeepsExplicitValues(t *testing.T) {
	hardware := &fakeHardware{found: DiscoveredHardware{SKU: "h100_sxm", GPUs: 8}}
	spec := v1beta2WithHardware(ptrTo("b200_sxm"), ptrTo(int32(4)))
	spec.Objective.Metric = ptrTo(nvidiacomv1beta2.ObjectiveMetric("TTFT"))
	spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Image: "my/image:1.0.0"}

	newV1Beta2Defaulter(hardware).DefaultSpec(context.Background(), spec)
	if hardware.calls != 0 {
		t.Fatal("discovered hardware although it was fully specified")
	}
	if *spec.Hardware.GPU.SKU != "b200_sxm" || *spec.Hardware.GPU.Budget != 4 ||
		*spec.Objective.Metric != "TTFT" || spec.AISimulate.Image != "my/image:1.0.0" {
		t.Fatalf("explicit values were overwritten: %+v", spec)
	}
}

func TestV1Beta2DefaultSpecResolvesHardwareField(t *testing.T) {
	const cluster = "h100_sxm"
	cases := []struct {
		name       string
		spec       *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
		found      DiscoveredHardware
		wantSKU    string
		wantBudget int32
		wantCalls  int
		wantFilter string
	}{
		{"nothing set", modelOnly(), DiscoveredHardware{SKU: cluster, GPUs: 16}, cluster, 16, 1, ""},
		{"sku set, budget missing", v1beta2WithHardware(ptrTo("h200_sxm"), nil), DiscoveredHardware{SKU: cluster, GPUs: 16}, "h200_sxm", 16, 1, "h200_sxm"},
		{"budget set, sku missing", v1beta2WithHardware(nil, ptrTo(int32(4))), DiscoveredHardware{SKU: cluster, GPUs: 16}, cluster, 4, 1, ""},
		{"budget is capped", modelOnly(), DiscoveredHardware{SKU: cluster, GPUs: 64}, cluster, 32, 1, ""},
		{"unknown model leaves the sku", modelOnly(), DiscoveredHardware{GPUs: 8}, "", 8, 1, ""},
		{"no gpus leaves the budget", v1beta2WithHardware(ptrTo(cluster), nil), DiscoveredHardware{SKU: cluster}, cluster, 0, 1, cluster},
		{"empty sku counts as missing", v1beta2WithHardware(ptrTo(""), ptrTo(int32(2))), DiscoveredHardware{SKU: cluster, GPUs: 16}, cluster, 2, 1, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			hardware := &fakeHardware{found: tc.found}
			newV1Beta2Defaulter(hardware).DefaultSpec(context.Background(), tc.spec)

			var gotSKU string
			var gotBudget int32
			if tc.spec.Hardware != nil && tc.spec.Hardware.GPU != nil {
				if tc.spec.Hardware.GPU.SKU != nil {
					gotSKU = *tc.spec.Hardware.GPU.SKU
				}
				if tc.spec.Hardware.GPU.Budget != nil {
					gotBudget = *tc.spec.Hardware.GPU.Budget
				}
			}
			if gotSKU != tc.wantSKU || gotBudget != tc.wantBudget {
				t.Fatalf("got %q %d want %q %d", gotSKU, gotBudget, tc.wantSKU, tc.wantBudget)
			}
			if hardware.calls != tc.wantCalls || hardware.filters[0] != tc.wantFilter {
				t.Fatalf("discovery calls %d filters %v, want %d filtered by %q", hardware.calls, hardware.filters, tc.wantCalls, tc.wantFilter)
			}
		})
	}
}

func TestV1Beta2DefaultSpecHonorsNativeHardware(t *testing.T) {
	spec := modelOnly()
	spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{
		Raw: []byte(`{"search_space":{"hardware_sku":"b200_sxm"}}`),
	}}
	hardware := &fakeHardware{found: DiscoveredHardware{SKU: "h100_sxm", GPUs: 8}}
	newV1Beta2Defaulter(hardware).DefaultSpec(context.Background(), spec)

	if hardware.filters[0] != "b200_sxm" {
		t.Fatalf("discovery should filter by the native SKU, got %q", hardware.filters[0])
	}
	if spec.Hardware.GPU.SKU != nil || *spec.Hardware.GPU.Budget != 8 {
		t.Fatalf("the native SKU must be kept and only the budget filled: %+v", spec.Hardware.GPU)
	}
}

func TestV1Beta2DefaultSpecMaxBudgetOption(t *testing.T) {
	spec := modelOnly()
	defaulter := NewDGDRV1Beta2Defaulter(DGDRV1Beta2DefaulterOptions{
		Hardware:  &fakeHardware{found: DiscoveredHardware{SKU: "x", GPUs: 20}},
		MaxBudget: 8,
	})
	defaulter.DefaultSpec(context.Background(), spec)
	if *spec.Hardware.GPU.Budget != 8 {
		t.Fatalf("got %d", *spec.Hardware.GPU.Budget)
	}
}

func TestV1Beta2DefaultSpecLeavesTheTargetToTheNativeInput(t *testing.T) {
	spec := modelOnly()
	spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{
		Parameters: &runtime.RawExtension{Raw: []byte(`{"goal":{"target":"ttft"}}`)},
	}
	newV1Beta2Defaulter(nil).DefaultSpec(context.Background(), spec)
	if spec.Objective.Metric != nil {
		t.Fatalf("defaulted a metric next to goal.target: %v", *spec.Objective.Metric)
	}
	if spec.AISimulate.Image == "" {
		t.Fatal("the image should still be defaulted next to native parameters")
	}
}

func TestV1Beta2DefaultSpecDiscoveryFailureLeavesHardwareUnresolved(t *testing.T) {
	spec := modelOnly()
	newV1Beta2Defaulter(&fakeHardware{err: errors.New("dcgm unreachable")}).DefaultSpec(context.Background(), spec)
	if spec.Hardware != nil {
		t.Fatalf("hardware set despite the failure: %+v", spec.Hardware)
	}
	if spec.Objective.Metric == nil {
		t.Fatal("the metric default must not depend on discovery")
	}
	if errs := dgdrinput.ValidateSpec(spec, dgdrinput.DefaultOptions()); len(errs) == 0 {
		t.Fatal("validation must report the unresolved hardware")
	}
}

func TestV1Beta2DefaultSpecMalformedParametersAreLeftToValidation(t *testing.T) {
	spec := modelOnly()
	spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{Parameters: &runtime.RawExtension{Raw: []byte(`[1]`)}}
	hardware := &fakeHardware{}
	newV1Beta2Defaulter(hardware).DefaultSpec(context.Background(), spec)
	if hardware.calls != 0 || spec.Hardware != nil {
		t.Fatal("hardware was resolved from malformed parameters")
	}
}

func TestV1Beta2DefaultSpecWithoutDiscovery(t *testing.T) {
	spec := modelOnly()
	newV1Beta2Defaulter(nil).DefaultSpec(context.Background(), spec)
	if spec.Hardware != nil {
		t.Fatal("hardware set without a discoverer")
	}
}

func TestV1Beta2DefaultSpecWithoutConfiguredImage(t *testing.T) {
	spec := modelOnly()
	NewDGDRV1Beta2Defaulter(DGDRV1Beta2DefaulterOptions{}).DefaultSpec(context.Background(), spec)
	if spec.AISimulate != nil {
		t.Fatalf("created aisimulate without an image to put in it: %+v", spec.AISimulate)
	}
}
