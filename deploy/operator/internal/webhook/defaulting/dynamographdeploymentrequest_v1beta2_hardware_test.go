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
	"math"
	"strconv"
	"strings"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	admissionv1 "k8s.io/api/admission/v1"
)

func hardwareOfV1beta2(sku *string, budget *int32, numGPUsPerNode ...*int32) *nvidiacomv1beta2.HardwareSpec {
	hardware := &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: sku, Budget: budget}}
	if len(numGPUsPerNode) > 0 {
		hardware.NumGPUsPerNode = numGPUsPerNode[0]
	}
	return hardware
}

func TestResolveHardwareCompletesTheTypedHardwareOrRejects(t *testing.T) {
	str := func(s string) *string { return &s }
	num := func(n int32) *int32 { return &n }
	h200x8 := gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1, TotalGPUs: 8, SelectedShapeGPUs: 8}
	h200x64 := gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 8, TotalGPUs: 64, SelectedShapeGPUs: 64}
	tests := []struct {
		name       string
		hardware   *nvidiacomv1beta2.HardwareSpec
		found      gpu.GPUInfo
		err        error
		wantSKU    string
		wantBudget int32
		wantWidth  int32
		wantCalls  int
		wantFilter string
		wantErr    string
	}{
		{"omitted hardware discovers the SKU budget and node width", nil, h200x8, nil, "h200_sxm", 8, 8, 1, "", ""},
		{"an empty hardware object opts out of discovery", &nvidiacomv1beta2.HardwareSpec{}, gpu.GPUInfo{}, errors.New("discovery must not run"), "", 0, 0, 0, "", ""},
		{"same-SKU nodes with different GPU counts use the selected node shape", nil,
			gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 2, TotalGPUs: 12, SelectedShapeGPUs: 8},
			nil, "h200_sxm", 8, 8, 1, "", ""},
		{"an empty gpu object is automatic discovery",
			&nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{}}, h200x8, nil, "h200_sxm", 8, 8, 1, "", ""},
		{"the discovered budget is capped", nil, h200x64, nil, "h200_sxm", 32, 8, 1, "", ""},
		{"a SKU alone keeps the SKU, filters the cluster and persists the capped budget",
			hardwareOfV1beta2(str("h200_sxm"), nil), h200x64, nil, "", 32, 8, 1, "h200_sxm", ""},
		{"a budget alone keeps the budget and discovers the SKU", hardwareOfV1beta2(nil, num(4)), h200x8, nil, "h200_sxm", 0, 8, 1, "", ""},
		{"an explicit budget above the cluster size is kept", hardwareOfV1beta2(nil, num(64)), h200x8, nil, "h200_sxm", 0, 8, 1, "", ""},
		{"an explicit SKU and budget discover the missing node width", hardwareOfV1beta2(str("h200_sxm"), num(64)), h200x8, nil, "", 0, 8, 1, "h200_sxm", ""},
		{"fully explicit hardware bypasses discovery", hardwareOfV1beta2(str("h100_sxm"), num(4), num(8)), gpu.GPUInfo{}, errors.New("down"), "", 0, 0, 0, "", ""},
		{"fully explicit unknown hardware is authoritative", hardwareOfV1beta2(str("future_accelerator"), num(4), num(2)), gpu.GPUInfo{}, errors.New("down"), "", 0, 0, 0, "", ""},
		{"an unknown SKU and budget require an explicit node width", hardwareOfV1beta2(str("future_accelerator"), num(4)), h200x8, nil, "", 0, 0, 0, "",
			`spec.hardware.numGpusPerNode: Required value: cannot discover a node width for SKU "future_accelerator" because it is not known to the bundled AISimulate catalog; set spec.hardware.numGpusPerNode explicitly`},
		{"an unknown SKU alone requires explicit capacity and topology", hardwareOfV1beta2(str("future_accelerator"), nil), h200x8, nil, "", 0, 0, 0, "",
			`spec.hardware.gpu.budget: Required value: cannot discover GPU capacity or topology for SKU "future_accelerator" because it is not known to the bundled AISimulate catalog; set spec.hardware.gpu.budget and spec.hardware.numGpusPerNode explicitly`},
		{"a failed discovery rejects", nil, gpu.GPUInfo{}, errors.New("no GPUs found"), "", 0, 0, 1, "",
			"spec.hardware.gpu: Required value: the cluster's GPUs could not be discovered (no GPUs found)"},
		{"a GPU model the catalog does not know rejects", nil,
			gpu.GPUInfo{Model: "Mystery GPU", GPUsPerNode: 8, NodesWithGPUs: 1}, nil, "", 0, 0, 1, "",
			`spec.hardware.gpu.sku: Required value: the cluster's GPU model "Mystery GPU" maps to no AISimulate SKU`},
		{"a legacy mapping outside the bundled catalog rejects", nil,
			gpu.GPUInfo{System: "gb10", Model: "GB10", GPUsPerNode: 1, NodesWithGPUs: 1, TotalGPUs: 1, SelectedShapeGPUs: 1}, nil, "", 0, 0, 1, "",
			`spec.hardware.gpu.sku: Required value: the cluster's GPU model "GB10" maps to no AISimulate SKU`},
		{"no GPUs of the SKU reject the missing budget", hardwareOfV1beta2(str("h100_sxm"), nil),
			gpu.GPUInfo{System: "h100_sxm"}, nil, "", 0, 0, 1, "h100_sxm",
			`spec.hardware.gpu.budget: Required value: no GPUs of SKU "h100_sxm" were found in the cluster`},
		{"a heterogeneous cluster rejects an omitted SKU", nil,
			gpu.GPUInfo{System: "h200_sxm", Model: "H200", GPUsPerNode: 8, NodesWithGPUs: 1, OtherModels: []string{"H100"}},
			nil, "", 0, 0, 1, "",
			"spec.hardware.gpu.sku: Required value: the cluster has several GPU types (H200, H100); set spec.hardware.gpu.sku"},
		{"a heterogeneous cluster rejects a budget alone too", hardwareOfV1beta2(nil, num(4)),
			gpu.GPUInfo{System: "h200_sxm", Model: "H200", GPUsPerNode: 8, NodesWithGPUs: 1, OtherModels: []string{"H100"}},
			nil, "", 0, 0, 1, "",
			"spec.hardware.gpu.sku: Required value: the cluster has several GPU types (H200, H100)"},
		{"a pinned SKU is not ambiguous in a heterogeneous cluster", hardwareOfV1beta2(str("h200_sxm"), nil),
			gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1, TotalGPUs: 8, SelectedShapeGPUs: 8}, nil, "", 8, 8, 1, "h200_sxm", ""},
		{"no GPUs at all reject the missing budget", nil, gpu.GPUInfo{System: "h200_sxm"}, nil, "", 0, 0, 1, "",
			`spec.hardware.gpu.budget: Required value: no GPUs of SKU "h200_sxm" were found in the cluster`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			probe := &fakeDiscoverer{found: tt.found, err: tt.err}
			spec := &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{}
			spec.Hardware = tt.hardware
			got, err := DGDRV1beta2Options{Hardware: probe}.resolveHardware(context.Background(), admissionv1.Create, nil, spec)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want it to contain %q", err, tt.wantErr)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if err == nil && (got.sku != tt.wantSKU || got.budget != tt.wantBudget || got.numGPUsPerNode != tt.wantWidth) {
				t.Fatalf("defaults = %+v, want sku %q budget %d numGpusPerNode %d", got, tt.wantSKU, tt.wantBudget, tt.wantWidth)
			}
			if probe.calls != tt.wantCalls {
				t.Fatalf("discovery calls = %d, want %d", probe.calls, tt.wantCalls)
			}
			if tt.wantCalls > 0 && probe.filter != tt.wantFilter {
				t.Fatalf("discovery filter = %q, want %q", probe.filter, tt.wantFilter)
			}
		})
	}
}

func TestResolveHardwareRejectsDiscoveredNodeWidthBeyondAPI(t *testing.T) {
	if strconv.IntSize <= 32 {
		t.Skip("the host int cannot represent a value above the API int32 maximum")
	}
	tooManyGPUs := int64(math.MaxInt32) + 1
	probe := &fakeDiscoverer{found: gpu.GPUInfo{
		System:            "h200_sxm",
		GPUsPerNode:       int(tooManyGPUs),
		SelectedShapeGPUs: 8,
	}}
	spec := &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{}
	_, err := DGDRV1beta2Options{Hardware: probe}.resolveHardware(
		context.Background(), admissionv1.Create, nil, spec,
	)
	want := "spec.hardware.numGpusPerNode: Invalid value: 2147483648: discovered node width exceeds the API maximum of 2147483647"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Fatalf("error = %v, want it to contain %q", err, want)
	}
	if probe.calls != 1 {
		t.Fatalf("discovery calls = %d, want 1", probe.calls)
	}
}

func TestResolveHardwareLeavesAnUnchangedUpdateAlone(t *testing.T) {
	probe := &fakeDiscoverer{err: errors.New("down")}
	spec := &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{}
	got, err := DGDRV1beta2Options{Hardware: probe}.resolveHardware(context.Background(), admissionv1.Update, spec.DeepCopy(), spec)
	if err != nil || got.sku != "" || got.budget != 0 || got.numGPUsPerNode != 0 || probe.calls != 0 {
		t.Fatalf("got %+v, %v after %d calls; an update that leaves spec.hardware unchanged must not read the cluster",
			got, err, probe.calls)
	}
}

func TestDefaultHardwareSpecV1beta2(t *testing.T) {
	t.Run("preserves the explicit empty-object opt-out", func(t *testing.T) {
		hardware := &nvidiacomv1beta2.HardwareSpec{}
		defaultHardwareSpecV1beta2(hardware, hardwareDefaultsV1beta2{})
		if hardware.GPU != nil || hardware.NumGPUsPerNode != nil {
			t.Fatalf("hardware = %#v, want the empty-object opt-out unchanged", hardware)
		}
	})

	t.Run("persists every discovered field", func(t *testing.T) {
		hardware := &nvidiacomv1beta2.HardwareSpec{}
		defaultHardwareSpecV1beta2(hardware, hardwareDefaultsV1beta2{
			sku: "h200_sxm", budget: 8, numGPUsPerNode: 8,
		})
		if hardware.GPU == nil || hardware.GPU.SKU == nil || *hardware.GPU.SKU != "h200_sxm" ||
			hardware.GPU.Budget == nil || *hardware.GPU.Budget != 8 ||
			hardware.NumGPUsPerNode == nil || *hardware.NumGPUsPerNode != 8 {
			t.Fatalf("hardware = %#v, want every discovered field", hardware)
		}
	})

	t.Run("preserves an explicit node width", func(t *testing.T) {
		hardware := &nvidiacomv1beta2.HardwareSpec{NumGPUsPerNode: new(int32(4))}
		defaultHardwareSpecV1beta2(hardware, hardwareDefaultsV1beta2{
			sku: "h200_sxm", budget: 8, numGPUsPerNode: 8,
		})
		if hardware.NumGPUsPerNode == nil || *hardware.NumGPUsPerNode != 4 {
			t.Fatalf("numGpusPerNode = %v, want explicit value 4", hardware.NumGPUsPerNode)
		}
	})
}
