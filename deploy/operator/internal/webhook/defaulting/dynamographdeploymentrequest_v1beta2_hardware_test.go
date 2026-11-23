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
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	admissionv1 "k8s.io/api/admission/v1"
)

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

func hardwareOfV1beta2(sku *string, budget *int32, numGPUsPerNode ...*int32) *nvidiacomv1beta2.HardwareSpec {
	hardware := &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: sku, Budget: budget}}
	if len(numGPUsPerNode) > 0 {
		hardware.NumGPUsPerNode = numGPUsPerNode[0]
	}
	return hardware
}

func hardwareSpecV1beta2(hardware *nvidiacomv1beta2.HardwareSpec) *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec {
	return &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{Hardware: hardware}
}

func TestResolveHardwareV1beta2(t *testing.T) {
	str := func(s string) *string { return &s }
	num := func(n int32) *int32 { return &n }
	h200x8 := gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1, TotalGPUs: 8, SelectedShapeGPUs: 8}
	h200x64 := gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 8, TotalGPUs: 64, SelectedShapeGPUs: 64}
	completeH100 := hardwareOfV1beta2(str("h100_sxm"), num(4), num(8))
	type testCase struct {
		name              string
		op                admissionv1.Operation
		oldHardware       *nvidiacomv1beta2.HardwareSpec
		hardware          *nvidiacomv1beta2.HardwareSpec
		found             gpu.GPUInfo
		discoveryErr      error
		withoutDiscoverer bool
		maxBudget         int32
		timeout           time.Duration
		wantSKU           string
		wantBudget        int32
		wantGPUsPerNode   int32
		wantCalls         int
		wantFilter        string
		wantErr           string
		wantMinTimeout    time.Duration
		wantMaxTimeout    time.Duration
	}
	tests := []testCase{
		{name: "no discoverer leaves omitted hardware unresolved", op: admissionv1.Create, withoutDiscoverer: true},
		{name: "omitted hardware discovers the SKU budget and node width", op: admissionv1.Create, found: h200x8, wantSKU: "h200_sxm", wantBudget: 8, wantGPUsPerNode: 8, wantCalls: 1, wantMinTimeout: time.Millisecond, wantMaxTimeout: defaultHardwareDiscoveryTimeout},
		{name: "an empty hardware object opts out of discovery", op: admissionv1.Create, hardware: &nvidiacomv1beta2.HardwareSpec{}, discoveryErr: errors.New("discovery must not run")},
		{
			name: "same-SKU nodes with different GPU counts use the selected node shape", op: admissionv1.Create,
			found:   gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 2, TotalGPUs: 12, SelectedShapeGPUs: 8},
			wantSKU: "h200_sxm", wantBudget: 8, wantGPUsPerNode: 8, wantCalls: 1,
		},
		{
			name: "an empty gpu object is automatic discovery", op: admissionv1.Create,
			hardware: &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{}}, found: h200x8,
			wantSKU: "h200_sxm", wantBudget: 8, wantGPUsPerNode: 8, wantCalls: 1,
		},
		{name: "the discovered budget is capped", op: admissionv1.Create, found: h200x64, wantSKU: "h200_sxm", wantBudget: 32, wantGPUsPerNode: 8, wantCalls: 1},
		{name: "the discovered budget honors a configured cap", op: admissionv1.Create, found: h200x64, maxBudget: 8, wantSKU: "h200_sxm", wantBudget: 8, wantGPUsPerNode: 8, wantCalls: 1},
		{
			name: "a SKU alone keeps the SKU, filters the cluster and persists the capped budget", op: admissionv1.Create,
			hardware: hardwareOfV1beta2(str("h200_sxm"), nil), found: h200x64,
			wantBudget: 32, wantGPUsPerNode: 8, wantCalls: 1, wantFilter: "h200_sxm",
		},
		{
			name: "a failed discovery for a SKU alone asks for capacity and topology", op: admissionv1.Create,
			hardware: hardwareOfV1beta2(str("h200_sxm"), nil), discoveryErr: errors.New("cluster unreadable"),
			wantCalls: 1, wantFilter: "h200_sxm",
			wantErr: `spec.hardware.gpu.budget: Required value: could not be discovered for SKU "h200_sxm" (cluster unreadable); set spec.hardware.gpu.budget and spec.hardware.numGpusPerNode`,
		},
		{name: "a budget alone keeps the budget and discovers the SKU", op: admissionv1.Create, hardware: hardwareOfV1beta2(nil, num(4)), found: h200x8, wantSKU: "h200_sxm", wantGPUsPerNode: 8, wantCalls: 1},
		{name: "an explicit budget above the cluster size is kept", op: admissionv1.Create, hardware: hardwareOfV1beta2(nil, num(64)), found: h200x8, wantSKU: "h200_sxm", wantGPUsPerNode: 8, wantCalls: 1},
		{name: "an explicit SKU and budget discover the missing node width", op: admissionv1.Create, hardware: hardwareOfV1beta2(str("h200_sxm"), num(64)), found: h200x8, wantGPUsPerNode: 8, wantCalls: 1, wantFilter: "h200_sxm"},
		{
			name: "a failed node-width discovery asks for the missing field", op: admissionv1.Create,
			hardware: hardwareOfV1beta2(str("h200_sxm"), num(64)), discoveryErr: errors.New("cluster unreadable"),
			wantCalls: 1, wantFilter: "h200_sxm",
			wantErr: `spec.hardware.numGpusPerNode: Required value: could not be discovered for SKU "h200_sxm" (cluster unreadable); set spec.hardware.numGpusPerNode`,
		},
		{
			name: "a discovered shape without a node width asks for the missing field", op: admissionv1.Create,
			hardware: hardwareOfV1beta2(str("h200_sxm"), num(64)),
			found:    gpu.GPUInfo{System: "h200_sxm", SelectedShapeGPUs: 8}, wantCalls: 1, wantFilter: "h200_sxm",
			wantErr: `spec.hardware.numGpusPerNode: Required value: could not be discovered for SKU "h200_sxm"; set spec.hardware.numGpusPerNode`,
		},
		{name: "fully explicit hardware bypasses discovery", op: admissionv1.Create, hardware: hardwareOfV1beta2(str("h200_sxm"), num(64), num(4)), discoveryErr: errors.New("down")},
		{
			name: "an explicit node width is authoritative while a missing budget triggers discovery", op: admissionv1.Create,
			hardware: hardwareOfV1beta2(str("h200_sxm"), nil, num(4)), found: h200x8,
			wantBudget: 8, wantCalls: 1, wantFilter: "h200_sxm",
		},
		{
			name: "an unknown explicit SKU and budget require an explicit node width", op: admissionv1.Create,
			hardware: hardwareOfV1beta2(str("future_accelerator"), num(4)),
			wantErr:  `spec.hardware.numGpusPerNode: Required value: cannot discover a node width for SKU "future_accelerator" because it is not known to the bundled AISimulate catalog; set spec.hardware.numGpusPerNode explicitly`,
		},
		{
			name: "fully explicit unknown hardware is authoritative", op: admissionv1.Create,
			hardware: hardwareOfV1beta2(str("future_accelerator"), num(4), num(2)), discoveryErr: errors.New("down"),
		},
		{
			name: "an unknown SKU alone requires explicit capacity and topology", op: admissionv1.Create,
			hardware: hardwareOfV1beta2(str("future_accelerator"), nil), found: h200x8,
			wantErr: `spec.hardware.gpu.budget: Required value: cannot discover GPU capacity or topology for SKU "future_accelerator" because it is not known to the bundled AISimulate catalog; set spec.hardware.gpu.budget and spec.hardware.numGpusPerNode explicitly`,
		},
		{name: "a failed discovery rejects", op: admissionv1.Create, discoveryErr: errors.New("no GPUs found"), wantCalls: 1,
			wantErr: "spec.hardware.gpu: Required value: the cluster's GPUs could not be discovered (no GPUs found)"},
		{
			name: "a GPU model the catalog does not know rejects", op: admissionv1.Create,
			found: gpu.GPUInfo{Model: "Mystery GPU", GPUsPerNode: 8, NodesWithGPUs: 1}, wantCalls: 1,
			wantErr: `spec.hardware.gpu.sku: Required value: the cluster's GPU model "Mystery GPU" maps to no AISimulate SKU`,
		},
		{
			name: "a legacy discovery mapping outside the bundled catalog rejects", op: admissionv1.Create,
			found: gpu.GPUInfo{System: "gb10", Model: "GB10", GPUsPerNode: 1, NodesWithGPUs: 1}, wantCalls: 1,
			wantErr: `spec.hardware.gpu.sku: Required value: the cluster's GPU model "GB10" maps to no AISimulate SKU`,
		},
		{
			name: "no GPUs of a known SKU require an explicit budget", op: admissionv1.Create,
			hardware: hardwareOfV1beta2(str("h100_sxm"), nil), found: gpu.GPUInfo{System: "h100_sxm"},
			wantCalls: 1, wantFilter: "h100_sxm",
			wantErr: `spec.hardware.gpu.budget: Required value: no GPUs of SKU "h100_sxm" were found in the cluster`,
		},
		{
			name: "a heterogeneous cluster rejects an omitted SKU", op: admissionv1.Create,
			found:     gpu.GPUInfo{System: "h200_sxm", Model: "H200", GPUsPerNode: 8, NodesWithGPUs: 1, OtherModels: []string{"H100"}},
			wantCalls: 1,
			wantErr:   "spec.hardware.gpu.sku: Required value: the cluster has several GPU types (H200, H100); set spec.hardware.gpu.sku",
		},
		{
			name: "a heterogeneous cluster rejects a budget alone too", op: admissionv1.Create,
			hardware:  hardwareOfV1beta2(nil, num(4)),
			found:     gpu.GPUInfo{System: "h200_sxm", Model: "H200", GPUsPerNode: 8, NodesWithGPUs: 1, OtherModels: []string{"H100"}},
			wantCalls: 1,
			wantErr:   "spec.hardware.gpu.sku: Required value: the cluster has several GPU types (H200, H100)",
		},
		{
			name: "a pinned SKU is not ambiguous in a heterogeneous cluster", op: admissionv1.Create,
			hardware: hardwareOfV1beta2(str("h200_sxm"), nil), found: h200x8,
			wantBudget: 8, wantGPUsPerNode: 8, wantCalls: 1, wantFilter: "h200_sxm",
		},
		{
			name: "no GPUs at all reject the missing budget", op: admissionv1.Create,
			found: gpu.GPUInfo{System: "h200_sxm"}, wantCalls: 1,
			wantErr: `spec.hardware.gpu.budget: Required value: no GPUs of SKU "h200_sxm" were found in the cluster`,
		},
		{name: "a configured timeout bounds discovery", op: admissionv1.Create, found: h200x8, timeout: time.Hour, wantSKU: "h200_sxm", wantBudget: 8, wantGPUsPerNode: 8, wantCalls: 1, wantMinTimeout: time.Minute, wantMaxTimeout: time.Hour},
		{name: "an unchanged incomplete update does not discover", op: admissionv1.Update, oldHardware: hardwareOfV1beta2(str("h100_sxm"), nil), hardware: hardwareOfV1beta2(str("h100_sxm"), nil), discoveryErr: errors.New("down")},
		{name: "an update without a stored spec does not discover", op: admissionv1.Update, found: h200x8},
		{name: "removing only the budget discovers the budget", op: admissionv1.Update, oldHardware: completeH100, hardware: hardwareOfV1beta2(str("h100_sxm"), nil, num(8)), found: h200x8, wantBudget: 8, wantCalls: 1, wantFilter: "h100_sxm"},
		{name: "removing only the SKU discovers the SKU", op: admissionv1.Update, oldHardware: completeH100, hardware: hardwareOfV1beta2(nil, num(4), num(8)), found: h200x8, wantSKU: "h200_sxm", wantCalls: 1},
		{name: "removing all hardware discovers all fields", op: admissionv1.Update, oldHardware: completeH100, found: h200x8, wantSKU: "h200_sxm", wantBudget: 8, wantGPUsPerNode: 8, wantCalls: 1},
		{name: "replacing hardware with an empty object opts out", op: admissionv1.Update, oldHardware: completeH100, hardware: &nvidiacomv1beta2.HardwareSpec{}, discoveryErr: errors.New("discovery must not run")},
		{name: "changing complete hardware keeps the explicit override", op: admissionv1.Update, oldHardware: completeH100, hardware: hardwareOfV1beta2(str("b200_sxm"), num(2), num(8)), discoveryErr: errors.New("down")},
		{name: "delete never discovers", op: admissionv1.Delete, oldHardware: completeH100, found: h200x8},
		{name: "connect never discovers", op: admissionv1.Connect, oldHardware: completeH100, found: h200x8},
		{name: "an unknown operation never discovers", oldHardware: completeH100, found: h200x8},
	}
	if strconv.IntSize > 32 {
		tooManyGPUs := int64(math.MaxInt32) + 1
		tests = append(tests, testCase{
			name:      "a discovered node width must fit the API integer",
			op:        admissionv1.Create,
			found:     gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: int(tooManyGPUs), SelectedShapeGPUs: 8},
			wantCalls: 1,
			wantErr:   "spec.hardware.numGpusPerNode: Invalid value: 2147483648: discovered node width exceeds the API maximum of 2147483647",
		})
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			probe := &fakeDiscoverer{found: tt.found, err: tt.discoveryErr}
			options := DGDRV1beta2Options{
				Hardware:                 probe,
				MaxHardwareBudget:        tt.maxBudget,
				HardwareDiscoveryTimeout: tt.timeout,
			}
			if tt.withoutDiscoverer {
				options.Hardware = nil
			}
			var old *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
			if tt.oldHardware != nil {
				old = hardwareSpecV1beta2(tt.oldHardware)
			}
			got, err := options.resolveHardware(context.Background(), tt.op, old, hardwareSpecV1beta2(tt.hardware))
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want it to contain %q", err, tt.wantErr)
				}
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if err == nil && (got.sku != tt.wantSKU || got.budget != tt.wantBudget || got.numGPUsPerNode != tt.wantGPUsPerNode) {
				t.Fatalf("defaults = %+v, want sku %q budget %d numGpusPerNode %d", got, tt.wantSKU, tt.wantBudget, tt.wantGPUsPerNode)
			}
			if probe.calls != tt.wantCalls {
				t.Fatalf("discovery calls = %d, want %d", probe.calls, tt.wantCalls)
			}
			if tt.wantCalls > 0 && probe.filter != tt.wantFilter {
				t.Fatalf("discovery filter = %q, want %q", probe.filter, tt.wantFilter)
			}
			if tt.wantMinTimeout > 0 && probe.timeout < tt.wantMinTimeout {
				t.Fatalf("discovery timeout = %v, want at least %v", probe.timeout, tt.wantMinTimeout)
			}
			if tt.wantMaxTimeout > 0 && probe.timeout > tt.wantMaxTimeout {
				t.Fatalf("discovery timeout = %v, want at most %v", probe.timeout, tt.wantMaxTimeout)
			}
		})
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
