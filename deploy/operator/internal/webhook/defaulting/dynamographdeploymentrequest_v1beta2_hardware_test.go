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
	"strings"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	admissionv1 "k8s.io/api/admission/v1"
)

func hardwareOfV1beta2(sku *string, budget *int32) *nvidiacomv1beta2.HardwareSpec {
	return &nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: sku, Budget: budget}}
}

func TestResolveHardwareCompletesTheTypedHardwareOrRejects(t *testing.T) {
	str := func(s string) *string { return &s }
	num := func(n int32) *int32 { return &n }
	h200x8 := gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1}
	h200x64 := gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 8}
	tests := []struct {
		name       string
		hardware   *nvidiacomv1beta2.HardwareSpec
		found      gpu.GPUInfo
		err        error
		wantSKU    string
		wantBudget int32
		wantCalls  int
		wantFilter string
		wantErr    string
	}{
		{"omitted hardware discovers the SKU and the budget", nil, h200x8, nil, "h200_sxm", 8, 1, "", ""},
		{"an empty hardware object is automatic discovery", &nvidiacomv1beta2.HardwareSpec{}, h200x8, nil, "h200_sxm", 8, 1, "", ""},
		{"an empty gpu object is automatic discovery",
			&nvidiacomv1beta2.HardwareSpec{GPU: &nvidiacomv1beta2.GPUHardwareSpec{}}, h200x8, nil, "h200_sxm", 8, 1, "", ""},
		{"the discovered budget is capped", nil, h200x64, nil, "h200_sxm", 32, 1, "", ""},
		{"a SKU alone keeps the SKU, filters the cluster and persists the capped budget",
			hardwareOfV1beta2(str("h200_sxm"), nil), h200x64, nil, "", 32, 1, "h200_sxm", ""},
		{"a budget alone keeps the budget and discovers the SKU", hardwareOfV1beta2(nil, num(4)), h200x8, nil, "h200_sxm", 0, 1, "", ""},
		{"an explicit budget above the cluster size is kept", hardwareOfV1beta2(nil, num(64)), h200x8, nil, "h200_sxm", 0, 1, "", ""},
		{"an explicit budget above the cap is kept", hardwareOfV1beta2(str("h200_sxm"), num(64)), h200x8, nil, "", 0, 0, "", ""},
		{"a SKU and a budget never reach the cluster", hardwareOfV1beta2(str("h100_sxm"), num(4)), gpu.GPUInfo{}, errors.New("down"), "", 0, 0, "", ""},
		{"a failed discovery rejects", nil, gpu.GPUInfo{}, errors.New("no GPUs found"), "", 0, 1, "",
			"spec.hardware.gpu: Required value: the cluster's GPUs could not be discovered (no GPUs found)"},
		{"a GPU model the catalog does not know rejects", nil,
			gpu.GPUInfo{Model: "Mystery GPU", GPUsPerNode: 8, NodesWithGPUs: 1}, nil, "", 0, 1, "",
			`spec.hardware.gpu.sku: Required value: the cluster's GPU model "Mystery GPU" maps to no AISimulate SKU`},
		{"no GPUs of the SKU reject the missing budget", hardwareOfV1beta2(str("h100_sxm"), nil),
			gpu.GPUInfo{System: "h100_sxm"}, nil, "", 0, 1, "h100_sxm",
			`spec.hardware.gpu.budget: Required value: no GPUs of SKU "h100_sxm" were found in the cluster`},
		{"a heterogeneous cluster rejects an omitted SKU", nil,
			gpu.GPUInfo{System: "h200_sxm", Model: "H200", GPUsPerNode: 8, NodesWithGPUs: 1, OtherModels: []string{"H100"}},
			nil, "", 0, 1, "",
			"spec.hardware.gpu.sku: Required value: the cluster has several GPU types (H200, H100); set spec.hardware.gpu.sku"},
		{"a heterogeneous cluster rejects a budget alone too", hardwareOfV1beta2(nil, num(4)),
			gpu.GPUInfo{System: "h200_sxm", Model: "H200", GPUsPerNode: 8, NodesWithGPUs: 1, OtherModels: []string{"H100"}},
			nil, "", 0, 1, "",
			"spec.hardware.gpu.sku: Required value: the cluster has several GPU types (H200, H100)"},
		{"a pinned SKU is not ambiguous in a heterogeneous cluster", hardwareOfV1beta2(str("h200_sxm"), nil),
			gpu.GPUInfo{System: "h200_sxm", GPUsPerNode: 8, NodesWithGPUs: 1}, nil, "", 8, 1, "h200_sxm", ""},
		{"no GPUs at all reject the missing budget", nil, gpu.GPUInfo{System: "h200_sxm"}, nil, "", 0, 1, "",
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
			if err == nil && (got.sku != tt.wantSKU || got.budget != tt.wantBudget) {
				t.Fatalf("defaults = %+v, want sku %q budget %d", got, tt.wantSKU, tt.wantBudget)
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

func TestResolveHardwareLeavesAnUnchangedUpdateAlone(t *testing.T) {
	probe := &fakeDiscoverer{err: errors.New("down")}
	spec := &nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec{}
	got, err := DGDRV1beta2Options{Hardware: probe}.resolveHardware(context.Background(), admissionv1.Update, spec.DeepCopy(), spec)
	if err != nil || got.sku != "" || got.budget != 0 || probe.calls != 0 {
		t.Fatalf("got %+v, %v after %d calls; an update that leaves spec.hardware unchanged must not read the cluster",
			got, err, probe.calls)
	}
}
