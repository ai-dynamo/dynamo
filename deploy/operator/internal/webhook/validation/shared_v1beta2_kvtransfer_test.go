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

package validation

import (
	"strings"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateNativeSearchInputKVTransfer(t *testing.T) {
	const aggregatedOnly = `{"engine":{"mode":"aggregated"}}`
	cases := []struct {
		name      string
		bandwidth *float64
		overrides string
		wantErr   string
		wantWarn  bool
	}{
		{name: "default modes without a bandwidth warn", wantWarn: true},
		{name: "default modes with a bandwidth are accepted", bandwidth: new(27.7)},
		{
			name:      "an empty override transfer model still warns",
			overrides: `{"engine":{"kv_transfer":{}}}`,
			wantWarn:  true,
		},
		{name: "aggregated-only search without a transfer model is accepted", overrides: aggregatedOnly},
		{
			name:      "aggregated-only search rejects the typed bandwidth",
			bandwidth: new(27.7),
			overrides: aggregatedOnly,
			wantErr:   "spec.hardware.interconnect.kvTransferBandwidthGBps",
		},
		{
			name:      "aggregated-only search rejects an override transfer model",
			overrides: `{"engine":{"mode":"aggregated","kv_transfer":{"bandwidth_gb_per_second":10}}}`,
			wantErr:   "spec.aisim.inputOverrides.engine.kv_transfer",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Build a defaulted Request search input with the case's bandwidth and overrides")
			spec := &nvidiacomv1beta2.NativeSearchInput{
				Model:    &nvidiacomv1beta2.ModelSpec{Name: "Qwen/Qwen3-32B"},
				Backends: []nvidiacomv1beta2.Backend{nvidiacomv1beta2.BackendVLLM},
				Hardware: &nvidiacomv1beta2.HardwareSpec{
					GPU: &nvidiacomv1beta2.GPUHardwareSpec{SKU: new("h100_sxm"), Budget: new(int32(8))},
				},
				Workload: &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
					InputTokens:  new(int32(4096)),
					OutputTokens: new(int32(256)),
					Load:         &nvidiacomv1beta2.StaticLoadSpec{Concurrency: new(int32(16))},
				}},
			}
			if tc.bandwidth != nil {
				spec.Hardware.Interconnect = &nvidiacomv1beta2.InterconnectSpec{KVTransferBandwidthGBps: tc.bandwidth}
			}
			if tc.overrides != "" {
				spec.AISim = &nvidiacomv1beta2.AISimSpec{InputOverrides: &runtime.RawExtension{Raw: []byte(tc.overrides)}}
			}

			t.Log("Validate the effective input, as admission does on create")
			v := &sharedValidation{}
			errs := v.validateNativeSearchInput(spec, field.NewPath("spec"), true)

			t.Log("Check the rejection names the field that introduced the transfer model")
			if tc.wantErr == "" && len(errs) != 0 {
				t.Fatalf("validateNativeSearchInput() errors = %v, want none", errs)
			}
			if tc.wantErr != "" && (len(errs) != 1 || errs[0].Field != tc.wantErr) {
				t.Fatalf("validateNativeSearchInput() errors = %v, want one at %s", errs, tc.wantErr)
			}

			t.Log("Check the warning about an instantaneous transfer")
			warned := len(v.warnings) == 1 && strings.Contains(v.warnings[0], "instantaneous prefill-to-decode KV transfer")
			if warned != tc.wantWarn || (!tc.wantWarn && len(v.warnings) != 0) {
				t.Fatalf("warnings = %v, want KV-transfer warning %v", v.warnings, tc.wantWarn)
			}
		})
	}
}
