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

package aisimulate

import (
	"encoding/json"
	"testing"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
)

func TestKVTransferBandwidthLowering(t *testing.T) {
	cases := []struct {
		name      string
		bandwidth *float64
		overrides string
		want      any
	}{
		{name: "typed bandwidth lowers to engine.kv_transfer", bandwidth: new(27.7), want: 27.7},
		{
			name:      "input override replaces the typed bandwidth",
			bandwidth: new(27.7),
			overrides: `{"engine":{"kv_transfer":{"bandwidth_gb_per_second":10}}}`,
			want:      json.Number("10"),
		},
		{
			name:      "input override deletes the typed transfer model",
			bandwidth: new(27.7),
			overrides: `{"engine":{"kv_transfer":null}}`,
			want:      nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Lower a defaulted Request with the case's interconnect and overrides")
			spec := defaultedSpec()
			if tc.bandwidth != nil {
				spec.Hardware.Interconnect = &v1beta2.InterconnectSpec{KVTransferBandwidthGBps: tc.bandwidth}
			}
			if tc.overrides != "" {
				spec = withInputOverrides(spec, tc.overrides)
			}
			input := lower(t, spec)

			t.Log("Only the bandwidth is lowered; AISimulate keeps its bytes-per-token and timing defaults")
			if got := nested(t, input, engineKVTransferBW...); got != tc.want {
				t.Fatalf("engine.kv_transfer.bandwidth_gb_per_second = %v, want %v: %s", got, tc.want, encode(t, input))
			}
			if tc.want != nil && len(nested(t, input, engineKVTransfer...).(map[string]any)) != 1 {
				t.Fatalf("engine.kv_transfer carries more than the bandwidth: %s", encode(t, input))
			}
		})
	}
}
