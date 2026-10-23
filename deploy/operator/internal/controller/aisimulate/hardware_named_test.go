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

import "testing"

func TestHardwareNamed(t *testing.T) {
	tests := []struct {
		name  string
		input map[string]any
		want  bool
	}{
		{"both named", map[string]any{"search_space": map[string]any{"hardware_sku": "h200_sxm", "gpu_budget": int64(8)}}, true},
		{"no search space", map[string]any{}, false},
		{"budget missing", map[string]any{"search_space": map[string]any{"hardware_sku": "h200_sxm"}}, false},
		{"sku missing", map[string]any{"search_space": map[string]any{"gpu_budget": int64(8)}}, false},
		{"null sku", map[string]any{"search_space": map[string]any{"hardware_sku": nil, "gpu_budget": int64(8)}}, false},
		{"blank sku", map[string]any{"search_space": map[string]any{"hardware_sku": "", "gpu_budget": int64(8)}}, false},
		{"null budget", map[string]any{"search_space": map[string]any{"hardware_sku": "h200_sxm", "gpu_budget": nil}}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := HardwareNamed(tt.input); got != tt.want {
				t.Fatalf("HardwareNamed() = %v, want %v", got, tt.want)
			}
		})
	}
}
