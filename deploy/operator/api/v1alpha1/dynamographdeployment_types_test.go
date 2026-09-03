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

package v1alpha1

import (
	"strings"
	"testing"
)

func TestExtraPodSpecMergeStrategy_IsValid(t *testing.T) {
	tests := []struct {
		name     string
		strategy ExtraPodSpecMergeStrategy
		want     bool
	}{
		{name: "override", strategy: ExtraPodSpecMergeStrategyOverride, want: true},
		{name: "strategic", strategy: ExtraPodSpecMergeStrategyStrategic, want: true},
		{name: "empty", strategy: ""},
		{name: "unknown", strategy: "replace"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Check whether the API enum value is supported")
			got := tt.strategy.IsValid()

			t.Log("Verify only the declared enum values are valid")
			if got != tt.want {
				t.Fatalf("ExtraPodSpecMergeStrategy(%q).IsValid() = %v, want %v", tt.strategy, got, tt.want)
			}
		})
	}
}

func TestResolveExtraPodSpecMergeStrategy(t *testing.T) {
	tests := []struct {
		name             string
		explicitStrategy ExtraPodSpecMergeStrategy
		defaultStrategy  ExtraPodSpecMergeStrategy
		want             ExtraPodSpecMergeStrategy
		wantErr          string
	}{
		{
			name:             "explicit strategy wins",
			explicitStrategy: ExtraPodSpecMergeStrategyStrategic,
			defaultStrategy:  ExtraPodSpecMergeStrategyOverride,
			want:             ExtraPodSpecMergeStrategyStrategic,
		},
		{
			name:            "operator default is used when explicit is empty",
			defaultStrategy: ExtraPodSpecMergeStrategyStrategic,
			want:            ExtraPodSpecMergeStrategyStrategic,
		},
		{
			name: "built-in default is used when both inputs are empty",
			want: DefaultExtraPodSpecMergeStrategy,
		},
		{
			name:             "invalid explicit strategy is rejected",
			explicitStrategy: "replace",
			defaultStrategy:  ExtraPodSpecMergeStrategyOverride,
			wantErr:          `invalid extraPodSpec merge strategy "replace"`,
		},
		{
			name:            "invalid operator default is rejected",
			defaultStrategy: "replace",
			wantErr:         `invalid extraPodSpec merge strategy "replace"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Resolve the component, operator, and built-in strategy precedence")
			got, err := ResolveExtraPodSpecMergeStrategy(tt.explicitStrategy, tt.defaultStrategy)

			t.Log("Verify invalid inputs fail and valid inputs resolve to the expected strategy")
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("ResolveExtraPodSpecMergeStrategy() error = %v, want error containing %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("ResolveExtraPodSpecMergeStrategy() unexpected error = %v", err)
			}
			if got != tt.want {
				t.Fatalf("ResolveExtraPodSpecMergeStrategy() = %q, want %q", got, tt.want)
			}
		})
	}
}
