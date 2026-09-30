/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import "testing"

func TestEffectiveSeccompProfile(t *testing.T) {
	tests := []struct {
		name string
		cfg  CheckpointConfiguration
		want string
	}{
		{
			name: "nil seccomp falls back to default",
			cfg:  CheckpointConfiguration{},
			want: DefaultSeccompProfile,
		},
		{
			name: "zero-value substruct falls back to default (default-on)",
			cfg: CheckpointConfiguration{
				Seccomp: &CheckpointSeccompConfiguration{},
			},
			want: DefaultSeccompProfile,
		},
		{
			name: "Disabled=true returns empty (regardless of Profile)",
			cfg: CheckpointConfiguration{
				Seccomp: &CheckpointSeccompConfiguration{Disabled: true, Profile: "ignored"},
			},
			want: "",
		},
		{
			name: "Profile-only override returns the custom profile",
			cfg: CheckpointConfiguration{
				Seccomp: &CheckpointSeccompConfiguration{Profile: "profiles/custom.json"},
			},
			want: "profiles/custom.json",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.cfg.EffectiveSeccompProfile(); got != tc.want {
				t.Errorf("EffectiveSeccompProfile() = %q, want %q", got, tc.want)
			}
		})
	}
}
