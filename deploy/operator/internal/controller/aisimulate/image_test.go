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

func TestResolveVersion(t *testing.T) {
	cases := []struct {
		name, image, override, want string
		wantErr                     bool
	}{
		{name: "version from the tag", image: "registry.example.com/aisimulate:1.2.3", want: "1.2.3"},
		{name: "v-prefixed tag", image: "registry.example.com/aisimulate:v1.2.3", want: "1.2.3"},
		{name: "override wins over the tag", image: "registry.example.com/aisimulate:1.2.3", override: "2.0.0", want: "2.0.0"},
		{name: "override rescues a custom tag", image: "registry.example.com/aisimulate:custom", override: "1.4.0", want: "1.4.0"},
		{name: "custom tag needs an override", image: "registry.example.com/aisimulate:custom", wantErr: true},
		{name: "untagged image needs an override", image: "registry.example.com/aisimulate", wantErr: true},
		{name: "registry port is not a tag", image: "registry.example.com:5000/aisimulate", wantErr: true},
		{name: "no image", wantErr: true},
		{name: "override without an image", override: "1.0.0", want: "1.0.0"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveVersion(tc.image, tc.override)
			if (err != nil) != tc.wantErr || got != tc.want {
				t.Fatalf("ResolveVersion(%q, %q) = %q, %v; want %q (error %v)", tc.image, tc.override, got, err, tc.want, tc.wantErr)
			}
		})
	}
}
