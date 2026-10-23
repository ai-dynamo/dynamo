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
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
)

func TestDefaultAISimulateFollowsTheOwnershipOfTheImage(t *testing.T) {
	const image = "registry.example/aisimulate:nightly"
	tests := []struct {
		name         string
		spec         nvidiacomv1beta2.AISimulateSpec
		wantImage    string
		wantOverride string
	}{
		{"the operator's pair is persisted together", nvidiacomv1beta2.AISimulateSpec{}, image, "1.2.0"},
		{"a caller's image gets no operator override",
			nvidiacomv1beta2.AISimulateSpec{Image: "example/own:v1.0.0"}, "example/own:v1.0.0", ""},
		{"a caller's override wins next to the operator's image",
			nvidiacomv1beta2.AISimulateSpec{VersionOverride: "2.0.0"}, image, "2.0.0"},
		{"a caller's pair is kept",
			nvidiacomv1beta2.AISimulateSpec{Image: "example/own:x", VersionOverride: "3.0.0"}, "example/own:x", "3.0.0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec := tt.spec
			defaultAISimulateSpecV1beta2(&spec, image, "1.2.0")
			if spec.Image != tt.wantImage || spec.VersionOverride != tt.wantOverride {
				t.Fatalf("got %q/%q, want %q/%q", spec.Image, spec.VersionOverride, tt.wantImage, tt.wantOverride)
			}
		})
	}
}

func TestValidateAISimulateDefault(t *testing.T) {
	tests := []struct {
		name, image, override string
		wantErr               bool
	}{
		{"nothing configured", "", "", false},
		{"semver tag", "registry.example/aisimulate:1.2.3", "", false},
		{"semver tag with a matching override", "registry.example/aisimulate:1.2.3", "1.9.0", false},
		{"non-semver tag with its override", "registry.example/aisimulate:nightly", "1.2.0", false},
		{"non-semver tag without an override", "registry.example/aisimulate:nightly", "", true},
		{"override that is not a version", "registry.example/aisimulate:nightly", "bad", true},
		{"override without an image", "", "1.2.0", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateAISimulateDefault(tt.image, tt.override); (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, want error: %v", err, tt.wantErr)
			}
		})
	}
}
