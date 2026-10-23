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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/aisimulate"
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
	const (
		mapped    = "registry.example/aisimulate:1.6.0"
		unmapped  = "registry.example/aisimulate:1.2.3"
		nightly   = "registry.example/aisimulate:nightly"
		bundled   = "0.13.0"
		unmappedV = "0.9.0"
	)
	tests := []struct {
		name, operatorVersion, image, override string
		wantErr                                bool
	}{
		{"the derived planner image of a mapped release", "1.6.0", "", "", false},
		{"the derived planner image of a prerelease", "1.6.0-rc.1", "", "", false},
		{"the derived planner image of a release without an entry", "1.99.0", "", "", true},
		{"an override rescues the derived image", "1.99.0", "", unmappedV, false},
		{"an override with an empty image configuration", "1.6.0", "", bundled, false},
		{"an operator version that names no release", "unknown", "", "", false},
		{"an override with nothing to pair with", "unknown", "", bundled, true},
		{"a mapped tag", "1.6.0", mapped, "", false},
		{"a mapped tag with an override", "1.6.0", mapped, "0.13.1", false},
		{"a release without an entry", "1.6.0", unmapped, "", true},
		{"a release without an entry, with its override", "1.6.0", unmapped, unmappedV, false},
		{"a non-release tag with its override", "1.6.0", nightly, bundled, false},
		{"a non-release tag without an override", "1.6.0", nightly, "", true},
		{"an override that is not a version", "1.6.0", nightly, "bad", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateAISimulateDefault(tt.operatorVersion, tt.image, tt.override); (err != nil) != tt.wantErr {
				t.Fatalf("error = %v, want error: %v", err, tt.wantErr)
			}
		})
	}
}

func TestWithV1beta2PersistsThePlannerImageWithItsBundledAISimulateVersion(t *testing.T) {
	derived := defaultImage + ":1.6.0"
	bundled, err := aisimulate.ResolveVersion(derived, "")
	if err != nil {
		t.Fatalf("the release contract must map the operator version used here: %v", err)
	}
	tests := []struct {
		name         string
		options      DGDRV1beta2Options
		wantImage    string
		wantOverride string
	}{
		{"the derived image gets the bundled version", DGDRV1beta2Options{}, derived, bundled},
		{"the explicit override wins with the derived image",
			DGDRV1beta2Options{AISimulateVersionOverride: "0.13.1"}, derived, "0.13.1"},
		{"a configured mapped image gets the bundled version",
			DGDRV1beta2Options{AISimulateImage: "registry.example/aisimulate:1.6.0"},
			"registry.example/aisimulate:1.6.0", bundled},
		{"a configured nightly image keeps its override",
			DGDRV1beta2Options{AISimulateImage: "registry.example/aisimulate:nightly", AISimulateVersionOverride: "0.13.0"},
			"registry.example/aisimulate:nightly", "0.13.0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := NewDGDRDefaulter("1.6.0", "").WithV1beta2(tt.options).v1beta2
			if got.AISimulateImage != tt.wantImage || got.AISimulateVersionOverride != tt.wantOverride {
				t.Fatalf("got %q/%q, want %q/%q", got.AISimulateImage, got.AISimulateVersionOverride, tt.wantImage, tt.wantOverride)
			}
		})
	}
}
