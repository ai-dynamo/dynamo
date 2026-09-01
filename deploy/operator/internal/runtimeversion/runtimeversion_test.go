/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtimeversion

import "testing"

func TestVersionCompare(t *testing.T) {
	t.Log("define the reference runtime compatibility core")
	reference := Version{Major: 1, Minor: 4, Patch: 1}

	t.Log("define comparisons below, equal to, and above the reference")
	tests := []struct {
		name    string
		version Version
		want    int
	}{
		{name: "below major", version: Version{Major: 0, Minor: 9, Patch: 9}, want: -1},
		{name: "below minor", version: Version{Major: 1, Minor: 3, Patch: 9}, want: -1},
		{name: "below patch", version: Version{Major: 1, Minor: 4, Patch: 0}, want: -1},
		{name: "equal", version: Version{Major: 1, Minor: 4, Patch: 1}, want: 0},
		{name: "above patch", version: Version{Major: 1, Minor: 4, Patch: 2}, want: 1},
		{name: "above minor", version: Version{Major: 1, Minor: 5, Patch: 0}, want: 1},
		{name: "above major", version: Version{Major: 2, Minor: 0, Patch: 0}, want: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("compare the normalized runtime compatibility cores")
			if got := tt.version.Compare(reference); got != tt.want {
				t.Fatalf("%s.Compare(%s) = %d, want %d", tt.version, reference, got, tt.want)
			}
		})
	}
}

func TestParse(t *testing.T) {
	t.Log("define valid and invalid runtime-version override strings")
	tests := []struct {
		name    string
		value   string
		want    Version
		wantErr bool
	}{
		{
			name:  "parses a canonical override",
			value: "1.2.3",
			want:  Version{Major: 1, Minor: 2, Patch: 3},
		},
		{
			name:    "rejects an incomplete override",
			value:   "1.2",
			wantErr: true,
		},
		{
			name:    "rejects a uint64-overflowing override segment",
			value:   "18446744073709551616.0.0",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("parse the runtime-version override")
			got, err := Parse(tt.value)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Parse(%q) error = %v, wantErr %t", tt.value, err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Fatalf("Parse(%q) = %+v, want %+v", tt.value, got, tt.want)
			}
		})
	}
}

func TestParseImageVersion(t *testing.T) {
	t.Log("define parseable and unparseable runtime image tags")
	tests := []struct {
		name    string
		image   string
		want    Version
		wantErr bool
	}{
		{
			name:  "parses a tag with a prefix and prerelease suffix",
			image: "nvcr.io/nvidia/ai-dynamo/vllm-runtime:v1.2.3-cuda13",
			want:  Version{Major: 1, Minor: 2, Patch: 3},
		},
		{
			name:    "rejects an unparseable image tag",
			image:   "registry.example/runtime:sha-123",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("parse the runtime version from the image tag")
			got, err := ParseImageVersion(tt.image)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ParseImageVersion(%q) error = %v, wantErr %t", tt.image, err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Fatalf("ParseImageVersion(%q) = %+v, want %+v", tt.image, got, tt.want)
			}
		})
	}
}

func TestResolve(t *testing.T) {
	t.Log("define image-derived and explicit runtime-version resolution cases")
	tests := []struct {
		name     string
		image    string
		override string
		want     Version
		wantErr  bool
	}{
		{
			name:  "uses the image tag when the override is empty",
			image: "nvcr.io/nvidia/ai-dynamo/runtime:v1.5.0-cuda13",
			want:  Version{Major: 1, Minor: 5, Patch: 0},
		},
		{
			name:     "the override is authoritative",
			image:    "nvcr.io/nvidia/ai-dynamo/runtime:1.5.0",
			override: "1.4.0",
			want:     Version{Major: 1, Minor: 4, Patch: 0},
		},
		{
			name:     "does not fall back when the override is invalid",
			image:    "nvcr.io/nvidia/ai-dynamo/runtime:1.5.0",
			override: "invalid",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("resolve the canonical runtime version")
			got, err := Resolve(tt.image, tt.override)
			if (err != nil) != tt.wantErr {
				t.Fatalf("Resolve(%q, %q) error = %v, wantErr %t", tt.image, tt.override, err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Fatalf("Resolve(%q, %q) = %+v, want %+v", tt.image, tt.override, got, tt.want)
			}
		})
	}
}
