/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtimeversion

import "testing"

func TestVersionAtLeast(t *testing.T) {
	minimum := Version{Major: 1, Minor: 4, Patch: 0}
	tests := []struct {
		version Version
		want    bool
	}{
		{version: Version{Major: 1, Minor: 3, Patch: 9}, want: false},
		{version: Version{Major: 1, Minor: 4, Patch: 0}, want: true},
		{version: Version{Major: 1, Minor: 4, Patch: 1}, want: true},
		{version: Version{Major: 2, Minor: 0, Patch: 0}, want: true},
	}

	for _, tt := range tests {
		if got := tt.version.AtLeast(minimum); got != tt.want {
			t.Errorf("%s.AtLeast(%s) = %t, want %t", tt.version, minimum, got, tt.want)
		}
	}
}

func TestResolve(t *testing.T) {
	t.Run("override takes precedence over image tag", func(t *testing.T) {
		got, err := Resolve("registry.example/runtime:1.3.0", "1.4.0")
		if err != nil {
			t.Fatal(err)
		}
		want := Version{Major: 1, Minor: 4, Patch: 0}
		if got != want {
			t.Fatalf("Resolve() = %+v, want %+v", got, want)
		}
	})

	t.Run("falls back to image tag", func(t *testing.T) {
		got, err := Resolve("registry.example/runtime:1.4.0", "")
		if err != nil {
			t.Fatal(err)
		}
		want := Version{Major: 1, Minor: 4, Patch: 0}
		if got != want {
			t.Fatalf("Resolve() = %+v, want %+v", got, want)
		}
	})
}

func TestParse(t *testing.T) {
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
