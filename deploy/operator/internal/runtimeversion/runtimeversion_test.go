/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package runtimeversion

import "testing"

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
