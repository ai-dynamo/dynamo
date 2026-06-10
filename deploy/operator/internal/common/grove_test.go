/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package common

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func TestIsGrovePathway(t *testing.T) {
	tests := []struct {
		name         string
		groveEnabled bool
		annotations  map[string]string
		want         bool
	}{
		{
			name:         "operator disabled",
			groveEnabled: false,
			want:         false,
		},
		{
			name:         "operator enabled and nil annotations",
			groveEnabled: true,
			want:         true,
		},
		{
			name:         "operator enabled and annotation absent",
			groveEnabled: true,
			annotations:  map[string]string{},
			want:         true,
		},
		{
			name:         "annotation false opts out",
			groveEnabled: true,
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove: "false",
			},
			want: false,
		},
		{
			name:         "annotation false is case insensitive",
			groveEnabled: true,
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove: "FALSE",
			},
			want: false,
		},
		{
			name:         "annotation true uses Grove",
			groveEnabled: true,
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove: "true",
			},
			want: true,
		},
		{
			name:         "unrecognized annotation value uses Grove",
			groveEnabled: true,
			annotations: map[string]string{
				consts.KubeAnnotationEnableGrove: "maybe",
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := IsGrovePathway(tt.groveEnabled, tt.annotations)
			if got != tt.want {
				t.Fatalf("IsGrovePathway() = %v, want %v", got, tt.want)
			}
		})
	}
}
