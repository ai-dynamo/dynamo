/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
)

func TestIsDisaggregatedSetPathwaySelected(t *testing.T) {
	tests := []struct {
		name        string
		annotations map[string]string
		dsEnabled   bool
		want        bool
	}{
		{
			name:        "no annotations, DS disabled at operator level",
			annotations: nil,
			dsEnabled:   false,
			want:        false,
		},
		{
			name:        "annotation true, DS enabled at operator level",
			annotations: map[string]string{commonconsts.KubeAnnotationEnableDisaggregatedSet: "true"},
			dsEnabled:   true,
			want:        true,
		},
		{
			name:        "annotation true, but DS disabled at operator level",
			annotations: map[string]string{commonconsts.KubeAnnotationEnableDisaggregatedSet: "true"},
			dsEnabled:   false,
			want:        false,
		},
		{
			name:        "annotation true (case-insensitive), DS enabled",
			annotations: map[string]string{commonconsts.KubeAnnotationEnableDisaggregatedSet: "True"},
			dsEnabled:   true,
			want:        true,
		},
		{
			name:        "annotation false, DS enabled",
			annotations: map[string]string{commonconsts.KubeAnnotationEnableDisaggregatedSet: "false"},
			dsEnabled:   true,
			want:        false,
		},
		{
			name:        "annotation unset, DS enabled",
			annotations: map[string]string{},
			dsEnabled:   true,
			want:        false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rc := &controller_common.RuntimeConfig{DisaggregatedSetEnabled: tt.dsEnabled}
			if got := IsDisaggregatedSetPathwaySelected(tt.annotations, rc); got != tt.want {
				t.Errorf("IsDisaggregatedSetPathwaySelected() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDSNameForDGD(t *testing.T) {
	got := DSNameForDGD("my-dgd")
	want := "my-dgd-disagg"
	if got != want {
		t.Errorf("DSNameForDGD() = %q, want %q", got, want)
	}
}
