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

package featuregate

import (
	"testing"

	semver "github.com/Masterminds/semver/v3"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

func TestOperatorOriginFeatureGate_IsEnabled(t *testing.T) {
	gate := OperatorOriginFeatureGate{
		Name:             "TestFeature",
		MinOriginVersion: *semver.MustParse("1.0.0-dev"),
	}

	tests := []struct {
		name        string
		annotations map[string]string
		want        bool
	}{
		{
			name:        "nil annotations = disabled (backward compat)",
			annotations: nil,
			want:        false,
		},
		{
			name:        "empty annotations = disabled (backward compat)",
			annotations: map[string]string{},
			want:        false,
		},
		{
			name: "origin version below threshold = disabled",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "0.9.0",
			},
			want: false,
		},
		{
			name: "origin version at threshold = enabled",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "1.0.0-dev",
			},
			want: true,
		},
		{
			name: "origin version above threshold (release > pre-release) = enabled",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "1.0.0",
			},
			want: true,
		},
		{
			name: "origin version well above threshold = enabled",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "2.0.0",
			},
			want: true,
		},
		{
			name: "pre-release below threshold = disabled",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "0.9.0-dev",
			},
			want: false,
		},
		{
			name: "invalid origin version = disabled (graceful fallback)",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "not-a-version",
			},
			want: false,
		},
		{
			name: "fallback version 0.0.0-unknown = disabled",
			annotations: map[string]string{
				consts.KubeAnnotationDynamoOperatorOriginVersion: "0.0.0-unknown",
			},
			want: false,
		},
		{
			name: "unrelated annotations without origin version = disabled",
			annotations: map[string]string{
				"some.other/annotation": "value",
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := gate.IsEnabled(tt.annotations)
			if got != tt.want {
				t.Errorf("IsEnabled() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestOperatorOriginFeatureGate_DifferentThresholds(t *testing.T) {
	annotations := map[string]string{
		consts.KubeAnnotationDynamoOperatorOriginVersion: "0.9.0",
	}

	tests := []struct {
		name             string
		minOriginVersion semver.Version
		want             bool
	}{
		{
			name:             "threshold below origin = enabled",
			minOriginVersion: *semver.MustParse("0.8.0"),
			want:             true,
		},
		{
			name:             "threshold equal to origin = enabled",
			minOriginVersion: *semver.MustParse("0.9.0"),
			want:             true,
		},
		{
			name:             "threshold above origin = disabled",
			minOriginVersion: *semver.MustParse("1.0.0"),
			want:             false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gate := OperatorOriginFeatureGate{
				Name:             "TestFeature",
				MinOriginVersion: tt.minOriginVersion,
			}
			got := gate.IsEnabled(annotations)
			if got != tt.want {
				t.Errorf("IsEnabled() with threshold %v = %v, want %v", tt.minOriginVersion, got, tt.want)
			}
		})
	}
}
