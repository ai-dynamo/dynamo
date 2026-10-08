/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLPURuntimeBuildRef(t *testing.T) {
	t.Parallel()

	t.Log("Define runtime build-reference selection contracts")
	tests := []struct {
		name     string
		snapshot string
		runtime  string
		want     string
	}{
		{name: "relative runtime", snapshot: "file:///snapshot", runtime: "model-build", want: "file:///models/model-build"},
		{name: "cleaned runtime", snapshot: "file:///snapshot", runtime: " a/../b ", want: "file:///models/b"},
		{name: "GCS snapshot", snapshot: "gs://bucket/snapshot", runtime: "model-build", want: "gs://bucket/snapshot"},
		{name: "malformed snapshot", snapshot: "%", runtime: "model-build", want: "%"},
		{name: "empty runtime", snapshot: "file:///snapshot", want: "file:///snapshot"},
		{name: "malformed runtime", snapshot: "file:///snapshot", runtime: "%", want: "file:///snapshot"},
		{name: "runtime URL", snapshot: "file:///snapshot", runtime: "gs://bucket/build", want: "file:///snapshot"},
		{name: "absolute runtime", snapshot: "file:///snapshot", runtime: "/model-build", want: "file:///snapshot"},
		{name: "dot runtime", snapshot: "file:///snapshot", runtime: ".", want: "file:///snapshot"},
		{name: "parent runtime", snapshot: "file:///snapshot", runtime: "..", want: "file:///snapshot"},
		{name: "escaping runtime", snapshot: "file:///snapshot", runtime: "a/../../b", want: "file:///snapshot"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Select the runtime build reference")
			got := lpuRuntimeBuildRef(&ModelProjection{
				runtimeBuildRef: test.runtime,
				configuredBuild: Build{Path: test.snapshot},
			}, "/models")

			t.Log("Preserve valid remapping and every fallback byte exactly")
			require.Equal(t, test.want, got)
		})
	}
}
