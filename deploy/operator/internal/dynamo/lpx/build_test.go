/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestBuildRuntimePath(t *testing.T) {
	t.Parallel()

	t.Log("Define snapshot remapping and final local or GCS runtime-path contracts")
	tests := []struct {
		name             string
		buildPath        string
		runtimeRef       string
		modelStoragePath string
		want             string
		wantErr          string
	}{
		{name: "absolute path", buildPath: "/snapshot/../build", want: "/build"},
		{name: "file URL", buildPath: "file:///snapshot/../build", want: "/build"},
		{name: "GCS URL", buildPath: "gs://bucket/model/build", runtimeRef: "model-build", modelStoragePath: "/models", want: "/models/gcs/bucket/model/build"},
		{name: "relative runtime", buildPath: "file:///snapshot", runtimeRef: "model-build", modelStoragePath: "/models", want: "/models/model-build"},
		{name: "cleaned runtime", buildPath: "file:///snapshot", runtimeRef: " a/../b ", modelStoragePath: "/custom/models", want: "/custom/models/b"},
		{name: "literal runtime escaping", buildPath: "file:///snapshot", runtimeRef: "model%20build", modelStoragePath: "/models", want: "/models/model%20build"},
		{name: "absolute snapshot", buildPath: "/snapshot", runtimeRef: "model-build", modelStoragePath: "/models", want: "/snapshot"},
		{name: "empty runtime", buildPath: "file:///snapshot", modelStoragePath: "/models", want: "/snapshot"},
		{name: "malformed runtime", buildPath: "file:///snapshot", runtimeRef: "%", modelStoragePath: "/models", want: "/snapshot"},
		{name: "runtime URL", buildPath: "file:///snapshot", runtimeRef: "gs://bucket/build", modelStoragePath: "/models", want: "/snapshot"},
		{name: "absolute runtime", buildPath: "file:///snapshot", runtimeRef: "/model-build", modelStoragePath: "/models", want: "/snapshot"},
		{name: "dot runtime", buildPath: "file:///snapshot", runtimeRef: ".", modelStoragePath: "/models", want: "/snapshot"},
		{name: "parent runtime", buildPath: "file:///snapshot", runtimeRef: "..", modelStoragePath: "/models", want: "/snapshot"},
		{name: "escaping runtime", buildPath: "file:///snapshot", runtimeRef: "a/../../b", modelStoragePath: "/models", want: "/snapshot"},
		{name: "malformed snapshot", buildPath: "%", runtimeRef: "model-build", modelStoragePath: "/models", wantErr: "unable to parse ref"},
		{name: "remapped path without storage", buildPath: "file:///snapshot", runtimeRef: "model-build", wantErr: "unexpected file host"},
		{name: "relative path", buildPath: "relative/build", wantErr: "ref \"relative/build\" must be an absolute path or URL"},
		{name: "GCS without storage", buildPath: "gs://bucket/model", wantErr: "model storage path is empty"},
		{name: "GCS without host", buildPath: "gs:///model", modelStoragePath: "/models", wantErr: "invalid ref \"gs:///model\": missing host"},
		{name: "GCS without object", buildPath: "gs://bucket", modelStoragePath: "/models", wantErr: "invalid GCS build path \"gs://bucket\": missing object path"},
		{name: "GCS empty segment", buildPath: "gs://bucket/a//b", modelStoragePath: "/models", wantErr: "bad path segment \"\""},
		{name: "GCS dot segment", buildPath: "gs://bucket/a/./b", modelStoragePath: "/models", wantErr: "bad path segment \".\""},
		{name: "GCS parent segment", buildPath: "gs://bucket/a/../b", modelStoragePath: "/models", wantErr: "bad path segment \"..\""},
		{name: "unsupported scheme", buildPath: "https://bucket/model", wantErr: "unsupported build path scheme \"https\""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve the build reference into its runtime filesystem path")
			got, err := buildRuntimePath(test.buildPath, test.runtimeRef, test.modelStoragePath)

			t.Log("Verify the resolved path or exact error contract")
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, test.want, got)
		})
	}
}
