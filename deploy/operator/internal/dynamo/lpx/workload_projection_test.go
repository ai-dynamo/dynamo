/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"testing"

	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
)

func TestWorkloadDigestIsIndependentOfBuildLocator(t *testing.T) {
	t.Parallel()

	t.Log("Acquire equal compiler contents under distinct local build paths")
	first := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	second := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	require.NotEqual(t, first.build.Path, second.build.Path)
	require.Equal(t, first.contentID, second.contentID)

	t.Log("Project either immutable snapshot through the same intent")
	firstProjection := projectTestBuild(t, first, PipelineSingle, `{"prop_sync":true}`)
	secondProjection := projectTestBuild(t, second, PipelineSingle, `{"prop_sync":true}`)

	t.Log("Publish the canonical compiler snapshot identity independently of build locator")
	require.Equal(t, "scheduling.lpu.nvidia.com/compiler-snapshot-digest", lpxv1alpha1.CompilerSnapshotDigestAnnotation)
	require.Equal(t, first.contentID, firstProjection.CompilerSnapshotDigest())
	require.Equal(t, second.contentID, secondProjection.CompilerSnapshotDigest())

	t.Log("Produce the same workload projection digest independent of build locator")
	require.Equal(t, firstProjection.Digest(), secondProjection.Digest())

	t.Log("Keep compiler identity separate from downstream workload projection identity")
	specDecodeProjection := projectTestBuild(t, first, PipelineSpecDecode, `{"prop_sync":true}`)
	require.Equal(t, firstProjection.CompilerSnapshotDigest(), specDecodeProjection.CompilerSnapshotDigest())
	require.NotEqual(t, firstProjection.Digest(), specDecodeProjection.Digest())
}

func TestCanonicalModelSettingsRetainRawAndNormalizeRuntimeNumbers(t *testing.T) {
	t.Log("Decode canonical model settings while retaining their exact raw bytes")
	raw, object := canonicalModelSettings(json.RawMessage(
		`{"exponent":1e3,"float":1.0,"integer":8192,"nested":{"array":[128001,1.5,9223372036854775808]}}`,
	))
	require.Equal(t, json.RawMessage(`{"exponent":1e3,"float":1.0,"integer":8192,"nested":{"array":[128001,1.5,9223372036854775808]}}`), raw)
	require.Equal(t, json.Number("8192"), object["integer"])

	t.Log("Normalize representable integers while retaining floating-point runtime values")
	merged, err := mergeRuntimeSettingOverride(nil, object, "settings")
	require.NoError(t, err)
	normalized := merged.(map[string]any)
	require.Equal(t, int64(8192), normalized["integer"])
	require.Equal(t, []any{int64(128001), float64(1.5), float64(9223372036854775808)}, normalized["nested"].(map[string]any)["array"])

	t.Log("Normalize explicit null settings to a present empty object")
	raw, object = canonicalModelSettings(json.RawMessage("null"))
	require.Nil(t, raw)
	require.NotNil(t, object)
	require.Empty(t, object)
}
