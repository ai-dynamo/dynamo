/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestWorkloadDigestIsIndependentOfBuildLocator(t *testing.T) {
	t.Parallel()

	t.Log("Acquire equal compiler contents under distinct local build paths")
	first := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	second := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	require.NotEqual(t, first.path, second.path)
	require.Equal(t, first.contentID, second.contentID)

	t.Log("Project either immutable snapshot through the same intent")
	firstProjection := projectTestModel(t, first, PipelineSingle)
	secondProjection := projectTestModel(t, second, PipelineSingle)

	t.Log("Publish the canonical compiler snapshot identity independently of build locator")
	require.Equal(t, first.contentID, firstProjection.CompilerSnapshotDigest())
	require.Equal(t, second.contentID, secondProjection.CompilerSnapshotDigest())

	t.Log("Produce the same workload projection digest independent of build locator")
	require.Equal(t, firstProjection.Digest(), secondProjection.Digest())

	t.Log("Keep compiler identity separate from downstream workload projection identity")
	specDecodeProjection := projectTestModel(t, first, PipelineSpecDecode)
	require.Equal(t, firstProjection.CompilerSnapshotDigest(), specDecodeProjection.CompilerSnapshotDigest())
	require.NotEqual(t, firstProjection.Digest(), specDecodeProjection.Digest())
}
