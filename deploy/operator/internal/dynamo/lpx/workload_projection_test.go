/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
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
	firstProjection := projectTestBuild(t, first, PipelineSingle)
	secondProjection := projectTestBuild(t, second, PipelineSingle)

	t.Log("Publish the canonical compiler snapshot identity independently of build locator")
	require.Equal(t, "scheduling.lpu.nvidia.com/compiler-snapshot-digest", lpxv1alpha1.CompilerSnapshotDigestAnnotation)
	require.Equal(t, first.contentID, firstProjection.CompilerSnapshotDigest())
	require.Equal(t, second.contentID, secondProjection.CompilerSnapshotDigest())

	t.Log("Produce the same workload projection digest independent of build locator")
	require.Equal(t, firstProjection.Digest(), secondProjection.Digest())

	t.Log("Keep compiler identity separate from downstream workload projection identity")
	specDecodeProjection := projectTestBuild(t, first, PipelineSpecDecode)
	require.Equal(t, firstProjection.CompilerSnapshotDigest(), specDecodeProjection.CompilerSnapshotDigest())
	require.NotEqual(t, firstProjection.Digest(), specDecodeProjection.Digest())
}
