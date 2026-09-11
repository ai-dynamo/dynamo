/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"path/filepath"
	"strconv"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/capnp/gbuild_manifest/v2"
	"github.com/stretchr/testify/require"
)

func TestRenderSelectedCyborgConfigMapServerNames(t *testing.T) {
	t.Parallel()

	t.Log("Build a hybrid V2 workload with two uncollapsed runtime partitions")
	fixture := newV2CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	fixture.selectedPropSyncChains = nil
	snapshot := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
	projectionBatch, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline:        PipelineLPX,
		Models:          []string{"default"},
		RuntimeBuildRef: "model-build",
		BuildSnapshot:   normalizeTestSnapshot(t, snapshot),
		ModelSettings:   json.RawMessage(`{"prop_sync":false}`),
	})
	require.NoError(t, err)
	projection := projectionBatch[0]
	projection.stage = testRenderComponentName
	workload := &SelectedWorkload{
		modelProjections:     []*ModelProjection{projection},
		scalingGroupReplicas: 1,
	}

	t.Log("Resolve the projected runtime path")
	plan, err := workload.PlanNodeLocalMaterialization("test-dgd")
	require.NoError(t, err)
	runtimePath, err := buildRuntimePath(projection.configuredBuild.Path, "/models")
	require.NoError(t, err)
	podSpec := renderTestPodSpec()
	initial, err := workload.RenderCyborgConfigMap("test-namespace", "test-dgd", podSpec)
	require.NoError(t, err)

	t.Log("Verify every engine replica addresses only its own Agents")
	for _, replicas := range []int32{1, 2, 10, 12} {
		t.Run(strconv.Itoa(int(replicas)), func(t *testing.T) {
			workload.scalingGroupReplicas = replicas
			configMap, err := workload.RenderCyborgConfigMap("test-namespace", "test-dgd", podSpec)
			require.NoError(t, err)
			require.Equal(t, initial, configMap)
			require.True(t, *configMap.Immutable)
			prefix := plan.LPXScalingGroupTemplate + "-${GROVE_PCSG_INDEX}-"
			require.Equal(t, prefix+"lpu-wkr-m-0-0\n"+prefix+"lpu-wkr-m-0-2", configMap.Data["lpu_servers"])
			require.Equal(t, filepath.Join(runtimePath, "tokenizer"), configMap.Data["tokenizer_dir"])
		})
	}

	t.Log("Render a Cap'n Proto build with a nested tokenizer path")
	projection.configuredBuild.RuntimeTokenizerPath = "metadata/tokenizer"
	configMap, err := workload.RenderCyborgConfigMap("test-namespace", "test-dgd", podSpec)

	t.Log("Verify the nested tokenizer path is rooted in the runtime build")
	require.NoError(t, err)
	require.Equal(t, filepath.Join(runtimePath, "metadata/tokenizer"), configMap.Data["tokenizer_dir"])
	require.NotEqual(t, initial.Name, configMap.Name)

	t.Log("Render a Cap'n Proto build without tokenizer metadata")
	projection.configuredBuild.RuntimeTokenizerPath = ""
	projection.configuredBuild.Path = "relative/build"
	_, err = workload.RenderCyborgConfigMap("test-namespace", "test-dgd", podSpec)

	t.Log("Verify missing tokenizer metadata is rejected")
	require.EqualError(t, err, "capnp manifest build is missing model.tokenizer.path")
}
