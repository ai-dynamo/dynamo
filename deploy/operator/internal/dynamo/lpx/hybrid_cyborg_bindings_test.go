/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/capnp/gbuild_manifest/v2"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
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

	t.Log("Reject oversized rendered Cyborg configuration before publication")
	projection.configuredBuild.RuntimeTokenizerPath = strings.Repeat("x", corev1.MaxSecretSize)
	_, err = workload.RenderCyborgConfigMap("test-namespace", "test-dgd", podSpec)
	require.ErrorContains(t, err, "rendered LPX ConfigMap")
	require.ErrorContains(t, err, "maximum is 1048576")

	t.Log("Render a Cap'n Proto build without tokenizer metadata")
	projection.configuredBuild.RuntimeTokenizerPath = ""
	projection.configuredBuild.Path = "relative/build"
	_, err = workload.RenderCyborgConfigMap("test-namespace", "test-dgd", podSpec)

	t.Log("Verify missing tokenizer metadata is rejected")
	require.EqualError(t, err, "capnp manifest build is missing model.tokenizer.path")
}

func TestRenderCyborgConfigMapPreservesProjectedEndpoints(t *testing.T) {
	t.Parallel()

	t.Log("Cover the legacy launcher's 14 endpoints and selected PropSync chain roots")
	for _, test := range []struct {
		name        string
		partitions  int
		chains      [][]uint32
		wantOffsets []int
	}{
		{name: "legacy 14 endpoints", partitions: 14, wantOffsets: []int{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26}},
		{name: "chain followed by endpoint", partitions: 4, chains: [][]uint32{{1, 2}}, wantOffsets: []int{0, 2, 6}},
		{name: "two chains followed by endpoint", partitions: 5, chains: [][]uint32{{0, 1}, {2, 3}}, wantOffsets: []int{0, 4, 8}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Project a hybrid compiler inventory with two-node physical partitions")
			fixture := newV2CompilerFixture()
			partition := fixture.partitions[0]
			fixture.compilationMode = manifestcapnp.CompilationMode_lpx
			fixture.numLPUNodes = uint32(2 * test.partitions)
			fixture.selectedPropSyncChains = test.chains
			fixture.partitions = make([]testV3CapnpPartition, test.partitions)
			for index := range fixture.partitions {
				fixture.partitions[index] = partition
				fixture.partitions[index].id = uint32(index)
			}
			snapshot := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
			projection := projectTestBuild(t, normalizeTestSnapshot(t, snapshot), PipelineLPX, `{"prop_sync":false}`)
			projection.stage = testRenderComponentName

			t.Log("Keep every physical scheduler partition and Agent, including chain followers")
			require.Len(t, projection.RequestSpec("test", "agents", nil).Partitions, test.partitions)
			require.Equal(t, 2*test.partitions, projection.agentReplicas)

			t.Log("Render only projected endpoints without compressing their physical Agent offsets")
			workload := &SelectedWorkload{modelProjections: []*ModelProjection{projection}, scalingGroupReplicas: 1}
			plan, err := workload.PlanNodeLocalMaterialization("test-dgd")
			require.NoError(t, err)
			configMap, err := workload.RenderCyborgConfigMap("test", "test-dgd", renderTestPodSpec())
			require.NoError(t, err)
			prefix := plan.LPXScalingGroupTemplate + "-${GROVE_PCSG_INDEX}-" + plan.Agents[0].TemplateName + "-"
			servers := make([]string, len(test.wantOffsets))
			for index, offset := range test.wantOffsets {
				servers[index] = prefix + strconv.Itoa(offset)
			}
			require.Equal(t, strings.Join(servers, "\n"), configMap.Data["lpu_servers"])
		})
	}
}
