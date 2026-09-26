/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"strconv"
	"strings"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
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
	})
	require.NoError(t, err)
	projection := projectionBatch[0]
	projection.stage = testRenderComponentName
	workload := &Workload{
		modelProjections:     []*ModelProjection{projection},
		scalingGroupReplicas: 1,
	}

	t.Log("Render the generated Agent endpoints")
	plan, err := workload.PlanNodeLocalMaterialization("test-dgd")
	require.NoError(t, err)
	initial, err := workload.renderCyborgConfigMap(plan)
	require.NoError(t, err)

	t.Log("Verify every workload replica addresses only its own Agents")
	for _, replicas := range []int32{1, 2, 10, 12} {
		t.Run(strconv.Itoa(int(replicas)), func(t *testing.T) {
			workload.scalingGroupReplicas = replicas
			scaledPlan, err := workload.PlanNodeLocalMaterialization("test-dgd")
			require.NoError(t, err)
			configMap, err := workload.renderCyborgConfigMap(scaledPlan)
			require.NoError(t, err)
			require.Equal(t, initial, configMap)
			require.True(t, *configMap.Immutable)
			prefix := lpxScalingGroupTemplateName + "-${GROVE_PCSG_INDEX}-"
			require.Equal(t, prefix+"agt-0\n"+prefix+"agt-2", configMap.Data["lpu_servers"])
			require.Len(t, configMap.Data, 1)

			t.Log("Resolve Cyborg server addresses to the last workload replica's actual Agent hostnames")
			lastReplica := scaledPlan.ForReplica(replicas - 1)
			servers := strings.Split(strings.ReplaceAll(configMap.Data["lpu_servers"], "${GROVE_PCSG_INDEX}", strconv.Itoa(int(replicas-1))), "\n")
			for index, offset := range []int{0, 2} {
				require.Equal(t, lastReplica.Agents[0].CliqueName+"-"+strconv.Itoa(offset), "test-dgd-0-"+servers[index])
			}
		})
	}
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
		{name: "one chain across all partitions", partitions: 3, chains: [][]uint32{{0, 1, 2}}, wantOffsets: []int{0}},
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
			projection := projectTestBuild(t, normalizeTestSnapshot(t, snapshot), PipelineLPX)
			projection.stage = testRenderComponentName

			t.Log("Render only projected endpoints without compressing their physical Agent offsets")
			workload := &Workload{modelProjections: []*ModelProjection{projection}, scalingGroupReplicas: 1}
			plan, err := workload.PlanNodeLocalMaterialization("test-dgd")
			require.NoError(t, err)
			require.Len(t, projection.RequestSpec(plan, "agents").Partitions, test.partitions)
			require.Equal(t, 2*test.partitions, projection.agentReplicas)
			configMap, err := workload.renderCyborgConfigMap(plan)
			require.NoError(t, err)
			prefix := lpxScalingGroupTemplateName + "-${GROVE_PCSG_INDEX}-" + plan.Agents[0].TemplateName + "-"
			servers := make([]string, len(test.wantOffsets))
			for index, offset := range test.wantOffsets {
				servers[index] = prefix + strconv.Itoa(offset)
			}
			require.Equal(t, strings.Join(servers, "\n"), configMap.Data["lpu_servers"])
		})
	}
}
