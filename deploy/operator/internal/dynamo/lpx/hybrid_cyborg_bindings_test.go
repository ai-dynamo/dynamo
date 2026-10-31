/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"strconv"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
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
		Pipeline:        PipelineHybrid,
		Models:          []string{"default"},
		RuntimeBuildRef: "model-build",
		Build:           snapshot,
	})
	require.NoError(t, err)
	projection := projectionBatch[0]
	projection.stage = testRenderComponentName
	serving := &dynamov1beta1.DynamoComponentDeploymentSharedSpec{
		Roles: []dynamov1beta1.ComponentRoleSpec{{Name: dynamov1beta1.ComponentRoleLPXConductor}},
	}
	workload := &Workload{
		modelProjections:     []*ModelProjection{projection},
		scalingGroupReplicas: 1,
	}

	t.Log("Render the generated Agent endpoints")
	plan, err := workload.PlanNodeLocalMaterialization("test-dgd", serving)
	require.NoError(t, err)
	initial, _, err := workload.renderCyborgConfigMap(plan)
	require.NoError(t, err)

	t.Log("Verify every workload replica addresses only its own Agents")
	for _, replicas := range []int32{1, 2, 10, 12} {
		t.Run(strconv.Itoa(int(replicas)), func(t *testing.T) {
			workload.scalingGroupReplicas = replicas
			scaledPlan, err := workload.PlanNodeLocalMaterialization("test-dgd", serving)
			require.NoError(t, err)
			configMap, _, err := workload.renderCyborgConfigMap(scaledPlan)
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

			t.Log("Expand the RDMA addresses as Kubernetes does with the Grove variables of the last workload replica")
			addrs := strings.NewReplacer(
				"$(GROVE_PCSG_NAME)", "test-dgd-0-"+scaledPlan.ScalingGroupTemplate,
				"$(GROVE_PCSG_INDEX)", strconv.Itoa(int(replicas-1)),
				"$(GROVE_HEADLESS_SERVICE)", "test-dgd-0",
			).Replace(workload.cyborgAgentRDMAAddrs(scaledPlan))
			require.Equal(t, strings.Join([]string{
				lastReplica.Agents[0].CliqueName + "-0.test-dgd-0:19877",
				lastReplica.Agents[0].CliqueName + "-2.test-dgd-0:19877",
			}, ","), addrs)
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
			fixture.selectedPropSyncChains = test.chains
			fixture.partitions = make([]testV3CapnpPartition, test.partitions)
			for index := range fixture.partitions {
				fixture.partitions[index] = partition
				fixture.partitions[index].id = uint32(index)
			}
			snapshot := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
			projection := projectTestBuild(t, snapshot, PipelineHybrid)
			projection.stage = testRenderComponentName

			t.Log("Render only projected endpoints without compressing their physical Agent offsets")
			serving := &dynamov1beta1.DynamoComponentDeploymentSharedSpec{
				Roles: []dynamov1beta1.ComponentRoleSpec{{Name: dynamov1beta1.ComponentRoleLPXConductor}},
			}
			workload := &Workload{modelProjections: []*ModelProjection{projection}, scalingGroupReplicas: 1}
			plan, err := workload.PlanNodeLocalMaterialization("test-dgd", serving)
			require.NoError(t, err)
			require.Len(t, projection.RequestSpec(plan, "agents").Partitions, test.partitions)
			require.Equal(t, 2*test.partitions, projection.agentReplicas)
			configMap, _, err := workload.renderCyborgConfigMap(plan)
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

func TestRenderHXHybridCyborgAgentServers(t *testing.T) {
	t.Parallel()

	prefix := lpxScalingGroupTemplateName + "-${GROVE_PCSG_INDEX}-" + testAgentTemplateName + "-"
	host := "$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-" + testAgentTemplateName + "-"
	port := ".$(GROVE_HEADLESS_SERVICE):19877"
	tests := []struct {
		name        string
		selection   *dynamov1beta1.LPXLocalPartitions
		wantServers string
		wantAddrs   string
	}{
		{
			name:        "every partition on LPUs addresses chain 2-3 through partition 2",
			wantServers: prefix + "0\n" + prefix + "1\n" + prefix + "3",
			wantAddrs:   host + "0" + port + "," + host + "1" + port + "," + host + "3" + port,
		},
		{
			name:        "partition 1 on the GPU",
			selection:   &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{1}},
			wantServers: prefix + "0\n" + prefix + "2",
			wantAddrs:   host + "0" + port + "," + host + "2" + port,
		},
		{
			name:        "chain 2-3 on the GPU",
			selection:   &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{2}},
			wantServers: prefix + "0\n" + prefix + "1",
			wantAddrs:   host + "0" + port + "," + host + "1" + port,
		},
		{
			name:      "every partition on the GPU",
			selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeAll},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Create a hybrid HX build with partitions 1-4, a selected chain 2-3, and one CUDA artifact")
			fixture := newV3CompilerFixture()
			fixture.compilationMode = manifestcapnp.CompilationMode_lpx
			for id := uint32(2); id <= 4; id++ {
				partition := fixture.partitions[0]
				partition.id = id
				fixture.partitions = append(fixture.partitions, partition)
			}
			fixture.partitions = append(fixture.partitions, testV3CapnpPartition{id: 5, deviceType: manifestcapnp.DeviceType_cuda})
			fixture.selectedPropSyncChains = [][]uint32{{2, 3}}
			normalized := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineHybrid, Models: []string{"default"}, Build: normalized,
				RuntimeBuildRef: "model-build", LocalPartitions: test.selection,
			})
			require.NoError(t, err)
			require.Equal(t, BuildFamilyHX, projections[0].configuredBuild.family)

			t.Log("Render the hybrid workload with an authored Cyborg configuration mount")
			projections[0].stage = testRenderComponentName
			serving := &dynamov1beta1.DynamoComponentDeploymentSharedSpec{
				Roles: []dynamov1beta1.ComponentRoleSpec{{Name: dynamov1beta1.ComponentRoleLPXConductor}},
			}
			workload := &Workload{modelProjections: projections, scalingGroupReplicas: 1}
			workload.digest, err = workloadSetDigest(projections)
			require.NoError(t, err)
			plan, err := workload.PlanNodeLocalMaterialization("test-pcs", serving)
			require.NoError(t, err)
			pcs := renderTestPCS(true)
			rendered, err := RenderNodeLocal(workload, plan, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}},
				Cyborg: pcs.Spec.Template.Cliques[0],
			})
			require.NoError(t, err)

			t.Log("Publish one Agent per remote runtime partition, as for XT")
			require.Len(t, rendered.Resources, 2)
			servers, ok := rendered.Resources[0].(*corev1.ConfigMap)
			require.True(t, ok)
			require.Equal(t, map[string]string{"lpu_servers": test.wantServers}, servers.Data)

			t.Log("Mount the Agent addresses, not the Agents' partition table, into Cyborg")
			cyborg := rendered.Cliques[0]
			require.Equal(t, "cond", cyborg.Name)
			require.Contains(t, cyborg.Spec.PodSpec.Containers[0].VolumeMounts, corev1.VolumeMount{Name: lpuConfigVolumeName, MountPath: "/configs"})
			require.Contains(t, cyborg.Spec.PodSpec.Volumes, corev1.Volume{
				Name: lpuConfigVolumeName,
				ConfigMap: &corev1.ConfigMapVolumeSource{
					Name: servers.Name,
				},
			})
			require.NotEmpty(t, cyborg.Annotations[v1alpha1.AnnotationExtraResourcesHash])

			t.Log("Publish the same Agents to Cyborg as RDMA host:port addresses")
			require.Contains(t, cyborg.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{Name: "LPX_LPU_RDMA_ADDRS", Value: test.wantAddrs})
		})
	}
}
