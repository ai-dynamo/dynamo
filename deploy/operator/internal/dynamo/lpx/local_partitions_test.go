/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestApplyLocalPartitionIDs(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		localIDs []int
		env      []corev1.EnvVar
		envFrom  []corev1.EnvFromSource
		wantEnv  []corev1.EnvVar
	}{
		{
			name:    "no local partitions publishes an empty selection over authored values",
			env:     []corev1.EnvVar{{Name: localPartitionIDsEnv, Value: "authored"}, {Name: "OTHER", Value: "kept"}},
			wantEnv: []corev1.EnvVar{{Name: localPartitionIDsEnv}, {Name: "OTHER", Value: "kept"}},
		},
		{
			name:     "local unit roots",
			localIDs: []int{0, 15},
			wantEnv:  []corev1.EnvVar{{Name: localPartitionIDsEnv, Value: "0,15"}},
		},
		{
			name:    "no local partitions shadows envFrom sources",
			envFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{Name: "authored"}}},
			wantEnv: []corev1.EnvVar{{Name: localPartitionIDsEnv}},
		},
		{
			name:     "resolved selection replaces authored values",
			localIDs: []int{0, 2, 12},
			env:      []corev1.EnvVar{{Name: localPartitionIDsEnv, Value: "authored"}, {Name: "OTHER", Value: "kept"}},
			wantEnv:  []corev1.EnvVar{{Name: localPartitionIDsEnv, Value: "0,2,12"}, {Name: "OTHER", Value: "kept"}},
		},
		{
			name:     "resolved selection precedes dependent values",
			localIDs: []int{11},
			env: []corev1.EnvVar{
				{Name: "FIRST", Value: "kept"},
				{Name: "LOCAL_IDS", Value: "$(" + localPartitionIDsEnv + ")"},
				{Name: localPartitionIDsEnv, Value: "authored"},
				{Name: "LAST", Value: "kept"},
			},
			wantEnv: []corev1.EnvVar{
				{Name: localPartitionIDsEnv, Value: "11"},
				{Name: "FIRST", Value: "kept"},
				{Name: "LOCAL_IDS", Value: "$(" + localPartitionIDsEnv + ")"},
				{Name: "LAST", Value: "kept"},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Publish the resolved local partitions into the runtime container")
			projection := &ModelProjection{localPartitionIDs: test.localIDs}
			container := &corev1.Container{Env: test.env, EnvFrom: test.envFrom}
			applyLocalPartitionIDs(container, projection)
			require.Equal(t, test.wantEnv, container.Env)
		})
	}
}

func TestRenderHybridLocalPartitions(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name             string
		selection        *dynamov1beta1.LPXLocalPartitions
		wantAgents       int32
		wantStartsAfter  []string
		wantGroupMembers []string
		wantLocalEnv     string
	}{
		{
			name: "partial selection renders Agents for remote partitions", selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeIDs, IDs: []int64{11}},
			wantAgents: 9, wantStartsAfter: []string{"agt"}, wantGroupMembers: []string{"agt", "cond"}, wantLocalEnv: "11",
		},
		{
			name: "all-local selection renders only Cyborg", selection: &dynamov1beta1.LPXLocalPartitions{Mode: dynamov1beta1.LPXLocalPartitionsModeAll},
			wantStartsAfter: []string{}, wantGroupMembers: []string{"cond"}, wantLocalEnv: "7,11",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Project a hybrid build with chain 7-8 and an independent partition 11")
			build, err := acquireTestUnitBundle(t, testUnitBundle{
				architecture: deploymentcapnpv1.LpuArchitecture_lp20, devicesPerNode: 8, hybrid: true,
				partitions: []testUnitPartition{{id: 7, numChips: 8}, {id: 8, numChips: 64}, {id: 11, numChips: 64}},
				units: []testLPUUnit{
					{reservations: []testLPUReservation{testNodesReservation(1, 7), testNodesReservation(8, 8)}, connectors: []testLPUConnector{{}}},
					{reservations: []testLPUReservation{testNodesReservation(8, 11)}},
				},
			})
			require.NoError(t, err)
			projections, err := appendModelProjections(nil, ModelProjectionInput{
				Pipeline: PipelineHybrid, Models: []string{"default"}, Build: build,
				RuntimeBuildRef: "model-build", LocalPartitions: test.selection,
			})
			require.NoError(t, err)

			t.Log("Render the hybrid workload")
			rendered, err := renderSelectedForTest(renderTestPCS(true), projections, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}},
			})
			require.NoError(t, err)

			t.Log("Render Agents only for remote partitions and start Cyborg after them")
			var agentReplicas int32
			for _, clique := range rendered.Spec.Template.Cliques {
				if clique.Name == testAgentTemplateName {
					agentReplicas = clique.Spec.Replicas
				}
			}
			require.Equal(t, test.wantAgents, agentReplicas)
			cyborg := namedClique(t, rendered, "cond")
			require.Equal(t, test.wantStartsAfter, cyborg.Spec.StartsAfter)
			require.ElementsMatch(t, test.wantGroupMembers, rendered.Spec.Template.PodCliqueScalingGroupConfigs[0].CliqueNames)

			t.Log("Publish the resolved local partitions to Cyborg")
			localEnv := ""
			for _, variable := range cyborg.Spec.PodSpec.Containers[0].Env {
				if variable.Name == localPartitionIDsEnv {
					localEnv = variable.Value
				}
			}
			require.Equal(t, test.wantLocalEnv, localEnv)
		})
	}
}
