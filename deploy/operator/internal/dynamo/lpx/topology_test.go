/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// topologyRenderInput returns a hybrid PCS and render input whose Agent and
// Cyborg templates each declare a setup init container and the runtime sidecar.
func topologyRenderInput(t *testing.T) (*ModelProjection, *grovev1alpha1.PodCliqueSet, RenderInput) {
	t.Helper()
	fixture := newV3CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	projection := projectRenderFixture(t, PipelineHybrid, acquireTestSnapshot(t, writeCompilerFixture(t, fixture)))
	projection.configuredBuild.ioFPGACount = 1
	projection.configuredBuild.ioFanoutFactor = 1

	pcs := renderTestPCS(true)
	cyborg := namedClique(t, pcs, "cond")
	cyborg.Spec.PodSpec.Containers[0].Env = append(cyborg.Spec.PodSpec.Containers[0].Env,
		corev1.EnvVar{Name: "CYBORG_LPU_TOPOLOGY_SOCKET", Value: "/authored.sock"})
	cyborg.Spec.PodSpec.InitContainers = []corev1.Container{{
		Name: commonconsts.RuntimeContainerName, Image: "sidecar", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
		Env: []corev1.EnvVar{{Name: "DYN_LPX_TOPOLOGY_ROLE", Value: "publisher"}},
	}}

	agent := renderTestPodSpec()
	agent.Containers[0].Env = []corev1.EnvVar{{Name: "LPX_DEPLOYMENT", Value: "stale"}, {Name: "RDMA_PORT", Value: "19877"}}
	agent.InitContainers = []corev1.Container{
		{Name: "setup", Image: "setup"},
		{
			Name: commonconsts.RuntimeContainerName, Image: "sidecar", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways),
			Env: []corev1.EnvVar{{Name: "DYN_LPX_TOPOLOGY_ROLE", Value: "consumer"}, {Name: "DYN_SYSTEM_PORT", Value: "18090"}},
		},
	}
	return projection, pcs, RenderInput{
		Stages:             map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: agent}},
		TopologyDeployment: "dgd-uid",
	}
}

func TestRenderHybridTopologyRoles(t *testing.T) {
	t.Parallel()

	t.Log("Render a hybrid workload whose Agent and Cyborg templates declare the runtime sidecar")
	projection, pcs, input := topologyRenderInput(t)
	rendered, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, input)
	require.NoError(t, err)
	agent, cyborg := namedClique(t, rendered, testAgentTemplateName), namedClique(t, rendered, "cond")

	t.Log("Only the Agent clique is selected as the topology publisher")
	require.Equal(t, "true", agent.Labels[TopologyPublisherLabel])
	require.NotContains(t, cyborg.Labels, TopologyPublisherLabel)

	t.Log("Every Agent container receives the deployment identity; main and runtime replace authored topology values")
	deployment := corev1.EnvVar{Name: "LPX_DEPLOYMENT", Value: "dgd-uid"}
	modelPath := corev1.EnvVar{Name: "LPX_MODEL_PATH", Value: "/models/model-build"}
	noLocalPartitions := corev1.EnvVar{Name: "LPX_LOCAL_PARTITION_IDS", Value: ""}
	podIndex := &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.labels['grove.io/podclique-pod-index']"}}
	require.Equal(t, []corev1.EnvVar{
		noLocalPartitions,
		modelPath,
		deployment,
		{Name: "LPX_TOPOLOGY_SOCKET", Value: "/run/lpx-topology/agent.sock"},
		{Name: "RDMA_PORT", Value: "19877"},
	}, agent.Spec.PodSpec.Containers[0].Env)
	require.Equal(t, []corev1.EnvVar{deployment, modelPath}, agent.Spec.PodSpec.InitContainers[0].Env)
	require.Equal(t, []corev1.EnvVar{
		deployment,
		{Name: "DYN_LPX_TOPOLOGY_ROLE", Value: "publisher"},
		{Name: "DYN_LPX_TOPOLOGY_AGENT_SOCKET", Value: "/run/lpx-topology/agent.sock"},
		{Name: "DYN_SIDECAR_NODE_RANK", ValueFrom: podIndex},
		modelPath,
		{Name: "DYN_SYSTEM_PORT", Value: "18090"},
	}, agent.Spec.PodSpec.InitContainers[1].Env)

	t.Log("Cyborg main reads the local topology API that its runtime sidecar serves")
	require.Equal(t, []corev1.EnvVar{
		{Name: "LPX_ALLOCATION", Value: testAgentTemplateName},
		noLocalPartitions,
		modelPath,
		deployment,
		{Name: "CYBORG_LPU_TOPOLOGY_SOCKET", Value: "/run/lpx-topology/topology.sock"},
		{Name: "GBUILD_MANIFEST_PATH", Value: "/models/model-build/manifest.v2.capnp.bin"},
	}, cyborg.Spec.PodSpec.Containers[0].Env)
	require.Equal(t, []corev1.EnvVar{
		deployment,
		{Name: "DYN_LPX_TOPOLOGY_ROLE", Value: "consumer"},
		{Name: "DYN_LPX_TOPOLOGY_SOCKET", Value: "/run/lpx-topology/topology.sock"},
		modelPath,
	}, cyborg.Spec.PodSpec.InitContainers[0].Env)

	t.Log("Both roles share sockets through one emptyDir mounted in main and runtime only")
	volume := corev1.Volume{Name: "lpx-topology", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}}
	mount := corev1.VolumeMount{Name: "lpx-topology", MountPath: "/run/lpx-topology"}
	for _, pod := range []corev1.PodSpec{agent.Spec.PodSpec, cyborg.Spec.PodSpec} {
		require.Contains(t, pod.Volumes, volume)
		require.Contains(t, pod.Containers[0].VolumeMounts, mount)
		runtime := pod.InitContainers[len(pod.InitContainers)-1]
		require.Equal(t, commonconsts.RuntimeContainerName, runtime.Name)
		require.Equal(t, []corev1.VolumeMount{mount}, runtime.VolumeMounts)
	}
	require.Empty(t, agent.Spec.PodSpec.InitContainers[0].VolumeMounts)
}

func TestRenderHybridWithoutTopologyLeavesRolesUnchanged(t *testing.T) {
	t.Parallel()

	t.Log("Render the same sidecar templates without a topology deployment")
	projection, pcs, input := topologyRenderInput(t)
	input.TopologyDeployment = ""
	rendered, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, input)
	require.NoError(t, err)

	t.Log("No role receives the publisher label or socket volume")
	agent, cyborg := namedClique(t, rendered, testAgentTemplateName), namedClique(t, rendered, "cond")
	for _, clique := range []*grovev1alpha1.PodCliqueTemplateSpec{agent, cyborg} {
		require.NotContains(t, clique.Labels, TopologyPublisherLabel)
		for _, volume := range clique.Spec.PodSpec.Volumes {
			require.NotEqual(t, "lpx-topology", volume.Name)
		}
	}

	t.Log("Authored environment stays as written behind the model bindings")
	modelPath := corev1.EnvVar{Name: "LPX_MODEL_PATH", Value: "/models/model-build"}
	noLocalPartitions := corev1.EnvVar{Name: "LPX_LOCAL_PARTITION_IDS", Value: ""}
	require.Equal(t, []corev1.EnvVar{noLocalPartitions, modelPath, {Name: "LPX_DEPLOYMENT", Value: "stale"}, {Name: "RDMA_PORT", Value: "19877"}}, agent.Spec.PodSpec.Containers[0].Env)
	require.Equal(t, []corev1.EnvVar{modelPath, {Name: "DYN_LPX_TOPOLOGY_ROLE", Value: "consumer"}, {Name: "DYN_SYSTEM_PORT", Value: "18090"}}, agent.Spec.PodSpec.InitContainers[1].Env)
	require.Equal(t, []corev1.EnvVar{
		{Name: "LPX_ALLOCATION", Value: testAgentTemplateName},
		noLocalPartitions,
		modelPath,
		{Name: "GBUILD_MANIFEST_PATH", Value: "/models/model-build/manifest.v2.capnp.bin"},
		{Name: "CYBORG_LPU_TOPOLOGY_SOCKET", Value: "/authored.sock"},
	}, cyborg.Spec.PodSpec.Containers[0].Env)
}

func TestRenderHybridTopologyRejectsAuthoredSocketVolume(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name      string
		author    func(agent *corev1.PodSpec, cyborg *corev1.PodSpec)
		wantError string
	}{
		{
			name: "Agent volume",
			author: func(agent *corev1.PodSpec, _ *corev1.PodSpec) {
				agent.Volumes = append(agent.Volumes, corev1.Volume{Name: "lpx-topology"})
			},
			wantError: `stage lpu: pod template already defines volume "lpx-topology", which LPX topology discovery adds`,
		},
		{
			name: "Agent main mount",
			author: func(agent *corev1.PodSpec, _ *corev1.PodSpec) {
				agent.Containers[0].VolumeMounts = append(agent.Containers[0].VolumeMounts, corev1.VolumeMount{Name: "sockets", MountPath: "/run/lpx-topology"})
			},
			wantError: `stage lpu: container "main" already mounts /run/lpx-topology, which LPX topology discovery owns`,
		},
		{
			name: "Cyborg runtime mount",
			author: func(_ *corev1.PodSpec, cyborg *corev1.PodSpec) {
				cyborg.InitContainers[0].VolumeMounts = []corev1.VolumeMount{{Name: "sockets", MountPath: "/run/lpx-topology/"}}
			},
			wantError: `stage lpu conductor: container "runtime" already mounts /run/lpx-topology, which LPX topology discovery owns`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Author a conflicting socket volume or mount")
			projection, pcs, input := topologyRenderInput(t)
			agent := input.Stages[testRenderComponentName]
			test.author(&agent.Spec, &namedClique(t, pcs, "cond").Spec.PodSpec)
			input.Stages[testRenderComponentName] = agent

			t.Log("Reject the template instead of sharing an authored path")
			_, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, input)
			require.EqualError(t, err, test.wantError)
		})
	}
}
