/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

func TestRenderHybridProjectsManifestRuntimeIO(t *testing.T) {
	t.Parallel()

	t.Log("Project a split-I/O selected workload")
	fixture := newV3CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	fixture.numLPUNodes = 1
	fixture.partitions[0].topology = "URSA_V2_1__Q8__8C__G_106__KP_FEC__GHZ_1_0__NO_FPGA"
	fixture.partitions[0].numChips = 8
	fixture.partitions[0].devicesPerNode = 8
	normalized := normalizeTestSnapshot(t, acquireTestSnapshot(t, writeCompilerFixture(t, fixture)))
	build := normalized.build
	build.BatchSize = 4
	build.CompilationMode = BuildCompilationModeHybrid
	build.IOFPGACount = 2
	build.IOFanoutFactor = 2
	projectionBatch, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline: PipelineLPX, Models: []string{"default"},
		BuildSnapshot:   normalized,
		RuntimeBuildRef: "model-build", ModelSettings: json.RawMessage(`{"prop_sync":false}`),
	})
	require.NoError(t, err)
	projection := projectionBatch[0]

	t.Log("Render the Cyborg runtime contract")
	pcs := renderTestPCS(true)
	decode := namedClique(t, pcs, "lpu-engine-gpu")
	decode.Spec.Replicas = 4
	decode.Spec.MinAvailable = ptr.To[int32](4)
	decode.Spec.PodSpec.ResourceClaims = nil
	decode.Spec.PodSpec.Containers[0].Resources.Limits = corev1.ResourceList{
		corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
	}
	decode.Spec.PodSpec.Containers[0].Command = []string{"/usr/local/bin/dynamo_main"}
	decode.Spec.PodSpec.Containers[0].Env = append(
		decode.Spec.PodSpec.Containers[0].Env,
		corev1.EnvVar{Name: "RDMA_PORT", Value: "12345"},
		corev1.EnvVar{Name: CyborgBatchSizeEnv, Value: "3"},
	)
	input := RenderInput{
		MaterializationName: "dgd",
		Stages:              map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}}, SSHSecretName: "ssh-secret",
	}
	rendered, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, input)
	require.NoError(t, err)
	cyborg := namedClique(t, rendered, "lpu-engine-gpu")

	t.Log("Verify the rendered Cyborg runtime I/O contract")
	require.Contains(t, cyborg.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
		Name: "CYBORG_FPGA_GPI_IO_FPGA_COUNT", Value: "2",
	})
	require.Contains(t, cyborg.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
		Name: "CYBORG_BATCH_SIZE", Value: "3",
	})
	require.Contains(t, cyborg.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
		Name: "CYBORG_FPGA_GPI_REPLICA_INDEX",
		ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{
			FieldPath: "metadata.labels['grove.io/podclique-pod-index']",
		}},
	})
	require.Contains(t, cyborg.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
		Name: "GBUILD_MANIFEST_PATH", Value: "/models/model-build/manifest.v2.capnp.bin",
	})
	require.Contains(t, cyborg.Spec.PodSpec.Containers[0].Env, corev1.EnvVar{
		Name: "RDMA_PORT", Value: "19877",
	})
	require.Contains(t, cyborg.Spec.PodSpec.Volumes, renderTestPodSpec().Volumes[0])
	require.Contains(t, cyborg.Spec.PodSpec.Containers[0].VolumeMounts, renderTestPodSpec().Containers[0].VolumeMounts[0])
	require.Equal(t, []string{"/bin/sh", "-ec"}, cyborg.Spec.PodSpec.Containers[0].Command)
	require.Contains(t, cyborg.Spec.PodSpec.Containers[0].Args[0], `export CYBORG_SWA_CACHE_IDS="${ids}"`)
	require.Equal(t, []string{"--", "/configs/lpu_servers", "/tmp/lpu_servers", "/usr/local/bin/dynamo_main"}, cyborg.Spec.PodSpec.Containers[0].Args[1:])

	t.Log("Preserve an image-owned entrypoint")
	imageEntrypointPCS := renderTestPCS(true)
	imageEntrypointCyborg := namedClique(t, imageEntrypointPCS, "lpu-engine-gpu")
	imageEntrypointCyborg.Spec.Replicas = 4
	imageEntrypointCyborg.Spec.MinAvailable = ptr.To[int32](4)
	imageEntrypointCyborg.Spec.PodSpec.Containers[0].Args = []string{"serve"}
	imageEntrypointCyborg.Spec.PodSpec.Containers[0].Env = append(
		imageEntrypointCyborg.Spec.PodSpec.Containers[0].Env,
		corev1.EnvVar{Name: CyborgBatchSizeEnv, Value: "3"},
	)

	t.Log("Render the agreed Cyborg binary when the command is omitted")
	input.Stages = map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}}
	_, err = renderSelectedForTest(imageEntrypointPCS, []*ModelProjection{projection}, input)
	require.NoError(t, err)
	require.Equal(t, []string{"/usr/local/bin/cyborg", "serve"}, imageEntrypointCyborg.Spec.PodSpec.Containers[0].Args[4:])

	t.Log("Reject incomplete endpoint and fanout coverage of the same split-I/O runtime")

	for _, test := range []struct {
		name      string
		replicas  int32
		wantError string
	}{
		{"incomplete endpoints", 1, "Cyborg replicas 1 must be divisible by ioFpgaCount 2"},
		{"incomplete fanout", 2, "Cyborg replicas 2 must provide fanoutFactor 2 clients"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Reject incomplete Cyborg coverage during runtime rendering")
			pcs := renderTestPCS(true)
			cyborg := namedClique(t, pcs, "lpu-engine-gpu")
			cyborg.Spec.Replicas = test.replicas
			cyborg.Spec.MinAvailable = ptr.To(test.replicas)
			cyborg.Spec.PodSpec.Containers[0].Command = []string{"/usr/local/bin/dynamo_main"}
			_, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, RenderInput{
				MaterializationName: "dgd",
				Stages:              map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}}, SSHSecretName: "ssh-secret",
			})
			require.ErrorContains(t, err, test.wantError)
		})
	}
}
