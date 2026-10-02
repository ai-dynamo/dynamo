/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestRenderHybridPreservesRuntimeEnvironment(t *testing.T) {
	t.Parallel()

	t.Log("Project a split-I/O selected workload")
	fixture := newV3CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	fixture.numLPUNodes = 1
	fixture.partitions[0].numChips = 8
	fixture.partitions[0].devicesPerNode = 8
	normalized := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
	build := normalized
	build.compilationMode = compilationModeHybrid
	build.ioFPGACount = 2
	build.ioFanoutFactor = 2
	projectionBatch, err := projectComponent(testRenderComponentName, "model-build", normalized, PipelineHybrid, []string{"default"})
	require.NoError(t, err)
	projection := projectionBatch[0]

	t.Log("Render the Cyborg runtime contract")
	decode := renderTestCyborgTemplate()
	decode.Spec.ResourceClaims = nil
	decode.Spec.Containers[0].Resources.Limits = corev1.ResourceList{
		corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
	}
	decode.Spec.Containers[0].Command = []string{"/custom-cyborg", "--wrapper-option"}
	decode.Spec.Containers[0].Args = []string{"argument with spaces", "literal $HOME", ""}
	authoredEnv := []corev1.EnvVar{
		{Name: "RDMA_PORT", Value: "12345"},
		{Name: "CYBORG_BATCH_SIZE", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{
			FieldPath: "metadata.labels['runtime-batch']",
		}}},
		{Name: "CYBORG_FPGA_GPI_IO_FPGA_COUNT", Value: "9"},
		{Name: "CYBORG_SWA_CACHE_IDS", Value: "8,9"},
		{Name: "TOKENIZER_DIR", Value: "$(GBUILD_MANIFEST_PATH)/../tokenizer"},
		{Name: "TOTAL_REPLICAS", Value: "9"},
	}
	decode.Spec.Containers[0].Env = authoredEnv

	t.Log("Use independently provisioned Cyborg model storage at the shared runtime path")
	decode.Spec.Volumes[0].PersistentVolumeClaim.ClaimName = "cyborg-models"
	decode.Spec.Volumes[0].PersistentVolumeClaim.ReadOnly = true
	decode.Spec.Containers[0].VolumeMounts[1].SubPath = "cyborg"
	decode.Spec.Containers[0].VolumeMounts[1].ReadOnly = true
	rendered, err := renderSelectedForTest(renderTestPCS(), []*Model{projection},
		map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}}, decode, 4)
	require.NoError(t, err)
	cyborg := namedClique(t, rendered, "cond")

	t.Log("Keep runtime environment opaque while supplying its manifest before authored references")
	manifestEnv := corev1.EnvVar{
		Name: "GBUILD_MANIFEST_PATH", Value: "/models/model-build/manifest.v2.capnp.bin",
	}
	require.Equal(t, append([]corev1.EnvVar{manifestEnv}, authoredEnv...), cyborg.Spec.PodSpec.Containers[0].Env)
	require.Equal(t, &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "cyborg-models", ReadOnly: true}, cyborg.Spec.PodSpec.Volumes[0].PersistentVolumeClaim)
	require.Contains(t, cyborg.Spec.PodSpec.Containers[0].VolumeMounts, corev1.VolumeMount{
		Name: "model-storage", MountPath: "/models", SubPath: "cyborg", ReadOnly: true,
	})
	require.Equal(t, []string{"/custom-cyborg", "--wrapper-option"}, cyborg.Spec.PodSpec.Containers[0].Command)
	require.Equal(t, []string{"argument with spaces", "literal $HOME", ""}, cyborg.Spec.PodSpec.Containers[0].Args)

	t.Log("Preserve an image-owned entrypoint")
	imageEntrypoint := renderTestCyborgTemplate()
	imageEntrypoint.Spec.Containers[0].Args = []string{"serve"}
	imageEntrypointEnv := append([]corev1.EnvVar{manifestEnv}, imageEntrypoint.Spec.Containers[0].Env...)

	t.Log("Leave the image ENTRYPOINT selected when command is omitted")
	rendered, err = renderSelectedForTest(renderTestPCS(), []*Model{projection},
		map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}}, imageEntrypoint, 4)
	require.NoError(t, err)
	imageEntrypointCyborg := namedClique(t, rendered, "cond").Spec.PodSpec.Containers[0]
	require.Nil(t, imageEntrypointCyborg.Command)
	require.Equal(t, []string{"serve"}, imageEntrypointCyborg.Args)
	require.Equal(t, imageEntrypointEnv, imageEntrypointCyborg.Env)

	t.Log("Reject invalid Cyborg runtime bindings")

	for _, test := range []struct {
		name      string
		replicas  int32
		mountPath string
		wantError string
	}{
		{"incomplete endpoints", 1, "/models", "Cyborg replicas 1 must be divisible by ioFpgaCount 2"},
		{"incomplete fanout", 2, "/models", "Cyborg replicas 2 must provide fanoutFactor 2 clients"},
		{"different storage path", 4, "/other-models", "conflicts with model storage mount"},
		{"empty storage path", 4, "", "has no mount path"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Reject incompatible runtime bindings during rendering")
			cyborg := renderTestCyborgTemplate()
			cyborg.Spec.Containers[0].Command = []string{"/usr/local/bin/dynamo_main"}
			cyborg.Spec.Containers[0].VolumeMounts[1].MountPath = test.mountPath
			_, err := renderSelectedForTest(renderTestPCS(), []*Model{projection},
				map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}}, cyborg, test.replicas)
			require.ErrorContains(t, err, test.wantError)
		})
	}
}
