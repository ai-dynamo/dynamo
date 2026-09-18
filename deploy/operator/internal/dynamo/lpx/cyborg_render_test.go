/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"strings"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/utils/ptr"
)

func TestRenderHybridBoundsActualGPUHostnames(t *testing.T) {
	t.Parallel()

	t.Log("Project a hybrid engine with the longest PCS name and maximum scheduling replica count")
	fixture := newV3CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	projection := projectRenderFixture(t, lpxv1alpha1.TargetFamilyHx16x8x2x3, PipelineLPX, acquireTestSnapshot(t, writeCompilerFixture(t, fixture)))
	projection.stage = testRenderComponentName
	projection.configuredBuild.IOFPGACount = 1
	projection.configuredBuild.IOFanoutFactor = 1
	workload := &SelectedWorkload{
		modelProjections:     []*ModelProjection{projection},
		scalingGroupReplicas: 2496,
	}
	pcsName := strings.Repeat("a", MaxPodCliqueSetNameLength)
	plan, err := workload.PlanNodeLocalMaterialization(pcsName)
	require.NoError(t, err)

	t.Log("Validate actual GPU widths rather than assuming the largest int32 pod index")
	for _, test := range []struct {
		name      string
		replicas  int32
		wantError bool
	}{
		{name: "one GPU per engine", replicas: 1},
		{name: "hostname at DNS limit", replicas: 100_000_000},
		{name: "hostname over DNS limit", replicas: 100_000_001, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Render the GPU width beneath the same readable PCS name")
			pcs := renderTestPCS(true)
			pcs.Name = pcsName
			cyborg := namedClique(t, pcs, "cond")
			cyborg.Spec.Replicas = test.replicas
			cyborg.Spec.MinAvailable = ptr.To(test.replicas)
			_, err := RenderSelectedNodeLocal(pcs, workload, plan, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}},
			})
			if test.wantError {
				require.ErrorContains(t, err, "materialized Cyborg Pod hostname")
				require.ErrorContains(t, err, "must be no more than 63 characters")
				return
			}
			require.NoError(t, err)
			hostname := materializedPodHostname(plan.ForReplica(plan.Replicas-1).CyborgClique, int(test.replicas)-1)
			require.Empty(t, validation.IsDNS1123Label(hostname))
			if test.replicas > 1 {
				require.Len(t, hostname, validation.DNS1123LabelMaxLength)
			}
		})
	}
}

func TestRenderHybridPreservesRuntimeEnvironment(t *testing.T) {
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
	decode := namedClique(t, pcs, "cond")
	decode.Spec.Replicas = 4
	decode.Spec.MinAvailable = ptr.To[int32](4)
	decode.Spec.PodSpec.ResourceClaims = nil
	decode.Spec.PodSpec.Containers[0].Resources.Limits = corev1.ResourceList{
		corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
	}
	decode.Spec.PodSpec.Containers[0].Command = []string{"/custom-cyborg", "--wrapper-option"}
	decode.Spec.PodSpec.Containers[0].Args = []string{"argument with spaces", "literal $HOME", ""}
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
	decode.Spec.PodSpec.Containers[0].Env = authoredEnv

	t.Log("Use independently provisioned Cyborg model storage at the shared runtime path")
	decode.Spec.PodSpec.Volumes[0].PersistentVolumeClaim.ClaimName = "cyborg-models"
	decode.Spec.PodSpec.Volumes[0].PersistentVolumeClaim.ReadOnly = true
	decode.Spec.PodSpec.Containers[0].VolumeMounts[1].SubPath = "cyborg"
	decode.Spec.PodSpec.Containers[0].VolumeMounts[1].ReadOnly = true
	input := RenderInput{
		Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}},
	}
	rendered, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, input)
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
	imageEntrypointPCS := renderTestPCS(true)
	imageEntrypointCyborg := namedClique(t, imageEntrypointPCS, "cond")
	imageEntrypointCyborg.Spec.Replicas = 4
	imageEntrypointCyborg.Spec.MinAvailable = ptr.To[int32](4)
	imageEntrypointCyborg.Spec.PodSpec.Containers[0].Args = []string{"serve"}
	imageEntrypointEnv := append([]corev1.EnvVar{manifestEnv}, imageEntrypointCyborg.Spec.PodSpec.Containers[0].Env...)

	t.Log("Leave the image ENTRYPOINT selected when command is omitted")
	input.Stages = map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}}
	_, err = renderSelectedForTest(imageEntrypointPCS, []*ModelProjection{projection}, input)
	require.NoError(t, err)
	require.Nil(t, imageEntrypointCyborg.Spec.PodSpec.Containers[0].Command)
	require.Equal(t, []string{"serve"}, imageEntrypointCyborg.Spec.PodSpec.Containers[0].Args)
	require.Equal(t, imageEntrypointEnv, imageEntrypointCyborg.Spec.PodSpec.Containers[0].Env)

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
			pcs := renderTestPCS(true)
			cyborg := namedClique(t, pcs, "cond")
			cyborg.Spec.Replicas = test.replicas
			cyborg.Spec.MinAvailable = ptr.To(test.replicas)
			cyborg.Spec.PodSpec.Containers[0].Command = []string{"/usr/local/bin/dynamo_main"}
			cyborg.Spec.PodSpec.Containers[0].VolumeMounts[1].MountPath = test.mountPath
			_, err := renderSelectedForTest(pcs, []*ModelProjection{projection}, RenderInput{
				Stages: map[string]corev1.PodTemplateSpec{testRenderComponentName: {Spec: renderTestPodSpec()}},
			})
			require.ErrorContains(t, err, test.wantError)
		})
	}
}
