/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"strings"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestApplyCyborgManifestPathPrecedesAuthoredReferences(t *testing.T) {
	t.Parallel()

	t.Log("Define authored bindings that depend on the generated manifest location")
	projection := &Model{component: &component{configuredBuild: Build{path: "file:///models/build"}}}
	authored := []corev1.EnvVar{
		{Name: "MODEL_PATH", Value: "$(GBUILD_MANIFEST_PATH)"},
		{Name: "OTHER", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
	}
	want := append([]corev1.EnvVar{{Name: gbuildManifestPathEnv, Value: "/models/build/manifest.v2.capnp.bin"}}, authored...)
	for _, test := range []struct {
		name string
		env  []corev1.EnvVar
	}{
		{name: "missing binding", env: authored},
		{name: "stale binding after reference", env: append(append([]corev1.EnvVar(nil), authored...), corev1.EnvVar{
			Name: gbuildManifestPathEnv, Value: "/stale/manifest",
		})},
		{name: "existing binding before reference", env: want},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Publish the authoritative path before references while preserving other bindings")
			container := corev1.Container{Env: test.env}
			require.NoError(t, applyCyborgManifestPath(&container, projection, "/models"))
			require.Equal(t, want, container.Env)

			t.Log("Repeat rendering without duplicating or reordering environment bindings")
			require.NoError(t, applyCyborgManifestPath(&container, projection, "/models"))
			require.Equal(t, want, container.Env)
		})
	}
}

func TestWorkloadBoundsActualGPUHostnames(t *testing.T) {
	t.Parallel()

	t.Log("Project a hybrid workload at the combined name limit and maximum scheduling replica count")
	fixture := newV3CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	projection := projectTestModel(t, acquireTestSnapshot(t, writeCompilerFixture(t, fixture)), PipelineHybrid)
	projection.component.configuredBuild.ioFPGACount = 1
	projection.component.configuredBuild.ioFanoutFactor = 1
	pcsName := strings.Repeat("a", 28) // target + target-cond consume the remaining Grove budget.
	lastReplica := int32(maxWorkloadReplicas - 1)

	t.Log("Bound the rendered Cyborg width at the authored scaling-group count")
	for _, test := range []struct {
		name      string
		width     int32
		wantError bool
	}{
		{name: "one GPU per engine", width: 1},
		{name: "hostname at DNS limit", width: 100_000_000},
		{name: "hostname over DNS limit", width: 100_000_001, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Name the workload using its authored capacity and Cyborg width")
			workload := &Workload{name: projection.component.name, models: []*Model{projection}, scalingGroupReplicas: maxWorkloadReplicas, conductorReplicas: test.width}
			err := workload.nameResources(pcsName, "target")
			if test.wantError {
				require.ErrorContains(t, err, "materialized Cyborg Pod hostname")
				require.ErrorContains(t, err, "must be no more than 63 characters")
				return
			}
			require.NoError(t, err)
			hostname := podHostname(workload.ConductorCliqueName(lastReplica), int(test.width)-1)
			require.Empty(t, validation.IsDNS1123Label(hostname))
			if test.width > 1 {
				require.Len(t, hostname, validation.DNS1123LabelMaxLength)
			}
		})
	}

	t.Log("Keep external capacity validation independent of Cyborg hostnames after initial naming")
	workload := &Workload{name: projection.component.name, models: []*Model{projection}, scalingGroupReplicas: 1, conductorReplicas: 100_000_001}
	require.NoError(t, workload.nameResources(pcsName, "target"))
	require.NoError(t, workload.ValidateReplicas(maxWorkloadReplicas))
	require.NoError(t, workload.ValidateCyborgReplicas(100_000_001))
}
