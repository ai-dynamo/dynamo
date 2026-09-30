/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestAllocationPrecedesAuthoredReferences(t *testing.T) {
	for _, test := range []struct {
		name     string
		bindings []corev1.EnvVar
	}{
		{name: "absent"},
		{name: "existing", bindings: []corev1.EnvVar{{Name: allocationEnvVar, Value: "stale"}}},
		{name: "duplicates", bindings: []corev1.EnvVar{
			{Name: allocationEnvVar, Value: "stale"},
			{Name: allocationEnvVar, ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Author sidecar, init, and main bindings before any stale allocation bindings")
			authored := []corev1.EnvVar{
				{Name: "NOVA_ALLOCATION", Value: "$(LPX_ALLOCATION)"},
				{Name: "OTHER", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
			}
			pod := corev1.PodSpec{
				InitContainers: []corev1.Container{{Name: "prepare", Image: "setup", Command: []string{"/custom-init"}}},
				Containers: []corev1.Container{
					{Name: "sidecar", Image: "helper", Env: []corev1.EnvVar{{Name: allocationEnvVar, Value: "sidecar-owned"}}},
					{
						Name: commonconsts.MainContainerName, Command: []string{"custom-conductor"}, Args: []string{"--workers", "$(LPX_ALLOCATION)"},
						Env: append(slices.Clone(authored), test.bindings...),
					},
				},
			}
			want := pod.DeepCopy()
			want.Containers[1].Env = append([]corev1.EnvVar{{Name: allocationEnvVar, Value: "agt0:agt1"}}, authored...)

			t.Log("Publish exactly one allocation on main without changing startup, other containers, or other environment")
			configureNodeLocalConductorRuntime(&pod, "agt0:agt1")
			require.Equal(t, *want, pod)
		})
	}
}

func TestModelPathsPrecedeAuthoredReferences(t *testing.T) {
	for _, test := range []struct {
		name     string
		pipeline Pipeline
		local    bool
	}{
		{name: "single GCS", pipeline: PipelineSingle},
		{name: "hybrid local", pipeline: PipelineHybrid, local: true},
		{name: "multiple drafts local", pipeline: PipelineSpecDecode, local: true},
		{name: "multiple drafts GCS", pipeline: PipelineSpecDecode},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve each model from the registry reference and the runtime's custom mount")
			count := 1
			names := []string{"LPX_MODEL_PATH"}
			if test.pipeline == PipelineSpecDecode {
				count = 3
				names = []string{"LPX_DRAFT_MODEL_PATH", "LPX_TARGET_MODEL_PATH"}
			}
			projections := make([]*ModelProjection, count)
			for index := range projections {
				buildID := fmt.Sprintf("model-%d", index)
				path := "gs://registry/" + buildID
				if test.local {
					path = "file:///operator-cache/" + buildID
				}
				projections[index] = &ModelProjection{
					pipeline: test.pipeline, runtimeBuildRef: buildID, configuredBuild: Build{Path: path},
				}
			}
			want := []corev1.EnvVar{}
			for index, name := range names {
				path := fmt.Sprintf("/custom/models/gcs/registry/model-%d", index*(count-1))
				if test.local {
					path = fmt.Sprintf("/custom/models/model-%d", index*(count-1))
				}
				want = append(want, corev1.EnvVar{Name: name, Value: path})
			}

			t.Log("Author dependent values before duplicate forged bindings")
			runtimeVariable := "A_RUNTIME_MODEL"
			if test.pipeline == PipelineHybrid {
				runtimeVariable = "GAS_DIR"
			}
			authored := []corev1.EnvVar{
				{Name: runtimeVariable, Value: "$(" + names[0] + ")"},
				{Name: "OTHER", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
			}
			container := corev1.Container{Command: []string{"custom-runtime"}, Args: []string{"--unchanged"}, Env: slices.Clone(authored)}
			for _, name := range names {
				container.Env = append(container.Env, corev1.EnvVar{Name: name, Value: "stale"}, corev1.EnvVar{Name: name, Value: "duplicate"})
			}
			wantContainer := container.DeepCopy()
			wantContainer.Env = append(want, authored...)

			t.Log("Publish only the first draft and final target once, before all authored references")
			require.NoError(t, applyModelPaths(&container, projections, "/custom/models"))
			require.Equal(t, *wantContainer, container)

			t.Log("Repeated rendering keeps environment order, startup and authoritative values unchanged")
			require.NoError(t, applyModelPaths(&container, projections, "/custom/models"))
			require.Equal(t, *wantContainer, container)
		})
	}
}

func TestModelPathsRejectInvalidReferences(t *testing.T) {
	for _, test := range []struct {
		pipeline Pipeline
		name     string
	}{
		{pipeline: PipelineSingle, name: "LPX_MODEL_PATH"},
		{pipeline: PipelineSpecDecode, name: "LPX_TARGET_MODEL_PATH"},
	} {
		t.Run(string(test.pipeline)+"/"+test.name, func(t *testing.T) {
			t.Log("Keep the target reference invalid after a valid speculative draft")
			projections := []*ModelProjection{{pipeline: test.pipeline, configuredBuild: Build{Path: "gs://registry/model"}}}
			if test.pipeline == PipelineSpecDecode {
				projections = append(projections, &ModelProjection{pipeline: test.pipeline})
			}
			projections[len(projections)-1].configuredBuild.Path = "gs://registry/../outside"
			container := corev1.Container{Env: []corev1.EnvVar{{Name: "KEEP", Value: "unchanged"}}}
			before := container.DeepCopy()

			t.Log("Report the failing binding without publishing a partial environment")
			err := applyModelPaths(&container, projections, "/custom/models")
			require.ErrorContains(t, err, "resolve "+test.name)
			require.ErrorContains(t, err, "bad path segment")
			require.Equal(t, *before, container)
		})
	}
}

func testContainerEnvValue(env []corev1.EnvVar, name string) string {
	index := slices.IndexFunc(env, func(value corev1.EnvVar) bool { return value.Name == name })
	if index < 0 {
		return ""
	}
	return env[index].Value
}
