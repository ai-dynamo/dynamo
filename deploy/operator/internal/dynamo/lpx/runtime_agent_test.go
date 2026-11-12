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

func TestRuntimePreservesAuthoredStartup(t *testing.T) {
	t.Parallel()

	for _, role := range []struct {
		name           string
		family         BuildFamily
		devicesPerNode int
		conductor      bool
	}{
		{name: "XT conductor", family: BuildFamilyXT, conductor: true},
		{name: "HX conductor", family: BuildFamilyHX, conductor: true},
		{name: "XT worker", family: BuildFamilyXT, devicesPerNode: 8},
		{name: "HX worker", family: BuildFamilyHX, devicesPerNode: 16},
	} {
		for _, startup := range []struct {
			name    string
			command []string
			args    []string
		}{
			{name: "image defaults"},
			{name: "image ENTRYPOINT with args", args: []string{"--instance-model-name", "custom-model", "--agent-env-vars=KEEP=1"}},
			{name: "explicit command with image arguments", command: []string{"/custom-launcher"}},
			{name: "explicit command", command: []string{"/custom-launcher", "wrapper-option"}, args: []string{"argument with spaces", "literal $HOME", ""}},
		} {
			t.Run(role.name+"/"+startup.name, func(t *testing.T) {
				t.Parallel()

				t.Log("Author startup, health, lifecycle and initialization independently of LPX binary names")
				pod := corev1.PodSpec{Containers: []corev1.Container{
					{
						Name: "sidecar", Image: "helper-runtime", Command: []string{"/custom-helper"}, Args: []string{"observe"},
						Env: []corev1.EnvVar{{Name: "HELPER_SETTING", Value: "retained"}}, ReadinessProbe: testExecProbe("helper-ready"),
						SecurityContext: &corev1.SecurityContext{RunAsUser: new(int64(1000)), RunAsNonRoot: new(true)},
					},
					{
						Name: "main", Image: "custom-runtime", Command: startup.command, Args: startup.args,
						Env: []corev1.EnvVar{{Name: "LPX_ALLOCATION", Value: "forged-allocation"}},
						EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{
							Name: "user-environment",
						}}},
					},
				}}
				if startup.args != nil {
					pod.Containers[1].SecurityContext = &corev1.SecurityContext{
						RunAsUser: new(int64(1000)), RunAsGroup: new(int64(2000)),
						RunAsNonRoot: new(true), ReadOnlyRootFilesystem: new(true),
					}
				}
				if startup.command == nil && startup.args != nil {
					pod.Volumes = []corev1.Volume{
						{Name: "credentials", Projected: &corev1.ProjectedVolumeSource{
							DefaultMode: new(int32(0400)), Sources: []corev1.VolumeProjection{{Secret: &corev1.SecretProjection{
								Name:  "custom.ssh",
								Items: []corev1.KeyToPath{{Key: "private.key", Path: "id"}},
							}}},
						}},
						{Name: "scratch", Secret: &corev1.SecretVolumeSource{SecretName: "readonly"}},
					}
					pod.Containers[1].VolumeMounts = []corev1.VolumeMount{
						{Name: "credentials", MountPath: "/custom/ssh", ReadOnly: true},
						{Name: "scratch", MountPath: "/tmp", ReadOnly: true},
					}
				}
				if startup.command != nil {
					pod.Containers[1].Env = append(pod.Containers[1].Env,
						corev1.EnvVar{Name: "TEMPLATE_SECRET", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{
							Name: "settings", Key: "value",
						}}},
					)
					pod.Containers[1].StartupProbe = testExecProbe("custom-started")
					pod.Containers[1].ReadinessProbe = testExecProbe("custom-ready")
					pod.Containers[1].LivenessProbe = testExecProbe("custom-live")
				}
				pod.Containers[1].Lifecycle = &corev1.Lifecycle{
					PreStop: &corev1.LifecycleHandler{Exec: &corev1.ExecAction{Command: []string{"custom-stop"}}},
				}
				pod.InitContainers = []corev1.Container{
					{Name: "prepare-ssh-key", Image: "custom-key-image", Command: []string{"/custom-key"}, Args: []string{"literal arg"}},
					{Name: "prepare-lp30", Image: "custom-init-image", Command: []string{"/custom-init"}},
				}
				before := pod.DeepCopy()

				t.Log("Apply runtime bindings for the selected family and role")
				if role.conductor {
					configureNodeLocalConductorRuntime(&pod, "allocation")
				} else {
					configureAgentScheduling(&pod, role.family, role.devicesPerNode)
				}

				t.Log("Retain the sidecar and bind only the authored main runtime")
				require.Len(t, pod.Containers, 2)
				require.Equal(t, before.Containers[0], pod.Containers[0])
				container, authored := pod.Containers[1], before.Containers[1]
				require.Equal(t, authored.Command, container.Command)
				require.Equal(t, authored.Args, container.Args)
				require.Equal(t, authored.Name, container.Name)
				if role.conductor {
					require.Equal(t, "allocation", testContainerEnvValue(container.Env, "LPX_ALLOCATION"))
				}
				require.Equal(t, authored.SecurityContext, container.SecurityContext)

				t.Log("Keep static environment values template-owned, including intentional omission")
				if role.conductor {
					container.Env = slices.DeleteFunc(slices.Clone(container.Env), func(variable corev1.EnvVar) bool { return variable.Name == "LPX_ALLOCATION" })
					authored.Env = slices.DeleteFunc(slices.Clone(authored.Env), func(variable corev1.EnvVar) bool { return variable.Name == "LPX_ALLOCATION" })
				}
				require.Equal(t, authored.Env, container.Env)
				require.Equal(t, authored.EnvFrom, container.EnvFrom)
				require.Equal(t, authored.StartupProbe, container.StartupProbe)
				require.Equal(t, authored.ReadinessProbe, container.ReadinessProbe)
				require.Equal(t, authored.LivenessProbe, container.LivenessProbe)
				require.Equal(t, authored.Lifecycle, container.Lifecycle)
				require.Equal(t, before.InitContainers, pod.InitContainers)

				t.Log("Keep authored storage, including omitted mounts and read-only projected credentials")
				require.Equal(t, before.Volumes, pod.Volumes)
				require.Equal(t, authored.VolumeMounts, container.VolumeMounts)
			})
		}
	}
}

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
			t.Log("Author environment references before any stale allocation bindings")
			authored := []corev1.EnvVar{
				{Name: "NOVA_ALLOCATION", Value: "$(LPX_ALLOCATION)"},
				{Name: "OTHER", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
			}
			pod := corev1.PodSpec{Containers: []corev1.Container{{
				Name: commonconsts.MainContainerName, Command: []string{"custom-conductor"}, Args: []string{"--workers", "$(LPX_ALLOCATION)"},
				Env: append(slices.Clone(authored), test.bindings...),
			}}}
			want := pod.Containers[0].DeepCopy()
			want.Env = append([]corev1.EnvVar{{Name: allocationEnvVar, Value: "agt0:agt1"}}, authored...)

			t.Log("Publish exactly one authoritative allocation before references without changing startup or other environment")
			configureNodeLocalConductorRuntime(&pod, "agt0:agt1")
			require.Equal(t, *want, pod.Containers[0])
		})
	}
}

func TestModelPathsPrecedeAuthoredReferences(t *testing.T) {
	for _, test := range []struct {
		name     string
		family   BuildFamily
		pipeline Pipeline
		local    bool
	}{
		{name: "XT Single GCS", family: BuildFamilyXT, pipeline: PipelineSingle},
		{name: "HX Single local", family: BuildFamilyHX, pipeline: PipelineSingle, local: true},
		{name: "XT hybrid local", family: BuildFamilyXT, pipeline: PipelineHybrid, local: true},
		{name: "HX hybrid GCS", family: BuildFamilyHX, pipeline: PipelineHybrid},
		{name: "XT multiple drafts local", family: BuildFamilyXT, pipeline: PipelineSpecDecode, local: true},
		{name: "HX multiple drafts GCS", family: BuildFamilyHX, pipeline: PipelineSpecDecode},
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
					pipeline: test.pipeline, runtimeBuildRef: buildID,
					configuredBuild: Build{family: test.family, path: path},
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
			wantContainer.Env = append(slices.Clone(want), authored...)

			t.Log("Include regular and init containers across Pods, including containers without model-path references")
			containerLists := [][]corev1.Container{
				{container},
				{{Name: "prepare", Env: []corev1.EnvVar{{Name: "KEEP", Value: "unchanged"}}}},
				{{Name: "metrics"}},
				{*container.DeepCopy()},
			}
			wantLists := [][]corev1.Container{
				{*wantContainer},
				{{Name: "prepare", Env: append(slices.Clone(want), corev1.EnvVar{Name: "KEEP", Value: "unchanged"})}},
				{{Name: "metrics", Env: slices.Clone(want)}},
				{*wantContainer.DeepCopy()},
			}

			t.Log("Publish only the first draft and final target once in every container, before all authored references")
			require.NoError(t, applyModelPaths(containerLists, projections, "/custom/models"))
			require.Equal(t, wantLists, containerLists)

			t.Log("Repeated rendering keeps environment order, startup and authoritative values unchanged")
			require.NoError(t, applyModelPaths(containerLists, projections, "/custom/models"))
			require.Equal(t, wantLists, containerLists)
		})
	}
}

func TestModelPathsRejectInvalidReferences(t *testing.T) {
	for _, test := range []struct {
		pipeline Pipeline
		name     string
	}{
		{pipeline: PipelineSingle, name: "LPX_MODEL_PATH"},
		{pipeline: PipelineHybrid, name: "LPX_MODEL_PATH"},
		{pipeline: PipelineSpecDecode, name: "LPX_TARGET_MODEL_PATH"},
	} {
		t.Run(string(test.pipeline)+"/"+test.name, func(t *testing.T) {
			t.Log("Keep the target reference invalid after a valid speculative draft")
			projections := []*ModelProjection{{pipeline: test.pipeline, configuredBuild: Build{path: "gs://registry/model"}}}
			if test.pipeline == PipelineSpecDecode {
				projections = append(projections, &ModelProjection{pipeline: test.pipeline})
			}
			projections[len(projections)-1].configuredBuild.path = "gs://registry/../outside"
			containerLists := [][]corev1.Container{
				{{Env: []corev1.EnvVar{{Name: "KEEP", Value: "unchanged"}}}},
				{{Env: []corev1.EnvVar{{Name: "INIT", Value: "unchanged"}}}},
			}
			before := [][]corev1.Container{{*containerLists[0][0].DeepCopy()}, {*containerLists[1][0].DeepCopy()}}

			t.Log("Report the failing binding without changing any container's environment")
			err := applyModelPaths(containerLists, projections, "/custom/models")
			require.ErrorContains(t, err, "resolve "+test.name)
			require.ErrorContains(t, err, "bad path segment")
			require.Equal(t, before, containerLists)
		})
	}
}

func testExecProbe(command string) *corev1.Probe {
	return &corev1.Probe{
		Exec: &corev1.ExecAction{Command: []string{command}},
	}
}

func testContainerEnvValue(env []corev1.EnvVar, name string) string {
	index := slices.IndexFunc(env, func(value corev1.EnvVar) bool { return value.Name == name })
	if index < 0 {
		return ""
	}
	return env[index].Value
}
