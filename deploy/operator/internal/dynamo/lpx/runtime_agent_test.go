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
	"k8s.io/utils/ptr"
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
						SecurityContext: &corev1.SecurityContext{RunAsUser: ptr.To(int64(1000)), RunAsNonRoot: ptr.To(true)},
					},
					{
						Name: "main", Image: "custom-runtime", Command: startup.command, Args: startup.args,
						Env: []corev1.EnvVar{{Name: "LPX_ALLOCATION", Value: "forged-allocation"}},
						EnvFrom: []corev1.EnvFromSource{{ConfigMapRef: &corev1.ConfigMapEnvSource{
							LocalObjectReference: corev1.LocalObjectReference{Name: "user-environment"},
						}}},
					},
				}}
				if startup.args != nil {
					pod.Containers[1].SecurityContext = &corev1.SecurityContext{
						RunAsUser: ptr.To(int64(1000)), RunAsGroup: ptr.To(int64(2000)),
						RunAsNonRoot: ptr.To(true), ReadOnlyRootFilesystem: ptr.To(true),
					}
				}
				if startup.command == nil && startup.args != nil {
					pod.Volumes = []corev1.Volume{
						{Name: "credentials", VolumeSource: corev1.VolumeSource{Projected: &corev1.ProjectedVolumeSource{
							DefaultMode: ptr.To(int32(0400)), Sources: []corev1.VolumeProjection{{Secret: &corev1.SecretProjection{
								LocalObjectReference: corev1.LocalObjectReference{Name: "custom.ssh"},
								Items:                []corev1.KeyToPath{{Key: "private.key", Path: "id"}},
							}}},
						}}},
						{Name: "scratch", VolumeSource: corev1.VolumeSource{Secret: &corev1.SecretVolumeSource{SecretName: "readonly"}}},
					}
					pod.Containers[1].VolumeMounts = []corev1.VolumeMount{
						{Name: "credentials", MountPath: "/custom/ssh", ReadOnly: true},
						{Name: "scratch", MountPath: "/tmp", ReadOnly: true},
					}
				}
				if startup.command != nil {
					pod.Containers[1].Env = append(pod.Containers[1].Env,
						corev1.EnvVar{Name: "TEMPLATE_SECRET", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{
							LocalObjectReference: corev1.LocalObjectReference{Name: "settings"}, Key: "value",
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

func TestReferencesEnvVar(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name      string
		container corev1.Container
		want      bool
	}{
		{name: "command", container: corev1.Container{Command: []string{"/run", "$(LPX_MODEL_PATH)"}}, want: true},
		{name: "shell syntax is not a Kubernetes reference", container: corev1.Container{Args: []string{"${LPX_MODEL_PATH}"}}},
		{name: "escaped reference", container: corev1.Container{Args: []string{"$$(LPX_MODEL_PATH)"}}},
		{name: "escaped dollar before a reference", container: corev1.Container{Args: []string{"$$$(LPX_MODEL_PATH)"}}, want: true},
		{name: "reference after an escaped one", container: corev1.Container{Args: []string{"$$(LPX_MODEL_PATH):$(LPX_MODEL_PATH)"}}, want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Report only references that Kubernetes expands")
			require.Equal(t, tc.want, referencesEnvVar(&tc.container, modelPathEnvVar))
		})
	}
}

func TestConfigureServingAgent(t *testing.T) {
	t.Parallel()

	const modelStoragePath = "/nfs"
	reference := []string{"lpx", "--model-path", "$(LPX_MODEL_PATH)/tokenizer"}
	systemPort := corev1.EnvVar{Name: "DYN_SYSTEM_PORT", Value: "18090"}
	for _, tc := range []struct {
		name        string
		args        []string
		env         []corev1.EnvVar
		mounts      []corev1.VolumeMount
		mainEnv     []corev1.EnvVar
		wantEnv     []corev1.EnvVar
		wantMainEnv []corev1.EnvVar
		wantErr     string
	}{
		{
			name: "neither expands the model path, only another variable; a model-storage mount elsewhere is the sidecar's own", args: []string{"lpx", "$(LPX_DRAFT_MODEL_PATH)"}, env: []corev1.EnvVar{systemPort},
			mounts:  []corev1.VolumeMount{{Name: "model-storage", MountPath: "/models"}},
			mainEnv: []corev1.EnvVar{systemPort}, wantEnv: []corev1.EnvVar{systemPort}, wantMainEnv: []corev1.EnvVar{systemPort},
		},
		{
			name: "reference with model storage elsewhere", args: reference,
			mounts:  []corev1.VolumeMount{{Name: "model-storage", MountPath: "/models"}},
			wantErr: `"runtime" init container references $(LPX_MODEL_PATH), so it requires volume mount "model-storage" at the model-storage mount path "/nfs"`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Author an Agent whose main and runtime sidecar may expand a local model's path")
			pod := corev1.PodSpec{
				InitContainers: []corev1.Container{{Name: commonconsts.RuntimeContainerName, Args: tc.args, Env: tc.env, VolumeMounts: tc.mounts}},
				Containers:     []corev1.Container{{Name: commonconsts.MainContainerName, Env: tc.mainEnv}},
			}
			projections := []*ModelProjection{{
				pipeline: PipelineSingle, runtimeBuildRef: "model",
				configuredBuild: Build{family: BuildFamilyHX, path: "file:///operator-cache/model"},
			}}

			t.Log("Bind the model path where a container reads it beneath the Agent's mount")
			err := configureServingAgent(&pod, projections, modelStoragePath)
			if tc.wantErr != "" {
				require.EqualError(t, err, tc.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tc.wantEnv, pod.InitContainers[0].Env)
			require.Equal(t, tc.wantMainEnv, pod.Containers[0].Env)
		})
	}
}

func testExecProbe(command string) *corev1.Probe {
	return &corev1.Probe{
		ProbeHandler: corev1.ProbeHandler{Exec: &corev1.ExecAction{Command: []string{command}}},
	}
}

func testContainerEnvValue(env []corev1.EnvVar, name string) string {
	index := slices.IndexFunc(env, func(value corev1.EnvVar) bool { return value.Name == name })
	if index < 0 {
		return ""
	}
	return env[index].Value
}
