/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

func TestRuntimePreservesAuthoredStartup(t *testing.T) {
	t.Parallel()

	for _, role := range []struct {
		name      string
		family    BuildFamily
		conductor bool
		direct    bool
	}{
		{name: "XT conductor", family: BuildFamilyXT, conductor: true},
		{name: "HX conductor", family: BuildFamilyHX, conductor: true},
		{name: "XT worker", family: BuildFamilyXT},
		{name: "HX worker", family: BuildFamilyHX},
		{name: "XT direct hybrid", family: BuildFamilyXT, direct: true},
		{name: "HX direct hybrid", family: BuildFamilyHX, direct: true},
	} {
		for _, startup := range []struct {
			name    string
			command []string
			args    []string
		}{
			{name: "image defaults"},
			{name: "image ENTRYPOINT with args", args: []string{"--instance-model-name", "custom-model", "--agent-env-vars=KEEP=1"}},
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
						corev1.EnvVar{Name: "GROQ_V2_RESET_ON_OPEN", Value: "template-value"},
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
				configureAgentScheduling(&pod, role.family)
				switch {
				case role.conductor:
					configureNodeLocalConductorRuntime(&pod, role.family, "allocation")
				case role.direct:
					configureDirectHybridAgentRuntime(&pod, "config")
				default:
					configureNodeLocalAgentRuntime(&pod, role.family)
				}

				t.Log("Retain the sidecar and bind only the authored main runtime")
				require.Len(t, pod.Containers, 2)
				require.Equal(t, before.Containers[0], pod.Containers[0])
				container, authored := pod.Containers[1], before.Containers[1]
				require.Equal(t, authored.Command, container.Command)
				require.Equal(t, authored.Args, container.Args)
				if role.conductor {
					require.Equal(t, "conductor", container.Name)
					require.Equal(t, "allocation", testContainerEnvValue(container.Env, "LPX_ALLOCATION"))
				} else {
					require.Equal(t, lpuAgentContainerName, container.Name)
				}
				if role.conductor || role.family == BuildFamilyHX && !role.direct {
					require.Equal(t, authored.SecurityContext, container.SecurityContext)
				}

				t.Log("Keep static environment values template-owned, including intentional omission")
				dynamicEnv := map[string]bool{
					"POD_IP": true, "LPX_ALLOCATION": role.conductor,
					"TOPOLOGIES": role.direct, "GAS_DIR": role.direct,
				}
				container.Env = slices.DeleteFunc(slices.Clone(container.Env), func(variable corev1.EnvVar) bool { return dynamicEnv[variable.Name] })
				if role.conductor {
					authored.Env = slices.DeleteFunc(slices.Clone(authored.Env), func(variable corev1.EnvVar) bool { return variable.Name == "LPX_ALLOCATION" })
				}
				require.Equal(t, authored.Env, container.Env)
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
