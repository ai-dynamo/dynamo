/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestRuntimeStartup(t *testing.T) {
	for _, file := range []string{"datacenter.toml", "lpu_servers"} {
		for _, index := range []string{"0", "42", ""} {
			t.Run(file+index, func(t *testing.T) {
				t.Log("Render fixed Grove variables before an explicit command with literal arguments")
				dir := t.TempDir()
				input, output := filepath.Join(dir, "input"), filepath.Join(dir, "output")
				require.NoError(t, os.WriteFile(input, []byte("group-${GROVE_PCSG_INDEX}-agent-0\n${GROVE_PCSG_NAME}.${GROVE_HEADLESS_SERVICE}"), 0600))
				container := &corev1.Container{Command: []string{"/bin/sh", "-c"}, Args: []string{`printf '%s\n' "$@"; exit 17`, "--", "argument with spaces", "literal '$HOME'", ""}}
				wrapRuntimeStartup(container, "/bin/nova", file, "")
				container.Args[2], container.Args[3] = input, output
				cmd := exec.CommandContext(t.Context(), container.Command[0], append(container.Command[1:], container.Args...)...)
				cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "GROVE_PCSG_NAME=pcs-0-group", "GROVE_PCSG_INDEX=" + index, "GROVE_HEADLESS_SERVICE=service"}
				actual, err := cmd.CombinedOutput()
				if index == "" {
					require.Error(t, err)
					require.Contains(t, string(actual), "missing GROVE_PCSG_INDEX")
					require.NoFileExists(t, output)
					return
				}
				t.Log("Keep exit status, argument boundaries and this engine's expanded file")
				require.EqualError(t, err, "exit status 17")
				require.Equal(t, "argument with spaces\nliteral '$HOME'\n\n", string(actual))
				data, err := os.ReadFile(output)
				require.NoError(t, err)
				require.Equal(t, "group-"+index+"-agent-0\npcs-0-group.service", string(data))
			})
		}
	}
}

func TestRuntimeConfigStorage(t *testing.T) {
	for _, source := range []corev1.VolumeSource{
		{EmptyDir: &corev1.EmptyDirVolumeSource{}},
		{PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: "shared"}},
		{Secret: &corev1.SecretVolumeSource{SecretName: "readonly"}},
	} {
		t.Log("Reuse only a writable Pod-local workspace for expanded runtime configuration")
		pod := corev1.PodSpec{Volumes: []corev1.Volume{{Name: "scratch", VolumeSource: source}}}
		container := corev1.Container{VolumeMounts: []corev1.VolumeMount{{Name: "scratch", MountPath: "/tmp"}}}
		err := addRuntimeConfigStorage(&pod, &container, "lpu_servers")
		if source.EmptyDir == nil {
			require.ErrorContains(t, err, "Pod-local emptyDir")
		} else {
			require.NoError(t, err)
			require.Len(t, pod.Volumes, 1)
		}
		container.VolumeMounts[0].ReadOnly = true
		require.ErrorContains(t, addRuntimeConfigStorage(&pod, &container, "lpu_servers"), "writable storage")
	}
}
