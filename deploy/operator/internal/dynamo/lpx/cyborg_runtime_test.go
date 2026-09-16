/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestWrapCyborgDecodeForSwaBatchSkipsBatchOne(t *testing.T) {
	t.Parallel()

	t.Log("Construct a single-item Cyborg container entrypoint")
	container := &corev1.Container{
		Command: []string{"/usr/local/bin/dynamo_main"},
		Args:    []string{"--model", "test"},
	}
	original := container.DeepCopy()

	t.Log("Apply the batch-one runtime contract")
	wrapCyborgStartup(container, 1, "")

	t.Log("Verify batch one leaves the container unchanged")
	require.Equal(t, original, container)
}

func TestWrapCyborgStartupPreservesArgumentVector(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name       string
		batchSize  int
		configFile string
		wantFlags  []string
	}{
		{"hosts", 1, "lpu_servers", []string{"--expand-hosts", "--"}},
		{"batch", 2, "", []string{"--swa-batch-ids", "--"}},
		{"hosts and batch", 2, "lpu_servers", []string{"--expand-hosts", "--swa-batch-ids", "--"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Select image helper options while preserving a caller-owned shell and literal arguments")
			command := []string{"/bin/sh", "-c"}
			args := []string{`printf '%s\n' "$@"`, "--", "argument with spaces", "literal '$HOME'", ""}
			container := &corev1.Container{Command: command, Args: args}
			want := append(append(append([]string{}, test.wantFlags...), command...), args...)
			wrapCyborgStartup(container, test.batchSize, test.configFile)

			t.Log("Keep the wrapped invocation independent of subsequent caller slice mutations")
			command[0], args[2] = "mutated-command", "mutated-argument"
			require.Equal(t, []string{"/usr/local/bin/cyborg-entrypoint"}, container.Command)
			require.Equal(t, want, container.Args)
		})
	}
}

func TestConfiguredCyborgBatchSizeRejectsInvalidEnvironment(t *testing.T) {
	t.Log("Define invalid literal and field-sourced Cyborg batch-size variables")
	tests := []corev1.EnvVar{
		{Name: CyborgBatchSizeEnv, Value: "0"},
		{Name: CyborgBatchSizeEnv, Value: "invalid"},
		{Name: CyborgBatchSizeEnv, ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
	}
	for _, variable := range tests {
		t.Logf("Reject invalid Cyborg batch-size environment %+v", variable)
		_, err := configuredCyborgBatchSize(&corev1.Container{Env: []corev1.EnvVar{variable}}, 1)
		require.Error(t, err)
	}
}

func TestPinnedCyborgLauncher(t *testing.T) {
	t.Log("Preserve the linked deployment's exact image and complete bash argument")
	container := &corev1.Container{Image: pinnedCyborgImage, Command: []string{"/bin/bash", "-lc"}, Args: []string{pinnedCyborgLauncher}}
	wrapCyborgStartup(container, 2, "lpu_servers")
	require.Equal(t, pinnedCyborgImage, container.Image)
	require.Equal(t, []string{"/bin/bash", "-lc", pinnedCyborgLauncher}, container.Args[3:])
	require.Equal(t, []string{"--expand-hosts", "--swa-batch-ids", "--"}, container.Args[:3])
}
