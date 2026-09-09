/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"os/exec"
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
	require.NoError(t, wrapCyborgDecodeForSwaBatch(container, 1))

	t.Log("Verify batch one leaves the container unchanged")
	require.Equal(t, original, container)
}

func TestWrapCyborgDecodeForSwaBatch(t *testing.T) {
	t.Parallel()

	t.Log("Define computed and explicitly overridden SWA cache ID scenarios")
	tests := []struct {
		name string
		env  []string
		want string
	}{
		{
			name: "computes replica-local range with global offset",
			env: []string{
				"CYBORG_FPGA_GPI_REPLICA_INDEX=3",
				"CYBORG_BATCH_SIZE=2",
			},
			want: "6,7",
		},
		{
			name: "keeps explicit override",
			env: []string{
				"CYBORG_FPGA_GPI_REPLICA_INDEX=3",
				"CYBORG_BATCH_SIZE=2",
				"CYBORG_SWA_CACHE_IDS=100,101",
			},
			want: "100,101",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Wrap the actual command and preserve literal argument boundaries")
			command := []string{"/bin/sh", "-c"}
			args := []string{`printf '%s\n' "${CYBORG_SWA_CACHE_IDS}" "$@"`, "--", "argument with spaces", "literal '$HOME'", ""}
			container := &corev1.Container{Command: command, Args: args}
			require.NoError(t, wrapCyborgDecodeForSwaBatch(container, 2))

			t.Log("Changing caller-owned slices must not change the wrapped invocation")
			command[0] = "mutated-command"
			args[2] = "mutated-argument"
			cmd := exec.Command(container.Command[0], append(container.Command[1:], container.Args...)...)
			cmd.Env = tt.env

			t.Log("Execute the wrapper and verify cache IDs and forwarded argument bytes")
			out, err := cmd.Output()
			require.NoError(t, err)
			require.Equal(t, tt.want+"\nargument with spaces\nliteral '$HOME'\n\n", string(out))
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
