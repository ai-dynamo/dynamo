/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
)

func TestHasSupportedRestoreEntrypoint(t *testing.T) {
	tests := []struct {
		name      string
		command   []string
		args      []string
		supported bool
	}{
		{
			name:      "vLLM module in command",
			command:   []string{"python3", "-m", "dynamo.vllm"},
			supported: true,
		},
		{
			name:      "SGLang module split across command and args",
			command:   []string{"python"},
			args:      []string{"-m", "dynamo.sglang", "--model", "test"},
			supported: true,
		},
		{
			name:      "TensorRT-LLM module with versioned Python path",
			command:   []string{"/usr/bin/python3.11", "-m", "dynamo.trtllm"},
			supported: true,
		},
		{
			name:      "vLLM module after operand-free interpreter flags",
			command:   []string{"python3", "-u", "-O", "-m", "dynamo.vllm"},
			supported: true,
		},
		{
			name:      "custom Python module",
			command:   []string{"python3", "-m", "autovllm.dynamo_server"},
			supported: true,
		},
		{
			name:    "no command",
			command: nil,
		},
		{
			name:    "shell wrapper",
			command: []string{"/bin/sh", "-c"},
			args:    []string{"python3 -m dynamo.vllm"},
		},
		{
			name:    "custom wrapper with module arguments",
			command: []string{"serve-model"},
			args:    []string{"-m", "dynamo.vllm"},
		},
		{
			name:    "python without -m",
			command: []string{"python3", "serve.py"},
		},
		{
			name:    "operand-taking interpreter option before -m",
			command: []string{"python3", "-X", "dev", "-m", "dynamo.vllm"},
		},
		{
			name:    "empty module name",
			command: []string{"python3", "-m", ""},
		},
		{
			name:    "option-like module name",
			command: []string{"python3", "-m", "-c"},
		},
		{
			name:    "missing module name",
			command: []string{"python3", "-m"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Given a restore destination with an explicit container entrypoint")
			container := &corev1.Container{Command: test.command, Args: test.args}

			t.Log("Then any direct python -m module is accepted and every other launcher is rejected")
			assert.Equal(t, test.supported, HasSupportedRestoreEntrypoint(container))
		})
	}
}
