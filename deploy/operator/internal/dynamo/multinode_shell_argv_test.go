/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestMultinodeShellWrappersPreserveArgv(t *testing.T) {
	t.Log("Stub python3 to print its argv NUL-delimited and ray to succeed, so the rendered wrapper can run locally")
	binDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(binDir, "python3"), []byte("#!/bin/sh\nprintf '%s\\000' \"$@\"\n"), 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(binDir, "ray"), []byte("#!/bin/sh\nexit 0\n"), 0o755))
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("GROVE_PCLQ_POD_INDEX", "0")

	// Every value here is legal in a container args list and must survive the
	// shell verbatim: a single quote, an empty string, whitespace, JSON,
	// characters the shell would otherwise expand, and a kubelet $(VAR) reference.
	userArgs := []string{
		"--model", "models/it's a model", tensorParallelSizeFlag, "8",
		"--served-model-name", "",
		"--chat-template", `{{ 'hello' }}`,
		"--kv-transfer-config", `{"kv_connector": "NixlConnector", "kv_role": "$ROLE"}`,
		"--literal", `glob*?[x]`,
		"--whitespace", "tab\tand\nnewline",
		"--download-dir", "$(MODEL_CACHE)",
	}
	kubelet := strings.NewReplacer("$(MODEL_CACHE)", "/cache/models", "$(GROVE_PCSG_NAME)", "pcs", "$(GROVE_PCSG_INDEX)", "0", "$(GROVE_HEADLESS_SERVICE)", "svc")

	tests := []struct {
		name     string
		backend  Backend
		module   string
		role     Role
		executor string
	}{
		{name: "vllm ray leader", backend: &VLLMBackend{}, module: "dynamo.vllm", role: RoleLeader, executor: "ray"},
		{name: "vllm mp grove worker", backend: &VLLMBackend{}, module: "dynamo.vllm", role: RoleWorker, executor: "mp"},
		{name: "sglang grove worker", backend: &SGLangBackend{}, module: "dynamo.sglang", role: RoleWorker},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Render the multinode wrapper for an argv-form python command")
			container := &corev1.Container{
				Command: []string{"python3", "-m", tt.module},
				Args:    append([]string(nil), userArgs...),
			}
			spec := &v1alpha1.DynamoComponentDeploymentSharedSpec{}
			if tt.executor != "" {
				spec.Annotations = map[string]string{commonconsts.KubeAnnotationVLLMDistributedExecutorBackend: tt.executor}
			}
			require.NoError(t, tt.backend.UpdateContainer(container, 2, tt.role, betaComponent(t, spec), "worker", &GroveMultinodeDeployer{}, staticContainerGPUCount(4)))
			require.Equal(t, "-c", container.Command[len(container.Command)-1], "expected a sh -c wrapper, got %v", container.Command)

			t.Log("Stand in for kubelet $(VAR) expansion, then execute the wrapper through the shell it targets")
			script := kubelet.Replace(container.Args[0])
			out, err := exec.Command(container.Command[0], append(container.Command[1:], script)...).CombinedOutput()
			require.NoError(t, err, "shell wrapper failed:\n%s\n%s", script, out)
			got := strings.Split(strings.TrimSuffix(string(out), "\x00"), "\x00")

			t.Log("The user's argv, after kubelet expansion, must lead the engine command unchanged, followed by the injected multinode flags")
			want := []string{"-m", tt.module}
			for _, arg := range userArgs {
				want = append(want, kubelet.Replace(arg))
			}
			require.GreaterOrEqual(t, len(got), len(want), "engine argv: %q", got)
			require.Equal(t, want, got[:len(want)])
			if tt.role == RoleWorker {
				require.Contains(t, got[len(want):], "1", "injected node rank should evaluate to 1, got %q", got[len(want):])
			}
		})
	}
}
