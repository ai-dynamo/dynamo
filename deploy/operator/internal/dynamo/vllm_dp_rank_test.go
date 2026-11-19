// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/stretchr/testify/require"
)

func TestVLLMDataParallelWorkerStartRankShell(t *testing.T) {
	t.Log("Use a stub engine to capture the argv delivered by the complete worker launch")
	binDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(binDir, "python3"), []byte("#!/bin/sh\nprintf '%s\\000' \"$@\"\n"), 0o755))

	// Resolve only standalone address references; rank arithmetic must reach the shell intact.
	addresses := strings.NewReplacer(
		"$(LWS_LEADER_ADDRESS)", "leader.svc",
		"$(GROVE_PCSG_NAME)", "pcs",
		"$(GROVE_PCSG_INDEX)", "0",
		"$(GROVE_HEADLESS_SERVICE)", "svc",
	)

	tests := []struct {
		name     string
		deployer MultinodeDeployer
		rankEnv  string
		tp       int64
		want     string
		wantHost string
	}{
		{name: "LWS first worker", deployer: &LWSMultinodeDeployer{}, rankEnv: "LWS_WORKER_INDEX=1", tp: 1, want: "8", wantHost: "leader.svc"},
		{name: "LWS second worker", deployer: &LWSMultinodeDeployer{}, rankEnv: "LWS_WORKER_INDEX=2", tp: 1, want: "16", wantHost: "leader.svc"},
		{name: "LWS TP2", deployer: &LWSMultinodeDeployer{}, rankEnv: "LWS_WORKER_INDEX=1", tp: 2, want: "4", wantHost: "leader.svc"},
		{name: "LWS one local rank", deployer: &LWSMultinodeDeployer{}, rankEnv: "LWS_WORKER_INDEX=1", tp: 8, want: "1", wantHost: "leader.svc"},
		{name: "Grove first worker", deployer: &GroveMultinodeDeployer{}, rankEnv: "GROVE_PCLQ_POD_INDEX=0", tp: 1, want: "8", wantHost: "pcs-0-worker-ldr-0.svc"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Render a multinode DP worker through the backend entry point")
			container := vllmContainer(tensorParallelSizeFlag, strconv.FormatInt(tt.tp, 10), dataParallelSizeFlag, strconv.FormatInt(24/tt.tp, 10))
			originalArgs := append([]string(nil), container.Args...)
			component := betaComponent(t, &v1alpha1.DynamoComponentDeploymentSharedSpec{})
			require.NoError(t, (&VLLMBackend{}).UpdateContainer(container, 3, RoleWorker, component, "worker", tt.deployer, staticContainerGPUCount(8)))
			require.Equal(t, []string{"sh", "-c"}, container.Command)
			require.Len(t, container.Args, 1)

			t.Log("Execute the complete generated command, resolving only its standalone leader address")
			script := addresses.Replace(container.Args[0])
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, container.Command[0], append(container.Command[1:], script)...)
			cmd.Env = append(os.Environ(), "PATH="+binDir+string(os.PathListSeparator)+os.Getenv("PATH"), tt.rankEnv)
			out, err := cmd.CombinedOutput()
			require.NoError(t, err, "worker launch failed: %s\n%s", script, out)

			t.Log("Check all original and injected engine arguments, including the evaluated start rank")
			got := strings.Split(strings.TrimSuffix(string(out), "\x00"), "\x00")
			want := append(originalArgs,
				"--data-parallel-hybrid-lb",
				dataParallelSizeLocalFlag, strconv.FormatInt(8/tt.tp, 10),
				"--data-parallel-start-rank", tt.want,
				"--data-parallel-address", tt.wantHost,
				"--data-parallel-rpc-port", dataParallelRPCPort,
			)
			require.Equal(t, want, got)
		})
	}
}
