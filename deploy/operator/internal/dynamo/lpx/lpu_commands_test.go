/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestLPUPartitionMetadataDerivesConfigRowAndRank(t *testing.T) {
	t.Parallel()

	t.Log("Resolve the Bash interpreter used by the embedded metadata command")
	bash, err := exec.LookPath("bash")
	require.NoError(t, err)

	t.Log("Define single-model, SpecDecode, and invalid pod-index metadata fixtures")
	tests := []struct {
		name      string
		files     map[string]string
		podIndex  string
		model     string
		want      string
		wantError string
	}{
		{
			name: "single model",
			files: map[string]string{
				"nodes_per_partition":    "2\n1\n",
				"partition_node_offsets": "0\n2\n",
			},
			podIndex: "1",
			want:     "0,0,1,2,0,0,1",
		},
		{
			name: "specdecode model with reset offsets",
			files: map[string]string{
				"nodes_per_partition":    "2\n2\n2\n2\n",
				"partition_node_offsets": "0\n2\n0\n2\n",
				"partition_models":       "draft0\ndraft0\ntarget\ntarget\n",
				"partition_indices":      "0\n1\n0\n1\n",
			},
			podIndex: "1",
			model:    "target",
			want:     "2,0,1,2,0,0,1",
		},
		{
			name: "rejects out of range pod index",
			files: map[string]string{
				"nodes_per_partition":    "2\n1\n",
				"partition_node_offsets": "0\n2\n",
			},
			podIndex:  "3",
			wantError: "matched 0 partition rows",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Write the node-local partition metadata files")
			configDir := t.TempDir()
			for name, contents := range tt.files {
				require.NoError(t, os.WriteFile(filepath.Join(configDir, name), []byte(contents), 0o600))
			}

			t.Log("Execute the embedded metadata and environment commands")
			command := lpuPartitionMetadataCommand + "\n" + lpuPartitionEnvironmentCommand + `
printf '%s,%s,%s,%s,%s,%s,%s' "${CONFIG_PARTITION_INDEX}" "${LOGICAL_PARTITION_INDEX}" "${PARTITION_RANK}" "${NODE_COUNT}" "${NODE_OFFSET}" "${PARTITION_ID}" "${RANK_IN_PARTITION}"`
			cmd := exec.Command(bash, "-c", command)
			cmd.Env = []string{
				"LPU_CONFIG_DIR=" + configDir,
				"GROVE_PCLQ_POD_INDEX=" + tt.podIndex,
				"LPU_MODEL_NAME=" + tt.model,
			}
			out, err := cmd.CombinedOutput()
			if tt.wantError != "" {
				t.Log("Verify invalid partition metadata is rejected")
				require.Error(t, err)
				require.Contains(t, string(out), tt.wantError)
				return
			}

			t.Log("Verify the derived configuration row, logical partition, and rank")
			require.NoError(t, err, string(out))
			require.Equal(t, tt.want, string(out))
		})
	}
}

func TestLPXSSHSetupReusesHostKeys(t *testing.T) {
	t.Parallel()

	t.Log("Extract the direct-hybrid SSH setup alongside the embedded worker setup")
	_, partitionSetup, found := strings.Cut(lpuPartitionRunCommandSource, "setup_ssh() {\n")
	require.True(t, found)
	partitionSetup, _, found = strings.Cut(partitionSetup, "\n}")
	require.True(t, found)
	for name, setup := range map[string]string{"worker": sshSetupCommands, "direct hybrid": partitionSetup} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			t.Log("Isolate the SSH Secret and writable home without changing the process home")
			dir := t.TempDir()
			for _, name := range []string{"private.key", "private.key.pub"} {
				require.NoError(t, os.WriteFile(filepath.Join(dir, name), []byte("secret"), 0o600))
			}
			setup = strings.NewReplacer("${HOME}", dir, "$HOME", dir, "/ssh-pk", dir).Replace(setup)
			command := `ssh-keygen() {
	while [ "$1" != -f ]; do shift; done
	[ ! -e "$2" ] || return 1
	printf 'host key' > "$2"
}
` + setup

			t.Log("Generate fresh keys, reuse them on restart, and fill a missing key")
			for attempt := 0; attempt < 3; attempt++ {
				if attempt == 2 {
					require.NoError(t, os.Remove(filepath.Join(dir, ".ssh/host_keys/ssh_host_ecdsa_key")))
				}
				cmd := exec.Command("bash", "-ec", command)
				out, err := cmd.CombinedOutput()
				require.NoError(t, err, "attempt %d: %s", attempt, out)
				for _, keyType := range []string{"rsa", "ecdsa", "ed25519"} {
					key, err := os.ReadFile(filepath.Join(dir, ".ssh/host_keys/ssh_host_"+keyType+"_key"))
					require.NoError(t, err)
					require.Equal(t, "host key", string(key))
				}
			}
		})
	}
}

func TestLPUCommandDigests(t *testing.T) {
	t.Parallel()

	t.Log("Define reviewed SHA-256 identities for assembled runtime commands")
	tests := []struct {
		name    string
		command string
		want    string
	}{
		{name: "partition", command: lpuPartitionRunCommand, want: "e9af66afa3dbcdd589582d3576100cbdb72ae131d647919ef6bd11269d1462dd"},
		{name: "worker", command: lpuWorkerRunCommand, want: "ac7e7dfcf47c1a81bb1e80b7f91f323cc00ceb11bb85e0833fc248334e2b5dd0"},
		{name: "partition worker", command: lpuPartitionWorkerRunCommand, want: "f280af0e42b7c229317251cded7aa9d71ef53dd661e631c520b457e0d04cb3a8"},
		{name: "probe", command: lpuV2ProbeCommand, want: "96ac74ae5e413a21fe98077f4d3d5b1dba7a741064e2734ae822ad77ab30e854"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			t.Log("Bind the assembled embedded command to its reviewed SHA-256 digest")
			require.Equal(t, test.want, fmt.Sprintf("%x", sha256.Sum256([]byte(test.command))))
		})
	}
}
