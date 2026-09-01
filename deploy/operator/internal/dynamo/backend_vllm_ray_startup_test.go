// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestRayWorkerWaitsForHead(t *testing.T) {
	for _, tt := range []struct {
		name      string
		deployer  MultinodeDeployer
		env       []string
		hostname  string
		failures  string
		attempts  string
		wantError bool
	}{
		{"LWS delayed head", &LWSMultinodeDeployer{}, []string{"LWS_LEADER_ADDRESS=head.test"}, "head.test", "2", "3", false},
		{"Grove ready head", &GroveMultinodeDeployer{}, []string{"GROVE_PCSG_NAME=group", "GROVE_PCSG_INDEX=0", "GROVE_HEADLESS_SERVICE=headless"}, "group-0-test-service-ldr-0.headless", "0", "1", false},
		{"head never starts", &LWSMultinodeDeployer{}, []string{"LWS_LEADER_ADDRESS=head.test"}, "head.test", "150", "150", true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Render the production worker command with provider environment references")
			container := &corev1.Container{}
			injectRayDistributedLaunchFlags(container, RoleWorker, "test-service", tt.deployer)

			t.Log("Control readiness failures and record whether Ray is launched")
			dir := t.TempDir()
			scripts := map[string]string{
				"python3": `#!/bin/sh
[ "$LEADER_HOST" = "$EXPECTED_HOST" ] || exit 2
n=0
[ ! -f "$TEST_DIR/attempts" ] || read -r n < "$TEST_DIR/attempts"
n=$((n+1))
printf '%s' "$n" > "$TEST_DIR/attempts"
[ "$n" -gt "$FAILURES" ]
`,
				"sleep": "#!/bin/sh\nexit 0\n",
				"ray":   "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$TEST_DIR/ray-args\"\n",
			}
			for name, script := range scripts {
				require.NoError(t, os.WriteFile(filepath.Join(dir, name), []byte(script), 0o700))
			}

			t.Log("Expand Kubernetes environment references before executing the shell")
			for _, env := range tt.env {
				name, value, _ := strings.Cut(env, "=")
				container.Args[0] = strings.ReplaceAll(container.Args[0], "$("+name+")", value)
			}

			t.Log("Execute the real shell command and check retry and join behavior")
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, container.Command[0], append(container.Command[1:], container.Args...)...)
			cmd.Env = append(os.Environ(), "PATH="+dir+":"+os.Getenv("PATH"), "TEST_DIR="+dir, "EXPECTED_HOST="+tt.hostname, "FAILURES="+tt.failures)
			cmd.Env = append(cmd.Env, tt.env...)
			output, err := cmd.CombinedOutput()
			require.NoError(t, ctx.Err(), string(output))
			if tt.wantError {
				require.Error(t, err, string(output))
				require.Contains(t, string(output), "Ray head did not become reachable after 150 attempts")
				_, statErr := os.Stat(filepath.Join(dir, "ray-args"))
				require.ErrorIs(t, statErr, os.ErrNotExist)
			} else {
				require.NoError(t, err, string(output))
				args, readErr := os.ReadFile(filepath.Join(dir, "ray-args"))
				require.NoError(t, readErr)
				require.Equal(t, "start\n--address="+tt.hostname+":"+VLLMPort+"\n--block\n", string(args))
			}
			attempts, err := os.ReadFile(filepath.Join(dir, "attempts"))
			require.NoError(t, err)
			require.Equal(t, tt.attempts, string(attempts))
		})
	}
}
