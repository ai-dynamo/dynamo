/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"text/template"
	"time"
)

// TestOutputCopierScript runs the rendered sidecar script as a real process. It must be
// bash: dash rejects `set -o pipefail`, which would exit early and pass this test vacuously.
func TestOutputCopierScript(t *testing.T) {
	// Programs the script calls; `command`, `echo` and `[` are bash builtins.
	utilities := []string{"bash", "date", "grep", "awk", "sed", "tr", "cat", "sleep"}

	const scriptTimeout = 15 * time.Second

	tests := []struct {
		name        string
		withKubectl bool
		wantExitOK  bool
		wantStdout  string
		wantStderr  string
	}{
		{
			name:        "kubectl missing",
			withKubectl: false,
			wantExitOK:  false,
			wantStderr:  "kubectl",
		},
		{
			name:        "kubectl available",
			withKubectl: true,
			wantExitOK:  true,
			wantStdout:  "Saved profiling output to ConfigMap",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a private PATH holding only the utilities the sidecar image is required to provide")
			binDir := t.TempDir()
			for _, utility := range utilities {
				resolved, err := exec.LookPath(utility)
				if err != nil {
					t.Skipf("the sidecar script needs %q, which is not on PATH: %v", utility, err)
				}
				if err := os.Symlink(resolved, filepath.Join(binDir, utility)); err != nil {
					t.Fatalf("linking %q into the private PATH: %v", utility, err)
				}
			}
			bashPath := filepath.Join(binDir, "bash")

			if tt.withKubectl {
				// Reports the profiler container as terminated for `get` and accepts every
				// `apply`, which is all the script asks of kubectl on the success path.
				stub := "#!" + bashPath + "\n" +
					"if [ \"$1\" = \"get\" ]; then echo '{\"terminated\":{\"exitCode\":0}}'; fi\n" +
					"exit 0\n"
				if err := os.WriteFile(filepath.Join(binDir, "kubectl"), []byte(stub), 0o755); err != nil {
					t.Fatalf("writing the kubectl stub: %v", err)
				}
			}

			t.Log("Lay down the profiler output the script reads on the success path")
			outputDir := t.TempDir()
			statusFile := "status: success\nphase: Done\nmessage: profiling complete\n"
			if err := os.WriteFile(filepath.Join(outputDir, "profiler_status.yaml"), []byte(statusFile), 0o644); err != nil {
				t.Fatalf("writing profiler_status.yaml: %v", err)
			}
			finalConfig := "apiVersion: nvidia.com/v1alpha1\nkind: DynamoGraphDeployment\n"
			if err := os.WriteFile(filepath.Join(outputDir, ProfilingOutputFile), []byte(finalConfig), 0o644); err != nil {
				t.Fatalf("writing %s: %v", ProfilingOutputFile, err)
			}

			t.Log("Render the sidecar script through the same text/template path the controller uses")
			tmpl, err := template.New("sidecar").Parse(sidecarScriptTemplate)
			if err != nil {
				t.Fatalf("parsing the sidecar script template: %v", err)
			}
			var script bytes.Buffer
			if err := tmpl.Execute(&script, map[string]string{
				"OutputPath":    outputDir,
				"OutputFile":    ProfilingOutputFile,
				"ConfigMapName": "dgdr-output-test",
				"Namespace":     "test-namespace",
				"DGDRName":      "test-dgdr",
				"DGDRuid":       "8f0c4d5e-1b2a-4c3d-9e8f-0a1b2c3d4e5f",
			}); err != nil {
				t.Fatalf("executing the sidecar script template: %v", err)
			}

			t.Log("Redirect the script's scratch files out of the shared temporary directory")
			// The sidecar owns /tmp inside its own container, but here the script runs on
			// the host, where those fixed paths race with concurrent `go test` processes
			// and clobber unrelated files. Give each subtest its own directory instead.
			// Only these exact paths are rewritten; the template also renders OutputPath,
			// which is itself a t.TempDir() under the shared directory and must survive.
			scratchDir := t.TempDir()
			rendered := script.String()
			for _, scratchPath := range []string{"/tmp/progress.yaml", "/tmp/cm.yaml"} {
				if !strings.Contains(rendered, scratchPath) {
					t.Fatalf("the rendered sidecar script no longer writes %s; point this redirect at the path it uses now, so the script keeps out of the shared temporary directory", scratchPath)
				}
				rendered = strings.ReplaceAll(rendered, scratchPath, filepath.Join(scratchDir, filepath.Base(scratchPath)))
			}

			t.Log("Run the rendered script as a real process and observe how it terminates")
			ctx, cancel := context.WithTimeout(context.Background(), scriptTimeout)
			defer cancel()

			cmd := exec.CommandContext(ctx, bashPath, "-c", rendered)
			cmd.Env = []string{"PATH=" + binDir, "HOSTNAME=profile-test-pod"}
			// Killing the shell does not close the pipes its `sleep` child inherited, so
			// bound the wait instead of blocking on that child.
			cmd.WaitDelay = 5 * time.Second
			var stdout, stderr bytes.Buffer
			cmd.Stdout = &stdout
			cmd.Stderr = &stderr
			runErr := cmd.Run()

			if ctx.Err() != nil {
				t.Fatalf("the sidecar script never exited within %s; it is stuck polling.\nstdout:\n%s\nstderr:\n%s",
					scriptTimeout, stdout.String(), stderr.String())
			}

			var exitErr *exec.ExitError
			switch {
			case tt.wantExitOK && runErr != nil:
				t.Fatalf("expected the sidecar script to exit 0, got %v\nstdout:\n%s\nstderr:\n%s",
					runErr, stdout.String(), stderr.String())
			case !tt.wantExitOK && runErr == nil:
				t.Fatalf("expected the sidecar script to exit non-zero, but it exited 0\nstdout:\n%s\nstderr:\n%s",
					stdout.String(), stderr.String())
			case !tt.wantExitOK && !errors.As(runErr, &exitErr):
				t.Fatalf("expected a non-zero exit status from the sidecar script, got %v", runErr)
			}

			if tt.wantStdout != "" && !strings.Contains(stdout.String(), tt.wantStdout) {
				t.Errorf("expected standard output to contain %q\nstdout:\n%s", tt.wantStdout, stdout.String())
			}
			if tt.wantStderr != "" && !strings.Contains(stderr.String(), tt.wantStderr) {
				t.Errorf("expected standard error to contain %q\nstderr:\n%s", tt.wantStderr, stderr.String())
			}
		})
	}
}
