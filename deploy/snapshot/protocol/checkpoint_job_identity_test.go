// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestCudaCheckpointLaunchJobWrapperPersistsJobFile(t *testing.T) {
	tempDir := t.TempDir()
	transientJobFile := filepath.Join(tempDir, "transient-job")
	stableJobFile := filepath.Join(tempDir, "checkpoint", CUDAJobFileName)
	observedEnvironment := filepath.Join(tempDir, "observed-environment")
	if err := os.WriteFile(transientJobFile, []byte("job-state"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(stableJobFile), 0700); err != nil {
		t.Fatal(err)
	}

	_, args := wrapWithCudaCheckpointLaunchJob(
		[]string{"/bin/sh", "-c"},
		[]string{`printf '%s' "$CUDA_CHECKPOINT_JOB_FILE" > "$1"`, "workload", observedEnvironment},
	)
	args[5] = stableJobFile
	cmd := exec.Command(args[1], args[2:]...)
	cmd.Env = append(os.Environ(), "CUDA_CHECKPOINT_JOB_FILE="+transientJobFile)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("launch-job wrapper failed: %v (output: %s)", err, output)
	}

	content, err := os.ReadFile(stableJobFile)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "job-state" {
		t.Fatalf("stable job file = %q", content)
	}
	info, err := os.Stat(stableJobFile)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0600 {
		t.Fatalf("stable job file mode = %o, want 600", got)
	}
	observed, err := os.ReadFile(observedEnvironment)
	if err != nil {
		t.Fatal(err)
	}
	if string(observed) != stableJobFile {
		t.Fatalf("workload observed %q, want %q", observed, stableJobFile)
	}
}

func TestCudaCheckpointLaunchJobWrapperReportsMissingJobFile(t *testing.T) {
	_, args := wrapWithCudaCheckpointLaunchJob([]string{"/bin/true"}, nil)
	cmd := exec.Command(args[1], args[2:]...)
	cmd.Env = []string{"PATH=" + os.Getenv("PATH")}
	output, err := cmd.CombinedOutput()
	if err == nil {
		t.Fatal("launch-job wrapper succeeded without CUDA_CHECKPOINT_JOB_FILE")
	}
	if got := string(output); !strings.Contains(got, "requires NVIDIA driver 610 or newer") {
		t.Fatalf("launch-job wrapper output = %q", got)
	}
}
