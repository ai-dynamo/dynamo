// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cuda

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

func TestStageJobFile(t *testing.T) {
	sourceRoot := t.TempDir()
	checkpointDir := t.TempDir()
	jobFile := snapshotprotocol.CUDAJobFilePath
	if err := os.MkdirAll(filepath.Join(sourceRoot, filepath.Dir(jobFile)), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(sourceRoot, jobFile), []byte("job-state"), 0600); err != nil {
		t.Fatal(err)
	}

	helperJobFile, err := StageJobFile(sourceRoot, checkpointDir, 2)
	if err != nil {
		t.Fatalf("StageJobFile() error = %v", err)
	}
	wantHelperJobFile := filepath.Join(sourceRoot, jobFile)
	if helperJobFile != wantHelperJobFile {
		t.Fatalf("StageJobFile() = %q, want %q", helperJobFile, wantHelperJobFile)
	}
	artifact := filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName)
	content, err := os.ReadFile(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "job-state" {
		t.Fatalf("staged content = %q", content)
	}
}

func TestStageJobFileRejectsSymlink(t *testing.T) {
	sourceRoot := t.TempDir()
	checkpointDir := t.TempDir()
	jobFile := snapshotprotocol.CUDAJobFilePath
	if err := os.MkdirAll(filepath.Join(sourceRoot, filepath.Dir(jobFile)), 0700); err != nil {
		t.Fatal(err)
	}
	secret := filepath.Join(sourceRoot, "secret")
	if err := os.WriteFile(secret, []byte("must-not-copy"), 0600); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(sourceRoot, jobFile)
	if err := os.Symlink(filepath.Join("..", "secret"), source); err != nil {
		t.Fatal(err)
	}

	_, err := StageJobFile(sourceRoot, checkpointDir, 1)
	if err == nil {
		t.Fatal("expected symlink source to be rejected")
	}
	if _, statErr := os.Stat(filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName)); !os.IsNotExist(statErr) {
		t.Fatalf("staged file exists after rejected symlink: %v", statErr)
	}
}

func TestRefreshJobFileArtifactCapturesPostCheckpointState(t *testing.T) {
	checkpointDir := t.TempDir()
	live := filepath.Join(t.TempDir(), snapshotprotocol.CUDAJobFileName)
	artifact := filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName)
	if err := os.WriteFile(live, []byte("pre-checkpoint-state"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(artifact, []byte("validation-copy"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(live, []byte("post-checkpoint-job-state"), 0600); err != nil {
		t.Fatal(err)
	}

	if err := refreshJobFileArtifact(live, checkpointDir); err != nil {
		t.Fatalf("refreshJobFileArtifact() error = %v", err)
	}
	content, err := os.ReadFile(artifact)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "post-checkpoint-job-state" {
		t.Fatalf("artifact content = %q", content)
	}
}

func TestStageJobFileRequiresLaunchJobStateForMultiGPU(t *testing.T) {
	sourceRoot := t.TempDir()

	jobFile, err := StageJobFile(sourceRoot, t.TempDir(), 1)
	if err != nil || jobFile != "" {
		t.Fatalf("legacy single-GPU StageJobFile() = %q, %v", jobFile, err)
	}
	_, err = StageJobFile(sourceRoot, t.TempDir(), 2)
	if err == nil || !strings.Contains(err.Error(), "source must be launched under cuda-checkpoint --launch-job") {
		t.Fatalf("expected missing multi-GPU launch-job error, got %v", err)
	}
}

func TestCheckpointProcessTreePersistsStateAfterEveryProcessCheckpoint(t *testing.T) {
	tempDir := t.TempDir()
	trace := filepath.Join(tempDir, "trace")
	helper := filepath.Join(tempDir, "cuda-checkpoint-helper")
	script := `#!/bin/sh
action=""
pid=""
job_file=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --action) action="$2"; shift 2 ;;
        --pid) pid="$2"; shift 2 ;;
        --job-file) job_file="$2"; shift 2 ;;
        *) shift ;;
    esac
done
if [ "$job_file" != "$DYNAMO_TEST_JOB_FILE" ]; then
    printf 'job file = %s, want %s\n' "$job_file" "$DYNAMO_TEST_JOB_FILE" >&2
    exit 1
fi
printf '%s %s\n' "$action" "$pid" >> "$DYNAMO_TEST_TRACE"
if [ "$action" = checkpoint ]; then printf '|%s' "$pid" >> "$job_file"; fi
`
	if err := os.WriteFile(helper, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	originalHelper := cudaCheckpointHelperBinary
	cudaCheckpointHelperBinary = helper
	t.Cleanup(func() { cudaCheckpointHelperBinary = originalHelper })

	liveJobFile := filepath.Join(tempDir, "live-job")
	checkpointDir := filepath.Join(tempDir, "checkpoint")
	if err := os.Mkdir(checkpointDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(liveJobFile, []byte("initial"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName), []byte("validation-copy"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("DYNAMO_TEST_TRACE", trace)
	t.Setenv("DYNAMO_TEST_JOB_FILE", liveJobFile)

	if _, err := CheckpointProcessTree(context.Background(), []int{101, 202}, liveJobFile, checkpointDir, logr.Discard()); err != nil {
		t.Fatalf("CheckpointProcessTree() error = %v", err)
	}
	traceContent, err := os.ReadFile(trace)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(traceContent), "lock 101\nlock 202\ncheckpoint 101\ncheckpoint 202\n"; got != want {
		t.Fatalf("helper call order = %q, want %q", got, want)
	}
	artifact, err := os.ReadFile(filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(artifact), "initial|101|202"; got != want {
		t.Fatalf("persisted job state = %q, want %q", got, want)
	}
}

func TestSetLiveJobFileOwner(t *testing.T) {
	jobFile := filepath.Join(t.TempDir(), "job-file")
	if err := os.WriteFile(jobFile, []byte("job-state"), 0600); err != nil {
		t.Fatal(err)
	}

	wantUID, wantGID := os.Getuid(), os.Getgid()
	if os.Geteuid() == 0 {
		wantUID, wantGID = 1234, 2345
	}
	if err := SetLiveJobFileOwner(jobFile, wantUID, wantGID); err != nil {
		t.Fatalf("SetLiveJobFileOwner() error = %v", err)
	}

	var stat unix.Stat_t
	if err := unix.Stat(jobFile, &stat); err != nil {
		t.Fatal(err)
	}
	if gotUID, gotGID := int(stat.Uid), int(stat.Gid); gotUID != wantUID || gotGID != wantGID {
		t.Fatalf("job file ownership = %d:%d, want %d:%d", gotUID, gotGID, wantUID, wantGID)
	}
}

func TestSetLiveJobFileOwnerRejectsSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	if err := os.WriteFile(target, []byte("job-state"), 0600); err != nil {
		t.Fatal(err)
	}
	jobFile := filepath.Join(dir, "job-file")
	if err := os.Symlink(target, jobFile); err != nil {
		t.Fatal(err)
	}

	if err := SetLiveJobFileOwner(jobFile, os.Getuid(), os.Getgid()); err == nil {
		t.Fatal("SetLiveJobFileOwner() accepted a symlink")
	}
}

func TestJobFileFromCheckpointRejectsSymlink(t *testing.T) {
	checkpointDir := t.TempDir()
	target := filepath.Join(checkpointDir, "target")
	if err := os.WriteFile(target, []byte("job-state"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName)); err != nil {
		t.Fatal(err)
	}

	_, err := JobFileFromCheckpoint(checkpointDir)
	if err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("expected symlink artifact to be rejected, got %v", err)
	}
}

func TestPrepareLiveJobFileReplacesMutatedState(t *testing.T) {
	staged := filepath.Join(t.TempDir(), snapshotprotocol.CUDAJobFileName)
	live := filepath.Join(t.TempDir(), snapshotprotocol.CUDAJobFileName)
	if err := os.WriteFile(staged, []byte("capture-time-state"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(live, []byte("longer-mutated-restore-state"), 0644); err != nil {
		t.Fatal(err)
	}

	if err := prepareLiveJobFile(staged, live); err != nil {
		t.Fatalf("prepareLiveJobFile() error = %v", err)
	}
	content, err := os.ReadFile(live)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "capture-time-state" {
		t.Fatalf("live content = %q", content)
	}
	info, err := os.Stat(live)
	if err != nil {
		t.Fatal(err)
	}
	if got := info.Mode().Perm(); got != 0600 {
		t.Fatalf("live mode = %o, want 600", got)
	}
}

func TestPrepareLiveJobFileKeepsRestoreTargetsIsolated(t *testing.T) {
	staged := filepath.Join(t.TempDir(), snapshotprotocol.CUDAJobFileName)
	if err := os.WriteFile(staged, []byte("capture-time-state"), 0600); err != nil {
		t.Fatal(err)
	}
	first := filepath.Join(t.TempDir(), snapshotprotocol.CUDAJobFileName)
	second := filepath.Join(t.TempDir(), snapshotprotocol.CUDAJobFileName)
	if err := prepareLiveJobFile(staged, first); err != nil {
		t.Fatal(err)
	}
	if err := prepareLiveJobFile(staged, second); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(first, []byte("first-restore-mutated-state"), 0600); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(second)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "capture-time-state" {
		t.Fatalf("second restore content changed to %q", content)
	}
}

func TestPrepareLiveJobFileRejectsSymlinkDestination(t *testing.T) {
	staged := filepath.Join(t.TempDir(), snapshotprotocol.CUDAJobFileName)
	if err := os.WriteFile(staged, []byte("capture-time-state"), 0600); err != nil {
		t.Fatal(err)
	}
	destinationDir := t.TempDir()
	target := filepath.Join(destinationDir, "target")
	if err := os.WriteFile(target, []byte("must-not-change"), 0600); err != nil {
		t.Fatal(err)
	}
	live := filepath.Join(destinationDir, snapshotprotocol.CUDAJobFileName)
	if err := os.Symlink(target, live); err != nil {
		t.Fatal(err)
	}

	if err := prepareLiveJobFile(staged, live); err == nil {
		t.Fatal("expected symlink destination to be rejected")
	}
	content, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "must-not-change" {
		t.Fatalf("symlink target changed to %q", content)
	}
}
