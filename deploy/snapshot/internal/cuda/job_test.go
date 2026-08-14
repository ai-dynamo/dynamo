// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cuda

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

func TestStageJobFile(t *testing.T) {
	procRoot := t.TempDir()
	checkpointDir := t.TempDir()
	jobFile := snapshotprotocol.CUDAJobFilePath

	for _, pid := range []string{"101", "202"} {
		processRoot := filepath.Join(procRoot, pid, "root")
		if err := os.MkdirAll(filepath.Join(processRoot, filepath.Dir(jobFile)), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(procRoot, pid, "environ"), []byte("OTHER=value\x00"+JobFileEnv+"="+jobFile+"\x00"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(procRoot, "101", "root", jobFile), []byte("job-state"), 0600); err != nil {
		t.Fatal(err)
	}

	helperJobFile, err := StageJobFile(procRoot, []int{101, 202}, checkpointDir, 2)
	if err != nil {
		t.Fatalf("StageJobFile() error = %v", err)
	}
	wantHelperJobFile := filepath.Join(procRoot, "101", "root", jobFile)
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

func TestStageJobFileRejectsTransientProcPath(t *testing.T) {
	procRoot := t.TempDir()
	if err := os.MkdirAll(filepath.Join(procRoot, "101"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(procRoot, "101", "environ"), []byte(JobFileEnv+"=/proc/1/fd/3\x00"), 0600); err != nil {
		t.Fatal(err)
	}

	_, err := StageJobFile(procRoot, []int{101}, t.TempDir(), 1)
	if err == nil || !strings.Contains(err.Error(), "persisted outside procfs") {
		t.Fatalf("expected transient procfs error, got %v", err)
	}
}

func TestStageJobFileRejectsUnexpectedContainerPath(t *testing.T) {
	procRoot := t.TempDir()
	if err := os.MkdirAll(filepath.Join(procRoot, "101"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(procRoot, "101", "environ"), []byte(JobFileEnv+"=/etc/shadow\x00"), 0600); err != nil {
		t.Fatal(err)
	}

	_, err := StageJobFile(procRoot, []int{101}, t.TempDir(), 1)
	if err == nil || !strings.Contains(err.Error(), "want checkpoint job file") {
		t.Fatalf("expected unexpected-path error, got %v", err)
	}
}

func TestStageJobFileRejectsSymlink(t *testing.T) {
	procRoot := t.TempDir()
	checkpointDir := t.TempDir()
	jobFile := snapshotprotocol.CUDAJobFilePath
	processRoot := filepath.Join(procRoot, "101", "root")
	if err := os.MkdirAll(filepath.Join(processRoot, filepath.Dir(jobFile)), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(procRoot, "101", "environ"), []byte(JobFileEnv+"="+jobFile+"\x00"), 0600); err != nil {
		t.Fatal(err)
	}
	secret := filepath.Join(processRoot, "secret")
	if err := os.WriteFile(secret, []byte("must-not-copy"), 0600); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(processRoot, jobFile)
	if err := os.Symlink(filepath.Join("..", "secret"), source); err != nil {
		t.Fatal(err)
	}

	_, err := StageJobFile(procRoot, []int{101}, checkpointDir, 1)
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
	procRoot := t.TempDir()
	for _, pid := range []string{"101", "202"} {
		if err := os.MkdirAll(filepath.Join(procRoot, pid), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(procRoot, pid, "environ"), []byte("OTHER=value\x00"), 0600); err != nil {
			t.Fatal(err)
		}
	}

	jobFile, err := StageJobFile(procRoot, []int{101}, t.TempDir(), 1)
	if err != nil || jobFile != "" {
		t.Fatalf("legacy single-GPU StageJobFile() = %q, %v", jobFile, err)
	}
	_, err = StageJobFile(procRoot, []int{101, 202}, t.TempDir(), 2)
	if err == nil || !strings.Contains(err.Error(), "multi-GPU CUDA processes are missing") {
		t.Fatalf("expected missing multi-GPU launch-job error, got %v", err)
	}
}

func TestCheckpointProcessTreePersistsStateThenRecoversEveryProcess(t *testing.T) {
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
if [ "$action" = restore ]; then printf '|restored-%s' "$pid" >> "$job_file"; fi
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

	capture := func() error {
		artifact, err := os.ReadFile(filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName))
		if err != nil {
			return err
		}
		if got, want := string(artifact), "initial|101|202"; got != want {
			return fmt.Errorf("persisted job state during capture = %q, want %q", got, want)
		}
		f, err := os.OpenFile(trace, os.O_APPEND|os.O_WRONLY, 0600)
		if err != nil {
			return err
		}
		defer f.Close()
		_, err = f.WriteString("persist artifact\n")
		return err
	}
	if _, err := CheckpointProcessTree(context.Background(), []int{101, 202}, liveJobFile, checkpointDir, time.Second, capture, logr.Discard()); err != nil {
		t.Fatalf("CheckpointProcessTree() error = %v", err)
	}
	traceContent, err := os.ReadFile(trace)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(traceContent), "lock 101\nlock 202\ncheckpoint 101\ncheckpoint 202\npersist artifact\nrestore 101\nrestore 202\nunlock 101\nunlock 202\n"; got != want {
		t.Fatalf("helper call order = %q, want %q", got, want)
	}
	artifact, err := os.ReadFile(filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(artifact), "initial|101|202"; got != want {
		t.Fatalf("persisted job state = %q, want %q", got, want)
	}
	liveState, err := os.ReadFile(liveJobFile)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(liveState), "initial|101|202|restored-101|restored-202"; got != want {
		t.Fatalf("live job state after recovery = %q, want %q", got, want)
	}
}

func TestCheckpointProcessTreeRecoversWithFreshContextAfterCaptureCancellation(t *testing.T) {
	tempDir := t.TempDir()
	trace := filepath.Join(tempDir, "trace")
	installFakeCUDAHelper(t, `
action=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --action) action="$2"; shift 2 ;;
        *) shift ;;
    esac
done
printf '%s\n' "$action" >> "$DYNAMO_TEST_TRACE"
`)
	t.Setenv("DYNAMO_TEST_TRACE", trace)
	ctx, cancel := context.WithCancel(context.Background())
	liveJobFile := filepath.Join(tempDir, "live-job")
	checkpointDir := filepath.Join(tempDir, "checkpoint")
	if err := os.Mkdir(checkpointDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(liveJobFile, []byte("job-state"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName), []byte("validation-copy"), 0600); err != nil {
		t.Fatal(err)
	}

	_, err := CheckpointProcessTree(
		ctx,
		[]int{101},
		liveJobFile,
		checkpointDir,
		time.Second,
		func() error {
			cancel()
			return ctx.Err()
		},
		logr.Discard(),
	)

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("CheckpointProcessTree() error = %v, want context.Canceled", err)
	}
	traceContent, readErr := os.ReadFile(trace)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if got, want := string(traceContent), "lock\ncheckpoint\nrestore\nunlock\n"; got != want {
		t.Fatalf("helper call order = %q, want %q", got, want)
	}
}

func TestCheckpointProcessTreeGetStateCancellationKillsChild(t *testing.T) {
	const (
		recoveryTimeout = 100 * time.Millisecond
		returnTimeout   = time.Second
	)
	tempDir := t.TempDir()
	childPIDFile := filepath.Join(tempDir, "child-pid")
	installFakeCUDAHelper(t, `
if [ "$1" = "--get-state" ]; then
    sleep 300 &
    printf '%s\n' "$!" > "$DYNAMO_TEST_CHILD_PID_FILE"
    wait
fi
action=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --action) action="$2"; shift 2 ;;
        *) shift ;;
    esac
done
if [ "$action" = unlock ]; then exit 1; fi
`)
	t.Setenv("DYNAMO_TEST_CHILD_PID_FILE", childPIDFile)
	liveJobFile := filepath.Join(tempDir, "live-job")
	checkpointDir := filepath.Join(tempDir, "checkpoint")
	if err := os.Mkdir(checkpointDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(liveJobFile, []byte("job-state"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName), []byte("validation-copy"), 0600); err != nil {
		t.Fatal(err)
	}

	result := make(chan error, 1)
	start := time.Now()
	go func() {
		_, err := CheckpointProcessTree(
			context.Background(),
			[]int{101},
			liveJobFile,
			checkpointDir,
			recoveryTimeout,
			func() error { return nil },
			logr.Discard(),
		)
		result <- err
	}()

	cleanupChild := true
	killChild := func() {
		content, err := os.ReadFile(childPIDFile)
		if err != nil {
			return
		}
		pid, err := strconv.Atoi(strings.TrimSpace(string(content)))
		if err == nil {
			_ = unix.Kill(pid, unix.SIGKILL)
		}
	}
	t.Cleanup(func() {
		if cleanupChild {
			killChild()
		}
	})

	var err error
	select {
	case err = <-result:
	case <-time.After(returnTimeout):
		killChild()
		select {
		case <-result:
		case <-time.After(time.Second):
		}
		t.Fatal("CheckpointProcessTree() did not return after recovery deadline")
	}
	if duration := time.Since(start); duration > returnTimeout {
		t.Fatalf("CheckpointProcessTree() took %s after recovery cancellation", duration)
	}
	if err == nil || !strings.Contains(err.Error(), "recover source CUDA process tree") {
		t.Fatalf("CheckpointProcessTree() error = %v, want recovery failure", err)
	}

	content, readErr := os.ReadFile(childPIDFile)
	if readErr != nil {
		t.Fatal(readErr)
	}
	childPID, parseErr := strconv.Atoi(strings.TrimSpace(string(content)))
	if parseErr != nil {
		t.Fatal(parseErr)
	}
	deadline := time.Now().Add(time.Second)
	for {
		processErr := unix.Kill(childPID, 0)
		if errors.Is(processErr, unix.ESRCH) {
			cleanupChild = false
			break
		}
		if processErr != nil {
			t.Fatalf("probe get-state child %d: %v", childPID, processErr)
		}
		if time.Now().After(deadline) {
			killChild()
			t.Fatalf("get-state child %d remains after recovery cancellation", childPID)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestCheckpointProcessTreeJoinsCaptureAndRecoveryErrors(t *testing.T) {
	tempDir := t.TempDir()
	trace := filepath.Join(tempDir, "trace")
	installFakeCUDAHelper(t, `
action=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --action) action="$2"; shift 2 ;;
        *) shift ;;
    esac
done
printf '%s\n' "$action" >> "$DYNAMO_TEST_TRACE"
if [ "$action" = restore ]; then
    sleep 300 &
    wait
fi
`)
	t.Setenv("DYNAMO_TEST_TRACE", trace)
	liveJobFile := filepath.Join(tempDir, "live-job")
	checkpointDir := filepath.Join(tempDir, "checkpoint")
	if err := os.Mkdir(checkpointDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(liveJobFile, []byte("job-state"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName), []byte("validation-copy"), 0600); err != nil {
		t.Fatal(err)
	}

	captureErr := errors.New("capture failed")
	_, err := CheckpointProcessTree(
		context.Background(),
		[]int{101},
		liveJobFile,
		checkpointDir,
		100*time.Millisecond,
		func() error { return captureErr },
		logr.Discard(),
	)

	if !errors.Is(err, captureErr) {
		t.Fatalf("CheckpointProcessTree() error = %v, want capture error", err)
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("CheckpointProcessTree() error = %v, want recovery deadline", err)
	}
	traceContent, readErr := os.ReadFile(trace)
	if readErr != nil {
		t.Fatal(readErr)
	}
	if got, want := string(traceContent), "lock\ncheckpoint\nrestore\n"; got != want {
		t.Fatalf("helper call order = %q, want %q", got, want)
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
