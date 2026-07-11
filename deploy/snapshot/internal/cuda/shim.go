package cuda

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
)

const (
	cudaCheckpointHelperBinary = "/usr/local/bin/cuda-checkpoint-helper"

	actionLock       = "lock"
	actionCheckpoint = "checkpoint"
	actionRestore    = "restore"
	actionUnlock     = "unlock"
)

type helperActionRunner interface {
	run(context.Context, int, string, string, string, string, logr.Logger) error
	state(context.Context, int) (string, error)
}

type commandHelperActionRunner struct{}

func (commandHelperActionRunner) run(
	ctx context.Context,
	pid int,
	action,
	deviceMap,
	storageMode,
	storageDir string,
	log logr.Logger,
) error {
	return runAction(ctx, pid, action, deviceMap, storageMode, storageDir, log)
}

func (commandHelperActionRunner) state(ctx context.Context, pid int) (string, error) {
	return getState(ctx, pid)
}

func getState(ctx context.Context, pid int) (string, error) {
	cmd := exec.CommandContext(ctx, cudaCheckpointHelperBinary, "--get-state", "--pid", strconv.Itoa(pid))
	output, err := cmd.CombinedOutput()
	state := strings.TrimSpace(string(output))
	if err != nil {
		return "", fmt.Errorf("cuda-checkpoint-helper --get-state failed for pid %d: %w (output: %s)", pid, err, state)
	}
	if state == "" {
		return "", fmt.Errorf("cuda-checkpoint-helper --get-state returned empty state for pid %d", pid)
	}
	return state, nil
}

func helperActionArgs(pid int, action, deviceMap, storageMode, storageDir string) []string {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid)}
	if action == actionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	if storageMode == "posix" {
		args = append(args, "--storage-mode", storageMode, "--storage-dir", storageDir)
	}
	return args
}

func runAction(ctx context.Context, pid int, action, deviceMap, storageMode, storageDir string, log logr.Logger) error {
	args := helperActionArgs(pid, action, deviceMap, storageMode, storageDir)
	cmd := exec.CommandContext(ctx, cudaCheckpointHelperBinary, args...)
	details := snapshotruntime.ProcessDetails{
		ObservedPID:   pid,
		OutermostPID:  pid,
		InnermostPID:  pid,
		NamespacePIDs: []int{pid},
	}
	if process, err := snapshotruntime.ReadProcessDetails("/proc", pid); err == nil {
		details = process
	}
	start := time.Now()
	output, err := cmd.CombinedOutput()
	duration := time.Since(start)
	out := strings.TrimSpace(string(output))
	if err != nil {
		log.Error(err, "cuda-checkpoint-helper command failed",
			"pid", pid,
			"outermost_pid", details.OutermostPID,
			"innermost_pid", details.InnermostPID,
			"cmdline", details.Cmdline,
			"action", action,
			"duration", duration,
			"output", out,
		)
		return fmt.Errorf("cuda-checkpoint-helper %v failed for pid %d after %s: %w (output: %s)", args, pid, duration, err, out)
	}
	log.V(1).Info("cuda-checkpoint-helper command succeeded",
		"pid", pid,
		"outermost_pid", details.OutermostPID,
		"innermost_pid", details.InnermostPID,
		"cmdline", details.Cmdline,
		"action", action,
		"duration", duration,
		"output", out,
	)
	return nil
}
