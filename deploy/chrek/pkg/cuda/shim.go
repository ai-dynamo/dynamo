package cuda

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
)

const (
	cudaCheckpointBinary = "/usr/local/sbin/cuda-checkpoint"

	actionLock       = "lock"
	actionCheckpoint = "checkpoint"
	actionRestore    = "restore"
	actionUnlock     = "unlock"
)

func lock(ctx context.Context, pid int, log logr.Logger) error {
	return runAction(ctx, pid, actionLock, "", log)
}

func checkpoint(ctx context.Context, pid int, log logr.Logger) error {
	return runAction(ctx, pid, actionCheckpoint, "", log)
}

func restoreProcess(ctx context.Context, pid int, deviceMap string, log logr.Logger) error {
	return runAction(ctx, pid, actionRestore, deviceMap, log)
}

func unlock(ctx context.Context, pid int, log logr.Logger) error {
	return runAction(ctx, pid, actionUnlock, "", log)
}

func getState(ctx context.Context, pid int) (string, error) {
	cmd := exec.CommandContext(ctx, cudaCheckpointBinary, "--get-state", "--pid", strconv.Itoa(pid))
	output, err := cmd.CombinedOutput()
	state := strings.TrimSpace(string(output))
	if err != nil {
		return "", fmt.Errorf("cuda-checkpoint --get-state failed for pid %d: %w (output: %s)", pid, err, state)
	}
	if state == "" {
		return "", fmt.Errorf("cuda-checkpoint --get-state returned empty state for pid %d", pid)
	}
	return state, nil
}

func runAction(ctx context.Context, pid int, action, deviceMap string, log logr.Logger) error {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid)}
	if action == actionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	cmd := exec.CommandContext(ctx, cudaCheckpointBinary, args...)
	start := time.Now()
	output, err := cmd.CombinedOutput()
	duration := time.Since(start)
	out := strings.TrimSpace(string(output))
	if err != nil {
		return fmt.Errorf("cuda-checkpoint %v failed for pid %d after %s: %w (output: %s)", args, pid, duration, err, out)
	}
	log.Info("cuda-checkpoint command succeeded",
		"pid", pid,
		"action", action,
		"duration", duration,
		"output", out,
	)
	return nil
}
