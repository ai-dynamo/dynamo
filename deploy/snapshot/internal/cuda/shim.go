package cuda

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/go-logr/logr"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
)

const (
	helperWaitDelay = 2 * time.Second

	actionLock       = "lock"
	actionCheckpoint = "checkpoint"
	actionRestore    = "restore"
	actionUnlock     = "unlock"
)

var cudaCheckpointHelperBinary = DefaultHelperBinaryPath

func lock(ctx context.Context, pid int, log logr.Logger) error {
	return runAction(ctx, pid, actionLock, "", DefaultHelperBinaryPath, log)
}

func checkpoint(ctx context.Context, pid int, log logr.Logger) error {
	return runAction(ctx, pid, actionCheckpoint, "", DefaultHelperBinaryPath, log)
}

func restoreProcess(ctx context.Context, pid int, deviceMap, helperBinaryPath string, log logr.Logger) error {
	return runAction(ctx, pid, actionRestore, deviceMap, helperBinaryPath, log)
}

func unlock(ctx context.Context, pid int, helperBinaryPath string, log logr.Logger) error {
	return runAction(ctx, pid, actionUnlock, "", helperBinaryPath, log)
}

func getState(ctx context.Context, pid int, helperBinaryPath string) (string, error) {
	cmd := exec.CommandContext(ctx, helperBinaryPath, "--get-state", "--pid", strconv.Itoa(pid))
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

func runAction(ctx context.Context, pid int, action, deviceMap, helperBinaryPath string, log logr.Logger) error {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid)}
	if action == actionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	cmd := exec.CommandContext(ctx, helperBinaryPath, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return normalizeProcessGroupKillError(syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL))
	}
	cmd.WaitDelay = helperWaitDelay
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
		if ctx.Err() != nil {
			err = ctx.Err()
		}
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

func normalizeProcessGroupKillError(err error) error {
	if errors.Is(err, syscall.ESRCH) {
		return os.ErrProcessDone
	}
	return err
}
