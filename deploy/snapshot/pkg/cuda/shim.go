package cuda

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
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
	return runAction(ctx, pid, actionLock, "", false, "", "", log)
}

func checkpoint(ctx context.Context, pid int, log logr.Logger) error {
	return runAction(ctx, pid, actionCheckpoint, "", false, "", "", log)
}

func restoreProcess(ctx context.Context, pid int, deviceMap string, debugPause bool, debugResumeMode string, debugContinueFile string, log logr.Logger) error {
	return runAction(ctx, pid, actionRestore, deviceMap, debugPause, debugResumeMode, debugContinueFile, log)
}

func unlock(ctx context.Context, pid int, log logr.Logger) error {
	return runAction(ctx, pid, actionUnlock, "", false, "", "", log)
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

func runAction(ctx context.Context, pid int, action, deviceMap string, debugPause bool, debugResumeMode string, debugContinueFile string, log logr.Logger) error {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid)}
	if action == actionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	cmd := exec.CommandContext(ctx, cudaCheckpointBinary, args...)
	start := time.Now()
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("cuda-checkpoint %v failed to start for pid %d: %w", args, pid, err)
	}
	logChildProcess(log, action, pid, cmd.Process.Pid, args)
	if action == actionRestore && debugPause {
		if err := pauseForDebugger(ctx, log, pid, cmd.Process, debugResumeMode, debugContinueFile); err != nil {
			return err
		}
	}
	err := cmd.Wait()
	duration := time.Since(start)
	out := strings.TrimSpace(strings.TrimSpace(stdout.String()) + "\n" + strings.TrimSpace(stderr.String()))
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

func pauseForDebugger(ctx context.Context, log logr.Logger, targetPID int, process *os.Process, resumeMode string, continueFile string) error {
	if continueFile == "" {
		continueFile = "/tmp/cuda-restore-continue"
	}
	if err := process.Signal(syscall.SIGSTOP); err != nil {
		return fmt.Errorf("failed to SIGSTOP cuda-checkpoint child for pid %d: %w", targetPID, err)
	}
	hostPID, nspids := readNSPIDs(process.Pid)
	log.Info(
		"Paused cuda-checkpoint restore for debugger attach",
		"target_pid", targetPID,
		"cuda_checkpoint_pid", process.Pid,
		"cuda_checkpoint_host_pid", hostPID,
		"cuda_checkpoint_nspids", nspids,
		"resume_mode", resumeMode,
		"continue_file", continueFile,
		"resume_hint", fmt.Sprintf("from the snapshot-agent pod, send SIGCONT to the host-side cuda-checkpoint process for --pid %d", targetPID),
	)
	if resumeMode == "signal" {
		return nil
	}
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			_ = process.Signal(syscall.SIGKILL)
			return fmt.Errorf("debug pause cancelled for cuda-checkpoint pid %d: %w", process.Pid, ctx.Err())
		case <-ticker.C:
			if _, err := os.Stat(continueFile); err == nil {
				_ = os.Remove(continueFile)
				if err := process.Signal(syscall.SIGCONT); err != nil {
					return fmt.Errorf("failed to SIGCONT cuda-checkpoint child for pid %d: %w", targetPID, err)
				}
				log.Info("Resumed cuda-checkpoint restore after debugger attach", "target_pid", targetPID, "cuda_checkpoint_pid", process.Pid, "continue_file", continueFile)
				return nil
			} else if !os.IsNotExist(err) {
				return fmt.Errorf("failed to stat continue file %s: %w", continueFile, err)
			}
		}
	}
}

func logChildProcess(log logr.Logger, action string, targetPID, childPID int, args []string) {
	hostPID, nspids := readNSPIDs(childPID)
	log.Info(
		"Started cuda-checkpoint child",
		"action", action,
		"target_pid", targetPID,
		"cuda_checkpoint_pid", childPID,
		"cuda_checkpoint_host_pid", hostPID,
		"cuda_checkpoint_nspids", nspids,
		"args", strings.Join(args, " "),
	)
}

func readNSPIDs(pid int) (int, []int) {
	data, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "status"))
	if err != nil {
		return pid, nil
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "NSpid:") {
			continue
		}
		fields := strings.Fields(strings.TrimPrefix(line, "NSpid:"))
		nspids := make([]int, 0, len(fields))
		for _, field := range fields {
			value, err := strconv.Atoi(field)
			if err == nil {
				nspids = append(nspids, value)
			}
		}
		if len(nspids) == 0 {
			return pid, nil
		}
		return nspids[0], nspids
	}
	return pid, nil
}
