package cuda

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
)

const (
	actionLock       = "lock"
	actionCheckpoint = "checkpoint"
	actionRestore    = "restore"
	actionUnlock     = "unlock"

	cudaShortActionTimeout = 2 * time.Minute
	cudaLongActionTimeout  = 10 * time.Minute
)

var cudaCheckpointHelperBinary = "/usr/local/bin/cuda-checkpoint-helper"

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

func runAction(ctx context.Context, pid int, action, deviceMap string, log logr.Logger) error {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid)}
	timeout := cudaShortActionTimeout
	if action == actionCheckpoint || action == actionRestore {
		timeout = cudaLongActionTimeout
	}
	if action == actionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	actionCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.Command(cudaCheckpointHelperBinary, args...)
	var output bytes.Buffer
	cmd.Stdout = &output
	cmd.Stderr = &output
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
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("cuda-checkpoint-helper %v failed to start for pid %d: %w", args, pid, err)
	}

	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()

	var err error
	timedOut := false
	select {
	case err = <-done:
	case <-actionCtx.Done():
		timedOut = actionCtx.Err() == context.DeadlineExceeded
		if timedOut {
			logCudaTimeoutDiagnostics(ctx, pid, cmd.Process.Pid, action, details, log)
		}
		if killErr := cmd.Process.Kill(); killErr != nil {
			log.Error(killErr, "Failed to kill timed out cuda-checkpoint-helper", "pid", pid, "helper_pid", cmd.Process.Pid, "action", action)
		}
		select {
		case err = <-done:
		case <-time.After(5 * time.Second):
			err = fmt.Errorf("cuda-checkpoint-helper did not exit after kill")
		}
		if timedOut && err == nil {
			err = context.DeadlineExceeded
		}
	}
	duration := time.Since(start)
	out := strings.TrimSpace(output.String())
	if err != nil {
		log.Error(err, "cuda-checkpoint-helper command failed",
			"pid", pid,
			"outermost_pid", details.OutermostPID,
			"innermost_pid", details.InnermostPID,
			"cmdline", details.Cmdline,
			"action", action,
			"duration", duration,
			"timeout", timeout,
			"timed_out", timedOut,
			"output", out,
		)
		if timedOut {
			return fmt.Errorf("cuda-checkpoint-helper %v timed out for pid %d after %s (output: %s)", args, pid, timeout, out)
		}
		return fmt.Errorf("cuda-checkpoint-helper %v failed for pid %d after %s: %w (output: %s)", args, pid, duration, err, out)
	}
	log.V(1).Info("cuda-checkpoint-helper command succeeded",
		"pid", pid,
		"outermost_pid", details.OutermostPID,
		"innermost_pid", details.InnermostPID,
		"cmdline", details.Cmdline,
		"action", action,
		"duration", duration,
		"timeout", timeout,
		"output", out,
	)
	return nil
}

func logCudaTimeoutDiagnostics(ctx context.Context, targetPID, helperPID int, action string, details snapshotruntime.ProcessDetails, log logr.Logger) {
	log.Info("cuda-checkpoint-helper timeout diagnostics started",
		"pid", targetPID,
		"helper_pid", helperPID,
		"outermost_pid", details.OutermostPID,
		"innermost_pid", details.InnermostPID,
		"cmdline", details.Cmdline,
		"action", action,
	)

	logProcessDiagnostics("target", targetPID, log)
	if helperPID > 0 {
		logProcessDiagnostics("helper", helperPID, log)
	}

	nvidiaSmiCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	cmd := exec.CommandContext(nvidiaSmiCtx, "nvidia-smi", "pmon", "-c", "1")
	output, err := cmd.CombinedOutput()
	nvidiaSmiError := ""
	if err != nil {
		nvidiaSmiError = err.Error()
	}
	log.Info("cuda-checkpoint-helper timeout nvidia-smi pmon",
		"error", nvidiaSmiError,
		"output", trimForLog(string(output), 12000),
	)
}

func logProcessDiagnostics(label string, pid int, log logr.Logger) {
	status := readProcFile(pid, "status", 12000)
	threadsTotal, threads := collectThreadDiagnostics(pid, 128)
	fdTotal, fdMatches := collectFDDiagnostics(pid, 128)
	mapMatches := collectMapsDiagnostics(pid, 160)

	log.Info("cuda-checkpoint-helper timeout process diagnostics",
		"label", label,
		"pid", pid,
		"cmdline", readProcFile(pid, "cmdline", 4000),
		"comm", readProcFile(pid, "comm", 1000),
		"wchan", readProcFile(pid, "wchan", 1000),
		"stack", readProcFile(pid, "stack", 12000),
		"status", status,
		"thread_count", threadsTotal,
		"threads", strings.Join(threads, "\n"),
		"fd_count", fdTotal,
		"cuda_relevant_fds", strings.Join(fdMatches, "\n"),
		"cuda_relevant_maps", strings.Join(mapMatches, "\n"),
	)
}

func collectThreadDiagnostics(pid int, limit int) (int, []string) {
	taskDir := filepath.Join("/proc", strconv.Itoa(pid), "task")
	entries, err := os.ReadDir(taskDir)
	if err != nil {
		return 0, []string{fmt.Sprintf("failed to read %s: %v", taskDir, err)}
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() < entries[j].Name()
	})

	threads := make([]string, 0, min(len(entries), limit))
	for _, entry := range entries {
		if len(threads) >= limit {
			break
		}
		if !entry.IsDir() {
			continue
		}
		tid := entry.Name()
		threadPath := filepath.Join(taskDir, tid)
		status := readFile(filepath.Join(threadPath, "status"), 4000)
		threads = append(threads, fmt.Sprintf(
			"tid=%s comm=%q state=%q wchan=%q",
			tid,
			readFile(filepath.Join(threadPath, "comm"), 1000),
			statusValue(status, "State:"),
			readFile(filepath.Join(threadPath, "wchan"), 1000),
		))
	}
	if len(entries) > limit {
		threads = append(threads, fmt.Sprintf("... truncated %d additional threads", len(entries)-limit))
	}
	return len(entries), threads
}

func collectFDDiagnostics(pid int, limit int) (int, []string) {
	fdDir := filepath.Join("/proc", strconv.Itoa(pid), "fd")
	entries, err := os.ReadDir(fdDir)
	if err != nil {
		return 0, []string{fmt.Sprintf("failed to read %s: %v", fdDir, err)}
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() < entries[j].Name()
	})

	matches := make([]string, 0)
	for _, entry := range entries {
		if len(matches) >= limit {
			break
		}
		target, err := os.Readlink(filepath.Join(fdDir, entry.Name()))
		if err != nil {
			continue
		}
		if isCudaRelevant(target) {
			matches = append(matches, fmt.Sprintf("%s -> %s", entry.Name(), target))
		}
	}
	return len(entries), matches
}

func collectMapsDiagnostics(pid int, limit int) []string {
	mapsPath := filepath.Join("/proc", strconv.Itoa(pid), "maps")
	data := readFile(mapsPath, 1_000_000)
	if strings.HasPrefix(data, "failed to read ") {
		return []string{data}
	}

	matches := make([]string, 0)
	for _, line := range strings.Split(data, "\n") {
		if len(matches) >= limit {
			break
		}
		if isCudaRelevant(line) {
			matches = append(matches, line)
		}
	}
	return matches
}

func readProcFile(pid int, name string, limit int) string {
	path := filepath.Join("/proc", strconv.Itoa(pid), name)
	if name == "cmdline" {
		return strings.ReplaceAll(readFile(path, limit), "\x00", " ")
	}
	return readFile(path, limit)
}

func readFile(path string, limit int) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Sprintf("failed to read %s: %v", path, err)
	}
	return trimForLog(string(data), limit)
}

func trimForLog(value string, limit int) string {
	value = strings.TrimSpace(value)
	if limit > 0 && len(value) > limit {
		return value[:limit] + "\n... truncated"
	}
	return value
}

func statusValue(status string, key string) string {
	for _, line := range strings.Split(status, "\n") {
		if strings.HasPrefix(line, key) {
			return strings.TrimSpace(strings.TrimPrefix(line, key))
		}
	}
	return ""
}

func isCudaRelevant(value string) bool {
	lower := strings.ToLower(value)
	relevantTerms := []string{
		"cuda",
		"nvidia",
		"uvm",
		"nccl",
		"/dev/shm",
		"memfd:",
		"anon_inode",
		"socket:",
	}
	for _, term := range relevantTerms {
		if strings.Contains(lower, term) {
			return true
		}
	}
	return false
}
