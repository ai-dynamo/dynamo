// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cuda

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/go-logr/logr"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
)

func lockWithJobFile(ctx context.Context, pid int, jobFile string, log logr.Logger) error {
	if jobFile == "" {
		return lock(ctx, pid, log)
	}
	return runActionWithJobFile(ctx, pid, actionLock, jobFile, log)
}

func checkpointWithJobFile(ctx context.Context, pid int, jobFile string, log logr.Logger) error {
	if jobFile == "" {
		return checkpoint(ctx, pid, log)
	}
	return runActionWithJobFile(ctx, pid, actionCheckpoint, jobFile, log)
}

func runActionWithJobFile(ctx context.Context, pid int, action, jobFile string, log logr.Logger) error {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid), "--job-file", jobFile}
	cmd := exec.CommandContext(ctx, cudaCheckpointHelperBinary, args...)
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
