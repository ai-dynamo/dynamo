// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cuda

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"

	snapshotruntime "github.com/ai-dynamo/dynamo/deploy/snapshot/internal/runtime"
)

func lockWithJobFile(ctx context.Context, pid int, jobFile string, log logr.Logger) error {
	if jobFile == "" {
		return lock(ctx, pid, log)
	}
	return runActionWithJobFile(ctx, pid, actionLock, "", jobFile, cudaCheckpointHelperBinary, log)
}

func checkpointWithJobFile(ctx context.Context, pid int, jobFile string, log logr.Logger) error {
	if jobFile == "" {
		return checkpoint(ctx, pid, log)
	}
	return runActionWithJobFile(ctx, pid, actionCheckpoint, "", jobFile, cudaCheckpointHelperBinary, log)
}

func restoreWithJobFile(ctx context.Context, pid int, deviceMap, jobFile, helperBinaryPath string, log logr.Logger) error {
	if jobFile == "" {
		return restoreProcess(ctx, pid, deviceMap, helperBinaryPath, log)
	}
	return runActionWithJobFile(ctx, pid, actionRestore, deviceMap, jobFile, helperBinaryPath, log)
}

func unlockWithJobFile(ctx context.Context, pid int, jobFile, helperBinaryPath string, log logr.Logger) error {
	if jobFile == "" {
		return unlock(ctx, pid, helperBinaryPath, log)
	}
	return runActionWithJobFile(ctx, pid, actionUnlock, "", jobFile, helperBinaryPath, log)
}

func runActionWithJobFile(ctx context.Context, pid int, action, deviceMap, jobFile, helperBinaryPath string, log logr.Logger) error {
	args := []string{"--action", action, "--pid", strconv.Itoa(pid), "--job-file", jobFile}
	if action == actionRestore && deviceMap != "" {
		args = append(args, "--device-map", deviceMap)
	}
	cmd := helperCommand(ctx, helperBinaryPath, args...)
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
