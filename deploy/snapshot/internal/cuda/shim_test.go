// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cuda

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/go-logr/logr"
)

func installFakeCUDAHelper(t *testing.T, script string) {
	t.Helper()
	helper := filepath.Join(t.TempDir(), "cuda-checkpoint-helper")
	if err := os.WriteFile(helper, []byte("#!/bin/sh\n"+script), 0700); err != nil {
		t.Fatal(err)
	}
	originalHelper := cudaCheckpointHelperBinary
	cudaCheckpointHelperBinary = helper
	t.Cleanup(func() { cudaCheckpointHelperBinary = originalHelper })
}

func TestRunActionCancellationIsBounded(t *testing.T) {
	installFakeCUDAHelper(t, "sleep 300 &\nwait\n")
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	started := time.Now()
	err := runAction(ctx, 11, actionRestore, "", logr.Discard())
	duration := time.Since(started)
	if err == nil || !strings.Contains(err.Error(), context.DeadlineExceeded.Error()) {
		t.Fatalf("runAction() error = %v", err)
	}
	if duration > helperWaitDelay+time.Second {
		t.Fatalf("runAction() took %s after cancellation", duration)
	}
}

func TestNormalizeProcessGroupKillErrorReportsFinishedProcess(t *testing.T) {
	if err := normalizeProcessGroupKillError(syscall.ESRCH); !errors.Is(err, os.ErrProcessDone) {
		t.Fatalf("normalizeProcessGroupKillError() error = %v, want %v", err, os.ErrProcessDone)
	}
}
