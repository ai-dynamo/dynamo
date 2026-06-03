// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package executor

import (
	"context"
	"strings"
	"testing"

	"github.com/go-logr/logr/testr"

	"github.com/run-ai/snapshot/agent/internal/types"
)

func TestExecNSRestoreRejectsRelativeContainerCheckpointLocation(t *testing.T) {
	_, err := execNSRestore(
		context.Background(),
		testr.New(t),
		RestoreRequest{
			ContainerCheckpointLocation: "relative/checkpoint",
			NSRestorePath:               "/usr/local/bin/nsrestore",
		},
		&types.RestoreContainerSnapshot{
			CheckpointPath: "/host/checkpoints/abc123",
			PlaceholderPID: 1,
		},
	)
	if err == nil {
		t.Fatal("expected relative container checkpoint location to be rejected")
	}
	if !strings.Contains(err.Error(), "absolute") {
		t.Fatalf("expected absolute-path validation error, got: %v", err)
	}
}
