// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cuda

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/go-logr/logr"
)

func TestRunActionPassesJobFile(t *testing.T) {
	trace := filepath.Join(t.TempDir(), "trace")
	installFakeCUDAHelper(t, "printf '%s\\n' \"$@\" > \""+trace+"\"\n")

	if err := runActionWithJobFile(context.Background(), 11, actionRestore, "", "/checkpoints/job", logr.Discard()); err != nil {
		t.Fatalf("runActionWithJobFile() error = %v", err)
	}
	content, err := os.ReadFile(trace)
	if err != nil {
		t.Fatal(err)
	}
	want := "--action\nrestore\n--pid\n11\n--job-file\n/checkpoints/job\n"
	if string(content) != want {
		t.Fatalf("helper arguments = %q, want %q", content, want)
	}
}
