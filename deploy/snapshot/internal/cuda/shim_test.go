package cuda

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/go-logr/logr"
)

func TestLockPassesDriverTimeout(t *testing.T) {
	dir := t.TempDir()
	argsPath := filepath.Join(dir, "args")
	helperPath := filepath.Join(dir, "cuda-checkpoint-helper")
	script := "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$HELPER_ARGS_PATH\"\n"
	if err := os.WriteFile(helperPath, []byte(script), 0700); err != nil {
		t.Fatalf("write helper: %v", err)
	}
	t.Setenv("HELPER_ARGS_PATH", argsPath)

	previousHelper := cudaCheckpointHelperBinary
	cudaCheckpointHelperBinary = helperPath
	t.Cleanup(func() {
		cudaCheckpointHelperBinary = previousHelper
	})

	if err := lock(context.Background(), os.Getpid(), logr.Discard()); err != nil {
		t.Fatalf("lock: %v", err)
	}
	data, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatalf("read helper args: %v", err)
	}
	got := strings.Split(strings.TrimSpace(string(data)), "\n")
	want := []string{
		"--action",
		"lock",
		"--pid",
		strconv.Itoa(os.Getpid()),
		"--timeout",
		strconv.FormatInt(cudaShortActionTimeout.Milliseconds(), 10),
	}
	if strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("helper args = %q, want %q", got, want)
	}
}
