package logging

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/go-logr/logr"
)

func TestRestoreLogSummary(t *testing.T) {
	log := []string{
		"(00.000) Error: an older unrelated failure",
		"(00.001) Error splicing data: Operation not supported",
		"(00.002) Error: Can't open file /tmp/model.bin on restore",
		"(00.003) Error: Unable to open fd=7 id=0x123",
		"(00.004) Error: 42 exited, status=1",
		"(00.005) Error: Restoring FAILED.",
		"(00.006) Error: teardown failed",
	}

	want := "(00.001) Error splicing data: Operation not supported" +
		" | (00.002) Error: Can't open file /tmp/model.bin on restore" +
		" | (00.003) Error: Unable to open fd=7 id=0x123" +
		" | (00.004) Error: 42 exited, status=1"
	if got := restoreLogSummary(log); got != want {
		t.Fatalf("restoreLogSummary() = %q, want %q", got, want)
	}

	fallback := restoreLogSummary([]string{"initializing", "  last\tstatus  "})
	if fallback != "last status" {
		t.Fatalf("restoreLogSummary() fallback = %q, want %q", fallback, "last status")
	}
}

func TestRestoreLogSummaryIsBoundedValidUTF8(t *testing.T) {
	first := append([]byte("Error first \xff "+strings.Repeat("界", 100)), []byte(" tail-one")...)
	second := append([]byte("Warning second \xfe "+strings.Repeat("界", 100)), []byte(" tail-two")...)

	got := restoreLogSummary([]string{string(first), string(second)})
	if len(got) > restoreLogSummaryLimit {
		t.Fatalf("restoreLogSummary() returned %d bytes, limit is %d", len(got), restoreLogSummaryLimit)
	}
	if !utf8.ValidString(got) {
		t.Fatalf("restoreLogSummary() returned invalid UTF-8: %q", got)
	}
	for _, want := range []string{"Error first �", "tail-one", "Warning second �", "tail-two", "界"} {
		if !strings.Contains(got, want) {
			t.Errorf("restoreLogSummary() = %q, want retained %q", got, want)
		}
	}
}

func TestLogRestoreErrorsFallbackAndPrecedence(t *testing.T) {
	checkpointPath := t.TempDir()
	workDir := t.TempDir()

	if err := os.WriteFile(filepath.Join(workDir, "restore.log"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(checkpointPath, "restore.log"),
		[]byte("Error: checkpoint fallback"),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	if got := LogRestoreErrors(checkpointPath, workDir, logr.Discard()); !strings.Contains(got, "checkpoint fallback") {
		t.Fatalf("LogRestoreErrors() = %q, want checkpoint fallback", got)
	}

	if err := os.WriteFile(
		filepath.Join(workDir, "restore.log"),
		[]byte("Warning: preferred workdir cause"),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	if got := LogRestoreErrors(checkpointPath, workDir, logr.Discard()); got != "Warning: preferred workdir cause" {
		t.Fatalf("LogRestoreErrors() = %q, want preferred workdir cause", got)
	}

	if got := LogRestoreErrors(t.TempDir(), t.TempDir(), logr.Discard()); got != "" {
		t.Fatalf("LogRestoreErrors() = %q with no readable logs, want empty", got)
	}
}
