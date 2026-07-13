package executor

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestNSRestoreSetupTimingsFinalize(t *testing.T) {
	timings := NSRestoreSetupTimings{
		ManifestReadDuration:        1 * time.Millisecond,
		InetRemapDuration:           2 * time.Millisecond,
		BuildRestoreOptsDuration:    3 * time.Millisecond,
		RootfsDiffStatDuration:      4 * time.Millisecond,
		RootfsDiffExtractDuration:   5 * time.Millisecond,
		RootfsReleaseMarkerDuration: 11 * time.Millisecond,
		DeletedFilesReadDuration:    6 * time.Millisecond,
		DeletedFilesParseDuration:   7 * time.Millisecond,
		DeletedFilesRemoveDuration:  8 * time.Millisecond,
		DevShmUnmountDuration:       9 * time.Millisecond,
		ProcSysRemountReadWrite:     10 * time.Millisecond,
	}
	const total = 71 * time.Millisecond

	timings.finalize(total)

	const wantSum = 66 * time.Millisecond
	if timings.SetupSubphasesDuration != wantSum {
		t.Errorf("SetupSubphasesDuration = %s, want %s", timings.SetupSubphasesDuration, wantSum)
	}
	if timings.SetupUnaccountedDuration != total-wantSum {
		t.Errorf(
			"SetupUnaccountedDuration = %s, want %s",
			timings.SetupUnaccountedDuration,
			total-wantSum,
		)
	}
	if timings.SetupSubphasesDuration+timings.SetupUnaccountedDuration != total {
		t.Error("setup timings do not reconcile")
	}
	const wantDeletedFiles = 21 * time.Millisecond
	if timings.DeletedFilesDuration != wantDeletedFiles {
		t.Errorf(
			"DeletedFilesDuration = %s, want %s",
			timings.DeletedFilesDuration,
			wantDeletedFiles,
		)
	}
}

func TestValidateRootfsGMSGate(t *testing.T) {
	root := t.TempDir()
	checkpoint := filepath.Join(root, "checkpoint", "versions", "1")
	valid := RestoreOptions{
		CheckpointPath:       checkpoint,
		RootfsGMSReadyFile:   filepath.Join(root, ".ab", "ready"),
		RootfsGMSReleaseFile: filepath.Join(root, ".ab", "release"),
		RootfsGMSReleaseMode: "control",
		RootfsGMSWaitTimeout: time.Second,
	}
	if enabled, err := validateRootfsGMSGate(valid); err != nil || !enabled {
		t.Fatalf("valid gate = enabled %t, err %v", enabled, err)
	}

	insideCheckpoint := valid
	insideCheckpoint.RootfsGMSReadyFile = filepath.Join(checkpoint, "ready")
	if _, err := validateRootfsGMSGate(insideCheckpoint); err == nil {
		t.Fatal("expected marker inside checkpoint to fail")
	}
}

func TestWaitForMarkerAndAtomicWrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "marker")
	errCh := make(chan error, 1)
	go func() {
		errCh <- waitForMarker(context.Background(), path, time.Second)
	}()

	if err := writeMarkerAtomically(path); err != nil {
		t.Fatalf("writeMarkerAtomically: %v", err)
	}
	if err := <-errCh; err != nil {
		t.Fatalf("waitForMarker: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil || len(data) == 0 {
		t.Fatalf("marker data = %q, err %v", data, err)
	}
	if err := writeMarkerAtomically(path); err == nil {
		t.Fatal("expected atomic marker write to reject an existing path")
	}
}

func TestWaitForMarkerTimesOut(t *testing.T) {
	path := filepath.Join(t.TempDir(), "missing")
	if err := waitForMarker(context.Background(), path, time.Millisecond); err == nil {
		t.Fatal("expected marker wait timeout")
	}
}
