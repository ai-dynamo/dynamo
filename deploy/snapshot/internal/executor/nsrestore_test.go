package executor

import (
	"testing"
	"time"
)

func TestNSRestoreSetupTimingsFinalize(t *testing.T) {
	timings := NSRestoreSetupTimings{
		ManifestReadDuration:       1 * time.Millisecond,
		InetRemapDuration:          2 * time.Millisecond,
		BuildRestoreOptsDuration:   3 * time.Millisecond,
		RootfsDiffStatDuration:     4 * time.Millisecond,
		RootfsDiffExtractDuration:  5 * time.Millisecond,
		DeletedFilesReadDuration:   6 * time.Millisecond,
		DeletedFilesParseDuration:  7 * time.Millisecond,
		DeletedFilesRemoveDuration: 8 * time.Millisecond,
		DevShmUnmountDuration:      9 * time.Millisecond,
		ProcSysRemountReadWrite:    10 * time.Millisecond,
	}
	const total = 60 * time.Millisecond

	timings.finalize(total)

	const wantSum = 55 * time.Millisecond
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
