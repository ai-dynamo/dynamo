package criu

import (
	"testing"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

func TestParseManageCgroupsMode(t *testing.T) {
	tests := []struct {
		raw      string
		wantMode criurpc.CriuCgMode
		wantErr  bool
	}{
		{raw: "ignore", wantMode: criurpc.CriuCgMode_IGNORE},
		{raw: "soft", wantMode: criurpc.CriuCgMode_SOFT},
		{raw: "full", wantMode: criurpc.CriuCgMode_FULL},
		{raw: "strict", wantMode: criurpc.CriuCgMode_STRICT},
		// Case insensitive + whitespace trimming
		{raw: "IGNORE", wantMode: criurpc.CriuCgMode_IGNORE},
		{raw: " Soft ", wantMode: criurpc.CriuCgMode_SOFT},
		{raw: "  FULL  ", wantMode: criurpc.CriuCgMode_FULL},
		// Empty string defaults to SOFT (matches Helm default)
		{raw: "", wantMode: criurpc.CriuCgMode_SOFT},
		// Invalid
		{raw: "bogus", wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.raw, func(t *testing.T) {
			mode, _, err := parseManageCgroupsMode(tc.raw)
			if tc.wantErr {
				if err == nil {
					t.Errorf("expected error for %q, got mode=%v", tc.raw, mode)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error for %q: %v", tc.raw, err)
			}
			if mode != tc.wantMode {
				t.Errorf("mode = %v, want %v", mode, tc.wantMode)
			}
		})
	}
}

func TestApplyCommonSettings(t *testing.T) {
	t.Run("valid mode sets all fields", func(t *testing.T) {
		opts := &criurpc.CriuOpts{}
		settings := &types.CRIUSettings{
			LogLevel:          4,
			ShellJob:          true,
			TcpClose:          true,
			FileLocks:         true,
			ExtUnixSk:         true,
			LinkRemap:         true,
			ManageCgroupsMode: "soft",
		}

		if err := applyCommonSettings(opts, settings); err != nil {
			t.Fatalf("applyCommonSettings: %v", err)
		}

		if opts.GetLogLevel() != 4 {
			t.Errorf("LogLevel = %d", opts.GetLogLevel())
		}
		if !opts.GetShellJob() {
			t.Error("ShellJob should be true")
		}
		if !opts.GetTcpClose() {
			t.Error("TcpClose should be true")
		}
		if !opts.GetFileLocks() {
			t.Error("FileLocks should be true")
		}
		if !opts.GetExtUnixSk() {
			t.Error("ExtUnixSk should be true")
		}
		if !opts.GetLinkRemap() {
			t.Error("LinkRemap should be true")
		}
		if !opts.GetManageCgroups() {
			t.Error("ManageCgroups should be true")
		}
		if opts.GetManageCgroupsMode() != criurpc.CriuCgMode_SOFT {
			t.Errorf("ManageCgroupsMode = %v, want SOFT", opts.GetManageCgroupsMode())
		}
	})

	t.Run("invalid mode returns error", func(t *testing.T) {
		opts := &criurpc.CriuOpts{}
		settings := &types.CRIUSettings{ManageCgroupsMode: "invalid"}
		if err := applyCommonSettings(opts, settings); err == nil {
			t.Error("expected error for invalid ManageCgroupsMode")
		}
	})
}

func TestBuildRestoreExtMounts(t *testing.T) {
	t.Run("normal manifest with ExtMnt", func(t *testing.T) {
		m := &types.CheckpointManifest{
			CRIUDump: types.CRIUDumpManifest{
				ExtMnt: map[string]string{
					"/etc/hostname": "/etc/hostname",
					"/proc/acpi":   "/dev/null",
				},
			},
		}
		mounts, err := buildRestoreExtMounts(m)
		if err != nil {
			t.Fatalf("buildRestoreExtMounts: %v", err)
		}

		// Should contain value→value self-mappings plus "/" → "."
		mountMap := make(map[string]string, len(mounts))
		for _, em := range mounts {
			mountMap[em.GetKey()] = em.GetVal()
		}

		if mountMap["/"] != "." {
			t.Errorf("root mapping: got %q, want %q", mountMap["/"], ".")
		}
		if mountMap["/etc/hostname"] != "/etc/hostname" {
			t.Errorf("/etc/hostname mapping: got %q", mountMap["/etc/hostname"])
		}
		if mountMap["/dev/null"] != "/dev/null" {
			t.Errorf("/dev/null mapping: got %q", mountMap["/dev/null"])
		}
	})

	t.Run("values of / or empty are skipped", func(t *testing.T) {
		m := &types.CheckpointManifest{
			CRIUDump: types.CRIUDumpManifest{
				ExtMnt: map[string]string{
					"/root_mount": "/",
					"/empty_val":  "",
					"/good":       "/good",
				},
			},
		}
		mounts, err := buildRestoreExtMounts(m)
		if err != nil {
			t.Fatalf("buildRestoreExtMounts: %v", err)
		}

		mountMap := make(map[string]string, len(mounts))
		for _, em := range mounts {
			mountMap[em.GetKey()] = em.GetVal()
		}

		// "/" and "" values should be skipped from the value→value mapping
		// but "/" → "." root mapping always exists
		if mountMap["/"] != "." {
			t.Errorf("root mapping missing")
		}
		if _, ok := mountMap[""]; ok {
			t.Error("empty string should not be a key in restore map")
		}
		if mountMap["/good"] != "/good" {
			t.Errorf("/good mapping missing")
		}
	})

	t.Run("empty ExtMnt returns error", func(t *testing.T) {
		m := &types.CheckpointManifest{
			CRIUDump: types.CRIUDumpManifest{},
		}
		_, err := buildRestoreExtMounts(m)
		if err == nil {
			t.Error("expected error for empty ExtMnt")
		}
	})
}
