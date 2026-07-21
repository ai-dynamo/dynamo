package criu

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

func compressionThreads(value uint32) *uint32 {
	return &value
}

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

func TestReadLogTail(t *testing.T) {
	t.Run("returns whole small log", func(t *testing.T) {
		path := t.TempDir() + "/dump.log"
		if err := os.WriteFile(path, []byte("short log"), 0644); err != nil {
			t.Fatalf("write log: %v", err)
		}

		if got := readLogTail(path); got != "short log" {
			t.Fatalf("readLogTail() = %q, want %q", got, "short log")
		}
	})

	t.Run("truncates large log", func(t *testing.T) {
		path := t.TempDir() + "/dump.log"
		content := "prefix-" + strings.Repeat("x", dumpLogTailMaxSize+1)
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			t.Fatalf("write log: %v", err)
		}

		got := readLogTail(path)
		if !strings.HasPrefix(got, "...<truncated>...\n") {
			t.Fatalf("readLogTail() missing truncation marker: %q", got[:min(len(got), 32)])
		}
		if !strings.HasSuffix(got, strings.Repeat("x", dumpLogTailMaxSize)) {
			t.Fatal("readLogTail() did not keep the log tail")
		}
	})
}

func TestApplyCommonSettings(t *testing.T) {
	t.Run("valid mode sets all fields", func(t *testing.T) {
		opts := &criurpc.CriuOpts{}
		settings := &types.CRIUSettings{
			LogLevel:          4,
			ShellJob:          true,
			TcpEstablished:    true,
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
		if !opts.GetTcpEstablished() {
			t.Error("TcpEstablished should be true")
		}
		if opts.GetTcpClose() {
			t.Error("TcpClose should be false")
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

	t.Run("imageIoMode direct sets IMAGE_IO_DIRECT", func(t *testing.T) {
		opts := &criurpc.CriuOpts{}
		settings := &types.CRIUSettings{ImageIoMode: "direct"}
		if err := applyCommonSettings(opts, settings); err != nil {
			t.Fatalf("applyCommonSettings: %v", err)
		}
		if opts.GetImageIoMode() != criurpc.CriuImageIoMode_IMAGE_IO_DIRECT {
			t.Errorf("ImageIoMode = %v, want IMAGE_IO_DIRECT", opts.GetImageIoMode())
		}
	})

	t.Run("imageIoMode empty defaults to IMAGE_IO_DIRECT", func(t *testing.T) {
		opts := &criurpc.CriuOpts{}
		settings := &types.CRIUSettings{}
		if err := applyCommonSettings(opts, settings); err != nil {
			t.Fatalf("applyCommonSettings: %v", err)
		}
		if opts.GetImageIoMode() != criurpc.CriuImageIoMode_IMAGE_IO_DIRECT {
			t.Errorf("ImageIoMode = %v, want IMAGE_IO_DIRECT", opts.GetImageIoMode())
		}
	})

	t.Run("imageIoMode writeback sets IMAGE_IO_WRITEBACK", func(t *testing.T) {
		opts := &criurpc.CriuOpts{}
		settings := &types.CRIUSettings{ImageIoMode: "writeback"}
		if err := applyCommonSettings(opts, settings); err != nil {
			t.Fatalf("applyCommonSettings: %v", err)
		}
		if opts.GetImageIoMode() != criurpc.CriuImageIoMode_IMAGE_IO_WRITEBACK {
			t.Errorf("ImageIoMode = %v, want IMAGE_IO_WRITEBACK", opts.GetImageIoMode())
		}
	})

	t.Run("invalid imageIoMode returns error", func(t *testing.T) {
		opts := &criurpc.CriuOpts{}
		settings := &types.CRIUSettings{ImageIoMode: "bogus"}
		if err := applyCommonSettings(opts, settings); err == nil {
			t.Error("expected error for invalid ImageIoMode")
		}
	})

	t.Run("invalid mode returns error", func(t *testing.T) {
		opts := &criurpc.CriuOpts{}
		settings := &types.CRIUSettings{ManageCgroupsMode: "invalid"}
		if err := applyCommonSettings(opts, settings); err == nil {
			t.Error("expected error for invalid ManageCgroupsMode")
		}
	})

	t.Run("conflicting tcp settings return error", func(t *testing.T) {
		opts := &criurpc.CriuOpts{}
		settings := &types.CRIUSettings{
			TcpClose:       true,
			TcpEstablished: true,
		}
		if err := applyCommonSettings(opts, settings); err == nil {
			t.Error("expected error for conflicting tcp settings")
		}
	})
}

func TestBuildRestoreExtMounts(t *testing.T) {
	t.Run("normal manifest with ExtMnt", func(t *testing.T) {
		m := &types.CheckpointManifest{
			CRIUDump: types.CRIUDumpManifest{
				ExtMnt: map[string]string{
					"/etc/hostname": "/etc/hostname",
					"/proc/acpi":    "/dev/null",
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

func TestBuildCRIUConfCompression(t *testing.T) {
	tests := []struct {
		name        string
		settings    types.CRIUSettings
		wantLines   []string
		unwantLines []string
	}{
		{
			name:        "off remains uncompressed",
			settings:    types.CRIUSettings{CompressionMode: types.CRIUCompressionModeOff},
			unwantLines: []string{"compress\n", "compress-region", "compress-acceleration"},
		},
		{
			name: "per page",
			settings: types.CRIUSettings{
				CompressionMode:         types.CRIUCompressionModePerPage,
				CompressionAcceleration: 2,
				DecompressThreads:       compressionThreads(0),
			},
			wantLines: []string{"compress\n", "compress-acceleration 2\n", "decompress-threads 0\n"},
		},
		{
			name: "region",
			settings: types.CRIUSettings{
				CompressionMode:         types.CRIUCompressionModeRegion,
				CompressionAcceleration: 4,
				CompressionRegionSize:   1024 * 1024,
			},
			wantLines:   []string{"compress-region 1048576\n", "compress-acceleration 4\n"},
			unwantLines: []string{"compress\n"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.settings.Validate(); err != nil {
				t.Fatalf("Validate() error = %v", err)
			}
			content := buildCRIUConf(&tc.settings)
			for _, line := range tc.wantLines {
				if !strings.Contains(content, line) {
					t.Errorf("config %q does not contain %q", content, line)
				}
			}
			for _, line := range tc.unwantLines {
				if strings.Contains(content, line) {
					t.Errorf("config %q unexpectedly contains %q", content, line)
				}
			}
		})
	}
}

func TestCheckCompressionSupport(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		binary := filepath.Join(t.TempDir(), "criu")
		script := `#!/bin/sh
if [ "$#" -ne 3 ] || [ "$1" != "check" ] || [ "$2" != "--feature" ] || [ "$3" != "compress" ]; then
	echo "unexpected arguments: $*" >&2
	exit 2
fi
exit 0
`
		if err := os.WriteFile(binary, []byte(script), 0o700); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		if err := checkCompressionSupport(binary); err != nil {
			t.Fatalf("checkCompressionSupport() error = %v", err)
		}
	})

	t.Run("failure includes output", func(t *testing.T) {
		binary := filepath.Join(t.TempDir(), "criu")
		if err := os.WriteFile(binary, []byte("#!/bin/sh\necho missing-lz4 >&2\nexit 1\n"), 0o700); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}
		err := checkCompressionSupport(binary)
		if err == nil || !strings.Contains(err.Error(), "missing-lz4") {
			t.Fatalf("checkCompressionSupport() error = %v, want command output", err)
		}
	})
}
