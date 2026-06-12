package criu

import (
	"os"
	"strings"
	"testing"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/cuda"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
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

func TestConfigureCUDAPluginRestoreEnv(t *testing.T) {
	t.Run("non-CUDA checkpoint leaves environment unchanged", func(t *testing.T) {
		t.Setenv(criuLibsDirEnv, "/existing")
		m := &types.CheckpointManifest{}

		cleanup, err := configureCUDAPluginRestoreEnv(m, "GPU-a=GPU-b", logr.Discard())
		if err != nil {
			t.Fatalf("configureCUDAPluginRestoreEnv: %v", err)
		}
		cleanup()
		if got := os.Getenv(criuLibsDirEnv); got != "/existing" {
			t.Fatalf("%s = %q, want /existing", criuLibsDirEnv, got)
		}
	})

	t.Run("CUDA checkpoint sets plugin environment and restores previous values", func(t *testing.T) {
		t.Setenv(cuda.CRIUCUDADeviceMapEnv, "old-map")
		t.Setenv(cuda.CRIUCUDAForceRestorePluginEnv, "old-force")
		t.Setenv(criuLibsDirEnv, "/old/criu")
		m := &types.CheckpointManifest{
			CUDA: types.CUDAManifest{PIDs: []int{1}},
		}

		cleanup, err := configureCUDAPluginRestoreEnv(m, "GPU-a=GPU-b", logr.Discard())
		if err != nil {
			t.Fatalf("configureCUDAPluginRestoreEnv: %v", err)
		}
		if got := os.Getenv(cuda.CRIUCUDADeviceMapEnv); got != "GPU-a=GPU-b" {
			t.Fatalf("%s = %q, want device map", cuda.CRIUCUDADeviceMapEnv, got)
		}
		if got := os.Getenv(cuda.CRIUCUDAForceRestorePluginEnv); got != "1" {
			t.Fatalf("%s = %q, want 1", cuda.CRIUCUDAForceRestorePluginEnv, got)
		}
		if got := os.Getenv(criuLibsDirEnv); got != cudaPluginLibDir {
			t.Fatalf("%s = %q, want %q", criuLibsDirEnv, got, cudaPluginLibDir)
		}

		cleanup()
		if got := os.Getenv(cuda.CRIUCUDADeviceMapEnv); got != "old-map" {
			t.Fatalf("%s after cleanup = %q, want old-map", cuda.CRIUCUDADeviceMapEnv, got)
		}
		if got := os.Getenv(cuda.CRIUCUDAForceRestorePluginEnv); got != "old-force" {
			t.Fatalf("%s after cleanup = %q, want old-force", cuda.CRIUCUDAForceRestorePluginEnv, got)
		}
		if got := os.Getenv(criuLibsDirEnv); got != "/old/criu" {
			t.Fatalf("%s after cleanup = %q, want /old/criu", criuLibsDirEnv, got)
		}
	})

	t.Run("CUDA checkpoint uses configured plugin libdir", func(t *testing.T) {
		m := &types.CheckpointManifest{
			CRIUDump: types.CRIUDumpManifest{
				CRIU: types.CRIUSettings{LibDir: "/custom/criu"},
			},
			CUDA: types.CUDAManifest{PIDs: []int{1}},
		}

		cleanup, err := configureCUDAPluginRestoreEnv(m, "", logr.Discard())
		if err != nil {
			t.Fatalf("configureCUDAPluginRestoreEnv: %v", err)
		}
		defer cleanup()
		if _, ok := os.LookupEnv(cuda.CRIUCUDADeviceMapEnv); ok {
			t.Fatalf("%s should be unset for empty device map", cuda.CRIUCUDADeviceMapEnv)
		}
		if got := os.Getenv(criuLibsDirEnv); got != "/custom/criu" {
			t.Fatalf("%s = %q, want /custom/criu", criuLibsDirEnv, got)
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
