package criu

import (
	"os"
	"path/filepath"
	"testing"

	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"

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
		mounts, stats, err := buildRestoreExtMounts(m, RestoreMountContext{})
		if err != nil {
			t.Fatalf("buildRestoreExtMounts: %v", err)
		}
		if stats.remappedSnapshotControl != 0 {
			t.Fatalf("remapped snapshot-control mounts = %d, want 0", stats.remappedSnapshotControl)
		}
		if stats.remappedKubeletVolumes != 0 {
			t.Fatalf("remapped kubelet volume mounts = %d, want 0", stats.remappedKubeletVolumes)
		}
		if stats.remappedHostNetns != 0 {
			t.Fatalf("remapped host netns mounts = %d, want 0", stats.remappedHostNetns)
		}
		if stats.remappedHostRunParent != 0 {
			t.Fatalf("remapped host run parent mounts = %d, want 0", stats.remappedHostRunParent)
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
		mounts, stats, err := buildRestoreExtMounts(m, RestoreMountContext{})
		if err != nil {
			t.Fatalf("buildRestoreExtMounts: %v", err)
		}
		if stats.remappedSnapshotControl != 0 {
			t.Fatalf("remapped snapshot-control mounts = %d, want 0", stats.remappedSnapshotControl)
		}
		if stats.remappedKubeletVolumes != 0 {
			t.Fatalf("remapped kubelet volume mounts = %d, want 0", stats.remappedKubeletVolumes)
		}
		if stats.remappedHostNetns != 0 {
			t.Fatalf("remapped host netns mounts = %d, want 0", stats.remappedHostNetns)
		}
		if stats.remappedHostRunParent != 0 {
			t.Fatalf("remapped host run parent mounts = %d, want 0", stats.remappedHostRunParent)
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
		_, _, err := buildRestoreExtMounts(m, RestoreMountContext{})
		if err == nil {
			t.Error("expected error for empty ExtMnt")
		}
	})

	t.Run("snapshot-control host mounts remap to live control mount", func(t *testing.T) {
		m := &types.CheckpointManifest{
			CRIUDump: types.CRIUDumpManifest{
				ExtMnt: map[string]string{
					"/snapshot-control": "/snapshot-control",
					"/source-subpath":   "/host/var/lib/kubelet/pods/source-pod/volume-subpaths/snapshot-control/main/2",
					"/source-rootfs":    "/host/run/containerd/io.containerd.runtime.v2.task/k8s.io/source-container/rootfs/snapshot-control",
				},
			},
		}
		mounts, stats, err := buildRestoreExtMounts(m, RestoreMountContext{})
		if err != nil {
			t.Fatalf("buildRestoreExtMounts: %v", err)
		}
		if stats.remappedSnapshotControl != 2 {
			t.Fatalf("remapped snapshot-control mounts = %d, want 2", stats.remappedSnapshotControl)
		}
		if stats.remappedKubeletVolumes != 0 {
			t.Fatalf("remapped kubelet volume mounts = %d, want 0", stats.remappedKubeletVolumes)
		}
		if stats.remappedHostNetns != 0 {
			t.Fatalf("remapped host netns mounts = %d, want 0", stats.remappedHostNetns)
		}
		if stats.remappedHostRunParent != 0 {
			t.Fatalf("remapped host run parent mounts = %d, want 0", stats.remappedHostRunParent)
		}

		mountMap := make(map[string]string, len(mounts))
		for _, em := range mounts {
			mountMap[em.GetKey()] = em.GetVal()
		}
		if mountMap["/host/var/lib/kubelet/pods/source-pod/volume-subpaths/snapshot-control/main/2"] != "/snapshot-control" {
			t.Errorf("snapshot-control subPath remap: got %q", mountMap["/host/var/lib/kubelet/pods/source-pod/volume-subpaths/snapshot-control/main/2"])
		}
		if mountMap["/host/run/containerd/io.containerd.runtime.v2.task/k8s.io/source-container/rootfs/snapshot-control"] != "/snapshot-control" {
			t.Errorf("snapshot-control rootfs remap: got %q", mountMap["/host/run/containerd/io.containerd.runtime.v2.task/k8s.io/source-container/rootfs/snapshot-control"])
		}
	})

	t.Run("kubelet pod volume mounts remap to restore pod UID", func(t *testing.T) {
		oldKubeletPodsRoot := kubeletPodsRoot
		kubeletPodsRoot = t.TempDir()
		t.Cleanup(func() {
			kubeletPodsRoot = oldKubeletPodsRoot
		})

		targetCSIPath := kubeletPodsRoot + "/target-pod/volumes/kubernetes.io~csi/pvc-abc/mount"
		targetProjectedPath := kubeletPodsRoot + "/target-pod/volumes/kubernetes.io~projected/kube-api-access-live"
		targetSubpath := kubeletPodsRoot + "/target-pod/volume-subpaths/config/main/7"
		for _, path := range []string{targetCSIPath, targetProjectedPath, targetSubpath} {
			if err := os.MkdirAll(path, 0755); err != nil {
				t.Fatalf("mkdir target kubelet volume path %s: %v", path, err)
			}
		}
		sourceCSIPath := kubeletPodsRoot + "/source-pod/volumes/kubernetes.io~csi/pvc-abc/mount"
		sourceProjectedPath := kubeletPodsRoot + "/source-pod/volumes/kubernetes.io~projected/kube-api-access-old"
		sourceSubpath := kubeletPodsRoot + "/source-pod/volume-subpaths/config/main/2"
		m := &types.CheckpointManifest{
			CRIUDump: types.CRIUDumpManifest{
				ExtMnt: map[string]string{
					"/checkpoints": sourceCSIPath,
					"/api-access":  sourceProjectedPath,
					"/config":      sourceSubpath,
				},
			},
		}
		mounts, stats, err := buildRestoreExtMounts(m, RestoreMountContext{PodUID: "target-pod", ContainerName: "main"})
		if err != nil {
			t.Fatalf("buildRestoreExtMounts: %v", err)
		}
		if stats.remappedSnapshotControl != 0 {
			t.Fatalf("remapped snapshot-control mounts = %d, want 0", stats.remappedSnapshotControl)
		}
		if stats.remappedKubeletVolumes != 3 {
			t.Fatalf("remapped kubelet volume mounts = %d, want 3", stats.remappedKubeletVolumes)
		}
		if stats.remappedHostNetns != 0 {
			t.Fatalf("remapped host netns mounts = %d, want 0", stats.remappedHostNetns)
		}
		if stats.remappedHostRunParent != 0 {
			t.Fatalf("remapped host run parent mounts = %d, want 0", stats.remappedHostRunParent)
		}

		mountMap := make(map[string]string, len(mounts))
		for _, em := range mounts {
			mountMap[em.GetKey()] = em.GetVal()
		}
		if mountMap[sourceCSIPath] != targetCSIPath {
			t.Errorf("kubelet CSI volume remap: got %q, want %q", mountMap[sourceCSIPath], targetCSIPath)
		}
		if mountMap[sourceProjectedPath] != targetProjectedPath {
			t.Errorf("kubelet projected volume remap: got %q, want %q", mountMap[sourceProjectedPath], targetProjectedPath)
		}
		if mountMap[sourceSubpath] != targetSubpath {
			t.Errorf("kubelet subpath remap: got %q, want %q", mountMap[sourceSubpath], targetSubpath)
		}
	})

	t.Run("host netns mounts remap to live restore namespace", func(t *testing.T) {
		oldFindHostRunNetnsTarget := findHostRunNetnsTarget
		findHostRunNetnsTarget = func() string {
			return "/host/run/netns/cni-live"
		}
		t.Cleanup(func() {
			findHostRunNetnsTarget = oldFindHostRunNetnsTarget
		})

		netnsPath := "/host/run/netns/cni-source"
		m := &types.CheckpointManifest{
			CRIUDump: types.CRIUDumpManifest{
				ExtMnt: map[string]string{
					"/netns": netnsPath,
				},
			},
		}
		mounts, stats, err := buildRestoreExtMounts(m, RestoreMountContext{})
		if err != nil {
			t.Fatalf("buildRestoreExtMounts: %v", err)
		}
		if stats.remappedSnapshotControl != 0 {
			t.Fatalf("remapped snapshot-control mounts = %d, want 0", stats.remappedSnapshotControl)
		}
		if stats.remappedKubeletVolumes != 0 {
			t.Fatalf("remapped kubelet volume mounts = %d, want 0", stats.remappedKubeletVolumes)
		}
		if stats.remappedHostNetns != 1 {
			t.Fatalf("remapped host netns mounts = %d, want 1", stats.remappedHostNetns)
		}
		if stats.remappedHostRunParent != 0 {
			t.Fatalf("remapped host run parent mounts = %d, want 0", stats.remappedHostRunParent)
		}

		mountMap := make(map[string]string, len(mounts))
		for _, em := range mounts {
			mountMap[em.GetKey()] = em.GetVal()
		}
		if mountMap[netnsPath] != "/host/run/netns/cni-live" {
			t.Errorf("host netns remap: got %q, want %q", mountMap[netnsPath], "/host/run/netns/cni-live")
		}
	})

	t.Run("host run parent remaps to scratch tree", func(t *testing.T) {
		oldFindHostRunNetnsTarget := findHostRunNetnsTarget
		oldScratchRoot := restoreHostRunScratchRoot
		findHostRunNetnsTarget = func() string {
			return "/host/run/netns/cni-live"
		}
		restoreHostRunScratchRoot = filepath.Join(t.TempDir(), "host-run")
		t.Cleanup(func() {
			findHostRunNetnsTarget = oldFindHostRunNetnsTarget
			restoreHostRunScratchRoot = oldScratchRoot
		})

		netnsPath := "/host/run/netns/cni-source"
		shmPath := "/host/run/nvidia/driver/dev/shm"
		m := &types.CheckpointManifest{
			CRIUDump: types.CRIUDumpManifest{
				ExtMnt: map[string]string{
					"/host-run":   hostRunRoot,
					"/driver-shm": shmPath,
					"/netns":      netnsPath,
				},
			},
		}
		mounts, stats, err := buildRestoreExtMounts(m, RestoreMountContext{})
		if err != nil {
			t.Fatalf("buildRestoreExtMounts: %v", err)
		}
		if stats.remappedHostRunParent != 1 {
			t.Fatalf("remapped host run parent mounts = %d, want 1", stats.remappedHostRunParent)
		}
		if stats.remappedHostNetns != 1 {
			t.Fatalf("remapped host netns mounts = %d, want 1", stats.remappedHostNetns)
		}
		if stats.hostNetnsTarget != "/host/run/netns/cni-live" {
			t.Fatalf("host netns target = %q, want live netns", stats.hostNetnsTarget)
		}
		if stats.hostRunScratchTarget != restoreHostRunScratchRoot {
			t.Fatalf("host run scratch target = %q, want %q", stats.hostRunScratchTarget, restoreHostRunScratchRoot)
		}

		mountMap := make(map[string]string, len(mounts))
		for _, em := range mounts {
			mountMap[em.GetKey()] = em.GetVal()
		}
		if mountMap[hostRunRoot] != restoreHostRunScratchRoot {
			t.Errorf("host run remap: got %q, want %q", mountMap[hostRunRoot], restoreHostRunScratchRoot)
		}
		if mountMap[netnsPath] != "/host/run/netns/cni-live" {
			t.Errorf("host netns remap: got %q, want live netns", mountMap[netnsPath])
		}

		mountpoints := remappedRestoreMountpoints(m, RestoreMountContext{})
		if mountpoints[restoreHostRunScratchRoot] != hostRunRoot {
			t.Fatalf("scratch root target = %q, want %q", mountpoints[restoreHostRunScratchRoot], hostRunRoot)
		}
		if mountpoints[filepath.Join(restoreHostRunScratchRoot, "netns", "cni-source")] != "/host/run/netns/cni-live" {
			t.Fatalf("scratch netns mountpoint target = %q, want live netns", mountpoints[filepath.Join(restoreHostRunScratchRoot, "netns", "cni-source")])
		}
		if mountpoints[filepath.Join(restoreHostRunScratchRoot, "nvidia", "driver", "dev", "shm")] != shmPath {
			t.Fatalf("scratch shm mountpoint target = %q, want %q", mountpoints[filepath.Join(restoreHostRunScratchRoot, "nvidia", "driver", "dev", "shm")], shmPath)
		}
		if _, ok := mountpoints[netnsPath]; ok {
			t.Fatal("real host netns mountpoint should not be prepared")
		}
		if _, ok := mountpoints[hostRunRoot]; ok {
			t.Fatal("real host run parent should not be prepared")
		}
	})
}

func TestRestoreNVIDIAExternalDevices(t *testing.T) {
	external := restoreNVIDIAExternalDevices([]string{
		"net[12345]:extNetNs",
		"dev[195/255]:nvidiactl",
		"dev[195/0]:nvidia0",
		"dev[510/1]:nvidia-cap1",
		"dev[1/3]:null",
		"dev[195/1]:nvidia-frontend",
	})

	got := map[string]struct{}{}
	for _, entry := range external {
		got[entry] = struct{}{}
	}

	want := []string{
		"dev[nvidia0]:/dev/nvidia0",
		"dev[nvidia-cap1]:/dev/nvidia-caps/nvidia-cap1",
		"dev[nvidiactl]:/dev/nvidiactl",
	}
	for _, entry := range want {
		if _, ok := got[entry]; !ok {
			t.Errorf("missing restore external device %q from %v", entry, external)
		}
	}
	if _, ok := got["dev[null]:/dev/null"]; ok {
		t.Errorf("non-NVIDIA device should not be restored: %v", external)
	}
	if _, ok := got["dev[nvidia-frontend]:/dev/nvidia-frontend"]; ok {
		t.Errorf("unknown NVIDIA-like name should not be restored: %v", external)
	}
}

func TestRemappedRestoreMountpointsIncludesHostContainerdOnly(t *testing.T) {
	containerdPath := "/host/run/containerd/io.containerd.runtime.v2.task/k8s.io/source/rootfs"
	netnsPath := "/host/run/netns/cni-source"
	m := &types.CheckpointManifest{
		CRIUDump: types.CRIUDumpManifest{
			ExtMnt: map[string]string{
				"/rootfs":       containerdPath,
				"/netns":        netnsPath,
				"/etc/hostname": "/etc/hostname",
			},
		},
	}

	mountpoints := remappedRestoreMountpoints(m, RestoreMountContext{})
	if mountpoints[containerdPath] != containerdPath {
		t.Fatalf("containerd mountpoint missing: got %q", mountpoints[containerdPath])
	}
	if _, ok := mountpoints[netnsPath]; ok {
		t.Fatal("netns mountpoint should not be prepared")
	}
	if _, ok := mountpoints["/etc/hostname"]; ok {
		t.Fatal("non-host containerd mountpoint should not be prepared")
	}
}

func TestRemappedRestoreMountpointsIncludesNestedRootfsChildren(t *testing.T) {
	oldFindHostRunNetnsTarget := findHostRunNetnsTarget
	findHostRunNetnsTarget = func() string {
		return "/host/run/netns/cni-live"
	}
	t.Cleanup(func() {
		findHostRunNetnsTarget = oldFindHostRunNetnsTarget
	})

	rootfsPath := "/host/run/containerd/io.containerd.runtime.v2.task/k8s.io/source/rootfs"
	kubeletPath := "/host/var/lib/kubelet/pods/source-pod/volumes/kubernetes.io~csi/pvc-abc/mount"
	netnsPath := "/host/run/netns/cni-source"
	m := &types.CheckpointManifest{
		CRIUDump: types.CRIUDumpManifest{
			ExtMnt: map[string]string{
				"/checkpoints": kubeletPath,
				"/netns":       netnsPath,
				"/rootfs":      rootfsPath,
			},
		},
	}

	mountpoints := remappedRestoreMountpoints(m, RestoreMountContext{})
	nestedKubeletPath := filepath.Join(rootfsPath, "var/lib/kubelet/pods/source-pod/volumes/kubernetes.io~csi/pvc-abc/mount")
	if mountpoints[nestedKubeletPath] != kubeletPath {
		t.Fatalf("nested kubelet mountpoint target = %q, want %q", mountpoints[nestedKubeletPath], kubeletPath)
	}
	nestedNetnsPath := filepath.Join(rootfsPath, "run/netns/cni-source")
	if mountpoints[nestedNetnsPath] != "/host/run/netns/cni-live" {
		t.Fatalf("nested netns mountpoint target = %q, want live netns", mountpoints[nestedNetnsPath])
	}
	if _, ok := mountpoints[netnsPath]; ok {
		t.Fatal("real host netns mountpoint should not be prepared")
	}
}

func TestCreateRestoreMountpoints(t *testing.T) {
	root := t.TempDir()
	targetDir := filepath.Join(root, "target-dir")
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		t.Fatalf("mkdir target dir: %v", err)
	}
	targetFile := filepath.Join(root, "target-file")
	if err := os.WriteFile(targetFile, []byte("x"), 0644); err != nil {
		t.Fatalf("write target file: %v", err)
	}

	sourceDir := filepath.Join(root, "old", "dir")
	sourceFile := filepath.Join(root, "old", "file")
	prepared, err := createRestoreMountpoints(map[string]string{
		sourceDir:                             targetDir,
		sourceFile:                            targetFile,
		filepath.Join(root, "missing-target"): filepath.Join(root, "does-not-exist"),
	})
	if err != nil {
		t.Fatalf("createRestoreMountpoints: %v", err)
	}
	if prepared != 2 {
		t.Fatalf("prepared = %d, want 2", prepared)
	}
	if info, err := os.Stat(sourceDir); err != nil || !info.IsDir() {
		t.Fatalf("source dir was not created: info=%v err=%v", info, err)
	}
	if info, err := os.Stat(sourceFile); err != nil || info.IsDir() {
		t.Fatalf("source file was not created: info=%v err=%v", info, err)
	}
}
