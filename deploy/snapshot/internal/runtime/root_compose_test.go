package runtime

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestMkdirAllAtWithoutLowerMetadata(t *testing.T) {
	root := t.TempDir()
	upper := filepath.Join(root, "upper")
	snapshot := filepath.Join(root, "snapshot")
	placeholder := filepath.Join(root, "placeholder")
	for _, path := range []string{upper, snapshot, placeholder} {
		if err := os.Mkdir(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	open := func(path string) int {
		t.Helper()
		fd, err := unix.Open(path, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { unix.Close(fd) })
		return fd
	}
	upperFD := open(upper)
	snapshotFD := open(snapshot)
	placeholderFD := open(placeholder)

	const missing = "home/dynamo/.cache/huggingface"
	createdFD, err := mkdirAllAt(upperFD, missing, snapshotFD, placeholderFD)
	if err != nil {
		t.Fatal(err)
	}
	unix.Close(createdFD)
	if info, err := os.Stat(filepath.Join(upper, missing)); err != nil {
		t.Fatal(err)
	} else if !info.IsDir() {
		t.Fatalf("created path mode = %v, want directory", info.Mode())
	}

	if err := os.Symlink(".", filepath.Join(placeholder, "symlink")); err != nil {
		t.Fatal(err)
	}
	if fd, err := mkdirAllAt(upperFD, "symlink/child", snapshotFD, placeholderFD); err == nil {
		unix.Close(fd)
		t.Fatal("mkdirAllAt unexpectedly ignored lower symlink")
	} else if !errors.Is(err, unix.ENOTDIR) && !errors.Is(err, unix.ELOOP) {
		t.Fatalf("mkdirAllAt lower symlink error = %v, want ENOTDIR or ELOOP", err)
	}
}

func TestPrivilegedWorkspaceHandleCloneAcrossMountNamespaces(t *testing.T) {
	const (
		ownerRootEnv = "DYNAMO_SNAPSHOT_WORKSPACE_OWNER_ROOT"
		cloneEnv     = "DYNAMO_SNAPSHOT_WORKSPACE_CLONE_HELPER"
	)
	if os.Getenv(cloneEnv) == "1" {
		fd, err := unix.OpenTree(
			4,
			"",
			unix.AT_EMPTY_PATH|unix.OPEN_TREE_CLONE|unix.OPEN_TREE_CLOEXEC,
		)
		if err != nil {
			t.Fatalf("clone workspace handle in owning mount namespace: %v", err)
		}
		unix.Close(fd)
		return
	}
	if root := os.Getenv(ownerRootEnv); root != "" {
		if err := unix.Mount("", "/", "", unix.MS_PRIVATE|unix.MS_REC, ""); err != nil {
			t.Fatal(err)
		}
		workspace := filepath.Join(root, "workspace")
		if err := unix.Mount("tmpfs", workspace, "tmpfs", 0, "mode=700"); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "ready"), nil, 0o600); err != nil {
			t.Fatal(err)
		}
		release := filepath.Join(root, "release")
		for {
			if _, err := os.Stat(release); err == nil {
				return
			} else if !errors.Is(err, os.ErrNotExist) {
				t.Fatal(err)
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	if os.Getenv("DYNAMO_SNAPSHOT_PRIVILEGED_TEST") != "1" || os.Geteuid() != 0 {
		t.Skip("set DYNAMO_SNAPSHOT_PRIVILEGED_TEST=1 and run as root")
	}

	root := t.TempDir()
	workspace := filepath.Join(root, "workspace")
	if err := os.Mkdir(workspace, 0o700); err != nil {
		t.Fatal(err)
	}
	var ownerOutput bytes.Buffer
	owner := exec.Command(
		os.Args[0],
		"-test.run=^TestPrivilegedWorkspaceHandleCloneAcrossMountNamespaces$",
	)
	owner.Env = append(os.Environ(), ownerRootEnv+"="+root)
	owner.SysProcAttr = &unix.SysProcAttr{Cloneflags: unix.CLONE_NEWNS}
	owner.Stdout = &ownerOutput
	owner.Stderr = &ownerOutput
	if err := owner.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.WriteFile(filepath.Join(root, "release"), nil, 0o600); err != nil {
			t.Error(err)
			_ = owner.Process.Kill()
		}
		if err := owner.Wait(); err != nil {
			t.Errorf("workspace owner: %v\n%s", err, ownerOutput.Bytes())
		}
	}()

	ready := filepath.Join(root, "ready")
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		} else if !errors.Is(err, os.ErrNotExist) {
			t.Fatal(err)
		}
		if time.Now().After(deadline) {
			t.Fatalf("workspace owner did not become ready\n%s", ownerOutput.Bytes())
		}
		time.Sleep(10 * time.Millisecond)
	}

	foreignPath := fmt.Sprintf("/proc/%d/root%s", owner.Process.Pid, workspace)
	fd, err := unix.OpenTree(
		unix.AT_FDCWD,
		foreignPath,
		unix.OPEN_TREE_CLONE|unix.OPEN_TREE_CLOEXEC,
	)
	if err == nil {
		unix.Close(fd)
		t.Fatal("foreign mount namespace clone unexpectedly succeeded")
	}
	if !errors.Is(err, unix.EINVAL) {
		t.Fatalf("foreign mount namespace clone error = %v, want EINVAL", err)
	}

	workspaceFD, err := unix.Open(
		foreignPath,
		unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	workspaceHandle := os.NewFile(uintptr(workspaceFD), "workspace-handle")
	defer workspaceHandle.Close()
	dummy, err := os.Open("/dev/null")
	if err != nil {
		t.Fatal(err)
	}
	defer dummy.Close()
	clone := exec.Command(
		"nsenter",
		"-t", fmt.Sprint(owner.Process.Pid),
		"-m",
		"--", os.Args[0],
		"-test.run=^TestPrivilegedWorkspaceHandleCloneAcrossMountNamespaces$",
	)
	clone.Env = append(os.Environ(), cloneEnv+"=1")
	clone.ExtraFiles = []*os.File{dummy, workspaceHandle}
	if output, err := clone.CombinedOutput(); err != nil {
		t.Fatalf("clone inherited workspace handle after nsenter: %v\n%s", err, output)
	}
}

func TestPrivilegedSnapshotOverPlaceholderRootComposition(t *testing.T) {
	if os.Getenv("DYNAMO_SNAPSHOT_PRIVILEGED_TEST") != "1" {
		t.Skip("set DYNAMO_SNAPSHOT_PRIVILEGED_TEST=1 and run as root")
	}
	root := os.Getenv("DYNAMO_SNAPSHOT_PRIVILEGED_HELPER_ROOT")
	if root == "" {
		if os.Geteuid() != 0 {
			t.Skip("privileged root composition test requires root")
		}
		requireTool(t, "mksquashfs")
		root = filepath.Join(t.TempDir(), "root")
		if err := os.Mkdir(root, 0o755); err != nil {
			t.Fatal(err)
		}
		probeXattr := "user.dynamo-snapshot-xattr-probe"
		if err := unix.Setxattr(root, probeXattr, []byte("probe"), 0); err != nil {
			if errors.Is(err, unix.ENOTSUP) {
				t.Skipf("test filesystem does not support user xattrs: %v", err)
			}
			t.Fatal(err)
		}
		if err := unix.Removexattr(root, probeXattr); err != nil {
			t.Fatal(err)
		}
		stagingDir := t.TempDir()
		command := exec.Command(
			os.Args[0],
			"-test.run=^TestPrivilegedSnapshotOverPlaceholderRootComposition$",
		)
		command.Env = append(
			os.Environ(),
			"DYNAMO_SNAPSHOT_PRIVILEGED_HELPER_ROOT="+root,
			"DYNAMO_SNAPSHOT_PRIVILEGED_STAGING_DIR="+stagingDir,
		)
		command.SysProcAttr = &unix.SysProcAttr{Cloneflags: unix.CLONE_NEWNS}
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("privileged root composition helper: %v\n%s", err, output)
		}
		return
	}
	stagingDir := os.Getenv("DYNAMO_SNAPSHOT_PRIVILEGED_STAGING_DIR")
	if stagingDir == "" {
		t.Fatal("privileged staging directory is required")
	}
	runPrivilegedRootComposition(t, root, stagingDir)
}

func runPrivilegedRootComposition(t *testing.T, root, stagingDir string) {
	t.Helper()
	const (
		placeholderUID  = 1234
		placeholderGID  = 2345
		placeholderMode = 0o7751
		snapshotUID     = 3456
		snapshotGID     = 4567
		snapshotMode    = 0o3750
		opaqueUID       = 5678
		opaqueGID       = 6789
		opaqueMode      = 0o1750
		fileXattr       = "user.dynamo-snapshot-test"
		fileXattrValue  = "checkpoint"
	)
	requireTool(t, "mksquashfs")
	if err := unix.Mount("", "/", "", unix.MS_PRIVATE|unix.MS_REC, ""); err != nil {
		t.Fatal(err)
	}
	base := filepath.Dir(root)
	placeholderLower := filepath.Join(base, "placeholder-lower")
	placeholderUpper := filepath.Join(base, "placeholder-upper")
	placeholderWork := filepath.Join(base, "placeholder-work")
	checkpoint := filepath.Join(base, "checkpoint")
	snapshotSource := filepath.Join(base, "snapshot")
	for _, path := range []string{
		placeholderLower,
		placeholderUpper,
		placeholderWork,
		checkpoint,
		filepath.Join(snapshotSource, "opaque/shared"),
	} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	write := func(root, name, content string) {
		t.Helper()
		path := filepath.Join(root, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(placeholderLower, "collision", "placeholder")
	write(placeholderLower, "placeholder-only", "placeholder")
	write(placeholderLower, "delete-me", "placeholder")
	write(placeholderLower, "recreated/old-child", "placeholder")
	write(placeholderLower, "opaque/placeholder-only", "placeholder")
	write(placeholderLower, "opaque/shared/placeholder-nested", "placeholder")
	write(placeholderLower, "file-mount", "")
	write(placeholderLower, "mount-source", "file-mount")
	write(snapshotSource, "collision", "snapshot")
	write(snapshotSource, "snapshot-only", "snapshot")
	write(snapshotSource, "recreated", "snapshot-file")
	write(snapshotSource, "opaque/snapshot-only", "snapshot")
	write(snapshotSource, "opaque/shared/snapshot-nested", "snapshot")
	if err := unix.Setxattr(
		filepath.Join(snapshotSource, "snapshot-only"),
		fileXattr,
		[]byte(fileXattrValue),
		0,
	); err != nil {
		t.Fatal(err)
	}
	if err := unix.Chown(
		filepath.Join(snapshotSource, "opaque"),
		opaqueUID,
		opaqueGID,
	); err != nil {
		t.Fatal(err)
	}
	if err := unix.Chmod(filepath.Join(snapshotSource, "opaque"), opaqueMode); err != nil {
		t.Fatal(err)
	}
	if err := unix.Chown(snapshotSource, snapshotUID, snapshotGID); err != nil {
		t.Fatal(err)
	}
	if err := unix.Chmod(snapshotSource, snapshotMode); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(checkpoint, deletedFilesFilename),
		[]byte(`{"whiteouts":["delete-me","mounted/user-data"],"opaqueDirectories":["opaque"]}`),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	imagePath := filepath.Join(checkpoint, RootfsDiffFilename)
	if err := runMksquashfs(
		context.Background(),
		snapshotSource,
		imagePath,
		nil,
	); err != nil {
		t.Fatal(err)
	}

	// Match the agent pod: only loop-control is host-mounted. Loop nodes live
	// in the test namespace's ephemeral tmpfs.
	if err := os.MkdirAll(hostDevicePath, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := unix.Mount("tmpfs", hostDevicePath, "tmpfs", 0, "mode=700"); err != nil {
		t.Fatal(err)
	}
	loopControl := filepath.Join(hostDevicePath, "loop-control")
	if err := os.WriteFile(loopControl, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := unix.Mount("/dev/loop-control", loopControl, "", unix.MS_BIND, ""); err != nil {
		t.Fatal(err)
	}
	image, err := os.Open(imagePath)
	if err != nil {
		t.Fatal(err)
	}
	defer image.Close()
	deleted, err := os.Open(filepath.Join(checkpoint, deletedFilesFilename))
	if err != nil {
		t.Fatal(err)
	}
	defer deleted.Close()
	type mountResult struct {
		mount *os.File
		err   error
	}
	results := make(chan mountResult, 9)
	for range 9 {
		go func() {
			mount, err := prepareDetachedRootfsMount(image, stagingDir)
			results <- mountResult{mount, err}
		}()
	}
	var preparationErr error
	var preparedMounts []*os.File
	for range 9 {
		result := <-results
		if result.err != nil {
			preparationErr = errors.Join(preparationErr, result.err)
			continue
		}
		preparedMounts = append(preparedMounts, result.mount)
	}
	for _, mount := range preparedMounts {
		if err := mount.Close(); err != nil {
			preparationErr = errors.Join(preparationErr, err)
		}
	}
	if preparationErr != nil {
		t.Fatal(preparationErr)
	}
	mountinfo, err := os.ReadFile("/proc/self/mountinfo")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(mountinfo), " "+stagingDir+"/") {
		t.Fatal("concurrent preparation leaked a staging mount")
	}
	assertDirectoryEmpty(t, stagingDir)
	firstRootfsMount, err := prepareDetachedRootfsMount(image, stagingDir)
	if err != nil {
		t.Fatal(err)
	}
	secondRootfsMount, err := prepareDetachedRootfsMount(image, stagingDir)
	if err != nil {
		t.Fatal(err)
	}

	for _, path := range []string{
		"mounted",
		strings.TrimPrefix(RootfsBackingWorkspaceMountPath, "/"),
	} {
		if err := os.MkdirAll(filepath.Join(placeholderLower, path), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	options := strings.Join([]string{
		"lowerdir=" + placeholderLower,
		"upperdir=" + placeholderUpper,
		"workdir=" + placeholderWork,
	}, ",")
	if err := unix.Mount("overlay", root, "overlay", 0, options); err != nil {
		t.Fatal(err)
	}
	mounted := filepath.Join(root, "mounted")
	if err := unix.Mount("tmpfs", mounted, "tmpfs", 0, "mode=700"); err != nil {
		t.Fatal(err)
	}
	write(mounted, "child-only", "mounted")
	if err := unix.Mount(
		filepath.Join(root, "mount-source"),
		filepath.Join(root, "file-mount"),
		"",
		unix.MS_BIND,
		"",
	); err != nil {
		t.Fatal(err)
	}
	backing := filepath.Join(root, RootfsBackingWorkspaceMountPath)
	if err := unix.Mount("tmpfs", backing, "tmpfs", 0, "mode=700"); err != nil {
		t.Fatal(err)
	}
	procMountinfo, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		t.Fatal(err)
	}
	defer procMountinfo.Close()
	if err := unix.Chown(root, placeholderUID, placeholderGID); err != nil {
		t.Fatal(err)
	}
	if err := unix.Chmod(root, placeholderMode); err != nil {
		t.Fatal(err)
	}
	if err := unix.Chroot(root); err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir("/"); err != nil {
		t.Fatal(err)
	}
	callerRoot, err := os.Stat("/")
	if err != nil {
		t.Fatal(err)
	}
	var placeholderMetadata unix.Stat_t
	if err := unix.Stat("/", &placeholderMetadata); err != nil {
		t.Fatal(err)
	}
	if placeholderMetadata.Uid != placeholderUID ||
		placeholderMetadata.Gid != placeholderGID ||
		placeholderMetadata.Mode&0o7777 != placeholderMode {
		t.Fatalf(
			"placeholder root metadata = uid %d gid %d mode %#o, want uid %d gid %d mode %#o",
			placeholderMetadata.Uid,
			placeholderMetadata.Gid,
			placeholderMetadata.Mode&0o7777,
			placeholderUID,
			placeholderGID,
			placeholderMode,
		)
	}
	mountCount := func() int {
		t.Helper()
		if _, err := procMountinfo.Seek(0, 0); err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(procMountinfo)
		if err != nil {
			t.Fatal(err)
		}
		count := 0
		for _, line := range strings.Split(string(data), "\n") {
			fields := strings.Fields(line)
			if len(fields) >= 5 &&
				(fields[4] == RootfsBackingWorkspaceMountPath ||
					strings.HasPrefix(fields[4], RootfsBackingWorkspaceMountPath+"/")) {
				count++
			}
		}
		return count
	}
	baselineMounts := mountCount()
	compose := func(rootfsMount *os.File) *RootComposition {
		t.Helper()
		rootfsMountFD, err := unix.Dup(int(rootfsMount.Fd()))
		if err != nil {
			t.Fatal(err)
		}
		workspaceFD, err := unix.Open(
			RootfsBackingWorkspaceMountPath,
			unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
			0,
		)
		if err != nil {
			unix.Close(rootfsMountFD)
			t.Fatal(err)
		}
		composition, composeErr := ComposeRoot(
			rootfsMountFD,
			workspaceFD,
			int(deleted.Fd()),
		)
		closeErr := rootfsMount.Close()
		if composeErr != nil {
			t.Fatal(composeErr)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		return composition
	}
	firstComposition := compose(firstRootfsMount)
	staleName := fmt.Sprintf("stale-after-failed-restore-%d", os.Getpid())
	if err := os.WriteFile(
		filepath.Join(firstComposition.RootPath, staleName),
		[]byte("stale"),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	if err := firstComposition.Close(); err != nil {
		t.Fatal(err)
	}
	if got := mountCount(); got != baselineMounts {
		t.Fatalf("mounts after failed restore cleanup = %d, want %d", got, baselineMounts)
	}

	composition := compose(secondRootfsMount)
	if _, err := os.Stat(filepath.Join(composition.RootPath, staleName)); !os.IsNotExist(err) {
		t.Fatalf("stale file survived retry composition: %v", err)
	}
	if composition.RootPath != restoreRoot {
		t.Fatalf("composed root = %q, want %q", composition.RootPath, restoreRoot)
	}
	if cwd, err := os.Getwd(); err != nil || cwd != "/" {
		t.Fatalf("caller cwd = %q, %v; want /", cwd, err)
	}
	currentRoot, err := os.Stat("/")
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(callerRoot, currentRoot) {
		t.Fatal("composition changed the caller root")
	}
	if _, err := os.Stat("/snapshot-only"); !os.IsNotExist(err) {
		t.Fatalf("snapshot content leaked into caller root: %v", err)
	}
	if _, err := os.Stat(RootfsBackingWorkspaceMountPath); err != nil {
		t.Fatalf("caller backing workspace is unavailable: %v", err)
	}
	var composedRoot unix.Stat_t
	if err := unix.Stat(composition.RootPath, &composedRoot); err != nil {
		t.Fatal(err)
	}
	if composedRoot.Uid != snapshotUID ||
		composedRoot.Gid != snapshotGID ||
		composedRoot.Mode&0o7777 != snapshotMode {
		t.Fatalf(
			"composed root metadata = uid %d gid %d mode %#o, want uid %d gid %d mode %#o",
			composedRoot.Uid,
			composedRoot.Gid,
			composedRoot.Mode&0o7777,
			snapshotUID,
			snapshotGID,
			snapshotMode,
		)
	}

	assertContent := func(path, want string) {
		t.Helper()
		content, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if string(content) != want {
			t.Fatalf("%s = %q, want %q", path, content, want)
		}
	}
	staged := func(path string) string {
		return filepath.Join(composition.RootPath, strings.TrimPrefix(path, "/"))
	}
	var replayedOpaque unix.Stat_t
	if err := unix.Stat(staged("/opaque"), &replayedOpaque); err != nil {
		t.Fatal(err)
	}
	if replayedOpaque.Uid != opaqueUID ||
		replayedOpaque.Gid != opaqueGID ||
		replayedOpaque.Mode&0o7777 != opaqueMode {
		t.Fatalf(
			"replayed opaque metadata = uid %d gid %d mode %#o, want uid %d gid %d mode %#o",
			replayedOpaque.Uid,
			replayedOpaque.Gid,
			replayedOpaque.Mode&0o7777,
			opaqueUID,
			opaqueGID,
			opaqueMode,
		)
	}
	value := make([]byte, len(fileXattrValue))
	size, err := unix.Getxattr(staged("/snapshot-only"), fileXattr, value)
	if err != nil {
		t.Fatal(err)
	}
	if string(value[:size]) != fileXattrValue {
		t.Fatalf("snapshot file xattr = %q, want %q", value[:size], fileXattrValue)
	}
	assertContent("/mounted/child-only", "mounted")
	assertContent("/file-mount", "file-mount")
	assertContent(staged("/collision"), "snapshot")
	assertContent(staged("/placeholder-only"), "placeholder")
	assertContent(staged("/snapshot-only"), "snapshot")
	assertContent(staged("/recreated"), "snapshot-file")
	var recreated unix.Stat_t
	if err := unix.Lstat(staged("/recreated"), &recreated); err != nil {
		t.Fatal(err)
	}
	if recreated.Mode&unix.S_IFMT != unix.S_IFREG {
		t.Fatalf("recreated path mode = %#o, want regular file", recreated.Mode)
	}
	assertContent(staged("/opaque/snapshot-only"), "snapshot")
	assertContent(staged("/opaque/shared/snapshot-nested"), "snapshot")
	assertContent(staged("/file-mount"), "")
	for _, path := range []string{
		"/delete-me",
		"/mounted/child-only",
		"/opaque/placeholder-only",
		"/opaque/shared/placeholder-nested",
	} {
		if _, err := os.Lstat(staged(path)); !os.IsNotExist(err) {
			t.Fatalf("%s should be hidden, got %v", path, err)
		}
	}
	for _, path := range []string{
		RootfsBackingWorkspaceMountPath,
		RootfsBackingWorkspaceMountPath + "/upper",
		RootfsBackingWorkspaceMountPath + "/work",
	} {
		if _, err := os.Stat(staged(path)); !os.IsNotExist(err) {
			t.Fatalf("rootfs backing path %s should be inaccessible, got %v", path, err)
		}
	}
	for _, path := range []string{"/mounted", "/file-mount"} {
		if _, err := os.Stat(staged(path)); err != nil {
			t.Fatalf("placeholder mountpoint %s is unavailable: %v", path, err)
		}
	}
	if err := composition.Close(); err != nil {
		t.Fatal(err)
	}
	if got := mountCount(); got != baselineMounts {
		t.Fatalf("mounts after composition close = %d, want %d", got, baselineMounts)
	}
}
