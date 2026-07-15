package runtime

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

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

func TestPrivilegedPlaceholderFirstRootComposition(t *testing.T) {
	if os.Getenv("DYNAMO_SNAPSHOT_PRIVILEGED_TEST") != "1" {
		t.Skip("set DYNAMO_SNAPSHOT_PRIVILEGED_TEST=1 and run as root")
	}
	root := os.Getenv("DYNAMO_SNAPSHOT_PRIVILEGED_HELPER_ROOT")
	if root == "" {
		if os.Geteuid() != 0 {
			t.Skip("privileged root composition test requires root")
		}
		root = filepath.Join(t.TempDir(), "root")
		if err := os.Mkdir(root, 0o755); err != nil {
			t.Fatal(err)
		}
		command := exec.Command(
			os.Args[0],
			"-test.run=^TestPrivilegedPlaceholderFirstRootComposition$",
		)
		command.Env = append(
			os.Environ(),
			"DYNAMO_SNAPSHOT_PRIVILEGED_HELPER_ROOT="+root,
		)
		command.SysProcAttr = &unix.SysProcAttr{Cloneflags: unix.CLONE_NEWNS}
		output, err := command.CombinedOutput()
		if err != nil {
			t.Fatalf("privileged root composition helper: %v\n%s", err, output)
		}
		return
	}
	runPrivilegedRootComposition(t, root)
}

func runPrivilegedRootComposition(t *testing.T, root string) {
	t.Helper()
	const (
		placeholderUID  = 1234
		placeholderGID  = 2345
		placeholderMode = 0o7751
		backingGID      = 3456
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
	write(placeholderLower, "delete-me", "placeholder")
	write(placeholderLower, "opaque/placeholder-only", "placeholder")
	write(placeholderLower, "opaque/shared/placeholder-nested", "placeholder")
	write(placeholderLower, "file-mount", "")
	write(placeholderLower, "mount-source", "file-mount")
	write(snapshotSource, "collision", "snapshot")
	write(snapshotSource, "snapshot-only", "snapshot")
	write(snapshotSource, "opaque/snapshot-only", "snapshot")
	write(snapshotSource, "opaque/shared/snapshot-nested", "snapshot")
	if err := os.WriteFile(
		filepath.Join(checkpoint, deletedFilesFilename),
		[]byte(`{"whiteouts":["delete-me","mounted/user-data"],"opaqueDirectories":["opaque"]}`),
		0o600,
	); err != nil {
		t.Fatal(err)
	}
	imagePath := filepath.Join(checkpoint, RootfsDiffFilename)
	command := exec.CommandContext(
		context.Background(),
		"mksquashfs",
		snapshotSource,
		imagePath,
		"-comp", "zstd",
		"-b", "1M",
		"-processors", "2",
		"-noappend",
		"-no-progress",
	)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("mksquashfs: %v\n%s", err, output)
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
	rootfsMount, err := PrepareDetachedRootfsMount(image)
	if err != nil {
		t.Fatal(err)
	}
	defer rootfsMount.Close()

	for _, path := range []string{
		"checkpoint",
		"proc",
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
	if err := unix.Mount(
		checkpoint,
		filepath.Join(root, "checkpoint"),
		"",
		unix.MS_BIND,
		"",
	); err != nil {
		t.Fatal(err)
	}
	if err := unix.Mount(
		"/proc",
		filepath.Join(root, "proc"),
		"",
		unix.MS_BIND|unix.MS_REC,
		"",
	); err != nil {
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
	if err := unix.Chown(backing, 0, backingGID); err != nil {
		t.Fatal(err)
	}
	if err := unix.Chmod(backing, 0o2700); err != nil {
		t.Fatal(err)
	}
	inheritanceProbe := filepath.Join(backing, "inheritance-probe")
	if err := os.Mkdir(inheritanceProbe, 0o700); err != nil {
		t.Fatal(err)
	}
	var inherited unix.Stat_t
	if err := unix.Stat(inheritanceProbe, &inherited); err != nil {
		t.Fatal(err)
	}
	if got := inherited.Mode & 0o7777; got != 0o2700 {
		t.Fatalf("setgid-inherited directory mode = %#o, want 02700", got)
	}
	if inherited.Gid != backingGID {
		t.Fatalf("setgid-inherited directory gid = %d, want %d", inherited.Gid, backingGID)
	}
	if err := os.Remove(inheritanceProbe); err != nil {
		t.Fatal(err)
	}
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
	workspaceFD, err := unix.Open(
		RootfsBackingWorkspaceMountPath,
		unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	deleted, err := os.Open("/checkpoint/" + deletedFilesFilename)
	if err != nil {
		t.Fatal(err)
	}
	defer deleted.Close()
	composition, err := ComposeRoot(
		int(rootfsMount.Fd()),
		workspaceFD,
		int(deleted.Fd()),
	)
	if err != nil {
		t.Fatal(err)
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
	for _, path := range []string{restoreUpper, composition.RootPath} {
		var got unix.Stat_t
		if err := unix.Stat(path, &got); err != nil {
			t.Fatal(err)
		}
		if got.Uid != placeholderMetadata.Uid ||
			got.Gid != placeholderMetadata.Gid ||
			got.Mode&0o7777 != placeholderMetadata.Mode&0o7777 {
			t.Fatalf(
				"%s metadata = uid %d gid %d mode %#o, want uid %d gid %d mode %#o",
				path,
				got.Uid,
				got.Gid,
				got.Mode&0o7777,
				placeholderMetadata.Uid,
				placeholderMetadata.Gid,
				placeholderMetadata.Mode&0o7777,
			)
		}
	}
	var replayedOpaque unix.Stat_t
	if err := unix.Stat(filepath.Join(restoreUpper, "opaque"), &replayedOpaque); err != nil {
		t.Fatal(err)
	}
	if replayedOpaque.Gid != placeholderMetadata.Gid {
		t.Fatalf(
			"replayed opaque directory gid = %d, want placeholder gid %d",
			replayedOpaque.Gid,
			placeholderMetadata.Gid,
		)
	}
	if replayedOpaque.Mode&unix.S_ISGID == 0 {
		t.Fatalf("replayed opaque directory mode = %#o, want setgid", replayedOpaque.Mode&0o7777)
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
	assertContent("/mounted/child-only", "mounted")
	assertContent("/file-mount", "file-mount")
	assertContent(staged("/collision"), "placeholder")
	assertContent(staged("/snapshot-only"), "snapshot")
	assertContent(staged("/opaque/snapshot-only"), "snapshot")
	assertContent(staged("/opaque/shared/snapshot-nested"), "snapshot")
	assertContent(staged("/file-mount"), "")
	for _, path := range []string{
		"/checkpoint/" + deletedFilesFilename,
		"/delete-me",
		"/mounted/child-only",
		"/opaque/placeholder-only",
		"/proc/self",
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
	for _, path := range []string{"/checkpoint", "/proc", "/mounted", "/file-mount"} {
		if _, err := os.Stat(staged(path)); err != nil {
			t.Fatalf("placeholder mountpoint %s is unavailable: %v", path, err)
		}
	}
}
