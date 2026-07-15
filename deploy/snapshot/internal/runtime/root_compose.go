package runtime

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"golang.org/x/sys/unix"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
)

const (
	RootfsBackingWorkspaceMountPath = snapshotprotocol.RootfsBackingWorkspaceMountPath
	restoreRoot                     = RootfsBackingWorkspaceMountPath + "/root"
	restoreSnapshotLower            = RootfsBackingWorkspaceMountPath + "/snapshot"
	restorePlaceholder              = RootfsBackingWorkspaceMountPath + "/placeholder"
	restoreUpper                    = RootfsBackingWorkspaceMountPath + "/upper"
	restoreWork                     = RootfsBackingWorkspaceMountPath + "/work"
)

// RootComposition identifies the staged root and its setup phases.
type RootComposition struct {
	RootPath             string
	MountAttachDuration  time.Duration
	OverlaySetupDuration time.Duration
}

// ComposeRoot consumes a detached rootfs mount and workspace path handle,
// and stages a placeholder-first OverlayFS without changing the caller root.
func ComposeRoot(rootfsMountFD, workspaceFD, deletedFilesFD int) (*RootComposition, error) {
	if rootfsMountFD < 3 || workspaceFD < 3 {
		return nil, errors.New("rootfs mount and workspace FDs are required")
	}
	workspaceMountFD, err := unix.OpenTree(
		workspaceFD,
		"",
		unix.AT_EMPTY_PATH|unix.OPEN_TREE_CLONE|unix.OPEN_TREE_CLOEXEC,
	)
	unix.Close(workspaceFD)
	if err != nil {
		return nil, fmt.Errorf("clone rootfs backing workspace: %w", err)
	}
	defer unix.Close(workspaceMountFD)

	deletions, err := readDeletedFilesFD(deletedFilesFD)
	if err != nil {
		return nil, err
	}
	composition := &RootComposition{RootPath: restoreRoot}

	// Clone only the placeholder root mount. CRIU reconstructs child mounts.
	placeholderFD, err := unix.OpenTree(
		unix.AT_FDCWD,
		"/",
		unix.OPEN_TREE_CLONE|unix.OPEN_TREE_CLOEXEC,
	)
	if err != nil {
		return nil, fmt.Errorf("clone placeholder root: %w", err)
	}
	defer unix.Close(placeholderFD)

	attachStart := time.Now()
	backingTarget, err := unix.Open(
		RootfsBackingWorkspaceMountPath,
		unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		return nil, fmt.Errorf("open rootfs backing mountpoint: %w", err)
	}
	if err := moveMountToFD(workspaceMountFD, backingTarget); err != nil {
		unix.Close(backingTarget)
		return nil, fmt.Errorf("attach rootfs backing workspace: %w", err)
	}
	unix.Close(backingTarget)

	for _, path := range []string{
		restoreRoot,
		restoreSnapshotLower,
		restorePlaceholder,
		restoreUpper,
		restoreWork,
	} {
		if err := os.Mkdir(path, 0o700); err != nil && !errors.Is(err, os.ErrExist) {
			return nil, fmt.Errorf("create root composition directory %s: %w", path, err)
		}
	}
	snapshotTarget, err := unix.Open(
		restoreSnapshotLower,
		unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		return nil, fmt.Errorf("open SquashFS mountpoint: %w", err)
	}
	if err := moveMountToFD(rootfsMountFD, snapshotTarget); err != nil {
		unix.Close(snapshotTarget)
		return nil, fmt.Errorf("attach inherited SquashFS mount: %w", err)
	}
	unix.Close(snapshotTarget)
	unix.Close(rootfsMountFD)

	placeholderTarget, err := unix.Open(
		restorePlaceholder,
		unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		return nil, fmt.Errorf("open placeholder mountpoint: %w", err)
	}
	if err := moveMountToFD(placeholderFD, placeholderTarget); err != nil {
		unix.Close(placeholderTarget)
		return nil, fmt.Errorf("attach placeholder root: %w", err)
	}
	unix.Close(placeholderTarget)
	if err := copyDirectoryMetadataAt(workspaceMountFD, "placeholder", "upper"); err != nil {
		return nil, fmt.Errorf("copy placeholder root metadata: %w", err)
	}
	composition.MountAttachDuration = time.Since(attachStart)

	overlayStart := time.Now()
	if err := replayDeletions(
		deletions,
		restoreUpper,
		restorePlaceholder,
		restoreSnapshotLower,
	); err != nil {
		return nil, err
	}
	if err := createWhiteout(filepath.Join(restoreUpper, strings.TrimPrefix(RootfsBackingWorkspaceMountPath, "/"))); err != nil {
		return nil, fmt.Errorf("hide rootfs backing workspace: %w", err)
	}
	options := strings.Join([]string{
		"lowerdir=" + restorePlaceholder + ":" + restoreSnapshotLower,
		"upperdir=" + restoreUpper,
		"workdir=" + restoreWork,
	}, ",")
	if err := unix.Mount("overlay", restoreRoot, "overlay", 0, options); err != nil {
		return nil, fmt.Errorf("mount placeholder-first composed root: %w", err)
	}
	composition.OverlaySetupDuration = time.Since(overlayStart)
	return composition, nil
}

func moveMountToFD(sourceFD, targetFD int) error {
	return unix.MoveMount(
		sourceFD,
		"",
		targetFD,
		"",
		unix.MOVE_MOUNT_F_EMPTY_PATH|unix.MOVE_MOUNT_T_EMPTY_PATH,
	)
}

func copyDirectoryMetadataAt(rootFD int, source, target string) error {
	const flags = unix.O_RDONLY | unix.O_DIRECTORY | unix.O_NOFOLLOW | unix.O_CLOEXEC
	sourceFD, err := unix.Openat(rootFD, source, flags, 0)
	if err != nil {
		return fmt.Errorf("open source directory %q: %w", source, err)
	}
	defer unix.Close(sourceFD)
	targetFD, err := unix.Openat(rootFD, target, flags, 0)
	if err != nil {
		return fmt.Errorf("open target directory %q: %w", target, err)
	}
	defer unix.Close(targetFD)

	var stat unix.Stat_t
	if err := unix.Fstat(sourceFD, &stat); err != nil {
		return fmt.Errorf("stat source directory %q: %w", source, err)
	}
	if err := unix.Fchown(targetFD, int(stat.Uid), int(stat.Gid)); err != nil {
		return fmt.Errorf("set target directory %q ownership: %w", target, err)
	}
	if err := unix.Fchmod(targetFD, stat.Mode&0o7777); err != nil {
		return fmt.Errorf("set target directory %q mode: %w", target, err)
	}
	return nil
}

func createWhiteout(path string) error {
	err := unix.Mknod(path, unix.S_IFCHR, int(unix.Mkdev(0, 0)))
	if !errors.Is(err, unix.EEXIST) {
		return err
	}
	var stat unix.Stat_t
	if err := unix.Lstat(path, &stat); err != nil {
		return err
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFCHR || stat.Rdev != unix.Mkdev(0, 0) {
		return fmt.Errorf("%s collides with reserved rootfs backing path", path)
	}
	return nil
}

func readDeletedFilesFD(deletedFilesFD int) (*deletedFiles, error) {
	if deletedFilesFD < 0 {
		return &deletedFiles{}, nil
	}
	duplicate, err := unix.Dup(deletedFilesFD)
	if err != nil {
		return nil, fmt.Errorf("duplicate deletion sidecar FD: %w", err)
	}
	file := os.NewFile(uintptr(duplicate), deletedFilesFilename)
	defer file.Close()
	return readDeletedFilesFile(file)
}

func replayDeletions(
	deletions *deletedFiles,
	upperDir, placeholderRoot, snapshotRoot string,
) error {
	rootFD, err := unix.Open(
		upperDir,
		unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		return fmt.Errorf("open restore upperdir: %w", err)
	}
	defer unix.Close(rootFD)
	for _, entry := range deletions.OpaqueDirectory {
		if err := replayOpaqueDirectory(
			rootFD,
			entry,
			placeholderRoot,
			snapshotRoot,
		); err != nil {
			return err
		}
	}
	for _, entry := range deletions.Whiteouts {
		parent, name := filepath.Split(entry)
		parentFD, err := mkdirAllAt(rootFD, strings.TrimSuffix(parent, "/"))
		if err != nil {
			return fmt.Errorf("create whiteout parent for %q: %w", entry, err)
		}
		err = createWhiteoutAt(parentFD, name)
		unix.Close(parentFD)
		if err != nil {
			return fmt.Errorf("create whiteout %q: %w", entry, err)
		}
	}
	return nil
}

// An opaque directory hides the placeholder layer but must not hide files
// supplied by the SquashFS layer below it.
func replayOpaqueDirectory(
	upperRootFD int,
	relative, placeholderRoot, snapshotRoot string,
) error {
	if relative == "." {
		relative = ""
	}
	if info, err := os.Lstat(filepath.Join(placeholderRoot, relative)); err == nil {
		if !info.IsDir() {
			return nil
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect placeholder opaque path %q: %w", relative, err)
	}
	if info, err := os.Lstat(filepath.Join(snapshotRoot, relative)); err == nil {
		if !info.IsDir() {
			return fmt.Errorf("snapshot opaque path %q is not a directory", relative)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("inspect snapshot opaque path %q: %w", relative, err)
	}
	upperFD, err := mkdirAllAt(upperRootFD, relative)
	if err != nil {
		return fmt.Errorf("create opaque directory %q: %w", relative, err)
	}
	defer unix.Close(upperFD)
	placeholderEntries, err := directoryEntries(placeholderRoot, relative)
	if err != nil {
		return err
	}
	snapshotEntries, err := directoryEntries(snapshotRoot, relative)
	if err != nil {
		return err
	}
	for name, placeholderType := range placeholderEntries {
		snapshotType, suppliedBySnapshot := snapshotEntries[name]
		if !suppliedBySnapshot {
			if err := createWhiteoutAt(upperFD, name); err != nil {
				return err
			}
		} else if placeholderType == unix.S_IFDIR && snapshotType == unix.S_IFDIR {
			if err := replayOpaqueDirectory(
				upperRootFD,
				filepath.Join(relative, name),
				placeholderRoot,
				snapshotRoot,
			); err != nil {
				return err
			}
		}
	}
	return nil
}

func directoryEntries(root, relative string) (map[string]uint32, error) {
	rootFD, err := unix.Open(
		root,
		unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		return nil, err
	}
	defer unix.Close(rootFD)
	fd, err := unix.Dup(rootFD)
	if err != nil {
		return nil, err
	}
	if relative != "" {
		unix.Close(fd)
		fd, err = unix.Openat2(rootFD, relative, &unix.OpenHow{
			Flags:   unix.O_RDONLY | unix.O_DIRECTORY | unix.O_CLOEXEC | unix.O_NOFOLLOW,
			Resolve: unix.RESOLVE_BENEATH | unix.RESOLVE_NO_SYMLINKS,
		})
	}
	if errors.Is(err, unix.ENOENT) {
		return map[string]uint32{}, nil
	}
	if err != nil {
		return nil, err
	}
	file := os.NewFile(uintptr(fd), relative)
	defer file.Close()
	entries, err := file.ReadDir(-1)
	if err != nil {
		return nil, err
	}
	result := make(map[string]uint32, len(entries))
	for _, entry := range entries {
		var stat unix.Stat_t
		if err := unix.Fstatat(
			int(file.Fd()),
			entry.Name(),
			&stat,
			unix.AT_SYMLINK_NOFOLLOW,
		); err != nil {
			return nil, err
		}
		result[entry.Name()] = stat.Mode & unix.S_IFMT
	}
	return result, nil
}

func createWhiteoutAt(parentFD int, name string) error {
	err := unix.Mknodat(parentFD, name, unix.S_IFCHR, int(unix.Mkdev(0, 0)))
	if !errors.Is(err, unix.EEXIST) {
		return err
	}
	var stat unix.Stat_t
	if err := unix.Fstatat(parentFD, name, &stat, unix.AT_SYMLINK_NOFOLLOW); err != nil {
		return err
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFCHR || stat.Rdev != unix.Mkdev(0, 0) {
		return unix.EEXIST
	}
	return nil
}

func mkdirAllAt(rootFD int, path string) (int, error) {
	current, err := unix.Dup(rootFD)
	if err != nil {
		return -1, err
	}
	if path == "" {
		return current, nil
	}
	for _, component := range strings.Split(path, "/") {
		if component == "" || component == "." || component == ".." {
			unix.Close(current)
			return -1, fmt.Errorf("invalid path component %q", component)
		}
		next, err := unix.Openat2(current, component, &unix.OpenHow{
			Flags:   unix.O_RDONLY | unix.O_DIRECTORY | unix.O_CLOEXEC,
			Resolve: unix.RESOLVE_BENEATH | unix.RESOLVE_NO_SYMLINKS,
		})
		if errors.Is(err, unix.ENOENT) {
			if err := unix.Mkdirat(current, component, 0o755); err != nil &&
				!errors.Is(err, unix.EEXIST) {
				unix.Close(current)
				return -1, err
			}
			next, err = unix.Openat2(current, component, &unix.OpenHow{
				Flags:   unix.O_RDONLY | unix.O_DIRECTORY | unix.O_CLOEXEC,
				Resolve: unix.RESOLVE_BENEATH | unix.RESOLVE_NO_SYMLINKS,
			})
		}
		unix.Close(current)
		if err != nil {
			return -1, err
		}
		current = next
	}
	return current, nil
}
