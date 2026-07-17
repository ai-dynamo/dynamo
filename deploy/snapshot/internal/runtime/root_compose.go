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

// RootComposition identifies the staged root and owns its backing mount.
type RootComposition struct {
	RootPath             string
	MountAttachDuration  time.Duration
	OverlaySetupDuration time.Duration
}

// ComposeRoot consumes a detached rootfs mount and workspace path handle,
// and stages a snapshot-over-placeholder OverlayFS without changing the caller root.
func ComposeRoot(rootfsMountFD, workspaceFD, deletedFilesFD int) (result *RootComposition, retErr error) {
	if rootfsMountFD >= 3 {
		defer unix.Close(rootfsMountFD)
	}
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
	composition := &RootComposition{RootPath: restoreRoot}
	defer func() {
		if retErr != nil {
			retErr = errors.Join(retErr, composition.Close())
			result = nil
		}
	}()
	entries, err := os.ReadDir(RootfsBackingWorkspaceMountPath)
	if err != nil {
		return nil, fmt.Errorf("read rootfs backing workspace for reset: %w", err)
	}
	for _, entry := range entries {
		if err := os.RemoveAll(filepath.Join(RootfsBackingWorkspaceMountPath, entry.Name())); err != nil {
			return nil, fmt.Errorf("reset rootfs backing workspace entry %q: %w", entry.Name(), err)
		}
	}

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
	snapshotRootFD, err := openDirectoryAt(workspaceMountFD, "snapshot")
	if err != nil {
		return nil, fmt.Errorf("open snapshot root metadata source: %w", err)
	}
	defer unix.Close(snapshotRootFD)
	upperRootFD, err := openDirectoryAt(workspaceMountFD, "upper")
	if err != nil {
		return nil, fmt.Errorf("open restore upper root: %w", err)
	}
	defer unix.Close(upperRootFD)
	if err := copyDirectoryOwnerAndMode(snapshotRootFD, upperRootFD); err != nil {
		return nil, fmt.Errorf("copy snapshot root owner and mode: %w", err)
	}
	placeholderRootFD, err := openDirectoryAt(workspaceMountFD, "placeholder")
	if err != nil {
		return nil, fmt.Errorf("open placeholder root metadata source: %w", err)
	}
	defer unix.Close(placeholderRootFD)
	composition.MountAttachDuration = time.Since(attachStart)

	overlayStart := time.Now()
	if err := replayDeletions(
		deletions,
		upperRootFD,
		snapshotRootFD,
		placeholderRootFD,
	); err != nil {
		return nil, err
	}
	if err := createWhiteout(filepath.Join(restoreUpper, strings.TrimPrefix(RootfsBackingWorkspaceMountPath, "/"))); err != nil {
		return nil, fmt.Errorf("hide rootfs backing workspace: %w", err)
	}
	options := strings.Join([]string{
		"lowerdir=" + restoreSnapshotLower + ":" + restorePlaceholder,
		"upperdir=" + restoreUpper,
		"workdir=" + restoreWork,
	}, ",")
	if err := unix.Mount("overlay", restoreRoot, "overlay", 0, options); err != nil {
		return nil, fmt.Errorf("mount snapshot-over-placeholder composed root: %w", err)
	}
	composition.OverlaySetupDuration = time.Since(overlayStart)
	return composition, nil
}

// Close lazily disconnects the composed mount tree.
func (composition *RootComposition) Close() error {
	if err := unmountPath(RootfsBackingWorkspaceMountPath, unix.MNT_DETACH); err != nil {
		return fmt.Errorf("unmount rootfs backing workspace: %w", err)
	}
	return nil
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

func copyDirectoryOwnerAndMode(sourceFD, targetFD int) error {
	var stat unix.Stat_t
	if err := unix.Fstat(sourceFD, &stat); err != nil {
		return fmt.Errorf("stat source directory: %w", err)
	}
	if err := unix.Fchown(targetFD, int(stat.Uid), int(stat.Gid)); err != nil {
		return fmt.Errorf("set target directory ownership: %w", err)
	}
	if err := unix.Fchmod(targetFD, stat.Mode&0o7777); err != nil {
		return fmt.Errorf("set target directory mode: %w", err)
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
	upperRootFD, snapshotRootFD, placeholderRootFD int,
) error {
	for _, entry := range deletions.OpaqueDirectory {
		if err := replayOpaqueDirectory(
			upperRootFD,
			entry,
			snapshotRootFD,
			placeholderRootFD,
		); err != nil {
			return err
		}
	}
	for _, entry := range deletions.Whiteouts {
		parent, name := filepath.Split(entry)
		parentFD, err := mkdirAllAt(
			upperRootFD,
			strings.TrimSuffix(parent, "/"),
			snapshotRootFD,
			placeholderRootFD,
		)
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

// An opaque directory hides placeholder children absent from the snapshot.
func replayOpaqueDirectory(
	upperRootFD int,
	relative string,
	snapshotRootFD, placeholderRootFD int,
) error {
	if relative == "." {
		relative = ""
	}
	snapshotIsDirectory := false
	snapshotFD, err := openDirectoryAt(snapshotRootFD, relative)
	if err == nil {
		snapshotIsDirectory = true
		unix.Close(snapshotFD)
	} else if !errors.Is(err, unix.ENOENT) {
		return fmt.Errorf("inspect snapshot opaque path %q: %w", relative, err)
	}
	placeholderFD, err := openDirectoryAt(placeholderRootFD, relative)
	if err == nil {
		unix.Close(placeholderFD)
	} else if snapshotIsDirectory &&
		(errors.Is(err, unix.ENOTDIR) || errors.Is(err, unix.ELOOP)) {
		// The snapshot directory wins, and there is no placeholder directory to traverse.
		return nil
	} else if !errors.Is(err, unix.ENOENT) {
		return fmt.Errorf("inspect placeholder opaque path %q: %w", relative, err)
	}
	upperFD, err := mkdirAllAt(
		upperRootFD,
		relative,
		snapshotRootFD,
		placeholderRootFD,
	)
	if err != nil {
		return fmt.Errorf("create opaque directory %q: %w", relative, err)
	}
	defer unix.Close(upperFD)
	placeholderEntries, err := directoryEntriesAt(placeholderRootFD, relative)
	if err != nil {
		return err
	}
	snapshotEntries, err := directoryEntriesAt(snapshotRootFD, relative)
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
				snapshotRootFD,
				placeholderRootFD,
			); err != nil {
				return err
			}
		}
	}
	return nil
}

func directoryEntriesAt(rootFD int, relative string) (map[string]uint32, error) {
	fd, err := openDirectoryAt(rootFD, relative)
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

func openDirectoryAt(rootFD int, relative string) (int, error) {
	if relative == "" {
		return unix.Dup(rootFD)
	}
	return unix.Openat2(rootFD, relative, &unix.OpenHow{
		Flags:   unix.O_RDONLY | unix.O_DIRECTORY | unix.O_CLOEXEC | unix.O_NOFOLLOW,
		Resolve: unix.RESOLVE_BENEATH | unix.RESOLVE_NO_SYMLINKS,
	})
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

func mkdirAllAt(
	rootFD int,
	path string,
	snapshotRootFD, placeholderRootFD int,
) (int, error) {
	current, err := unix.Dup(rootFD)
	if err != nil {
		return -1, err
	}
	if path == "" {
		return current, nil
	}
	var relative string
	for _, component := range strings.Split(path, "/") {
		if component == "" || component == "." || component == ".." {
			unix.Close(current)
			return -1, fmt.Errorf("invalid path component %q", component)
		}
		relative = filepath.Join(relative, component)
		next, err := openDirectoryAt(current, component)
		if errors.Is(err, unix.ENOENT) {
			if err := unix.Mkdirat(current, component, 0o755); err != nil &&
				!errors.Is(err, unix.EEXIST) {
				unix.Close(current)
				return -1, err
			}
			next, err = openDirectoryAt(current, component)
		}
		unix.Close(current)
		if err != nil {
			return -1, err
		}
		sourceFD, sourceErr := openDirectoryAt(snapshotRootFD, relative)
		if errors.Is(sourceErr, unix.ENOENT) {
			sourceFD, sourceErr = openDirectoryAt(placeholderRootFD, relative)
		}
		if sourceErr != nil {
			unix.Close(next)
			return -1, fmt.Errorf("open winning lower directory %q: %w", relative, sourceErr)
		}
		copyErr := copyDirectoryOwnerAndMode(sourceFD, next)
		unix.Close(sourceFD)
		if copyErr != nil {
			unix.Close(next)
			return -1, fmt.Errorf("copy lower directory owner and mode for %q: %w", relative, copyErr)
		}
		current = next
	}
	return current, nil
}
