package runtime

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const (
	deletedFilesFilename    = "deleted-files.json"
	rootfsDirectoryFilename = "rootfs-diff"
	rootfsMetadataFilename  = "rootfs-diff.meta.json"
	rootfsDirectoryFormat   = "dynamo-rootfs-directory"
	rootfsDirectoryVersion  = 1
	maxMetadataSize         = 4096
)

type rootfsMetadata struct {
	Format  string `json:"format"`
	Version int    `json:"version"`
	Entries int64  `json:"entries"`
	Bytes   int64  `json:"bytes"`
}

type rootfsExclusion struct {
	pattern string
	exact   bool
}

type rootfsStats struct {
	entries int64
	bytes   int64
}

type rootfsEntry struct {
	path string
	mode uint32
	stat unix.Statx_t
	link string
}

// GetRootFS returns the container's root filesystem path via /host/proc.
func GetRootFS(pid int) (string, error) {
	rootPath := fmt.Sprintf("%s/%d/root", HostProcPath, pid)
	if _, err := os.Stat(rootPath); err != nil {
		return "", fmt.Errorf("rootfs not accessible at %s: %w", rootPath, err)
	}
	return rootPath, nil
}

// GetOverlayUpperDir extracts the overlay upperdir from mountinfo.
func GetOverlayUpperDir(pid int) (string, error) {
	mountInfo, err := ReadMountInfo(pid)
	if err != nil {
		return "", fmt.Errorf("failed to parse mountinfo: %w", err)
	}
	for _, mount := range mountInfo {
		if mount.MountPoint != "/" || mount.FSType != "overlay" {
			continue
		}
		for _, opt := range strings.Split(mount.VFSOptions, ",") {
			if strings.HasPrefix(opt, "upperdir=") {
				return strings.TrimPrefix(opt, "upperdir="), nil
			}
		}
	}
	return "", fmt.Errorf("overlay upperdir not found for pid %d", pid)
}

// CaptureRootfsDiff captures an overlay upperdir as a directory artifact.
func CaptureRootfsDiff(
	ctx context.Context,
	upperDir string,
	checkpointDir string,
	exclusions types.OverlaySettings,
	bindMountDests []string,
	workers int,
	log logr.Logger,
) (string, error) {
	if upperDir == "" {
		return "", fmt.Errorf("upperdir is empty")
	}
	effective, err := effectiveRootfsWorkers(workers)
	if err != nil {
		return "", err
	}
	start := time.Now()
	entries, stats, deleted, err := scanRootfs(
		ctx,
		upperDir,
		exclusions,
		bindMountDests,
		true,
	)
	if err != nil {
		return "", fmt.Errorf("scan overlay upperdir: %w", err)
	}

	staging, err := os.MkdirTemp(checkpointDir, ".rootfs-diff-")
	if err != nil {
		return "", fmt.Errorf("create rootfs staging directory: %w", err)
	}
	removeStaging := true
	defer func() {
		if removeStaging {
			_ = os.RemoveAll(staging)
		}
	}()
	if err := copyRootfs(ctx, upperDir, staging, entries, effective, false); err != nil {
		return "", fmt.Errorf("copy rootfs artifact: %w", err)
	}

	finalPath := filepath.Join(checkpointDir, rootfsDirectoryFilename)
	if err := os.Rename(staging, finalPath); err != nil {
		return "", fmt.Errorf("publish rootfs directory: %w", err)
	}
	removeStaging = false
	metadata := rootfsMetadata{
		Format:  rootfsDirectoryFormat,
		Version: rootfsDirectoryVersion,
		Entries: stats.entries,
		Bytes:   stats.bytes,
	}
	if err := publishDeletedFiles(checkpointDir, deleted); err != nil {
		_ = os.RemoveAll(finalPath)
		return "", err
	}
	if err := writeRootfsMetadata(checkpointDir, metadata); err != nil {
		_ = os.RemoveAll(finalPath)
		return "", err
	}
	log.Info(
		"Captured directory rootfs diff",
		"configured_workers", workers,
		"effective_workers", effective,
		"entries", stats.entries,
		"bytes", stats.bytes,
		"elapsed", time.Since(start),
	)
	return finalPath, nil
}

// ApplyRootfsDiff applies a complete directory artifact to the target root.
func ApplyRootfsDiff(
	ctx context.Context,
	checkpointPath string,
	targetRoot string,
	workers int,
	log logr.Logger,
) error {
	effective, err := effectiveRootfsWorkers(workers)
	if err != nil {
		return err
	}
	metadata, err := readRootfsMetadata(checkpointPath)
	if err != nil {
		return err
	}
	artifactPath := filepath.Join(checkpointPath, rootfsDirectoryFilename)
	start := time.Now()
	entries, stats, _, err := scanRootfs(
		ctx,
		artifactPath,
		types.OverlaySettings{},
		nil,
		false,
	)
	if err != nil {
		return fmt.Errorf("scan rootfs directory artifact: %w", err)
	}
	if stats.entries != metadata.Entries || stats.bytes != metadata.Bytes {
		return fmt.Errorf(
			"rootfs artifact metadata mismatch: marker entries=%d bytes=%d, directory entries=%d bytes=%d",
			metadata.Entries,
			metadata.Bytes,
			stats.entries,
			stats.bytes,
		)
	}
	if err := copyRootfs(
		ctx,
		artifactPath,
		targetRoot,
		entries,
		effective,
		true,
	); err != nil {
		return fmt.Errorf("apply rootfs directory artifact: %w", err)
	}
	log.Info(
		"Applied directory rootfs diff",
		"configured_workers", workers,
		"effective_workers", effective,
		"entries", stats.entries,
		"bytes", stats.bytes,
		"elapsed", time.Since(start),
	)
	return nil
}

func effectiveRootfsWorkers(configured int) (int, error) {
	if configured < 1 || configured > types.MaxRootFSWorkers {
		return 0, fmt.Errorf(
			"rootfs workers must be between 1 and %d",
			types.MaxRootFSWorkers,
		)
	}
	var limit unix.Rlimit
	if err := unix.Getrlimit(unix.RLIMIT_NOFILE, &limit); err != nil {
		return 0, fmt.Errorf("read RLIMIT_NOFILE: %w", err)
	}
	const (
		reservedFDs  = 32
		fdsPerWorker = 4
	)
	if limit.Cur < reservedFDs+fdsPerWorker {
		return 0, fmt.Errorf(
			"RLIMIT_NOFILE %d is too small for rootfs copy",
			limit.Cur,
		)
	}
	limitWorkers := (limit.Cur - reservedFDs) / fdsPerWorker
	if uint64(configured) <= limitWorkers {
		return configured, nil
	}
	return max(1, int(limitWorkers)), nil
}

func scanRootfs(
	ctx context.Context,
	root string,
	exclusions types.OverlaySettings,
	bindMountDests []string,
	skipWhiteouts bool,
) ([]rootfsEntry, rootfsStats, []string, error) {
	patterns, err := exclusionPatterns(exclusions, bindMountDests)
	if err != nil {
		return nil, rootfsStats{}, nil, err
	}
	var rootStat unix.Statx_t
	if err := statxPath(root, &rootStat); err != nil {
		return nil, rootfsStats{}, nil, fmt.Errorf("stat root: %w", err)
	}
	if uint32(rootStat.Mode)&unix.S_IFMT != unix.S_IFDIR {
		return nil, rootfsStats{}, nil, fmt.Errorf("root is not a directory")
	}
	if err := rejectXattrs(root); err != nil {
		return nil, rootfsStats{}, nil, fmt.Errorf("inspect root xattrs: %w", err)
	}

	entries := make([]rootfsEntry, 0, 1024)
	stats := rootfsStats{}
	var deleted []string
	err = filepath.WalkDir(root, func(
		entryPath string,
		dirEntry os.DirEntry,
		walkErr error,
	) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		rel, err := filepath.Rel(root, entryPath)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}
		if err := validateRelativePath(rel); err != nil {
			return err
		}
		if excludedRootfsPath(rel, patterns) {
			if dirEntry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		var stat unix.Statx_t
		if err := statxPath(entryPath, &stat); err != nil {
			return fmt.Errorf("stat %s: %w", rel, err)
		}
		if stat.Mnt_id != rootStat.Mnt_id {
			return fmt.Errorf("rootfs entry %s crosses a mount boundary", rel)
		}
		mode := uint32(stat.Mode)
		if skipWhiteouts && strings.HasPrefix(dirEntry.Name(), ".wh.") {
			if dirEntry.Name() == ".wh..wh..opq" {
				return fmt.Errorf(
					"unsupported opaque whiteout marker %s",
					rel,
				)
			}
			if mode&unix.S_IFMT != unix.S_IFREG {
				return fmt.Errorf("unsupported native whiteout %s", rel)
			}
			deletedPath := filepath.Join(
				filepath.Dir(rel),
				strings.TrimPrefix(dirEntry.Name(), ".wh."),
			)
			if err := validateRelativePath(deletedPath); err != nil {
				return err
			}
			deleted = append(deleted, deletedPath)
			return nil
		}
		if err := rejectXattrs(entryPath); err != nil {
			return fmt.Errorf("unsupported xattrs on %s: %w", rel, err)
		}
		entry := rootfsEntry{path: rel, mode: mode, stat: stat}
		switch mode & unix.S_IFMT {
		case unix.S_IFDIR:
		case unix.S_IFREG:
			if stat.Nlink != 1 {
				return fmt.Errorf("unsupported hardlink at %s", rel)
			}
			if stat.Size > uint64(^uint64(0)>>1) ||
				stats.bytes > int64(^uint64(0)>>1)-int64(stat.Size) {
				return fmt.Errorf("rootfs byte count overflow at %s", rel)
			}
			if stat.Size > 0 && stat.Blocks < (stat.Size+511)/512 {
				return fmt.Errorf("unsupported sparse file at %s", rel)
			}
			stats.bytes += int64(stat.Size)
		case unix.S_IFLNK:
			if stat.Nlink != 1 {
				return fmt.Errorf("unsupported hard-linked symlink at %s", rel)
			}
			entry.link, err = os.Readlink(entryPath)
			if err != nil {
				return fmt.Errorf("read symlink %s: %w", rel, err)
			}
		default:
			return fmt.Errorf(
				"unsupported rootfs entry %s with mode %#o",
				rel,
				mode,
			)
		}
		entries = append(entries, entry)
		stats.entries++
		return nil
	})
	if err != nil {
		return nil, rootfsStats{}, nil, err
	}
	sort.Strings(deleted)
	return entries, stats, deleted, nil
}

func statxPath(entryPath string, stat *unix.Statx_t) error {
	return unix.Statx(
		unix.AT_FDCWD,
		entryPath,
		unix.AT_SYMLINK_NOFOLLOW,
		unix.STATX_BASIC_STATS|unix.STATX_MNT_ID,
		stat,
	)
}

func rejectXattrs(entryPath string) error {
	size, err := unix.Llistxattr(entryPath, nil)
	if errors.Is(err, unix.ENOTSUP) {
		return nil
	}
	if err != nil {
		return err
	}
	if size != 0 {
		return fmt.Errorf("%d bytes of extended attribute names", size)
	}
	return nil
}

func exclusionPatterns(
	settings types.OverlaySettings,
	bindMountDests []string,
) ([]rootfsExclusion, error) {
	patterns := make(
		[]rootfsExclusion,
		0,
		len(settings.Exclusions)+len(bindMountDests),
	)
	for _, pattern := range settings.Exclusions {
		patterns = append(patterns, rootfsExclusion{pattern: pattern})
	}
	for _, destination := range bindMountDests {
		patterns = append(
			patterns,
			rootfsExclusion{pattern: destination, exact: true},
		)
	}
	for i, exclusion := range patterns {
		pattern := exclusion.pattern
		pattern = filepath.ToSlash(pattern)
		pattern = strings.TrimPrefix(pattern, "./")
		pattern = strings.TrimPrefix(pattern, "/")
		if pattern == "" || pattern == "." || strings.HasPrefix(pattern, "../") {
			return nil, fmt.Errorf(
				"invalid rootfs exclusion %q",
				exclusion.pattern,
			)
		}
		if !exclusion.exact {
			if _, err := path.Match(pattern, "validate"); err != nil {
				return nil, fmt.Errorf(
					"invalid rootfs exclusion %q: %w",
					exclusion.pattern,
					err,
				)
			}
		}
		patterns[i].pattern = pattern
	}
	return patterns, nil
}

func excludedRootfsPath(rel string, patterns []rootfsExclusion) bool {
	rel = filepath.ToSlash(rel)
	for _, exclusion := range patterns {
		if exclusion.exact {
			if rel == exclusion.pattern ||
				strings.HasPrefix(rel, exclusion.pattern+"/") {
				return true
			}
			continue
		}
		if matchRootfsPattern(exclusion.pattern, rel) {
			return true
		}
	}
	return false
}

func matchRootfsPattern(pattern, rel string) bool {
	for candidate := rel; ; {
		for prefix := candidate; ; {
			if matched, _ := path.Match(pattern, prefix); matched {
				return true
			}
			index := strings.LastIndexByte(prefix, '/')
			if index < 0 {
				break
			}
			prefix = prefix[:index]
		}
		if !strings.HasPrefix(pattern, "*/") {
			break
		}
		index := strings.IndexByte(candidate, '/')
		if index < 0 {
			break
		}
		candidate = candidate[index+1:]
	}
	if !strings.Contains(pattern, "/") {
		matched, _ := path.Match(pattern, path.Base(rel))
		return matched
	}
	return false
}

func copyRootfs(
	ctx context.Context,
	sourceRoot string,
	targetRoot string,
	entries []rootfsEntry,
	workers int,
	skipExisting bool,
) error {
	sourceFD, err := openRoot(sourceRoot, false)
	if err != nil {
		return fmt.Errorf("open source root: %w", err)
	}
	defer unix.Close(sourceFD)
	targetFD, err := openRoot(targetRoot, true)
	if err != nil {
		return fmt.Errorf("open target root: %w", err)
	}
	defer unix.Close(targetFD)

	createdDirs := make(map[string]bool)
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.mode&unix.S_IFMT != unix.S_IFDIR {
			continue
		}
		created, err := createDirectory(targetFD, entry.path, skipExisting)
		if err != nil {
			return fmt.Errorf("create directory %s: %w", entry.path, err)
		}
		createdDirs[entry.path] = created
	}
	for _, entry := range entries {
		if entry.mode&unix.S_IFMT != unix.S_IFLNK {
			continue
		}
		if err := copySymlink(
			sourceFD,
			targetFD,
			entry,
			skipExisting,
		); err != nil {
			return fmt.Errorf("copy symlink %s: %w", entry.path, err)
		}
	}
	if err := copyRegularFiles(
		ctx,
		sourceFD,
		targetFD,
		entries,
		workers,
		skipExisting,
	); err != nil {
		return err
	}
	for i := len(entries) - 1; i >= 0; i-- {
		entry := entries[i]
		if entry.mode&unix.S_IFMT != unix.S_IFDIR ||
			!createdDirs[entry.path] {
			continue
		}
		sourceDir, err := openBeneath(
			sourceFD,
			entry.path,
			unix.O_RDONLY|unix.O_DIRECTORY,
		)
		if err != nil {
			return fmt.Errorf("open source directory %s: %w", entry.path, err)
		}
		targetDir, err := openBeneath(
			targetFD,
			entry.path,
			unix.O_RDONLY|unix.O_DIRECTORY,
		)
		if err != nil {
			unix.Close(sourceDir)
			return fmt.Errorf("open target directory %s: %w", entry.path, err)
		}
		err = applyFDMetadata(targetDir, entry)
		unix.Close(targetDir)
		unix.Close(sourceDir)
		if err != nil {
			return fmt.Errorf("set directory metadata %s: %w", entry.path, err)
		}
	}
	return nil
}

func copyRegularFiles(
	ctx context.Context,
	sourceRootFD int,
	targetRootFD int,
	entries []rootfsEntry,
	workers int,
	skipExisting bool,
) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	jobs := make(chan rootfsEntry)
	errs := make(chan error, 1)
	var group sync.WaitGroup
	for range workers {
		group.Add(1)
		go func() {
			defer group.Done()
			buffer := make([]byte, 1<<20)
			for {
				select {
				case <-ctx.Done():
					return
				case entry, ok := <-jobs:
					if !ok {
						return
					}
					if err := copyRegularFile(
						ctx,
						sourceRootFD,
						targetRootFD,
						entry,
						skipExisting,
						buffer,
					); err != nil {
						select {
						case errs <- fmt.Errorf(
							"copy regular file %s: %w",
							entry.path,
							err,
						):
						default:
						}
						cancel()
						return
					}
				}
			}
		}()
	}

enqueue:
	for _, entry := range entries {
		if entry.mode&unix.S_IFMT != unix.S_IFREG {
			continue
		}
		select {
		case jobs <- entry:
		case <-ctx.Done():
			break enqueue
		}
	}
	close(jobs)
	group.Wait()
	select {
	case err := <-errs:
		return err
	default:
		return ctx.Err()
	}
}

func copyRegularFile(
	ctx context.Context,
	sourceRootFD int,
	targetRootFD int,
	entry rootfsEntry,
	skipExisting bool,
	buffer []byte,
) error {
	sourceFD, err := openBeneath(sourceRootFD, entry.path, unix.O_RDONLY)
	if err != nil {
		return fmt.Errorf("open source: %w", err)
	}
	defer unix.Close(sourceFD)
	if err := validateOpenSource(sourceFD, entry); err != nil {
		return err
	}
	parentFD, name, err := openParent(targetRootFD, entry.path)
	if err != nil {
		return err
	}
	defer unix.Close(parentFD)
	targetFD, err := unix.Openat(
		parentFD,
		name,
		unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0600,
	)
	if errors.Is(err, unix.EEXIST) && skipExisting {
		return nil
	}
	if err != nil {
		return fmt.Errorf("create target: %w", err)
	}
	removeOnError := true
	defer func() {
		_ = unix.Close(targetFD)
		if removeOnError {
			_ = unix.Unlinkat(parentFD, name, 0)
		}
	}()
	if err := copyDenseFile(
		ctx,
		sourceFD,
		targetFD,
		int64(entry.stat.Size),
		buffer,
	); err != nil {
		return err
	}
	if err := applyFDMetadata(targetFD, entry); err != nil {
		return err
	}
	removeOnError = false
	return nil
}

func copyDenseFile(
	ctx context.Context,
	sourceFD int,
	targetFD int,
	size int64,
	buffer []byte,
) error {
	for offset := int64(0); offset < size; {
		if err := ctx.Err(); err != nil {
			return err
		}
		want := int(min(int64(len(buffer)), size-offset))
		read, err := unix.Pread(sourceFD, buffer[:want], offset)
		if err != nil {
			return fmt.Errorf("read source: %w", err)
		}
		if read == 0 {
			return io.ErrUnexpectedEOF
		}
		for written := 0; written < read; {
			count, err := unix.Pwrite(
				targetFD,
				buffer[written:read],
				offset+int64(written),
			)
			if err != nil {
				return fmt.Errorf("write target: %w", err)
			}
			if count == 0 {
				return io.ErrShortWrite
			}
			written += count
		}
		offset += int64(read)
	}
	return nil
}

func createDirectory(
	targetRootFD int,
	entryPath string,
	skipExisting bool,
) (bool, error) {
	parentFD, name, err := openParent(targetRootFD, entryPath)
	if err != nil {
		return false, err
	}
	defer unix.Close(parentFD)
	if err := unix.Mkdirat(parentFD, name, 0700); err != nil {
		if !errors.Is(err, unix.EEXIST) || !skipExisting {
			return false, err
		}
		fd, err := openBeneath(
			parentFD,
			name,
			unix.O_RDONLY|unix.O_DIRECTORY,
		)
		if err != nil {
			return false, fmt.Errorf(
				"existing entry is not a safe directory: %w",
				err,
			)
		}
		unix.Close(fd)
		return false, nil
	}
	return true, nil
}

func copySymlink(
	sourceRootFD int,
	targetRootFD int,
	entry rootfsEntry,
	skipExisting bool,
) error {
	sourceParent, sourceName, err := openParent(sourceRootFD, entry.path)
	if err != nil {
		return err
	}
	defer unix.Close(sourceParent)
	link, err := readlinkAt(sourceParent, sourceName)
	if err != nil {
		return err
	}
	if link != entry.link {
		return fmt.Errorf("source changed while copying")
	}
	targetParent, targetName, err := openParent(targetRootFD, entry.path)
	if err != nil {
		return err
	}
	defer unix.Close(targetParent)
	if err := unix.Symlinkat(link, targetParent, targetName); err != nil {
		if errors.Is(err, unix.EEXIST) && skipExisting {
			return nil
		}
		return err
	}
	removeOnError := true
	defer func() {
		if removeOnError {
			_ = unix.Unlinkat(targetParent, targetName, 0)
		}
	}()
	if err := applySymlinkMetadata(targetParent, targetName, entry); err != nil {
		return err
	}
	removeOnError = false
	return nil
}

func validateOpenSource(fd int, entry rootfsEntry) error {
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil {
		return fmt.Errorf("stat source: %w", err)
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFREG ||
		uint64(stat.Ino) != entry.stat.Ino ||
		stat.Size != int64(entry.stat.Size) {
		return fmt.Errorf("source changed while copying")
	}
	return nil
}

func applyFDMetadata(fd int, entry rootfsEntry) error {
	if err := chownFD(fd, entry.stat.Uid, entry.stat.Gid); err != nil {
		return fmt.Errorf("set ownership: %w", err)
	}
	if err := unix.Fchmod(fd, entry.mode&07777); err != nil {
		return fmt.Errorf("set mode: %w", err)
	}
	times := []unix.Timespec{
		unix.NsecToTimespec(statxNsec(entry.stat.Atime)),
		unix.NsecToTimespec(statxNsec(entry.stat.Mtime)),
	}
	if err := unix.UtimesNanoAt(fd, "", times, unix.AT_EMPTY_PATH); err != nil {
		return fmt.Errorf("set timestamps: %w", err)
	}
	return nil
}

func chownFD(fd int, uid, gid uint32) error {
	err := unix.Fchown(fd, int(uid), int(gid))
	if !errors.Is(err, unix.EPERM) {
		return err
	}
	var stat unix.Stat_t
	if statErr := unix.Fstat(fd, &stat); statErr != nil {
		return statErr
	}
	if stat.Uid != uid || stat.Gid != gid {
		return err
	}
	return nil
}

func applySymlinkMetadata(
	parentFD int,
	name string,
	entry rootfsEntry,
) error {
	err := unix.Fchownat(
		parentFD,
		name,
		int(entry.stat.Uid),
		int(entry.stat.Gid),
		unix.AT_SYMLINK_NOFOLLOW,
	)
	if errors.Is(err, unix.EPERM) {
		var stat unix.Stat_t
		if statErr := unix.Fstatat(
			parentFD,
			name,
			&stat,
			unix.AT_SYMLINK_NOFOLLOW,
		); statErr != nil {
			return fmt.Errorf("verify ownership: %w", statErr)
		}
		if stat.Uid != entry.stat.Uid || stat.Gid != entry.stat.Gid {
			return fmt.Errorf("set ownership: %w", err)
		}
	} else if err != nil {
		return fmt.Errorf("set ownership: %w", err)
	}
	times := []unix.Timespec{
		unix.NsecToTimespec(statxNsec(entry.stat.Atime)),
		unix.NsecToTimespec(statxNsec(entry.stat.Mtime)),
	}
	if err := unix.UtimesNanoAt(
		parentFD,
		name,
		times,
		unix.AT_SYMLINK_NOFOLLOW,
	); err != nil {
		return fmt.Errorf("set timestamps: %w", err)
	}
	return nil
}

func statxNsec(timestamp unix.StatxTimestamp) int64 {
	return timestamp.Sec*1e9 + int64(timestamp.Nsec)
}

func writeRootfsMetadata(
	checkpointDir string,
	metadata rootfsMetadata,
) error {
	file, err := os.CreateTemp(checkpointDir, ".rootfs-meta-")
	if err != nil {
		return fmt.Errorf("create rootfs completion metadata: %w", err)
	}
	tempPath := file.Name()
	defer os.Remove(tempPath)
	if err := file.Chmod(0644); err != nil {
		file.Close()
		return err
	}
	if err := json.NewEncoder(file).Encode(metadata); err != nil {
		file.Close()
		return fmt.Errorf("write rootfs completion metadata: %w", err)
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return fmt.Errorf("sync rootfs completion metadata: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close rootfs completion metadata: %w", err)
	}
	if err := os.Rename(
		tempPath,
		filepath.Join(checkpointDir, rootfsMetadataFilename),
	); err != nil {
		return fmt.Errorf("publish rootfs completion metadata: %w", err)
	}
	parentFD, err := openRoot(checkpointDir, false)
	if err != nil {
		return fmt.Errorf("open checkpoint directory: %w", err)
	}
	defer unix.Close(parentFD)
	if err := unix.Fsync(parentFD); err != nil {
		return fmt.Errorf("sync checkpoint directory: %w", err)
	}
	return nil
}

func readRootfsMetadata(checkpointPath string) (rootfsMetadata, error) {
	artifactPath := filepath.Join(checkpointPath, rootfsDirectoryFilename)
	metadataPath := filepath.Join(checkpointPath, rootfsMetadataFilename)
	artifactInfo, artifactErr := os.Lstat(artifactPath)
	metadataInfo, metadataErr := os.Lstat(metadataPath)
	artifactMissing := errors.Is(artifactErr, os.ErrNotExist)
	metadataMissing := errors.Is(metadataErr, os.ErrNotExist)
	if artifactMissing && metadataMissing {
		return rootfsMetadata{}, fmt.Errorf(
			"missing rootfs directory artifact %s and completion metadata %s; legacy tar checkpoints are unsupported",
			rootfsDirectoryFilename,
			rootfsMetadataFilename,
		)
	}
	if artifactErr != nil && !artifactMissing {
		return rootfsMetadata{}, fmt.Errorf(
			"inspect rootfs directory artifact: %w",
			artifactErr,
		)
	}
	if metadataErr != nil && !metadataMissing {
		return rootfsMetadata{}, fmt.Errorf(
			"inspect rootfs completion metadata: %w",
			metadataErr,
		)
	}
	if artifactMissing || metadataMissing {
		return rootfsMetadata{}, fmt.Errorf(
			"incomplete rootfs directory artifact: %s and %s must both exist",
			rootfsDirectoryFilename,
			rootfsMetadataFilename,
		)
	}
	if !artifactInfo.IsDir() {
		return rootfsMetadata{}, fmt.Errorf(
			"%s is not a directory",
			rootfsDirectoryFilename,
		)
	}
	if !metadataInfo.Mode().IsRegular() || metadataInfo.Size() > maxMetadataSize {
		return rootfsMetadata{}, fmt.Errorf(
			"%s is not valid completion metadata",
			rootfsMetadataFilename,
		)
	}
	metadataFD, err := unix.Open(
		metadataPath,
		unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW|unix.O_NONBLOCK,
		0,
	)
	if err != nil {
		return rootfsMetadata{}, fmt.Errorf(
			"open rootfs completion metadata: %w",
			err,
		)
	}
	file := os.NewFile(uintptr(metadataFD), metadataPath)
	defer file.Close()
	var metadataStat unix.Stat_t
	if err := unix.Fstat(metadataFD, &metadataStat); err != nil {
		return rootfsMetadata{}, fmt.Errorf(
			"stat rootfs completion metadata: %w",
			err,
		)
	}
	if metadataStat.Mode&unix.S_IFMT != unix.S_IFREG ||
		metadataStat.Size < 0 ||
		metadataStat.Size > maxMetadataSize {
		return rootfsMetadata{}, fmt.Errorf(
			"%s is not valid completion metadata",
			rootfsMetadataFilename,
		)
	}
	data, err := io.ReadAll(io.LimitReader(file, maxMetadataSize+1))
	if err != nil {
		return rootfsMetadata{}, fmt.Errorf(
			"read rootfs completion metadata: %w",
			err,
		)
	}
	if len(data) > maxMetadataSize {
		return rootfsMetadata{}, fmt.Errorf(
			"rootfs completion metadata exceeds %d bytes",
			maxMetadataSize,
		)
	}
	var metadata rootfsMetadata
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&metadata); err != nil {
		return rootfsMetadata{}, fmt.Errorf(
			"parse rootfs completion metadata: %w",
			err,
		)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return rootfsMetadata{}, fmt.Errorf(
			"parse rootfs completion metadata: trailing data",
		)
	}
	if metadata.Format != rootfsDirectoryFormat ||
		metadata.Version != rootfsDirectoryVersion {
		return rootfsMetadata{}, fmt.Errorf(
			"unsupported rootfs directory format %q version %d",
			metadata.Format,
			metadata.Version,
		)
	}
	if metadata.Entries < 0 || metadata.Bytes < 0 {
		return rootfsMetadata{}, fmt.Errorf(
			"invalid rootfs directory artifact counts",
		)
	}
	return metadata, nil
}

func publishDeletedFiles(checkpointDir string, deleted []string) error {
	if len(deleted) == 0 {
		return nil
	}
	data, err := json.Marshal(deleted)
	if err != nil {
		return err
	}
	if err := os.WriteFile(
		filepath.Join(checkpointDir, deletedFilesFilename),
		data,
		0644,
	); err != nil {
		return fmt.Errorf("write deleted files: %w", err)
	}
	return nil
}

// ApplyDeletedFiles removes entries without following symlinks or mounts.
func ApplyDeletedFiles(
	checkpointPath string,
	targetRoot string,
	log logr.Logger,
) error {
	data, err := os.ReadFile(filepath.Join(checkpointPath, deletedFilesFilename))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read deleted files: %w", err)
	}
	var deleted []string
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&deleted); err != nil {
		return fmt.Errorf("parse deleted files: %w", err)
	}
	if deleted == nil {
		return fmt.Errorf("parse deleted files: expected a JSON array")
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return fmt.Errorf("parse deleted files: trailing data")
	}
	rootFD, err := openRoot(targetRoot, true)
	if err != nil {
		return fmt.Errorf("open deletion target root: %w", err)
	}
	defer unix.Close(rootFD)
	for _, entryPath := range deleted {
		if err := validateRelativePath(entryPath); err != nil {
			return fmt.Errorf("invalid deleted file entry: %w", err)
		}
		if err := validateRemoval(rootFD, entryPath); err != nil {
			return fmt.Errorf("validate deletion %s: %w", entryPath, err)
		}
	}
	count := 0
	for _, entryPath := range deleted {
		removed, err := removeBeneath(rootFD, entryPath)
		if err != nil {
			return fmt.Errorf("delete %s: %w", entryPath, err)
		}
		if removed {
			count++
		}
	}
	log.Info("Deleted files applied", "count", count)
	return nil
}

func validateRemoval(rootFD int, entryPath string) error {
	parentFD, name, err := openParent(rootFD, entryPath)
	if errors.Is(err, unix.ENOENT) {
		return nil
	}
	if err != nil {
		return err
	}
	defer unix.Close(parentFD)
	return walkDirectoryAt(parentFD, name, false)
}

func removeBeneath(rootFD int, entryPath string) (bool, error) {
	parentFD, name, err := openParent(rootFD, entryPath)
	if errors.Is(err, unix.ENOENT) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer unix.Close(parentFD)
	var stat unix.Stat_t
	if err := unix.Fstatat(
		parentFD,
		name,
		&stat,
		unix.AT_SYMLINK_NOFOLLOW,
	); errors.Is(err, unix.ENOENT) {
		return false, nil
	} else if err != nil {
		return false, err
	}
	if stat.Mode&unix.S_IFMT == unix.S_IFDIR {
		if err := walkDirectoryAt(parentFD, name, true); err != nil {
			return false, err
		}
		return true, unix.Unlinkat(parentFD, name, unix.AT_REMOVEDIR)
	}
	return true, unix.Unlinkat(parentFD, name, 0)
}

func walkDirectoryAt(parentFD int, name string, remove bool) error {
	var stat unix.Stat_t
	if err := unix.Fstatat(
		parentFD,
		name,
		&stat,
		unix.AT_SYMLINK_NOFOLLOW,
	); errors.Is(err, unix.ENOENT) {
		return nil
	} else if err != nil {
		return err
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFDIR {
		return nil
	}
	fd, err := openBeneath(parentFD, name, unix.O_RDONLY|unix.O_DIRECTORY)
	if err != nil {
		return err
	}
	file := os.NewFile(uintptr(fd), name)
	defer file.Close()
	names, err := file.Readdirnames(-1)
	if err != nil {
		return err
	}
	sort.Strings(names)
	if !remove {
		for _, child := range names {
			if err := walkDirectoryAt(fd, child, false); err != nil {
				return err
			}
		}
		return nil
	}
	for _, child := range names {
		var childStat unix.Stat_t
		if err := unix.Fstatat(
			fd,
			child,
			&childStat,
			unix.AT_SYMLINK_NOFOLLOW,
		); err != nil {
			return err
		}
		if childStat.Mode&unix.S_IFMT == unix.S_IFDIR {
			if err := walkDirectoryAt(fd, child, true); err != nil {
				return err
			}
			if err := unix.Unlinkat(fd, child, unix.AT_REMOVEDIR); err != nil {
				return err
			}
		} else if err := unix.Unlinkat(fd, child, 0); err != nil {
			return err
		}
	}
	return nil
}

func validateRelativePath(entryPath string) error {
	if entryPath == "" || entryPath == "." ||
		filepath.IsAbs(entryPath) ||
		filepath.Clean(entryPath) != entryPath {
		return fmt.Errorf("unsafe rootfs path %q", entryPath)
	}
	for _, component := range strings.Split(entryPath, string(os.PathSeparator)) {
		if component == "" || component == "." || component == ".." {
			return fmt.Errorf("unsafe rootfs path %q", entryPath)
		}
	}
	return nil
}

func openRoot(root string, trusted bool) (int, error) {
	flags := unix.O_RDONLY | unix.O_DIRECTORY | unix.O_CLOEXEC
	if !trusted {
		flags |= unix.O_NOFOLLOW
	}
	return unix.Open(root, flags, 0)
}

func openBeneath(rootFD int, entryPath string, flags int) (int, error) {
	if entryPath != "." {
		if err := validateRelativePath(entryPath); err != nil {
			return -1, err
		}
	}
	return unix.Openat2(rootFD, entryPath, &unix.OpenHow{
		Flags: uint64(flags | unix.O_CLOEXEC | unix.O_NOFOLLOW),
		Resolve: unix.RESOLVE_BENEATH |
			unix.RESOLVE_NO_MAGICLINKS |
			unix.RESOLVE_NO_SYMLINKS |
			unix.RESOLVE_NO_XDEV,
	})
}

func openParent(rootFD int, entryPath string) (int, string, error) {
	if err := validateRelativePath(entryPath); err != nil {
		return -1, "", err
	}
	parent := filepath.Dir(entryPath)
	fd, err := openBeneath(
		rootFD,
		parent,
		unix.O_RDONLY|unix.O_DIRECTORY,
	)
	if err != nil {
		return -1, "", err
	}
	return fd, filepath.Base(entryPath), nil
}

func readlinkAt(parentFD int, name string) (string, error) {
	size := 256
	for {
		buffer := make([]byte, size)
		read, err := unix.Readlinkat(parentFD, name, buffer)
		if err != nil {
			return "", err
		}
		if read < len(buffer) {
			return string(buffer[:read]), nil
		}
		size *= 2
	}
}
