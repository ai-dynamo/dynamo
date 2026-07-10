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
	rootfsDirectoryVersion  = 2
	maxMetadataSize         = 4096
	maxDeletedFilesSize     = 64 << 20
	maxRootfsPathLength     = 4096
	maxDeletedFiles         = 1_000_000
	maxRootfsEntries        = 1_000_000
	maxSkippedSocketSamples = 10
	maxXattrListBytes       = 64 << 10
	maxXattrNames           = 1024
	maxXattrNameLength      = 255
	maxXattrListAttempts    = 3
	overlayXattrPrefix      = "trusted.overlay."
)

type rootfsXattrPolicy uint8

const (
	rejectAllRootfsXattrs rootfsXattrPolicy = iota
	allowSourceOverlayXattrs
)

type rootfsMetadata struct {
	Format              string `json:"format"`
	Version             int    `json:"version"`
	Entries             int64  `json:"entries"`
	Bytes               int64  `json:"bytes"`
	DeletedFilesPresent bool   `json:"deletedFilesPresent"`
	Deletions           int64  `json:"deletions"`
}

type rootfsExclusion struct {
	pattern string
	exact   bool
}

type rootfsStats struct {
	entries             int64
	bytes               int64
	ignoredOverlayAttrs map[string]int
	skippedSockets      int64
	skippedSocketSample []string
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
		true,
		allowSourceOverlayXattrs,
	)
	logSkippedSourceSockets(log, stats)
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
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := os.Rename(staging, finalPath); err != nil {
		return "", fmt.Errorf("publish rootfs directory: %w", err)
	}
	removeStaging = false
	published := false
	defer func() {
		if !published {
			_ = os.RemoveAll(finalPath)
			_ = os.Remove(filepath.Join(checkpointDir, deletedFilesFilename))
			_ = os.Remove(filepath.Join(checkpointDir, rootfsMetadataFilename))
		}
	}()
	metadata := rootfsMetadata{
		Format:              rootfsDirectoryFormat,
		Version:             rootfsDirectoryVersion,
		Entries:             stats.entries,
		Bytes:               stats.bytes,
		DeletedFilesPresent: true,
		Deletions:           int64(len(deleted)),
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := publishDeletedFiles(ctx, checkpointDir, deleted); err != nil {
		return "", err
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if err := writeRootfsMetadata(ctx, checkpointDir, metadata); err != nil {
		return "", err
	}
	published = true
	logIgnoredOverlayXattrs(log, stats.ignoredOverlayAttrs)
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
	checkpointFD, err := openRoot(checkpointPath, false)
	if err != nil {
		return fmt.Errorf("open checkpoint directory: %w", err)
	}
	defer unix.Close(checkpointFD)
	rootfsFD, metadata, _, err := readRootfsRestoreArtifacts(ctx, checkpointFD)
	if err != nil {
		return err
	}
	defer unix.Close(rootfsFD)
	return applyRootfsDiff(ctx, rootfsFD, targetRoot, workers, metadata, log)
}

// ApplyRootfsRestore applies rootfs additions and deletions from one checkpoint
// directory generation.
func ApplyRootfsRestore(
	ctx context.Context,
	checkpointPath string,
	targetRoot string,
	workers int,
	log logr.Logger,
) error {
	checkpointFD, err := openRoot(checkpointPath, false)
	if err != nil {
		return fmt.Errorf("open checkpoint directory: %w", err)
	}
	defer unix.Close(checkpointFD)
	rootfsFD, metadata, deleted, err := readRootfsRestoreArtifacts(
		ctx,
		checkpointFD,
	)
	if err != nil {
		return err
	}
	defer unix.Close(rootfsFD)
	if err := applyRootfsDiff(
		ctx,
		rootfsFD,
		targetRoot,
		workers,
		metadata,
		log,
	); err != nil {
		return fmt.Errorf("rootfs diff failed: %w", err)
	}
	if err := applyDeletedFiles(ctx, deleted, targetRoot, log); err != nil {
		return fmt.Errorf("deleted files failed: %w", err)
	}
	return nil
}

func applyRootfsDiff(
	ctx context.Context,
	rootfsFD int,
	targetRoot string,
	workers int,
	metadata rootfsMetadata,
	log logr.Logger,
) error {
	effective, err := effectiveRootfsWorkers(workers)
	if err != nil {
		return err
	}
	start := time.Now()
	entries, stats, _, err := scanRootfsFD(
		ctx,
		rootfsFD,
		types.OverlaySettings{},
		nil,
		false,
		false,
		rejectAllRootfsXattrs,
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
	if err := copyRootfsFD(
		ctx,
		rootfsFD,
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
	skipSockets bool,
	xattrPolicy rootfsXattrPolicy,
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

	entries := make([]rootfsEntry, 0, 1024)
	stats := rootfsStats{}
	ignored, err := inspectRootfsXattrs(root, xattrPolicy)
	if err != nil {
		return nil, rootfsStats{}, nil, fmt.Errorf(
			"unsupported xattrs on .: %w",
			err,
		)
	}
	addIgnoredOverlayXattrs(&stats, ignored)
	var deleted []string
	var totalEntries int64
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
		if totalEntries >= maxRootfsEntries {
			return fmt.Errorf(
				"rootfs contains more than %d entries",
				maxRootfsEntries,
			)
		}
		totalEntries++
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
		if shouldSkipRootfsEntry(mode, skipSockets) {
			stats.skippedSockets++
			stats.skippedSocketSample = append(stats.skippedSocketSample, rel)
			sort.Strings(stats.skippedSocketSample)
			if len(stats.skippedSocketSample) > maxSkippedSocketSamples {
				stats.skippedSocketSample =
					stats.skippedSocketSample[:maxSkippedSocketSamples]
			}
			return nil
		}
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
			target := strings.TrimPrefix(dirEntry.Name(), ".wh.")
			if target == "" || target == "." || target == ".." {
				return fmt.Errorf(
					"unsafe whiteout target %q at %s",
					target,
					rel,
				)
			}
			deletedPath := filepath.Join(
				filepath.Dir(rel),
				target,
			)
			if err := validateRelativePath(deletedPath); err != nil {
				return err
			}
			if len(deleted) >= maxDeletedFiles {
				return fmt.Errorf(
					"rootfs contains more than %d deletions",
					maxDeletedFiles,
				)
			}
			ignored, err := inspectRootfsXattrs(entryPath, xattrPolicy)
			if err != nil {
				return fmt.Errorf("unsupported xattrs on %s: %w", rel, err)
			}
			addIgnoredOverlayXattrs(&stats, ignored)
			deleted = append(deleted, deletedPath)
			return nil
		}
		ignored, err := inspectRootfsXattrs(entryPath, xattrPolicy)
		if err != nil {
			return fmt.Errorf("unsupported xattrs on %s: %w", rel, err)
		}
		addIgnoredOverlayXattrs(&stats, ignored)
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
		return nil, stats, nil, err
	}
	sort.Strings(deleted)
	return entries, stats, deleted, nil
}

func scanRootfsFD(
	ctx context.Context,
	rootFD int,
	exclusions types.OverlaySettings,
	bindMountDests []string,
	skipWhiteouts bool,
	skipSockets bool,
	xattrPolicy rootfsXattrPolicy,
) ([]rootfsEntry, rootfsStats, []string, error) {
	return scanRootfs(
		ctx,
		rootfsFDPath(rootFD),
		exclusions,
		bindMountDests,
		skipWhiteouts,
		skipSockets,
		xattrPolicy,
	)
}

func shouldSkipRootfsEntry(mode uint32, skipSockets bool) bool {
	return skipSockets && mode&unix.S_IFMT == unix.S_IFSOCK
}

func rootfsFDPath(fd int) string {
	return fmt.Sprintf("/proc/self/fd/%d/.", fd)
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

func inspectRootfsXattrs(
	entryPath string,
	policy rootfsXattrPolicy,
) ([]string, error) {
	list, err := listRootfsXattrs(entryPath)
	if err != nil {
		return nil, err
	}
	return parseRootfsXattrs(list, policy)
}

func listRootfsXattrs(entryPath string) ([]byte, error) {
	for range maxXattrListAttempts {
		size, err := unix.Llistxattr(entryPath, nil)
		if errors.Is(err, unix.ENOTSUP) {
			return nil, nil
		}
		if errors.Is(err, unix.ERANGE) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("list extended attributes: %w", err)
		}
		if size < 0 {
			return nil, fmt.Errorf(
				"invalid extended attribute list size %d",
				size,
			)
		}
		if size > maxXattrListBytes {
			return nil, fmt.Errorf(
				"extended attribute list is %d bytes, maximum is %d",
				size,
				maxXattrListBytes,
			)
		}
		if size == 0 {
			return nil, nil
		}
		list := make([]byte, size)
		size, err = unix.Llistxattr(entryPath, list)
		if errors.Is(err, unix.ENOTSUP) {
			return nil, nil
		}
		if errors.Is(err, unix.ERANGE) {
			continue
		}
		if err != nil {
			return nil, fmt.Errorf("read extended attributes: %w", err)
		}
		if size < 0 || size > len(list) {
			return nil, fmt.Errorf(
				"invalid extended attribute list size %d",
				size,
			)
		}
		return list[:size], nil
	}
	return nil, fmt.Errorf("extended attribute list changed during inspection")
}

func parseRootfsXattrs(
	list []byte,
	policy rootfsXattrPolicy,
) ([]string, error) {
	if len(list) > maxXattrListBytes {
		return nil, fmt.Errorf(
			"extended attribute list is %d bytes, maximum is %d",
			len(list),
			maxXattrListBytes,
		)
	}
	var names []string
	for len(list) != 0 {
		if len(names) >= maxXattrNames {
			return nil, fmt.Errorf(
				"more than %d extended attribute names",
				maxXattrNames,
			)
		}
		end := bytes.IndexByte(list, 0)
		if end < 0 {
			return nil, fmt.Errorf("unterminated extended attribute name")
		}
		if end == 0 {
			return nil, fmt.Errorf("empty extended attribute name")
		}
		if end > maxXattrNameLength {
			return nil, fmt.Errorf(
				"extended attribute name is %d bytes, maximum is %d",
				end,
				maxXattrNameLength,
			)
		}
		name := string(list[:end])
		switch policy {
		case allowSourceOverlayXattrs:
			if !isSourceOverlayXattr(name) {
				return nil, fmt.Errorf(
					"extended attribute %q is not allowed",
					name,
				)
			}
			names = append(names, name)
		case rejectAllRootfsXattrs:
			return nil, fmt.Errorf(
				"extended attribute %q is not allowed",
				name,
			)
		default:
			return nil, fmt.Errorf("invalid rootfs xattr policy %d", policy)
		}
		list = list[end+1:]
	}
	return names, nil
}

func isSourceOverlayXattr(name string) bool {
	return len(name) > len(overlayXattrPrefix) &&
		len(name) <= maxXattrNameLength &&
		strings.HasPrefix(name, overlayXattrPrefix) &&
		!strings.ContainsRune(name, 0)
}

func addIgnoredOverlayXattrs(stats *rootfsStats, names []string) {
	if len(names) == 0 {
		return
	}
	if stats.ignoredOverlayAttrs == nil {
		stats.ignoredOverlayAttrs = make(map[string]int)
	}
	for _, name := range names {
		stats.ignoredOverlayAttrs[name]++
	}
}

func logIgnoredOverlayXattrs(log logr.Logger, counts map[string]int) {
	if len(counts) == 0 {
		return
	}
	names := make([]string, 0, len(counts))
	for name := range counts {
		names = append(names, name)
	}
	sort.Strings(names)
	for i, name := range names {
		names[i] = fmt.Sprintf("%q:%d", name, counts[name])
	}
	log.Info(
		"Ignored source overlay xattrs for benchmark-compatible rootfs capture",
		"xattr_name_counts", names,
	)
}

func logSkippedSourceSockets(log logr.Logger, stats rootfsStats) {
	if stats.skippedSockets == 0 {
		return
	}
	log.Info(
		"Skipped source Unix-domain sockets for benchmark-compatible rootfs capture",
		"skipped_socket_count", stats.skippedSockets,
		"skipped_socket_path_sample", stats.skippedSocketSample,
	)
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
	return copyRootfsFD(
		ctx,
		sourceFD,
		targetRoot,
		entries,
		workers,
		skipExisting,
	)
}

func copyRootfsFD(
	ctx context.Context,
	sourceFD int,
	targetRoot string,
	entries []rootfsEntry,
	workers int,
	skipExisting bool,
) error {
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
		if err := ctx.Err(); err != nil {
			return err
		}
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
		if err := ctx.Err(); err != nil {
			return err
		}
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
	ctx context.Context,
	checkpointDir string,
	metadata rootfsMetadata,
) error {
	data, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("marshal rootfs completion metadata: %w", err)
	}
	if len(data) > maxMetadataSize {
		return fmt.Errorf(
			"rootfs completion metadata exceeds %d bytes",
			maxMetadataSize,
		)
	}
	if err := writeAtomicFile(
		ctx,
		checkpointDir,
		".rootfs-meta-",
		rootfsMetadataFilename,
		data,
	); err != nil {
		return fmt.Errorf("publish rootfs completion metadata: %w", err)
	}
	return nil
}

func writeAtomicFile(
	ctx context.Context,
	directory string,
	prefix string,
	name string,
	data []byte,
) error {
	file, err := os.CreateTemp(directory, prefix)
	if err != nil {
		return err
	}
	tempPath := file.Name()
	defer os.Remove(tempPath)
	if err := file.Chmod(0644); err != nil {
		file.Close()
		return err
	}
	if _, err := file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := os.Rename(tempPath, filepath.Join(directory, name)); err != nil {
		return err
	}
	parentFD, err := openRoot(directory, false)
	if err != nil {
		return err
	}
	defer unix.Close(parentFD)
	if err := unix.Fsync(parentFD); err != nil {
		return err
	}
	return nil
}

func readRootfsRestoreArtifacts(
	ctx context.Context,
	checkpointFD int,
) (int, rootfsMetadata, []string, error) {
	if err := ctx.Err(); err != nil {
		return -1, rootfsMetadata{}, nil, err
	}
	rootfsFD, artifactErr := openBeneath(
		checkpointFD,
		rootfsDirectoryFilename,
		unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NONBLOCK,
	)
	metadataFD, metadataErr := unix.Openat(
		checkpointFD,
		rootfsMetadataFilename,
		unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW|unix.O_NONBLOCK,
		0,
	)
	artifactMissing := errors.Is(artifactErr, unix.ENOENT)
	metadataMissing := errors.Is(metadataErr, unix.ENOENT)
	if artifactMissing && metadataMissing {
		return -1, rootfsMetadata{}, nil, fmt.Errorf(
			"missing rootfs directory artifact %s and completion metadata %s; legacy tar checkpoints are unsupported",
			rootfsDirectoryFilename,
			rootfsMetadataFilename,
		)
	}
	if artifactErr != nil && !artifactMissing {
		if metadataErr == nil {
			unix.Close(metadataFD)
		}
		return -1, rootfsMetadata{}, nil, fmt.Errorf(
			"inspect rootfs directory artifact: %w",
			artifactErr,
		)
	}
	if metadataErr != nil && !metadataMissing {
		if artifactErr == nil {
			unix.Close(rootfsFD)
		}
		return -1, rootfsMetadata{}, nil, fmt.Errorf(
			"open rootfs completion metadata: %w",
			metadataErr,
		)
	}
	if artifactMissing || metadataMissing {
		if artifactErr == nil {
			unix.Close(rootfsFD)
		}
		if metadataErr == nil {
			unix.Close(metadataFD)
		}
		return -1, rootfsMetadata{}, nil, fmt.Errorf(
			"incomplete rootfs directory artifact: %s and %s must both exist",
			rootfsDirectoryFilename,
			rootfsMetadataFilename,
		)
	}
	metadata, err := readRootfsMetadataFD(ctx, metadataFD)
	unix.Close(metadataFD)
	if err != nil {
		unix.Close(rootfsFD)
		return -1, rootfsMetadata{}, nil, err
	}
	deleted, err := readDeletedFilesAt(ctx, checkpointFD, metadata.Deletions)
	if err != nil {
		unix.Close(rootfsFD)
		return -1, rootfsMetadata{}, nil, err
	}
	return rootfsFD, metadata, deleted, nil
}

func readRootfsMetadata(checkpointPath string) (rootfsMetadata, error) {
	checkpointFD, err := openRoot(checkpointPath, false)
	if err != nil {
		return rootfsMetadata{}, fmt.Errorf("open checkpoint directory: %w", err)
	}
	defer unix.Close(checkpointFD)
	rootfsFD, metadata, _, err := readRootfsRestoreArtifacts(
		context.Background(),
		checkpointFD,
	)
	if err != nil {
		return rootfsMetadata{}, err
	}
	unix.Close(rootfsFD)
	return metadata, nil
}

func readRootfsMetadataFD(
	ctx context.Context,
	metadataFD int,
) (rootfsMetadata, error) {
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
	data, err := readAllContext(ctx, fdReader(metadataFD), maxMetadataSize)
	if err != nil {
		return rootfsMetadata{}, fmt.Errorf(
			"read rootfs completion metadata: %w",
			err,
		)
	}
	var metadata rootfsMetadata
	decoder := json.NewDecoder(contextReader{
		ctx:    ctx,
		reader: bytes.NewReader(data),
	})
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&metadata); err != nil {
		return rootfsMetadata{}, fmt.Errorf(
			"parse rootfs completion metadata: %w",
			err,
		)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return rootfsMetadata{}, ctxErr
		}
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
	if metadata.Entries < 0 ||
		metadata.Entries > maxRootfsEntries ||
		metadata.Bytes < 0 ||
		!metadata.DeletedFilesPresent ||
		metadata.Deletions < 0 ||
		metadata.Deletions > maxDeletedFiles {
		return rootfsMetadata{}, fmt.Errorf(
			"invalid rootfs directory artifact counts",
		)
	}
	return metadata, nil
}

func publishDeletedFiles(
	ctx context.Context,
	checkpointDir string,
	deleted []string,
) error {
	if deleted == nil {
		deleted = []string{}
	}
	if len(deleted) > maxDeletedFiles {
		return fmt.Errorf("more than %d deleted files", maxDeletedFiles)
	}
	for _, entryPath := range deleted {
		if err := validateRelativePath(entryPath); err != nil {
			return fmt.Errorf("invalid deleted file entry: %w", err)
		}
	}
	data, err := json.Marshal(deleted)
	if err != nil {
		return err
	}
	if len(data) > maxDeletedFilesSize {
		return fmt.Errorf(
			"deleted files metadata exceeds %d bytes",
			maxDeletedFilesSize,
		)
	}
	if err := writeAtomicFile(
		ctx,
		checkpointDir,
		".deleted-files-",
		deletedFilesFilename,
		data,
	); err != nil {
		return fmt.Errorf("publish deleted files: %w", err)
	}
	return nil
}

// ApplyDeletedFiles removes entries without following symlinks or mounts.
func ApplyDeletedFiles(
	ctx context.Context,
	checkpointPath string,
	targetRoot string,
	log logr.Logger,
) error {
	checkpointFD, err := openRoot(checkpointPath, false)
	if err != nil {
		return fmt.Errorf("open checkpoint directory: %w", err)
	}
	defer unix.Close(checkpointFD)
	rootfsFD, _, deleted, err := readRootfsRestoreArtifacts(ctx, checkpointFD)
	if err != nil {
		return err
	}
	unix.Close(rootfsFD)
	return applyDeletedFiles(ctx, deleted, targetRoot, log)
}

func applyDeletedFiles(
	ctx context.Context,
	deleted []string,
	targetRoot string,
	log logr.Logger,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	rootFD, err := openRoot(targetRoot, true)
	if err != nil {
		return fmt.Errorf("open deletion target root: %w", err)
	}
	defer unix.Close(rootFD)
	for _, entryPath := range deleted {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := validateRelativePath(entryPath); err != nil {
			return fmt.Errorf("invalid deleted file entry: %w", err)
		}
		if err := validateRemoval(ctx, rootFD, entryPath); err != nil {
			return fmt.Errorf("validate deletion %s: %w", entryPath, err)
		}
	}
	count := 0
	for _, entryPath := range deleted {
		if err := ctx.Err(); err != nil {
			return err
		}
		removed, err := removeBeneath(ctx, rootFD, entryPath)
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

func readDeletedFiles(
	checkpointPath string,
	expected int64,
) ([]string, error) {
	checkpointFD, err := openRoot(checkpointPath, false)
	if err != nil {
		return nil, fmt.Errorf("open checkpoint directory: %w", err)
	}
	defer unix.Close(checkpointFD)
	return readDeletedFilesAt(context.Background(), checkpointFD, expected)
}

func readDeletedFilesAt(
	ctx context.Context,
	checkpointFD int,
	expected int64,
) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	deletedFD, err := unix.Openat(
		checkpointFD,
		deletedFilesFilename,
		unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW|unix.O_NONBLOCK,
		0,
	)
	if err != nil {
		return nil, fmt.Errorf("open deleted files metadata: %w", err)
	}
	defer unix.Close(deletedFD)
	var stat unix.Stat_t
	if err := unix.Fstat(deletedFD, &stat); err != nil {
		return nil, fmt.Errorf("stat deleted files metadata: %w", err)
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFREG ||
		stat.Size < 0 ||
		stat.Size > maxDeletedFilesSize {
		return nil, fmt.Errorf(
			"%s is not valid deletion metadata",
			deletedFilesFilename,
		)
	}
	data, err := readAllContext(ctx, fdReader(deletedFD), maxDeletedFilesSize)
	if err != nil {
		return nil, fmt.Errorf("read deleted files metadata: %w", err)
	}
	deleted, err := parseDeletedFiles(ctx, data)
	if err != nil {
		return nil, err
	}
	if int64(len(deleted)) != expected {
		return nil, fmt.Errorf(
			"deleted files metadata count mismatch: marker=%d file=%d",
			expected,
			len(deleted),
		)
	}
	return deleted, nil
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

type fdReader int

func (fd fdReader) Read(buffer []byte) (int, error) {
	read, err := unix.Read(int(fd), buffer)
	if read == 0 && err == nil {
		return 0, io.EOF
	}
	return read, err
}

func (r contextReader) Read(buffer []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(buffer)
}

func readAllContext(
	ctx context.Context,
	reader io.Reader,
	limit int64,
) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(
		contextReader{ctx: ctx, reader: reader},
		limit+1,
	))
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("metadata exceeds %d bytes", limit)
	}
	return data, nil
}

func parseDeletedFiles(ctx context.Context, data []byte) ([]string, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(contextReader{
		ctx:    ctx,
		reader: bytes.NewReader(data),
	})
	token, err := decoder.Token()
	if err != nil {
		return nil, fmt.Errorf("parse deleted files: %w", err)
	}
	if delimiter, ok := token.(json.Delim); !ok || delimiter != '[' {
		return nil, fmt.Errorf("parse deleted files: expected a JSON array")
	}
	deleted := make([]string, 0)
	for decoder.More() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(deleted) >= maxDeletedFiles {
			return nil, fmt.Errorf(
				"parse deleted files: more than %d entries",
				maxDeletedFiles,
			)
		}
		var entryPath string
		if err := decoder.Decode(&entryPath); err != nil {
			return nil, fmt.Errorf("parse deleted files: %w", err)
		}
		if err := validateRelativePath(entryPath); err != nil {
			return nil, fmt.Errorf("invalid deleted file entry: %w", err)
		}
		deleted = append(deleted, entryPath)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := decoder.Token(); err != nil {
		return nil, fmt.Errorf("parse deleted files: %w", err)
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, ctxErr
		}
		return nil, fmt.Errorf("parse deleted files: trailing data")
	}
	return deleted, nil
}

func validateRemoval(
	ctx context.Context,
	rootFD int,
	entryPath string,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	parentFD, name, err := openParent(rootFD, entryPath)
	if errors.Is(err, unix.ENOENT) {
		return nil
	}
	if err != nil {
		return err
	}
	defer unix.Close(parentFD)
	return walkDirectoryAt(ctx, parentFD, name, false)
}

func removeBeneath(
	ctx context.Context,
	rootFD int,
	entryPath string,
) (bool, error) {
	if err := ctx.Err(); err != nil {
		return false, err
	}
	parentFD, name, err := openParent(rootFD, entryPath)
	if errors.Is(err, unix.ENOENT) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer unix.Close(parentFD)
	if err := ctx.Err(); err != nil {
		return false, err
	}
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
		if err := walkDirectoryAt(ctx, parentFD, name, true); err != nil {
			return false, err
		}
		if err := ctx.Err(); err != nil {
			return false, err
		}
		return true, unix.Unlinkat(parentFD, name, unix.AT_REMOVEDIR)
	}
	if err := ctx.Err(); err != nil {
		return false, err
	}
	return true, unix.Unlinkat(parentFD, name, 0)
}

func walkDirectoryAt(
	ctx context.Context,
	parentFD int,
	name string,
	remove bool,
) error {
	if err := ctx.Err(); err != nil {
		return err
	}
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
	names := make([]string, 0)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		batch, err := file.Readdirnames(256)
		names = append(names, batch...)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return err
		}
	}
	sort.Strings(names)
	if !remove {
		for _, child := range names {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := walkDirectoryAt(ctx, fd, child, false); err != nil {
				return err
			}
		}
		return nil
	}
	for _, child := range names {
		if err := ctx.Err(); err != nil {
			return err
		}
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
			if err := walkDirectoryAt(ctx, fd, child, true); err != nil {
				return err
			}
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := unix.Unlinkat(fd, child, unix.AT_REMOVEDIR); err != nil {
				return err
			}
		} else {
			if err := ctx.Err(); err != nil {
				return err
			}
			if err := unix.Unlinkat(fd, child, 0); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateRelativePath(entryPath string) error {
	if len(entryPath) > maxRootfsPathLength ||
		entryPath == "" || entryPath == "." ||
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
