package runtime

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const (
	// RootfsWorkersEnv configures the bounded directory copy worker pool.
	RootfsWorkersEnv = "DYN_SNAPSHOT_ROOTFS_WORKERS"

	deletedFilesFilename    = "deleted-files.json"
	rootfsDirectoryFilename = "rootfs-diff"
	rootfsMetadataFilename  = "rootfs-diff.meta.json"
	rootfsDirectoryFormat   = "dynamo-rootfs-directory"
	rootfsDirectoryVersion  = 1
	defaultRootfsWorkers    = 16
	maxRootfsWorkers        = 256
	maxRootfsMetadataSize   = 4096
)

type rootfsDirectoryMetadata struct {
	Format  string `json:"format"`
	Version int    `json:"version"`
	Entries int64  `json:"entries"`
	Bytes   int64  `json:"bytes"`
}

type directoryCopyStats struct {
	Entries int64
	Bytes   int64
}

type artifactEntryType int

const (
	artifactDirectory artifactEntryType = iota
	artifactRegular
	artifactSymlink
)

type artifactEntry struct {
	path       string
	entryType  artifactEntryType
	stat       unix.Stat_t
	linkTarget string
}

type inodeKey struct {
	dev uint64
	ino uint64
}

type regularFileGroup struct {
	entries []artifactEntry
}

type exclusionMatcher struct {
	re *regexp.Regexp
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

// CaptureRootfsDiff captures the overlay upperdir to a directory artifact.
func CaptureRootfsDiff(
	upperDir string,
	checkpointDir string,
	exclusions types.OverlaySettings,
	bindMountDests []string,
) (string, error) {
	return CaptureRootfsDiffContext(
		context.Background(),
		upperDir,
		checkpointDir,
		exclusions,
		bindMountDests,
	)
}

// CaptureRootfsDiffContext captures the overlay upperdir to a directory artifact.
func CaptureRootfsDiffContext(
	ctx context.Context,
	upperDir string,
	checkpointDir string,
	exclusions types.OverlaySettings,
	bindMountDests []string,
) (string, error) {
	if upperDir == "" {
		return "", fmt.Errorf("upperdir is empty")
	}

	return captureRootfsDirectory(
		ctx,
		upperDir,
		checkpointDir,
		exclusions,
		bindMountDests,
	)
}

// buildExclusions merges exclusion lists and normalizes root-relative patterns.
func buildExclusions(s types.OverlaySettings) []string {
	exclusions := append([]string(nil), s.Exclusions...)
	for i, path := range exclusions {
		if strings.HasPrefix(path, "*") {
			continue
		}
		path = strings.TrimPrefix(path, ".")
		path = strings.TrimPrefix(path, "/")
		exclusions[i] = "./" + path
	}
	return exclusions
}

// CaptureDeletedFiles finds whiteout files and saves them to a JSON file.
func CaptureDeletedFiles(upperDir, checkpointDir string) (bool, error) {
	if upperDir == "" {
		return false, nil
	}

	whiteouts, err := findWhiteoutFiles(upperDir)
	if err != nil {
		return false, fmt.Errorf("failed to find whiteout files: %w", err)
	}

	if len(whiteouts) == 0 {
		return false, nil
	}

	deletedFilesPath := filepath.Join(checkpointDir, deletedFilesFilename)
	data, err := json.Marshal(whiteouts)
	if err != nil {
		return false, fmt.Errorf("failed to marshal whiteouts: %w", err)
	}

	if err := os.WriteFile(deletedFilesPath, data, 0644); err != nil {
		return false, fmt.Errorf("failed to write deleted files: %w", err)
	}

	return true, nil
}

// ApplyRootfsDiff applies a rootfs directory artifact to the target root.
func ApplyRootfsDiff(
	checkpointPath string,
	targetRoot string,
	log logr.Logger,
) error {
	return ApplyRootfsDiffContext(
		context.Background(),
		checkpointPath,
		targetRoot,
		log,
	)
}

// ApplyRootfsDiffContext applies a rootfs directory artifact.
func ApplyRootfsDiffContext(
	ctx context.Context,
	checkpointPath string,
	targetRoot string,
	log logr.Logger,
) error {
	metadata, present, err := rootfsDirectoryArtifact(checkpointPath)
	if err != nil {
		return err
	}
	if present {
		return applyRootfsDirectory(
			ctx,
			checkpointPath,
			targetRoot,
			log,
			metadata,
		)
	}
	log.V(1).Info("No rootfs directory artifact, skipping")
	return nil
}

func inspectExistingEntry(rootFD int, path string) (bool, bool, error) {
	fd, err := openBeneath(rootFD, path, unix.O_PATH)
	if err != nil {
		if errors.Is(err, unix.ENOENT) {
			return false, false, nil
		}
		if errors.Is(err, unix.ELOOP) {
			return true, false, nil
		}
		return false, false, err
	}
	defer unix.Close(fd)

	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil {
		return false, false, err
	}
	return true, stat.Mode&unix.S_IFMT == unix.S_IFREG, nil
}

// ApplyDeletedFiles removes files marked as deleted in the checkpoint.
func ApplyDeletedFiles(
	checkpointPath string,
	targetRoot string,
	log logr.Logger,
) error {
	deletedFilesPath := filepath.Join(checkpointPath, deletedFilesFilename)
	data, err := os.ReadFile(deletedFilesPath)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to read deleted files: %w", err)
	}

	var deletedFiles []string
	if err := json.Unmarshal(data, &deletedFiles); err != nil {
		return fmt.Errorf("failed to parse deleted files: %w", err)
	}

	count := 0
	targetRootAbs, err := filepath.Abs(targetRoot)
	if err != nil {
		return fmt.Errorf("failed to resolve target root %s: %w", targetRoot, err)
	}
	targetRootPrefix := targetRootAbs + string(os.PathSeparator)
	for _, file := range deletedFiles {
		if file == "" {
			continue
		}
		target := filepath.Join(targetRoot, file)
		targetAbs, err := filepath.Abs(target)
		if err != nil ||
			(targetAbs != targetRootAbs &&
				!strings.HasPrefix(targetAbs, targetRootPrefix)) {
			log.V(1).Info(
				"Skipping out-of-root deleted file entry",
				"entry",
				file,
			)
			continue
		}
		if _, err := os.Stat(target); os.IsNotExist(err) {
			continue
		} else if err != nil {
			log.V(1).Info(
				"Could not stat deleted file target",
				"path",
				target,
				"error",
				err,
			)
			continue
		}
		if err := os.RemoveAll(target); err != nil {
			log.V(1).Info(
				"Could not delete file",
				"path",
				target,
				"error",
				err,
			)
			continue
		}
		count++
	}
	log.Info("Deleted files applied", "count", count)
	return nil
}

// findWhiteoutFiles finds overlay whiteout files in the upperdir.
func findWhiteoutFiles(upperDir string) ([]string, error) {
	var whiteouts []string

	err := filepath.Walk(
		upperDir,
		func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			name := info.Name()
			if strings.HasPrefix(name, ".wh.") {
				relPath, err := filepath.Rel(upperDir, path)
				if err != nil {
					return fmt.Errorf(
						"failed to compute relative path for %s: %w",
						path,
						err,
					)
				}
				dir := filepath.Dir(relPath)
				deletedFile := strings.TrimPrefix(name, ".wh.")
				deletedPath := deletedFile
				if dir != "." {
					deletedPath = filepath.Join(dir, deletedFile)
				}
				whiteouts = append(whiteouts, deletedPath)
			}
			return nil
		},
	)

	return whiteouts, err
}

func rootfsWorkerCount(value string) (int, error) {
	if value == "" {
		return defaultRootfsWorkers, nil
	}
	workers, err := strconv.Atoi(value)
	if err != nil || workers < 1 || workers > maxRootfsWorkers {
		return 0, fmt.Errorf(
			"invalid %s value %q (want an integer from 1 to %d)",
			RootfsWorkersEnv,
			value,
			maxRootfsWorkers,
		)
	}
	return workers, nil
}

func captureRootfsDirectory(
	ctx context.Context,
	upperDir string,
	checkpointDir string,
	exclusions types.OverlaySettings,
	bindMountDests []string,
) (string, error) {
	workers, err := rootfsWorkerCount(os.Getenv(RootfsWorkersEnv))
	if err != nil {
		return "", err
	}
	matchers, err := compileExclusionMatchers(exclusions, bindMountDests)
	if err != nil {
		return "", err
	}

	sourceFD, err := openDirectoryRoot(upperDir)
	if err != nil {
		return "", fmt.Errorf("open overlay upperdir: %w", err)
	}
	defer unix.Close(sourceFD)

	entries, stats, err := scanArtifact(
		ctx,
		stablePath(sourceFD, "."),
		matchers,
		true,
	)
	if err != nil {
		return "", fmt.Errorf("scan overlay upperdir: %w", err)
	}

	tmpDir, err := os.MkdirTemp(checkpointDir, ".rootfs-diff.tmp-")
	if err != nil {
		return "", fmt.Errorf("create rootfs directory staging area: %w", err)
	}
	cleanupPath := tmpDir
	defer func() {
		if cleanupPath != "" {
			_ = os.RemoveAll(cleanupPath)
		}
	}()

	targetFD, err := openDirectoryRoot(tmpDir)
	if err != nil {
		return "", fmt.Errorf("open rootfs directory staging area: %w", err)
	}
	if err := copyArtifact(
		ctx,
		sourceFD,
		targetFD,
		entries,
		workers,
		false,
		true,
	); err != nil {
		unix.Close(targetFD)
		return "", fmt.Errorf("copy rootfs directory artifact: %w", err)
	}
	if err := unix.Fsync(targetFD); err != nil {
		unix.Close(targetFD)
		return "", fmt.Errorf("sync rootfs directory artifact: %w", err)
	}
	if err := unix.Close(targetFD); err != nil {
		return "", fmt.Errorf("close rootfs directory artifact: %w", err)
	}

	finalDir := filepath.Join(checkpointDir, rootfsDirectoryFilename)
	if err := os.Rename(tmpDir, finalDir); err != nil {
		return "", fmt.Errorf("publish rootfs directory artifact: %w", err)
	}
	cleanupPath = finalDir

	metadata := rootfsDirectoryMetadata{
		Format:  rootfsDirectoryFormat,
		Version: rootfsDirectoryVersion,
		Entries: stats.Entries,
		Bytes:   stats.Bytes,
	}
	if err := writeRootfsMetadata(checkpointDir, metadata); err != nil {
		_ = os.Remove(filepath.Join(checkpointDir, rootfsMetadataFilename))
		return "", err
	}
	cleanupPath = ""
	return finalDir, nil
}

func applyRootfsDirectory(
	ctx context.Context,
	checkpointPath string,
	targetRoot string,
	log logr.Logger,
	metadata rootfsDirectoryMetadata,
) error {
	workers, err := rootfsWorkerCount(os.Getenv(RootfsWorkersEnv))
	if err != nil {
		return err
	}
	artifactPath := filepath.Join(checkpointPath, rootfsDirectoryFilename)
	sourceFD, err := openDirectoryRoot(artifactPath)
	if err != nil {
		return fmt.Errorf("open rootfs directory artifact: %w", err)
	}
	defer unix.Close(sourceFD)

	entries, stats, err := scanArtifact(
		ctx,
		stablePath(sourceFD, "."),
		nil,
		false,
	)
	if err != nil {
		return fmt.Errorf("scan rootfs directory artifact: %w", err)
	}
	if stats.Entries != metadata.Entries || stats.Bytes != metadata.Bytes {
		return fmt.Errorf(
			"rootfs directory metadata mismatch: marker has entries=%d bytes=%d, artifact has entries=%d bytes=%d",
			metadata.Entries,
			metadata.Bytes,
			stats.Entries,
			stats.Bytes,
		)
	}

	targetFD, err := openTrustedDirectoryRoot(targetRoot)
	if err != nil {
		return fmt.Errorf("open rootfs restore target: %w", err)
	}
	defer unix.Close(targetFD)

	start := time.Now()
	if err := copyArtifact(
		ctx,
		sourceFD,
		targetFD,
		entries,
		workers,
		true,
		false,
	); err != nil {
		return err
	}
	log.Info(
		"Applied directory rootfs diff",
		"target", targetRoot,
		"workers", workers,
		"entries", stats.Entries,
		"bytes", stats.Bytes,
		"elapsed", time.Since(start),
	)
	return nil
}

func rootfsDirectoryArtifact(
	checkpointPath string,
) (rootfsDirectoryMetadata, bool, error) {
	artifactPath := filepath.Join(checkpointPath, rootfsDirectoryFilename)
	metadataPath := filepath.Join(checkpointPath, rootfsMetadataFilename)

	artifactInfo, artifactErr := os.Lstat(artifactPath)
	metadataInfo, metadataErr := os.Lstat(metadataPath)
	artifactMissing := errors.Is(artifactErr, os.ErrNotExist)
	metadataMissing := errors.Is(metadataErr, os.ErrNotExist)
	if artifactMissing && metadataMissing {
		return rootfsDirectoryMetadata{}, false, nil
	}
	if artifactErr != nil && !artifactMissing {
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("inspect rootfs directory artifact: %w", artifactErr)
	}
	if metadataErr != nil && !metadataMissing {
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("inspect rootfs directory metadata: %w", metadataErr)
	}
	if artifactMissing || metadataMissing {
		return rootfsDirectoryMetadata{}, false, fmt.Errorf(
			"incomplete rootfs directory artifact: %s and %s must both exist",
			rootfsDirectoryFilename,
			rootfsMetadataFilename,
		)
	}
	if !artifactInfo.IsDir() {
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("%s is not a directory", rootfsDirectoryFilename)
	}
	if !metadataInfo.Mode().IsRegular() {
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("%s is not a regular file", rootfsMetadataFilename)
	}

	metadataFD, err := unix.Open(
		metadataPath,
		unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW|unix.O_NONBLOCK,
		0,
	)
	if err != nil {
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("open rootfs directory metadata: %w", err)
	}
	var metadataStat unix.Stat_t
	if err := unix.Fstat(metadataFD, &metadataStat); err != nil {
		unix.Close(metadataFD)
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("stat rootfs directory metadata: %w", err)
	}
	if metadataStat.Mode&unix.S_IFMT != unix.S_IFREG {
		unix.Close(metadataFD)
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("%s is not a regular file", rootfsMetadataFilename)
	}
	if metadataStat.Size < 0 || metadataStat.Size > maxRootfsMetadataSize {
		unix.Close(metadataFD)
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf(
				"rootfs directory metadata size %d is invalid",
				metadataStat.Size,
			)
	}
	metadataFile := os.NewFile(uintptr(metadataFD), metadataPath)
	defer metadataFile.Close()
	data, err := io.ReadAll(io.LimitReader(
		metadataFile,
		maxRootfsMetadataSize+1,
	))
	if err != nil {
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("read rootfs directory metadata: %w", err)
	}
	if len(data) > maxRootfsMetadataSize {
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("rootfs directory metadata exceeds %d bytes", maxRootfsMetadataSize)
	}
	var metadata rootfsDirectoryMetadata
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&metadata); err != nil {
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("parse rootfs directory metadata: %w", err)
	}
	if err := ensureJSONEOF(decoder); err != nil {
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("parse rootfs directory metadata: %w", err)
	}
	if metadata.Format != rootfsDirectoryFormat ||
		metadata.Version != rootfsDirectoryVersion {
		return rootfsDirectoryMetadata{}, false, fmt.Errorf(
			"unsupported rootfs directory artifact format %q version %d",
			metadata.Format,
			metadata.Version,
		)
	}
	if metadata.Entries < 1 || metadata.Bytes < 0 {
		return rootfsDirectoryMetadata{}, false,
			fmt.Errorf("invalid rootfs directory artifact counts")
	}
	return metadata, true, nil
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("metadata contains multiple JSON values")
		}
		return err
	}
	return nil
}

func writeRootfsMetadata(
	checkpointDir string,
	metadata rootfsDirectoryMetadata,
) error {
	tmp, err := os.CreateTemp(checkpointDir, ".rootfs-diff.meta-")
	if err != nil {
		return fmt.Errorf("create rootfs directory metadata: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)

	if err := tmp.Chmod(0644); err != nil {
		tmp.Close()
		return fmt.Errorf("set rootfs directory metadata mode: %w", err)
	}
	encoder := json.NewEncoder(tmp)
	if err := encoder.Encode(metadata); err != nil {
		tmp.Close()
		return fmt.Errorf("write rootfs directory metadata: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return fmt.Errorf("sync rootfs directory metadata: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close rootfs directory metadata: %w", err)
	}
	metadataPath := filepath.Join(checkpointDir, rootfsMetadataFilename)
	if err := os.Rename(tmpPath, metadataPath); err != nil {
		return fmt.Errorf("publish rootfs directory metadata: %w", err)
	}

	parentFD, err := openDirectoryRoot(checkpointDir)
	if err != nil {
		return fmt.Errorf("open checkpoint directory for sync: %w", err)
	}
	defer unix.Close(parentFD)
	if err := unix.Fsync(parentFD); err != nil {
		return fmt.Errorf("sync checkpoint directory: %w", err)
	}
	return nil
}

func scanArtifact(
	ctx context.Context,
	root string,
	exclusions []exclusionMatcher,
	skipWhiteouts bool,
) ([]artifactEntry, directoryCopyStats, error) {
	entries := make([]artifactEntry, 0, 1024)
	stats := directoryCopyStats{}
	symlinkInodes := make(map[inodeKey]int)

	err := filepath.WalkDir(root, func(path string, dirEntry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return fmt.Errorf("resolve relative artifact path %s: %w", path, err)
		}
		if err := validateRelativePath(rel); err != nil {
			return err
		}
		if rel != "." && skipWhiteouts &&
			strings.HasPrefix(dirEntry.Name(), ".wh.") {
			return nil
		}
		if rel != "." && excludedArtifactPath(rel, exclusions) {
			if dirEntry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		var stat unix.Stat_t
		if err := unix.Lstat(path, &stat); err != nil {
			return fmt.Errorf("lstat %s: %w", rel, err)
		}
		entry := artifactEntry{path: rel, stat: stat}
		switch stat.Mode & unix.S_IFMT {
		case unix.S_IFDIR:
			entry.entryType = artifactDirectory
		case unix.S_IFREG:
			entry.entryType = artifactRegular
			newBytes := stats.Bytes + stat.Size
			if stat.Size < 0 || newBytes < stats.Bytes {
				return fmt.Errorf("rootfs entry size overflow at %s", rel)
			}
			stats.Bytes = newBytes
		case unix.S_IFLNK:
			entry.entryType = artifactSymlink
			entry.linkTarget, err = os.Readlink(path)
			if err != nil {
				return fmt.Errorf("read symlink %s: %w", rel, err)
			}
			symlinkInodes[inodeKey{dev: uint64(stat.Dev), ino: stat.Ino}]++
		default:
			return fmt.Errorf(
				"unsupported rootfs entry %s with mode %#o",
				rel,
				stat.Mode,
			)
		}
		entries = append(entries, entry)
		stats.Entries++
		return nil
	})
	if err != nil {
		return nil, directoryCopyStats{}, err
	}
	for key, count := range symlinkInodes {
		if count > 1 {
			return nil, directoryCopyStats{}, fmt.Errorf(
				"hard-linked symlinks are unsupported (device=%d inode=%d)",
				key.dev,
				key.ino,
			)
		}
	}
	return entries, stats, nil
}

func validateRelativePath(path string) error {
	if path == "." {
		return nil
	}
	if path == "" || filepath.IsAbs(path) || filepath.Clean(path) != path {
		return fmt.Errorf("unsafe rootfs artifact path %q", path)
	}
	for _, component := range strings.Split(path, string(os.PathSeparator)) {
		if component == "" || component == "." || component == ".." {
			return fmt.Errorf("unsafe rootfs artifact path %q", path)
		}
	}
	return nil
}

func compileExclusionMatchers(
	settings types.OverlaySettings,
	bindMountDests []string,
) ([]exclusionMatcher, error) {
	patterns := buildExclusions(settings)
	for _, dest := range bindMountDests {
		patterns = append(patterns, "."+dest)
	}
	matchers := make([]exclusionMatcher, 0, len(patterns))
	for _, pattern := range patterns {
		expression, err := rootfsGlobRegexp(pattern)
		if err != nil {
			return nil, fmt.Errorf("invalid rootfs exclusion %q: %w", pattern, err)
		}
		compiled, err := regexp.Compile(expression)
		if err != nil {
			return nil, fmt.Errorf("invalid rootfs exclusion %q: %w", pattern, err)
		}
		matchers = append(matchers, exclusionMatcher{re: compiled})
	}
	return matchers, nil
}

func rootfsGlobRegexp(pattern string) (string, error) {
	var expression strings.Builder
	expression.WriteString(`^(?:.*/)?`)
	for i := 0; i < len(pattern); i++ {
		switch pattern[i] {
		case '*':
			expression.WriteString(".*")
		case '?':
			expression.WriteByte('.')
		case '[':
			end := i + 1
			if end < len(pattern) && (pattern[end] == '!' || pattern[end] == '^') {
				end++
			}
			if end < len(pattern) && pattern[end] == ']' {
				end++
			}
			for end < len(pattern) && pattern[end] != ']' {
				end++
			}
			if end == len(pattern) {
				return "", fmt.Errorf("unterminated character class")
			}
			class := pattern[i+1 : end]
			if strings.HasPrefix(class, "!") {
				class = "^" + class[1:]
			}
			expression.WriteByte('[')
			expression.WriteString(class)
			expression.WriteByte(']')
			i = end
		case '\\':
			if i+1 >= len(pattern) {
				return "", fmt.Errorf("trailing escape")
			}
			i++
			expression.WriteString(regexp.QuoteMeta(pattern[i : i+1]))
		default:
			expression.WriteString(regexp.QuoteMeta(pattern[i : i+1]))
		}
	}
	expression.WriteString(`(?:/.*)?$`)
	return expression.String(), nil
}

func excludedArtifactPath(path string, matchers []exclusionMatcher) bool {
	artifactPath := "./" + filepath.ToSlash(path)
	for _, matcher := range matchers {
		if matcher.re.MatchString(artifactPath) {
			return true
		}
	}
	return false
}

func copyArtifact(
	ctx context.Context,
	sourceRootFD int,
	targetRootFD int,
	entries []artifactEntry,
	workers int,
	skipExisting bool,
	preserveRoot bool,
) error {
	createdDirectories := make(map[string]bool)
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.entryType != artifactDirectory || entry.path == "." {
			continue
		}
		created, err := createDirectory(targetRootFD, entry, skipExisting)
		if err != nil {
			return fmt.Errorf("create directory %s: %w", entry.path, err)
		}
		createdDirectories[entry.path] = created
	}

	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.entryType != artifactSymlink {
			continue
		}
		if err := copySymlink(
			sourceRootFD,
			targetRootFD,
			entry,
			skipExisting,
		); err != nil {
			return fmt.Errorf("copy symlink %s: %w", entry.path, err)
		}
	}

	groups := regularGroups(entries)
	if err := copyRegularGroups(
		ctx,
		sourceRootFD,
		targetRootFD,
		groups,
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
		if entry.entryType != artifactDirectory {
			continue
		}
		if entry.path == "." {
			if !preserveRoot {
				continue
			}
		} else if !createdDirectories[entry.path] {
			continue
		}
		sourceFD, err := openBeneath(
			sourceRootFD,
			entry.path,
			unix.O_RDONLY|unix.O_DIRECTORY,
		)
		if err != nil {
			return fmt.Errorf("open source directory %s: %w", entry.path, err)
		}
		targetFD, err := openBeneath(
			targetRootFD,
			entry.path,
			unix.O_RDONLY|unix.O_DIRECTORY,
		)
		if err != nil {
			unix.Close(sourceFD)
			return fmt.Errorf("open target directory %s: %w", entry.path, err)
		}
		err = copyFDMetadata(sourceFD, targetFD, entry.stat)
		unix.Close(targetFD)
		unix.Close(sourceFD)
		if err != nil {
			return fmt.Errorf("copy directory metadata %s: %w", entry.path, err)
		}
	}
	return nil
}

func createDirectory(
	targetRootFD int,
	entry artifactEntry,
	skipExisting bool,
) (bool, error) {
	parentFD, name, err := openParent(targetRootFD, entry.path)
	if err != nil {
		return false, err
	}
	defer unix.Close(parentFD)

	if err := unix.Mkdirat(parentFD, name, 0700); err != nil {
		if !errors.Is(err, unix.EEXIST) || !skipExisting {
			return false, err
		}
		fd, openErr := openBeneath(
			parentFD,
			name,
			unix.O_RDONLY|unix.O_DIRECTORY,
		)
		if openErr != nil {
			return false, fmt.Errorf("existing entry is not a safe directory: %w", openErr)
		}
		unix.Close(fd)
		return false, nil
	}
	return true, nil
}

func copySymlink(
	sourceRootFD int,
	targetRootFD int,
	entry artifactEntry,
	skipExisting bool,
) error {
	sourceParentFD, sourceName, err := openParent(sourceRootFD, entry.path)
	if err != nil {
		return err
	}
	defer unix.Close(sourceParentFD)
	target, err := readlinkAt(sourceParentFD, sourceName)
	if err != nil {
		return err
	}
	if target != entry.linkTarget {
		return fmt.Errorf("source changed while being copied")
	}

	targetParentFD, targetName, err := openParent(targetRootFD, entry.path)
	if err != nil {
		return err
	}
	defer unix.Close(targetParentFD)
	if err := unix.Symlinkat(target, targetParentFD, targetName); err != nil {
		if errors.Is(err, unix.EEXIST) && skipExisting {
			return nil
		}
		return err
	}
	removeOnError := true
	defer func() {
		if removeOnError {
			_ = unix.Unlinkat(targetParentFD, targetName, 0)
		}
	}()

	if err := unix.Fchownat(
		targetParentFD,
		targetName,
		int(entry.stat.Uid),
		int(entry.stat.Gid),
		unix.AT_SYMLINK_NOFOLLOW,
	); err != nil {
		return fmt.Errorf("set ownership: %w", err)
	}
	if err := copyPathXattrs(
		stablePath(sourceParentFD, sourceName),
		stablePath(targetParentFD, targetName),
	); err != nil {
		return fmt.Errorf("copy xattrs: %w", err)
	}
	times := []unix.Timespec{entry.stat.Atim, entry.stat.Mtim}
	if err := unix.UtimesNanoAt(
		targetParentFD,
		targetName,
		times,
		unix.AT_SYMLINK_NOFOLLOW,
	); err != nil {
		return fmt.Errorf("set timestamps: %w", err)
	}
	removeOnError = false
	return nil
}

func regularGroups(entries []artifactEntry) []regularFileGroup {
	groupIndexes := make(map[inodeKey]int)
	groups := make([]regularFileGroup, 0)
	for _, entry := range entries {
		if entry.entryType != artifactRegular {
			continue
		}
		key := inodeKey{dev: uint64(entry.stat.Dev), ino: entry.stat.Ino}
		index, ok := groupIndexes[key]
		if !ok {
			index = len(groups)
			groupIndexes[key] = index
			groups = append(groups, regularFileGroup{})
		}
		groups[index].entries = append(groups[index].entries, entry)
	}
	return groups
}

func copyRegularGroups(
	ctx context.Context,
	sourceRootFD int,
	targetRootFD int,
	groups []regularFileGroup,
	workers int,
	skipExisting bool,
) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	jobs := make(chan regularFileGroup)
	errs := make(chan error, 1)
	var workerGroup sync.WaitGroup

	for range workers {
		workerGroup.Add(1)
		go func() {
			defer workerGroup.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case group, ok := <-jobs:
					if !ok {
						return
					}
					if err := copyRegularGroup(
						ctx,
						sourceRootFD,
						targetRootFD,
						group,
						skipExisting,
					); err != nil {
						select {
						case errs <- err:
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
	for _, group := range groups {
		select {
		case jobs <- group:
		case <-ctx.Done():
			break enqueue
		}
	}
	close(jobs)
	workerGroup.Wait()

	select {
	case err := <-errs:
		return err
	default:
		return ctx.Err()
	}
}

func copyRegularGroup(
	ctx context.Context,
	sourceRootFD int,
	targetRootFD int,
	group regularFileGroup,
	skipExisting bool,
) error {
	var copiedPath string
	if skipExisting {
		for _, entry := range group.entries {
			if err := ctx.Err(); err != nil {
				return err
			}
			exists, regular, err := inspectExistingEntry(
				targetRootFD,
				entry.path,
			)
			if err != nil {
				return fmt.Errorf(
					"inspect existing entry %s: %w",
					entry.path,
					err,
				)
			}
			if exists && regular {
				copiedPath = entry.path
				break
			}
		}
	}

	var deferredLinks []artifactEntry
	for _, entry := range group.entries {
		if err := ctx.Err(); err != nil {
			return err
		}
		if copiedPath != "" {
			if entry.path != copiedPath {
				deferredLinks = append(deferredLinks, entry)
			}
			continue
		}
		created, err := copyRegularFile(
			ctx,
			sourceRootFD,
			targetRootFD,
			entry,
			skipExisting,
		)
		if err != nil {
			return fmt.Errorf("copy regular file %s: %w", entry.path, err)
		}
		if created {
			copiedPath = entry.path
		}
	}
	if copiedPath == "" {
		return nil
	}
	for _, entry := range deferredLinks {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := validateSourceRegular(sourceRootFD, entry); err != nil {
			return fmt.Errorf(
				"validate hardlink source %s: %w",
				entry.path,
				err,
			)
		}
		if err := createHardlink(
			targetRootFD,
			copiedPath,
			entry.path,
			skipExisting,
		); err != nil {
			return fmt.Errorf(
				"create hardlink %s to %s: %w",
				entry.path,
				copiedPath,
				err,
			)
		}
	}
	return nil
}

func copyRegularFile(
	ctx context.Context,
	sourceRootFD int,
	targetRootFD int,
	entry artifactEntry,
	skipExisting bool,
) (bool, error) {
	sourceFD, err := openBeneath(
		sourceRootFD,
		entry.path,
		unix.O_RDONLY,
	)
	if err != nil {
		return false, fmt.Errorf("open source: %w", err)
	}
	defer unix.Close(sourceFD)
	if err := validateSourceStat(sourceFD, entry); err != nil {
		return false, err
	}

	parentFD, name, err := openParent(targetRootFD, entry.path)
	if err != nil {
		return false, err
	}
	defer unix.Close(parentFD)
	targetFD, err := unix.Openat(
		parentFD,
		name,
		unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0600,
	)
	if err != nil {
		if errors.Is(err, unix.EEXIST) && skipExisting {
			return false, nil
		}
		return false, fmt.Errorf("create target: %w", err)
	}
	removeOnError := true
	defer func() {
		unix.Close(targetFD)
		if removeOnError {
			_ = unix.Unlinkat(parentFD, name, 0)
		}
	}()

	if err := copyRegularData(ctx, sourceFD, targetFD, entry.stat); err != nil {
		return false, err
	}
	if err := copyFDMetadata(sourceFD, targetFD, entry.stat); err != nil {
		return false, err
	}
	removeOnError = false
	return true, nil
}

func validateSourceRegular(sourceRootFD int, entry artifactEntry) error {
	sourceFD, err := openBeneath(
		sourceRootFD,
		entry.path,
		unix.O_PATH,
	)
	if err != nil {
		return err
	}
	defer unix.Close(sourceFD)
	return validateSourceStat(sourceFD, entry)
}

func validateSourceStat(sourceFD int, entry artifactEntry) error {
	var current unix.Stat_t
	if err := unix.Fstat(sourceFD, &current); err != nil {
		return fmt.Errorf("stat source: %w", err)
	}
	if current.Mode&unix.S_IFMT != unix.S_IFREG ||
		current.Dev != entry.stat.Dev ||
		current.Ino != entry.stat.Ino ||
		current.Size != entry.stat.Size {
		return fmt.Errorf("source changed while being copied")
	}
	return nil
}

func createHardlink(
	targetRootFD int,
	sourcePath string,
	targetPath string,
	skipExisting bool,
) error {
	sourceFD, err := openBeneath(targetRootFD, sourcePath, unix.O_PATH)
	if err != nil {
		return err
	}
	defer unix.Close(sourceFD)
	var sourceStat unix.Stat_t
	if err := unix.Fstat(sourceFD, &sourceStat); err != nil {
		return err
	}
	if sourceStat.Mode&unix.S_IFMT != unix.S_IFREG {
		return fmt.Errorf("hardlink source is not a regular file")
	}
	targetParentFD, targetName, err := openParent(targetRootFD, targetPath)
	if err != nil {
		return err
	}
	defer unix.Close(targetParentFD)
	if err := unix.Linkat(
		unix.AT_FDCWD,
		fmt.Sprintf("/proc/self/fd/%d", sourceFD),
		targetParentFD,
		targetName,
		unix.AT_SYMLINK_FOLLOW,
	); err != nil {
		if errors.Is(err, unix.EEXIST) && skipExisting {
			return nil
		}
		return err
	}
	return nil
}

func copyRegularData(
	ctx context.Context,
	sourceFD int,
	targetFD int,
	stat unix.Stat_t,
) error {
	if stat.Size == 0 {
		return unix.Ftruncate(targetFD, 0)
	}
	potentiallySparse := stat.Blocks >= 0 && stat.Blocks*512 < stat.Size
	position := int64(0)
	for position < stat.Size {
		dataOffset, err := unix.Seek(sourceFD, position, unix.SEEK_DATA)
		if errors.Is(err, unix.ENXIO) {
			break
		}
		if errors.Is(err, unix.EINVAL) || errors.Is(err, unix.ENOTSUP) {
			if potentiallySparse {
				return fmt.Errorf("sparse extent discovery is unsupported")
			}
			return copyFileRange(ctx, sourceFD, targetFD, 0, stat.Size)
		}
		if err != nil {
			return fmt.Errorf("find data extent: %w", err)
		}
		holeOffset, err := unix.Seek(sourceFD, dataOffset, unix.SEEK_HOLE)
		if errors.Is(err, unix.EINVAL) || errors.Is(err, unix.ENOTSUP) {
			if potentiallySparse {
				return fmt.Errorf("sparse extent discovery is unsupported")
			}
			return copyFileRange(ctx, sourceFD, targetFD, 0, stat.Size)
		}
		if err != nil {
			return fmt.Errorf("find hole extent: %w", err)
		}
		if holeOffset < dataOffset || holeOffset > stat.Size {
			return fmt.Errorf(
				"invalid sparse extent [%d,%d) for size %d",
				dataOffset,
				holeOffset,
				stat.Size,
			)
		}
		if err := copyFileRange(
			ctx,
			sourceFD,
			targetFD,
			dataOffset,
			holeOffset-dataOffset,
		); err != nil {
			return err
		}
		position = holeOffset
	}
	return unix.Ftruncate(targetFD, stat.Size)
}

func copyFileRange(
	ctx context.Context,
	sourceFD int,
	targetFD int,
	offset int64,
	length int64,
) error {
	sourceOffset := offset
	targetOffset := offset
	remaining := length
	for remaining > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk := int(min(remaining, int64(1<<30)))
		written, err := unix.CopyFileRange(
			sourceFD,
			&sourceOffset,
			targetFD,
			&targetOffset,
			chunk,
			0,
		)
		if err == nil && written > 0 {
			remaining -= int64(written)
			continue
		}
		if err == nil || errors.Is(err, unix.EXDEV) ||
			errors.Is(err, unix.EINVAL) ||
			errors.Is(err, unix.ENOSYS) ||
			errors.Is(err, unix.EOPNOTSUPP) ||
			errors.Is(err, unix.EPERM) {
			return copyFileRangeFallback(
				ctx,
				sourceFD,
				targetFD,
				sourceOffset,
				remaining,
			)
		}
		return fmt.Errorf("copy_file_range: %w", err)
	}
	return nil
}

func copyFileRangeFallback(
	ctx context.Context,
	sourceFD int,
	targetFD int,
	offset int64,
	length int64,
) error {
	buffer := make([]byte, 1<<20)
	for copied := int64(0); copied < length; {
		if err := ctx.Err(); err != nil {
			return err
		}
		want := int(min(int64(len(buffer)), length-copied))
		read, err := unix.Pread(sourceFD, buffer[:want], offset+copied)
		if err != nil {
			return fmt.Errorf("read source: %w", err)
		}
		if read == 0 {
			return io.ErrUnexpectedEOF
		}
		for written := 0; written < read; {
			n, err := unix.Pwrite(
				targetFD,
				buffer[written:read],
				offset+copied+int64(written),
			)
			if err != nil {
				return fmt.Errorf("write target: %w", err)
			}
			if n == 0 {
				return io.ErrShortWrite
			}
			written += n
		}
		copied += int64(read)
	}
	return nil
}

func copyFDMetadata(sourceFD int, targetFD int, stat unix.Stat_t) error {
	if err := unix.Fchown(targetFD, int(stat.Uid), int(stat.Gid)); err != nil {
		return fmt.Errorf("set ownership: %w", err)
	}
	if err := unix.Fchmod(targetFD, stat.Mode&07777); err != nil {
		return fmt.Errorf("set mode: %w", err)
	}
	if err := copyFDXattrs(sourceFD, targetFD); err != nil {
		return fmt.Errorf("copy xattrs: %w", err)
	}
	times := []unix.Timespec{stat.Atim, stat.Mtim}
	if err := unix.UtimesNanoAt(
		targetFD,
		"",
		times,
		unix.AT_EMPTY_PATH,
	); err != nil {
		return fmt.Errorf("set timestamps: %w", err)
	}
	return nil
}

func copyFDXattrs(sourceFD int, targetFD int) error {
	names, err := listFDXattrs(sourceFD)
	if err != nil {
		return err
	}
	for _, name := range names {
		value, err := getFDXattr(sourceFD, name)
		if err != nil {
			return fmt.Errorf("read %s: %w", name, err)
		}
		if err := unix.Fsetxattr(targetFD, name, value, 0); err != nil {
			return fmt.Errorf("write %s: %w", name, err)
		}
	}
	return nil
}

func listFDXattrs(fd int) ([]string, error) {
	size, err := unix.Flistxattr(fd, nil)
	if err != nil {
		return nil, err
	}
	if size == 0 {
		return nil, nil
	}
	for {
		buffer := make([]byte, size)
		read, err := unix.Flistxattr(fd, buffer)
		if errors.Is(err, unix.ERANGE) {
			size *= 2
			continue
		}
		if err != nil {
			return nil, err
		}
		return splitXattrNames(buffer[:read]), nil
	}
}

func getFDXattr(fd int, name string) ([]byte, error) {
	size, err := unix.Fgetxattr(fd, name, nil)
	if err != nil {
		return nil, err
	}
	for {
		value := make([]byte, size)
		read, err := unix.Fgetxattr(fd, name, value)
		if errors.Is(err, unix.ERANGE) {
			size = max(256, size*2)
			continue
		}
		if err != nil {
			return nil, err
		}
		return value[:read], nil
	}
}

func copyPathXattrs(sourcePath string, targetPath string) error {
	names, err := listPathXattrs(sourcePath)
	if err != nil {
		return err
	}
	for _, name := range names {
		size, err := unix.Lgetxattr(sourcePath, name, nil)
		if err != nil {
			return fmt.Errorf("size %s: %w", name, err)
		}
		for {
			value := make([]byte, size)
			read, err := unix.Lgetxattr(sourcePath, name, value)
			if errors.Is(err, unix.ERANGE) {
				size = max(256, size*2)
				continue
			}
			if err != nil {
				return fmt.Errorf("read %s: %w", name, err)
			}
			if err := unix.Lsetxattr(targetPath, name, value[:read], 0); err != nil {
				return fmt.Errorf("write %s: %w", name, err)
			}
			break
		}
	}
	return nil
}

func listPathXattrs(path string) ([]string, error) {
	size, err := unix.Llistxattr(path, nil)
	if err != nil {
		return nil, err
	}
	if size == 0 {
		return nil, nil
	}
	for {
		buffer := make([]byte, size)
		read, err := unix.Llistxattr(path, buffer)
		if errors.Is(err, unix.ERANGE) {
			size *= 2
			continue
		}
		if err != nil {
			return nil, err
		}
		return splitXattrNames(buffer[:read]), nil
	}
}

func splitXattrNames(buffer []byte) []string {
	var names []string
	for len(buffer) > 0 {
		index := bytes.IndexByte(buffer, 0)
		if index < 0 {
			break
		}
		if index > 0 {
			names = append(names, string(buffer[:index]))
		}
		buffer = buffer[index+1:]
	}
	return names
}

func openDirectoryRoot(path string) (int, error) {
	return unix.Open(
		path,
		unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
}

// openTrustedDirectoryRoot follows the caller-resolved root itself, including
// /proc/<pid>/root, then all descendant operations are descriptor-relative.
func openTrustedDirectoryRoot(path string) (int, error) {
	return unix.Open(path, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
}

func openBeneath(rootFD int, path string, flags int) (int, error) {
	if err := validateRelativePath(path); err != nil {
		return -1, err
	}
	return unix.Openat2(rootFD, path, &unix.OpenHow{
		Flags: uint64(flags | unix.O_CLOEXEC | unix.O_NOFOLLOW),
		Resolve: unix.RESOLVE_BENEATH |
			unix.RESOLVE_NO_MAGICLINKS |
			unix.RESOLVE_NO_SYMLINKS,
	})
}

func openParent(rootFD int, path string) (int, string, error) {
	if err := validateRelativePath(path); err != nil {
		return -1, "", err
	}
	if path == "." {
		return -1, "", fmt.Errorf("artifact root has no parent")
	}
	parent := filepath.Dir(path)
	name := filepath.Base(path)
	fd, err := openBeneath(
		rootFD,
		parent,
		unix.O_RDONLY|unix.O_DIRECTORY,
	)
	if err != nil {
		return -1, "", err
	}
	return fd, name, nil
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

func stablePath(parentFD int, name string) string {
	return fmt.Sprintf("/proc/self/fd/%d/%s", parentFD, name)
}
