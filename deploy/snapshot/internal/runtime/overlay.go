package runtime

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const (
	rootfsDiffFilename   = "rootfs-diff.tar"
	deletedFilesFilename = "deleted-files.json"
)

// RootfsDiffApplyStats contains low-overhead restore measurements.
type RootfsDiffApplyStats struct {
	SizeBytes       int64
	StatDuration    time.Duration
	ExtractDuration time.Duration
}

// DeletedFilesApplyStats contains low-overhead restore measurements.
type DeletedFilesApplyStats struct {
	SizeBytes      int64
	Entries        int
	Removed        int
	Skipped        int
	ReadDuration   time.Duration
	ParseDuration  time.Duration
	RemoveDuration time.Duration
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

// CaptureRootfsDiff captures the overlay upperdir to a tar file.
func CaptureRootfsDiff(upperDir, checkpointDir string, exclusions types.OverlaySettings, bindMountDests []string) (string, error) {
	if upperDir == "" {
		return "", fmt.Errorf("upperdir is empty")
	}

	rootfsDiffPath := filepath.Join(checkpointDir, rootfsDiffFilename)

	tarArgs := []string{"--xattrs"}
	for _, excl := range buildExclusions(exclusions) {
		tarArgs = append(tarArgs, "--exclude="+excl)
	}
	for _, dest := range bindMountDests {
		tarArgs = append(tarArgs, "--exclude=."+dest)
	}
	tarArgs = append(tarArgs, "-C", upperDir, "-cf", rootfsDiffPath, ".")

	cmd := exec.Command("tar", tarArgs...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("tar failed: %w (output: %s)", err, string(output))
	}

	return rootfsDiffPath, nil
}

// buildExclusions merges exclusion lists and normalizes paths for tar --exclude patterns.
func buildExclusions(s types.OverlaySettings) []string {
	exclusions := append([]string(nil), s.Exclusions...)
	for i, p := range exclusions {
		if strings.HasPrefix(p, "*") {
			continue
		}
		p = strings.TrimPrefix(p, ".")
		p = strings.TrimPrefix(p, "/")
		exclusions[i] = "./" + p
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

// ApplyRootfsDiff extracts rootfs-diff.tar into the target root.
func ApplyRootfsDiff(checkpointPath, targetRoot string, log logr.Logger) error {
	_, err := ApplyRootfsDiffWithStats(checkpointPath, targetRoot, log)
	return err
}

// ApplyRootfsDiffWithStats extracts rootfs-diff.tar and reports restore measurements.
func ApplyRootfsDiffWithStats(checkpointPath, targetRoot string, log logr.Logger) (RootfsDiffApplyStats, error) {
	var stats RootfsDiffApplyStats
	rootfsDiffPath := filepath.Join(checkpointPath, rootfsDiffFilename)
	statStart := time.Now()
	info, err := os.Stat(rootfsDiffPath)
	stats.StatDuration = time.Since(statStart)
	if os.IsNotExist(err) {
		log.V(1).Info("No rootfs-diff.tar, skipping")
		return stats, nil
	}
	if err == nil {
		stats.SizeBytes = info.Size()
	}

	// --skip-old-files: silently skip files that already exist in the restore target.
	// The rootfs diff only contains overlay upperdir changes (runtime-generated files
	// like triton caches, tmp files) — base image files should not be overwritten.
	log.Info("Applying rootfs diff", "target", targetRoot, "size_bytes", stats.SizeBytes)
	cmd := exec.Command("tar", "--skip-old-files", "-C", targetRoot, "-xf", rootfsDiffPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	extractStart := time.Now()
	err = cmd.Run()
	stats.ExtractDuration = time.Since(extractStart)
	if err != nil {
		return stats, fmt.Errorf("tar extract failed: %w", err)
	}
	return stats, nil
}

// ApplyDeletedFiles removes files marked as deleted in the checkpoint.
func ApplyDeletedFiles(checkpointPath, targetRoot string, log logr.Logger) error {
	_, err := ApplyDeletedFilesWithStats(checkpointPath, targetRoot, log)
	return err
}

// ApplyDeletedFilesWithStats removes deleted files and reports restore measurements.
func ApplyDeletedFilesWithStats(checkpointPath, targetRoot string, log logr.Logger) (DeletedFilesApplyStats, error) {
	var stats DeletedFilesApplyStats
	deletedFilesPath := filepath.Join(checkpointPath, deletedFilesFilename)
	readStart := time.Now()
	data, err := os.ReadFile(deletedFilesPath)
	stats.ReadDuration = time.Since(readStart)
	if os.IsNotExist(err) {
		return stats, nil
	}
	if err != nil {
		return stats, fmt.Errorf("failed to read deleted files: %w", err)
	}
	stats.SizeBytes = int64(len(data))

	var deletedFiles []string
	parseStart := time.Now()
	err = json.Unmarshal(data, &deletedFiles)
	stats.ParseDuration = time.Since(parseStart)
	if err != nil {
		return stats, fmt.Errorf("failed to parse deleted files: %w", err)
	}
	stats.Entries = len(deletedFiles)

	removeStart := time.Now()
	targetRootAbs, err := filepath.Abs(targetRoot)
	if err != nil {
		stats.RemoveDuration = time.Since(removeStart)
		return stats, fmt.Errorf("failed to resolve target root %s: %w", targetRoot, err)
	}
	targetRootPrefix := targetRootAbs + string(os.PathSeparator)
	for _, f := range deletedFiles {
		if f == "" {
			stats.Skipped++
			continue
		}
		target := filepath.Join(targetRoot, f)
		targetAbs, err := filepath.Abs(target)
		if err != nil || (targetAbs != targetRootAbs && !strings.HasPrefix(targetAbs, targetRootPrefix)) {
			log.V(1).Info("Skipping out-of-root deleted file entry", "entry", f)
			stats.Skipped++
			continue
		}
		if _, err := os.Stat(target); os.IsNotExist(err) {
			stats.Skipped++
			continue
		} else if err != nil {
			log.V(1).Info("Could not stat deleted file target", "path", target, "error", err)
			stats.Skipped++
			continue
		}
		if err := os.RemoveAll(target); err != nil {
			log.V(1).Info("Could not delete file", "path", target, "error", err)
			stats.Skipped++
			continue
		}
		stats.Removed++
	}
	stats.RemoveDuration = time.Since(removeStart)
	log.Info("Deleted files applied",
		"count", stats.Removed,
		"entries", stats.Entries,
		"skipped", stats.Skipped,
	)
	return stats, nil
}

// findWhiteoutFiles finds overlay whiteout files in the upperdir.
func findWhiteoutFiles(upperDir string) ([]string, error) {
	var whiteouts []string

	err := filepath.Walk(upperDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		name := info.Name()
		if strings.HasPrefix(name, ".wh.") {
			relPath, err := filepath.Rel(upperDir, path)
			if err != nil {
				return fmt.Errorf("failed to compute relative path for %s: %w", path, err)
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
	})

	return whiteouts, err
}
