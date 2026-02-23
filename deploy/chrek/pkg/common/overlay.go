package common

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/go-logr/logr"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

const (
	rootfsDiffFilename   = "rootfs-diff.tar"
	deletedFilesFilename = "deleted-files.json"
)

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
	total := len(s.SystemDirs) + len(s.CacheDirs) + len(s.AdditionalExclusions)
	exclusions := make([]string, 0, total)
	exclusions = append(exclusions, s.SystemDirs...)
	exclusions = append(exclusions, s.CacheDirs...)
	exclusions = append(exclusions, s.AdditionalExclusions...)
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
	rootfsDiffPath := filepath.Join(checkpointPath, rootfsDiffFilename)
	if _, err := os.Stat(rootfsDiffPath); os.IsNotExist(err) {
		log.V(1).Info("No rootfs-diff.tar, skipping")
		return nil
	}

	// --skip-old-files: silently skip files that already exist in the restore target.
	// The rootfs diff only contains overlay upperdir changes (runtime-generated files
	// like triton caches, tmp files) — base image files should not be overwritten.
	log.Info("Applying rootfs diff", "target", targetRoot)
	cmd := exec.Command("tar", "--skip-old-files", "-C", targetRoot, "-xf", rootfsDiffPath)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("tar extract failed: %w", err)
	}
	return nil
}

// ApplyDeletedFiles removes files marked as deleted in the checkpoint.
func ApplyDeletedFiles(checkpointPath, targetRoot string, log logr.Logger) error {
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
	for _, f := range deletedFiles {
		if f == "" {
			continue
		}
		target := filepath.Join(targetRoot, f)
		targetAbs, err := filepath.Abs(target)
		if err != nil || (targetAbs != targetRootAbs && !strings.HasPrefix(targetAbs, targetRootPrefix)) {
			log.V(1).Info("Skipping out-of-root deleted file entry", "entry", f)
			continue
		}
		if _, err := os.Stat(target); os.IsNotExist(err) {
			continue
		} else if err != nil {
			log.V(1).Info("Could not stat deleted file target", "path", target, "error", err)
			continue
		}
		if err := os.RemoveAll(target); err != nil {
			log.V(1).Info("Could not delete file", "path", target, "error", err)
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
