// rootfs provides container rootfs introspection via /proc for CRIU checkpoint.
package checkpoint

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/sirupsen/logrus"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// GetRootFS returns the container's root filesystem path
// For containers using overlayfs, this extracts the upperdir
func GetRootFS(pid int, hostProc string) (string, error) {
	if hostProc == "" {
		hostProc = "/proc"
	}

	// The rootfs is accessible via /proc/<pid>/root
	// But for CRIU, we need the actual filesystem path
	rootPath := fmt.Sprintf("%s/%d/root", hostProc, pid)

	// Verify it exists
	if _, err := os.Stat(rootPath); err != nil {
		return "", fmt.Errorf("rootfs not accessible at %s: %w", rootPath, err)
	}

	return rootPath, nil
}

// GetOverlayUpperDir extracts the overlay upperdir from mountinfo
// This is the writable layer of the container's filesystem
func GetOverlayUpperDir(pid int, hostProc string) (string, error) {
	if hostProc == "" {
		hostProc = "/proc"
	}

	mountinfoPath := fmt.Sprintf("%s/%d/mountinfo", hostProc, pid)
	file, err := os.Open(mountinfoPath)
	if err != nil {
		return "", fmt.Errorf("failed to open mountinfo: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)

		// Look for the root mount (mount point is /)
		// mountinfo format: id parent major:minor root mount-point options ... - fstype source super-options
		if len(fields) < 5 {
			continue
		}

		mountPoint := fields[4]
		if mountPoint != "/" {
			continue
		}

		// Find the separator (-) to get fstype and options
		sepIdx := -1
		for i, f := range fields {
			if f == "-" {
				sepIdx = i
				break
			}
		}

		if sepIdx == -1 || sepIdx+2 >= len(fields) {
			continue
		}

		fsType := fields[sepIdx+1]
		if fsType != "overlay" {
			continue
		}

		// Parse super options to find upperdir
		superOptions := fields[sepIdx+3]
		for _, opt := range strings.Split(superOptions, ",") {
			if strings.HasPrefix(opt, "upperdir=") {
				return strings.TrimPrefix(opt, "upperdir="), nil
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("error reading mountinfo: %w", err)
	}

	return "", fmt.Errorf("overlay upperdir not found for pid %d", pid)
}

// DefaultRootfsDiffExclusions are paths excluded from the rootfs diff capture.
// These directories are injected/bind-mounted by NVIDIA GPU Operator at container
// start time, so they already exist in the restore target and cause conflicts
// (especially socket files which cannot be overwritten).
var DefaultRootfsDiffExclusions = []string{
	// NVIDIA GPU Operator injects drivers, libraries, and config here
	"./usr",
	"./etc",
	"./opt",
	"./var",

	// NVIDIA GPU Operator creates runtime sockets and firmware mounts here
	// Socket files cause fatal tar errors even with --keep-old-files
	"./run",
}

// CaptureRootfsDiff captures the overlay upperdir to a tar file.
// The upperdir contains all filesystem modifications made by the container.
// Excludes bind mount destinations and system directories to avoid conflicts during restore.
// Returns the path to the tar file or empty string if capture failed.
func CaptureRootfsDiff(upperDir, checkpointDir string, excludePaths []string) (string, error) {
	if upperDir == "" {
		return "", fmt.Errorf("upperdir is empty")
	}

	rootfsDiffPath := filepath.Join(checkpointDir, "rootfs-diff.tar")

	// Build tar arguments with xattrs and exclusions
	tarArgs := []string{"--xattrs"}

	// Add default exclusions for system directories and caches
	for _, excl := range DefaultRootfsDiffExclusions {
		tarArgs = append(tarArgs, "--exclude="+excl)
	}

	// Add bind mount exclusions passed from caller
	for _, dest := range excludePaths {
		// Convert absolute path to relative for tar (e.g., /etc/hosts -> ./etc/hosts)
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

// CaptureDeletedFiles finds whiteout files and saves them to a JSON file.
// Returns true if deleted files were found and saved.
func CaptureDeletedFiles(upperDir, checkpointDir string) (bool, error) {
	if upperDir == "" {
		return false, nil
	}

	whiteouts, err := FindWhiteoutFiles(upperDir)
	if err != nil {
		return false, fmt.Errorf("failed to find whiteout files: %w", err)
	}

	if len(whiteouts) == 0 {
		return false, nil
	}

	deletedFilesPath := filepath.Join(checkpointDir, "deleted-files.json")
	data, err := json.Marshal(whiteouts)
	if err != nil {
		return false, fmt.Errorf("failed to marshal whiteouts: %w", err)
	}

	if err := os.WriteFile(deletedFilesPath, data, 0644); err != nil {
		return false, fmt.Errorf("failed to write deleted files: %w", err)
	}

	return true, nil
}

// FindWhiteoutFiles finds overlay whiteout files in the upperdir.
// Overlay filesystems use .wh.<filename> to mark deleted files.
// Returns a list of paths that were deleted in the container.
func FindWhiteoutFiles(upperDir string) ([]string, error) {
	var whiteouts []string

	err := filepath.Walk(upperDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		name := info.Name()
		if strings.HasPrefix(name, ".wh.") {
			// Convert whiteout marker to actual deleted path
			relPath, _ := filepath.Rel(upperDir, path)
			dir := filepath.Dir(relPath)
			deletedFile := strings.TrimPrefix(name, ".wh.")
			if dir == "." {
				whiteouts = append(whiteouts, deletedFile)
			} else {
				whiteouts = append(whiteouts, filepath.Join(dir, deletedFile))
			}
		}
		return nil
	})

	return whiteouts, err
}

// CaptureRootfsState captures the overlay upperdir and deleted files after CRIU dump.
// Updates the metadata with rootfs diff information and saves it.
func CaptureRootfsState(upperDir, checkpointDir string, meta *common.CheckpointMetadata, log *logrus.Entry) {
	if upperDir == "" {
		return
	}

	// Capture rootfs diff
	log.WithFields(logrus.Fields{
		"default_exclusions":    DefaultRootfsDiffExclusions,
		"bind_mount_exclusions": meta.BindMountDests,
	}).Debug("Rootfs diff exclusions")
	rootfsDiffPath, err := CaptureRootfsDiff(upperDir, checkpointDir, meta.BindMountDests)
	if err != nil {
		log.WithError(err).Warn("Failed to capture rootfs diff")
	} else {
		meta.RootfsDiffPath = rootfsDiffPath
		meta.HasRootfsDiff = true
		log.WithFields(logrus.Fields{
			"upperdir": upperDir,
			"tar_path": rootfsDiffPath,
		}).Info("Captured rootfs diff")
	}

	// Capture deleted files (whiteouts)
	hasDeletedFiles, err := CaptureDeletedFiles(upperDir, checkpointDir)
	if err != nil {
		log.WithError(err).Warn("Failed to capture deleted files")
	} else if hasDeletedFiles {
		meta.HasDeletedFiles = true
		log.Info("Recorded deleted files (whiteouts)")
	}

	// Update metadata with rootfs diff info
	if err := common.SaveMetadata(checkpointDir, meta); err != nil {
		log.WithError(err).Warn("Failed to update metadata with rootfs diff info")
	}
}
