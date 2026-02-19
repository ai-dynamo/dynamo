// filesystem.go provides container rootfs introspection, filesystem config/metadata types,
// and rootfs diff capture for CRIU checkpoint.
package checkpoint

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
)

// FilesystemConfig is the static config for rootfs exclusions (from values.yaml).
type FilesystemConfig struct {
	// SystemDirs are system directories that should be excluded from rootfs diff.
	// These directories are typically injected/bind-mounted by NVIDIA GPU Operator
	// at container start time, so they already exist in the restore target.
	// Excluding them prevents conflicts (especially socket files which cannot be overwritten).
	// Default: ["./usr", "./etc", "./opt", "./var", "./run"]
	SystemDirs []string `yaml:"systemDirs"`

	// CacheDirs are cache directories that can safely be excluded to reduce checkpoint size.
	// Model weights and other cached data are typically re-downloaded if needed.
	// Default: ["./.cache/huggingface", "./.cache/torch"]
	CacheDirs []string `yaml:"cacheDirs"`

	// AdditionalExclusions are custom paths to exclude from the rootfs diff.
	// Use this for application-specific exclusions.
	// Paths should be relative with "./" prefix (e.g., "./data/temp").
	AdditionalExclusions []string `yaml:"additionalExclusions"`
}

// GetAllExclusions returns all exclusion paths combined.
// This is used when building tar arguments for rootfs diff capture.
func (c *FilesystemConfig) GetAllExclusions() []string {
	if c == nil {
		return nil
	}
	total := len(c.SystemDirs) + len(c.CacheDirs) + len(c.AdditionalExclusions)
	exclusions := make([]string, 0, total)
	exclusions = append(exclusions, c.SystemDirs...)
	exclusions = append(exclusions, c.CacheDirs...)
	exclusions = append(exclusions, c.AdditionalExclusions...)
	return exclusions
}

// Validate checks that the FilesystemConfig has valid values.
func (c *FilesystemConfig) Validate() error {
	if c == nil {
		return nil
	}
	// All paths should start with "./" for tar relative path handling
	for _, path := range c.GetAllExclusions() {
		if !strings.HasPrefix(path, "./") {
			return &ConfigError{
				Field:   "rootfsExclusions",
				Message: "all exclusion paths must start with './' (got: " + path + ")",
			}
		}
	}
	return nil
}

// FilesystemManifest holds runtime filesystem state captured at checkpoint time.
type FilesystemManifest struct {
	Exclusions      FilesystemConfig `yaml:"exclusions"`
	UpperDir        string           `yaml:"upperDir,omitempty"`
	ExternalPaths   []string         `yaml:"externalPaths,omitempty"`
	BindMountDests  []string         `yaml:"bindMountDests,omitempty"`
	HasRootfsDiff   bool             `yaml:"hasRootfsDiff"`
	HasDeletedFiles bool             `yaml:"hasDeletedFiles"`
}

// NewFilesystemManifest constructs FilesystemManifest from config, overlay state, and OCI spec.
func NewFilesystemManifest(exclusions FilesystemConfig, upperDir string, ociSpec *specs.Spec) FilesystemManifest {
	meta := FilesystemManifest{
		Exclusions: exclusions,
		UpperDir:   upperDir,
	}
	if ociSpec == nil {
		return meta
	}

	if ociSpec.Linux != nil {
		meta.ExternalPaths = make([]string, 0, len(ociSpec.Linux.MaskedPaths)+len(ociSpec.Linux.ReadonlyPaths))
		meta.ExternalPaths = append(meta.ExternalPaths, ociSpec.Linux.MaskedPaths...)
		meta.ExternalPaths = append(meta.ExternalPaths, ociSpec.Linux.ReadonlyPaths...)
	}
	for _, m := range ociSpec.Mounts {
		if m.Type == "bind" {
			meta.BindMountDests = append(meta.BindMountDests, m.Destination)
		}
	}
	return meta
}

// GetRootFS returns the container's root filesystem path.
func GetRootFS(pid int) (string, error) {
	rootPath := fmt.Sprintf("%s/%d/root", HostProcPath, pid)

	if _, err := os.Stat(rootPath); err != nil {
		return "", fmt.Errorf("rootfs not accessible at %s: %w", rootPath, err)
	}

	return rootPath, nil
}

// GetOverlayUpperDir extracts the overlay upperdir from mountinfo.
// This is the writable layer of the container's filesystem.
func GetOverlayUpperDir(pid int) (string, error) {
	mountInfo, err := ReadMountInfoFromHostProcPath(pid)
	if err != nil {
		return "", fmt.Errorf("failed to parse mountinfo: %w", err)
	}

	for _, mount := range mountInfo {
		if mount.MountPoint != "/" || mount.FSType != "overlay" {
			continue
		}

		// Parse super options to find upperdir
		for _, opt := range strings.Split(mount.SuperOptions, ",") {
			if strings.HasPrefix(opt, "upperdir=") {
				return strings.TrimPrefix(opt, "upperdir="), nil
			}
		}
	}

	return "", fmt.Errorf("overlay upperdir not found for pid %d", pid)
}

// CaptureRootfsDiff captures the overlay upperdir to a tar file.
// The upperdir contains all filesystem modifications made by the container.
// Excludes bind mount destinations and configured directories to avoid conflicts during restore.
// Returns the path to the tar file or empty string if capture failed.
func CaptureRootfsDiff(upperDir, checkpointDir string, exclusions *FilesystemConfig, bindMountDests []string) (string, error) {
	if upperDir == "" {
		return "", fmt.Errorf("upperdir is empty")
	}

	rootfsDiffPath := filepath.Join(checkpointDir, RootfsDiffFilename)

	// Build tar arguments with xattrs and exclusions
	tarArgs := []string{"--xattrs"}

	// Add configured exclusions (systemDirs, cacheDirs, additionalExclusions from values.yaml)
	if exclusions != nil {
		for _, excl := range exclusions.GetAllExclusions() {
			tarArgs = append(tarArgs, "--exclude="+excl)
		}
	}

	// Add bind mount exclusions passed from caller
	for _, dest := range bindMountDests {
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

	deletedFilesPath := filepath.Join(checkpointDir, DeletedFilesFilename)
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

// CaptureRootfsState captures the overlay upperdir and deleted files after CRIU dump.
// Updates the checkpoint manifest with rootfs diff information and saves it.
func CaptureRootfsState(upperDir, checkpointDir string, data *CheckpointManifest, log *logrus.Entry) {
	if upperDir == "" || data == nil {
		return
	}

	// Capture rootfs diff using exclusions from the checkpoint manifest.
	configuredExclusions := data.Filesystem.Exclusions.GetAllExclusions()
	log.WithFields(logrus.Fields{
		"configured_exclusions": configuredExclusions,
		"bind_mount_exclusions": data.Filesystem.BindMountDests,
	}).Debug("Rootfs diff exclusions")
	rootfsDiffPath, err := CaptureRootfsDiff(upperDir, checkpointDir, &data.Filesystem.Exclusions, data.Filesystem.BindMountDests)
	if err != nil {
		log.WithError(err).Warn("Failed to capture rootfs diff")
	} else {
		data.Filesystem.HasRootfsDiff = true
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
		data.Filesystem.HasDeletedFiles = true
		log.Info("Recorded deleted files (whiteouts)")
	}

	// Update checkpoint manifest with rootfs diff info.
	if err := WriteCheckpointManifest(checkpointDir, data); err != nil {
		log.WithError(err).Warn("Failed to update checkpoint manifest with rootfs diff info")
	}
}
