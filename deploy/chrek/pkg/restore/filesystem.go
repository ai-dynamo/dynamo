package restore

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/sirupsen/logrus"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

// ApplyRootfsDiff extracts the rootfs-diff.tar from the checkpoint to the target root.
// This restores filesystem changes that were made in the original container.
func ApplyRootfsDiff(checkpointPath, targetRoot string, log *logrus.Entry) error {
	rootfsDiffPath := filepath.Join(checkpointPath, checkpoint.RootfsDiffFilename)

	// Check if rootfs-diff.tar exists
	if _, err := os.Stat(rootfsDiffPath); os.IsNotExist(err) {
		log.Info("No rootfs-diff.tar found, skipping filesystem restoration")
		return nil
	}

	log.WithField("path", rootfsDiffPath).Info("Applying rootfs diff")

	// Exclusions are already applied at checkpoint time (bind mounts, system dirs, etc.)
	// so we just extract with --keep-old-files to avoid overwriting existing files.
	cmd := exec.Command("tar",
		"--keep-old-files",
		"-C", targetRoot,
		"-xf", rootfsDiffPath,
	)

	output, err := cmd.CombinedOutput()
	if err != nil {
		// Some failures are expected (read-only mounts, existing files)
		// tar returns exit code 1 for "file exists" which is not fatal for us
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
			log.WithField("output", string(output)).Info("Rootfs diff applied (some files may have been skipped due to mounts)")
			return nil
		}
		return fmt.Errorf("failed to extract rootfs diff: %w (output: %s)", err, string(output))
	}

	log.Info("Rootfs diff applied successfully")
	return nil
}

// ApplyDeletedFiles removes files that were deleted in the original container.
// These are tracked via overlay whiteout markers (.wh.<filename>).
func ApplyDeletedFiles(checkpointPath, targetRoot string, log *logrus.Entry) error {
	deletedFilesPath := filepath.Join(checkpointPath, checkpoint.DeletedFilesFilename)

	// Check if deleted-files.json exists
	data, err := os.ReadFile(deletedFilesPath)
	if os.IsNotExist(err) {
		log.Debug("No deleted-files.json found")
		return nil
	}
	if err != nil {
		return fmt.Errorf("failed to read deleted files list: %w", err)
	}

	log.Info("Applying deleted files from whiteout list")

	// Parse JSON array of deleted file paths
	var deletedFiles []string
	if err := json.Unmarshal(data, &deletedFiles); err != nil {
		return fmt.Errorf("failed to parse deleted files JSON: %w", err)
	}

	deletedCount := 0
	for _, filePath := range deletedFiles {
		if filePath == "" {
			continue
		}

		targetPath := filepath.Join(targetRoot, filePath)

		// Check if file exists before attempting deletion
		if _, err := os.Stat(targetPath); os.IsNotExist(err) {
			continue
		}

		if err := os.RemoveAll(targetPath); err != nil {
			log.WithError(err).WithField("path", targetPath).Debug("Could not delete file")
			continue
		}
		deletedCount++
	}

	log.WithField("count", deletedCount).Info("Deleted files applied")
	return nil
}

// CheckpointFilesExist verifies that the checkpoint directory contains valid checkpoint files.
func CheckpointFilesExist(checkpointPath string) bool {
	// Check for CRIU image files (core-*.img is always present)
	matches, err := filepath.Glob(filepath.Join(checkpointPath, "core-*.img"))
	return err == nil && len(matches) > 0
}
