// Package checkpoint provides CRIU checkpoint (dump) operations.
package checkpoint

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/sirupsen/logrus"
)

// CaptureDevShm captures files from /dev/shm to the checkpoint directory.
// This is needed because /dev/shm is a tmpfs mount that is not part of the
// container's overlay filesystem, so rootfs diff doesn't capture it.
//
// Semaphores (sem.* files) are included so that sem_unlink() calls succeed
// after restore. The semaphore kernel state won't be perfectly restored,
// but the files will exist for cleanup operations.
//
// The files are saved to <checkpointDir>/dev-shm/ and can be restored
// using RestoreDevShm before CRIU restore.
func CaptureDevShm(pid int, checkpointDir string, log *logrus.Entry) error {
	// Access container's /dev/shm via /proc/<pid>/root
	shmPath := filepath.Join(HostProcPath, fmt.Sprintf("%d/root/dev/shm", pid))

	entries, err := os.ReadDir(shmPath)
	if err != nil {
		if os.IsNotExist(err) {
			log.Debug("Container /dev/shm does not exist, skipping capture")
			return nil
		}
		return fmt.Errorf("failed to read container /dev/shm: %w", err)
	}

	// Filter out directories
	var filesToCapture []os.DirEntry
	for _, entry := range entries {
		// Skip directories (unlikely in /dev/shm but be safe)
		if entry.IsDir() {
			log.WithField("dir", entry.Name()).Debug("Skipping directory in /dev/shm")
			continue
		}

		filesToCapture = append(filesToCapture, entry)
	}

	if len(filesToCapture) == 0 {
		log.Debug("No files to capture from /dev/shm")
		return nil
	}

	// Create destination directory
	destDir := filepath.Join(checkpointDir, DevShmDirName)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("failed to create dev-shm directory: %w", err)
	}

	var captured []string
	var totalSize int64

	for _, entry := range filesToCapture {
		name := entry.Name()
		srcPath := filepath.Join(shmPath, name)
		destPath := filepath.Join(destDir, name)

		info, err := entry.Info()
		if err != nil {
			log.WithError(err).WithField("file", name).Warn("Failed to get file info, skipping")
			continue
		}

		size := info.Size()

		// Copy the file
		if err := copyFile(srcPath, destPath, info.Mode()); err != nil {
			log.WithError(err).WithField("file", name).Warn("Failed to copy file, skipping")
			continue
		}

		captured = append(captured, name)
		totalSize += size

		log.WithFields(logrus.Fields{
			"file": name,
			"size": size,
		}).Debug("Captured /dev/shm file")
	}

	if len(captured) > 0 {
		log.WithFields(logrus.Fields{
			"count":      len(captured),
			"total_size": totalSize,
			"files":      captured,
		}).Info("Captured /dev/shm files")
	}

	return nil
}

// copyFile copies a file from src to dest with the given permissions.
func copyFile(src, dest string, mode os.FileMode) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open source: %w", err)
	}
	defer srcFile.Close()

	destFile, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return fmt.Errorf("failed to create destination: %w", err)
	}
	defer destFile.Close()

	if _, err := io.Copy(destFile, srcFile); err != nil {
		return fmt.Errorf("failed to copy contents: %w", err)
	}

	// Sync to ensure durability for checkpoint data
	if err := destFile.Sync(); err != nil {
		return fmt.Errorf("failed to sync destination: %w", err)
	}

	return nil
}
