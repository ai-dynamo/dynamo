// Package restore provides CRIU restore operations.
package restore

import (
    "fmt"
    "io"
    "os"
    "path/filepath"

    "github.com/sirupsen/logrus"

    "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// RestoreDevShm restores files from the checkpoint's dev-shm directory to /dev/shm.
// This must be called BEFORE CRIU restore so that the shared memory files exist
// when CRIU tries to restore file descriptors pointing to them.
//
// /dev/shm may contain various types of files including:
//   - POSIX semaphores (sem.*)
//   - PSM/shared memory files used by communication libraries
//   - Other tmpfs-backed shared memory regions
//
// The kernel state for these files won't be perfectly restored, but the files
// will exist for CRIU to restore file descriptors and for cleanup operations.
func RestoreDevShm(checkpointPath string, log *logrus.Entry) error {
    srcDir := filepath.Join(checkpointPath, config.DevShmDirName)

    // Check if dev-shm directory exists in checkpoint
    entries, err := os.ReadDir(srcDir)
    if err != nil {
        if os.IsNotExist(err) {
            log.Debug("No dev-shm directory in checkpoint, skipping restore")
            return nil
        }
        return fmt.Errorf("failed to read checkpoint dev-shm directory: %w", err)
    }

    if len(entries) == 0 {
        log.Debug("Checkpoint dev-shm directory is empty")
        return nil
    }

    // Ensure /dev/shm exists and is writable
    destDir := "/dev/shm"
    if err := os.MkdirAll(destDir, 0777); err != nil {
        return fmt.Errorf("failed to ensure /dev/shm exists: %w", err)
    }

    var restored []string
    var totalSize int64

    for _, entry := range entries {
        if entry.IsDir() {
            continue
        }

        name := entry.Name()
        srcPath := filepath.Join(srcDir, name)
        destPath := filepath.Join(destDir, name)

        info, err := entry.Info()
        if err != nil {
            log.WithError(err).WithField("file", name).Warn("Failed to get file info, skipping")
            continue
        }

        size := info.Size()

        // Copy the file to /dev/shm
        if err := copyFileToShm(srcPath, destPath, info.Mode()); err != nil {
            log.WithError(err).WithField("file", name).Warn("Failed to restore file, skipping")
            continue
        }

        restored = append(restored, name)
        totalSize += size

        log.WithFields(logrus.Fields{
            "file": name,
            "size": size,
        }).Debug("Restored /dev/shm file")
    }

    if len(restored) > 0 {
        log.WithFields(logrus.Fields{
            "count":      len(restored),
            "total_size": totalSize,
            "files":      restored,
        }).Info("Restored /dev/shm files from checkpoint")
    }

    return nil
}

// copyFileToShm copies a file from src to dest in /dev/shm.
// Uses mode 0666 as default when mode is 0, otherwise preserves the original mode.
func copyFileToShm(src, dest string, mode os.FileMode) error {
    srcFile, err := os.Open(src)
    if err != nil {
        return fmt.Errorf("failed to open source: %w", err)
    }
    defer srcFile.Close()

    // Default to 0666 when mode is not set (mode == 0)
    if mode == 0 {
        mode = 0666
    }

    destFile, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
    if err != nil {
        return fmt.Errorf("failed to create destination: %w", err)
    }
    defer destFile.Close()

    if _, err := io.Copy(destFile, srcFile); err != nil {
        return fmt.Errorf("failed to copy contents: %w", err)
    }

    return nil
}
