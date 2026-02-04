// Package checkpoint provides CRIU checkpoint (dump) operations.
package checkpoint

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/sirupsen/logrus"
)

// RemoveSemaphores removes POSIX semaphores from the container's /dev/shm.
// Semaphores cause CRIU restore to fail with "Can't link dev/shm/link_remap.X -> dev/shm/sem.Y"
// because they maintain kernel state that cannot be correctly restored.
// This accesses the container's filesystem via /proc/<pid>/root/dev/shm/.
func RemoveSemaphores(pid int, hostProc string, log *logrus.Entry) {
	if hostProc == "" {
		hostProc = "/proc"
	}

	shmPath := filepath.Join(hostProc, fmt.Sprintf("%d/root/dev/shm", pid))

	entries, err := os.ReadDir(shmPath)
	if err != nil {
		log.WithError(err).Debug("Could not read container /dev/shm (may not exist)")
		return
	}

	var removed []string
	for _, entry := range entries {
		name := entry.Name()
		// Check for both "sem." and "sem_" prefixes for consistency with CaptureDevShm
		if strings.HasPrefix(name, "sem.") || strings.HasPrefix(name, "sem_") {
			semPath := filepath.Join(shmPath, name)
			if err := os.Remove(semPath); err != nil {
				log.WithError(err).WithField("semaphore", name).Warn("Failed to remove semaphore")
			} else {
				removed = append(removed, name)
			}
		}
	}

	if len(removed) > 0 {
		log.WithFields(logrus.Fields{
			"count":      len(removed),
			"semaphores": removed,
		}).Info("Removed semaphores from container /dev/shm before checkpoint")
	} else {
		log.Debug("No semaphores found in container /dev/shm")
	}
}

const (
    // DevShmDirName is the directory name for captured /dev/shm contents
    DevShmDirName = "dev-shm"
)

// CaptureDevShm captures files from /dev/shm to the checkpoint directory.
// This is needed because /dev/shm is a tmpfs mount that is not part of the
// container's overlay filesystem, so rootfs diff doesn't capture it.
//
// Files starting with "sem." are excluded because they are POSIX semaphores
// whose kernel state cannot be correctly restored by copying the file.
//
// The files are saved to <checkpointDir>/dev-shm/ and can be restored
// using RestoreDevShm before CRIU restore.
func CaptureDevShm(pid int, hostProc, checkpointDir string, log *logrus.Entry) error {
    if hostProc == "" {
        hostProc = "/proc"
    }

    // Access container's /dev/shm via /proc/<pid>/root
    shmPath := filepath.Join(hostProc, fmt.Sprintf("%d/root/dev/shm", pid))

    entries, err := os.ReadDir(shmPath)
    if err != nil {
        if os.IsNotExist(err) {
            log.Debug("Container /dev/shm does not exist, skipping capture")
            return nil
        }
        return fmt.Errorf("failed to read container /dev/shm: %w", err)
    }

    // Filter out semaphores and empty entries
    var filesToCapture []os.DirEntry
    for _, entry := range entries {
        name := entry.Name()

        // Skip semaphores - they have kernel state that can't be restored correctly
        if strings.HasPrefix(name, "sem.") || strings.HasPrefix(name, "sem_") {
            log.WithField("file", name).Debug("Skipping semaphore file")
            continue
        }

        // Skip directories (unlikely in /dev/shm but be safe)
        if entry.IsDir() {
            log.WithField("dir", name).Debug("Skipping directory in /dev/shm")
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

    return nil
}
