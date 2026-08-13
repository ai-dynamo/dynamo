// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package cuda

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"golang.org/x/sys/unix"
)

// JobFileEnv is the CUDA launch-job environment variable consumed by the driver.
const JobFileEnv = "CUDA_CHECKPOINT_JOB_FILE"

// StageJobFile copies a launch-job file into the checkpoint artifact and
// returns the host-visible path to the source pod's live job file. Capture
// helpers must use that live file so they join the same CUDA job as the target
// processes; the artifact copy is only a seed for later restore pods. A process
// may only name the job file created for this checkpoint; the privileged agent
// must not copy an arbitrary container path into shared storage.
func StageJobFile(hostProcPath string, cudaPIDs []int, checkpointDir string, sourceGPUCount int) (string, error) {
	if len(cudaPIDs) == 0 {
		return "", nil
	}

	jobFile := ""
	jobFilePID := 0
	missingPIDs := make([]int, 0, len(cudaPIDs))
	for _, pid := range cudaPIDs {
		value, err := processEnvironmentValue(hostProcPath, pid, JobFileEnv)
		if err != nil {
			return "", err
		}
		if value == "" {
			missingPIDs = append(missingPIDs, pid)
			continue
		}
		if jobFile == "" {
			jobFile = value
			jobFilePID = pid
			continue
		}
		if value != jobFile {
			return "", fmt.Errorf("CUDA processes do not share one %s: %q != %q", JobFileEnv, jobFile, value)
		}
	}
	if jobFile == "" {
		if sourceGPUCount > 1 {
			return "", fmt.Errorf("multi-GPU CUDA processes are missing %s", JobFileEnv)
		}
		return "", nil
	}
	if len(missingPIDs) > 0 {
		return "", fmt.Errorf("CUDA processes %v are missing %s while other CUDA processes use it", missingPIDs, JobFileEnv)
	}
	if !filepath.IsAbs(jobFile) || filepath.Clean(jobFile) != jobFile {
		return "", fmt.Errorf("%s must be an absolute, clean path, got %q", JobFileEnv, jobFile)
	}
	if jobFile == "/proc" || strings.HasPrefix(jobFile, "/proc/") {
		return "", fmt.Errorf("%s must be persisted outside procfs before checkpoint, got %q", JobFileEnv, jobFile)
	}
	if jobFile != snapshotprotocol.CUDAJobFilePath {
		return "", fmt.Errorf("%s is %q, want checkpoint job file %q", JobFileEnv, jobFile, snapshotprotocol.CUDAJobFilePath)
	}

	sourcePath := filepath.Join(hostProcPath, strconv.Itoa(jobFilePID), "root", strings.TrimPrefix(jobFile, string(os.PathSeparator)))
	destinationPath := filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName)
	if err := copyJobFile(sourcePath, destinationPath); err != nil {
		return "", fmt.Errorf("stage CUDA checkpoint job file: %w", err)
	}
	return sourcePath, nil
}

// refreshJobFileArtifact captures the job state after every CUDA process has
// reached CHECKPOINTED. CUDA mutates the launch-job file while checkpointing,
// so the earlier validation copy is not a valid restore seed.
func refreshJobFileArtifact(liveJobFile, checkpointDir string) error {
	if liveJobFile == "" {
		return nil
	}
	destinationPath := filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName)
	if err := prepareLiveJobFile(liveJobFile, destinationPath); err != nil {
		return fmt.Errorf("refresh CUDA checkpoint job file: %w", err)
	}
	return nil
}

// PrepareLiveJobFile materializes the immutable capture-time launch-job state
// at the stable path recorded in the checkpointed process environment. It runs
// inside the restore container's namespaces before CRIU recreates processes.
// The returned path is the per-restore working copy that CUDA helpers must use;
// the staged artifact remains immutable so it can seed later restores.
func PrepareLiveJobFile(stagedJobFile string) (string, error) {
	if err := prepareLiveJobFile(stagedJobFile, snapshotprotocol.CUDAJobFilePath); err != nil {
		return "", err
	}
	return snapshotprotocol.CUDAJobFilePath, nil
}

// JobFileFromCheckpoint returns the staged job file when an artifact contains one.
func JobFileFromCheckpoint(checkpointDir string) (string, error) {
	jobFile := filepath.Join(checkpointDir, snapshotprotocol.CUDAJobFileName)
	info, err := os.Lstat(jobFile)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("stat CUDA checkpoint job file: %w", err)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("CUDA checkpoint job file %q is not a regular file", jobFile)
	}
	return jobFile, nil
}

func processEnvironmentValue(hostProcPath string, pid int, name string) (string, error) {
	content, err := os.ReadFile(filepath.Join(hostProcPath, strconv.Itoa(pid), "environ"))
	if err != nil {
		return "", fmt.Errorf("read environment for CUDA process %d: %w", pid, err)
	}
	prefix := name + "="
	for _, entry := range strings.Split(string(content), "\x00") {
		if strings.HasPrefix(entry, prefix) {
			return strings.TrimPrefix(entry, prefix), nil
		}
	}
	return "", nil
}

func copyJobFile(sourcePath, destinationPath string) error {
	return copyRegularFile(sourcePath, destinationPath, unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL)
}

func prepareLiveJobFile(sourcePath, destinationPath string) error {
	return copyRegularFile(sourcePath, destinationPath, unix.O_WRONLY|unix.O_CREAT|unix.O_TRUNC)
}

func copyRegularFile(sourcePath, destinationPath string, destinationFlags int) (err error) {
	fd, err := unix.Open(sourcePath, unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0)
	if err != nil {
		return err
	}
	source := os.NewFile(uintptr(fd), sourcePath)
	defer func() {
		if closeErr := source.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("close source %q: %w", sourcePath, closeErr)
		}
	}()

	info, err := source.Stat()
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("source %q is not a regular file", sourcePath)
	}

	destinationFD, err := unix.Open(destinationPath, destinationFlags|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0600)
	if err != nil {
		return err
	}
	destination := os.NewFile(uintptr(destinationFD), destinationPath)
	defer func() {
		if closeErr := destination.Close(); closeErr != nil && err == nil {
			err = fmt.Errorf("close destination %q: %w", destinationPath, closeErr)
		}
	}()

	if err := destination.Chmod(0600); err != nil {
		return err
	}
	if _, err := io.Copy(destination, source); err != nil {
		return err
	}
	return destination.Sync()
}

// SetLiveJobFileOwner makes the restore-time working copy accessible to the
// restored workload without following a replacement symlink.
func SetLiveJobFileOwner(jobFile string, uid, gid int) error {
	fd, err := unix.Open(jobFile, unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0)
	if err != nil {
		return fmt.Errorf("open live CUDA checkpoint job file %q: %w", jobFile, err)
	}

	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil {
		_ = unix.Close(fd)
		return fmt.Errorf("stat live CUDA checkpoint job file %q: %w", jobFile, err)
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFREG {
		_ = unix.Close(fd)
		return fmt.Errorf("live CUDA checkpoint job file %q is not a regular file", jobFile)
	}
	if err := unix.Fchown(fd, uid, gid); err != nil {
		_ = unix.Close(fd)
		return fmt.Errorf("set live CUDA checkpoint job file %q owner to %d:%d: %w", jobFile, uid, gid, err)
	}
	if err := unix.Close(fd); err != nil {
		return fmt.Errorf("close live CUDA checkpoint job file %q: %w", jobFile, err)
	}
	return nil
}
