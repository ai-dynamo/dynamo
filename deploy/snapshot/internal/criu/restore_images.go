package criu

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/checkpoint-restore/go-criu/v8/crit"
	"github.com/checkpoint-restore/go-criu/v8/crit/images/fdinfo"
	"golang.org/x/sys/unix"
)

const (
	filesImageFilename          = "files.img"
	placeholderPIDNamespacePath = "/proc/self/ns/pid"
	cudaSocketNameStart         = "\x00cuda-uvmfd-"
)

func prepareRestoreImageDir(checkpointPath string) (string, func(), error) {
	// CRIU creates the restored child namespace too late to rewrite images, but restored
	// CUDA endpoints do not rediscover this name. Only collision-free binding in the shared
	// network namespace is required, and sibling placeholders have distinct PID namespaces.
	var stat unix.Stat_t
	if err := unix.Stat(placeholderPIDNamespacePath, &stat); err != nil {
		return "", nil, fmt.Errorf("failed to stat placeholder PID namespace at %s: %w", placeholderPIDNamespacePath, err)
	}
	return prepareRestoreImageDirForSocketScope(checkpointPath, stat.Ino)
}

func prepareRestoreImageDirForSocketScope(checkpointPath string, restoreSocketScope uint64) (string, func(), error) {
	checkpointPath, err := filepath.Abs(checkpointPath)
	if err != nil {
		return "", nil, fmt.Errorf("failed to resolve checkpoint path: %w", err)
	}

	filesImage, err := os.Open(filepath.Join(checkpointPath, filesImageFilename))
	if err != nil {
		return "", nil, fmt.Errorf("failed to open %s: %w", filesImageFilename, err)
	}
	defer filesImage.Close()

	image, err := crit.New(filesImage, nil, "", false, false).Decode(&fdinfo.FileEntry{})
	if err != nil {
		return "", nil, fmt.Errorf("failed to decode %s: %w", filesImageFilename, err)
	}

	rewritten := false
	for _, entry := range image.Entries {
		fileEntry := entry.Message.(*fdinfo.FileEntry)
		if fileEntry.GetType() != fdinfo.FdTypes_UNIXSK || fileEntry.Usk == nil {
			continue
		}
		if name, ok := rewriteCUDASocketName(fileEntry.Usk.Name, restoreSocketScope); ok {
			fileEntry.Usk.Name = name
			rewritten = true
		}
	}
	if !rewritten {
		return checkpointPath, func() {}, nil
	}

	privateDir, err := os.MkdirTemp("", "dynamo-criu-restore-*")
	if err != nil {
		return "", nil, fmt.Errorf("failed to create private CRIU image directory: %w", err)
	}
	cleanup := func() {
		_ = os.RemoveAll(privateDir)
	}
	fail := func(err error) (string, func(), error) {
		cleanup()
		return "", nil, err
	}

	entries, err := os.ReadDir(checkpointPath)
	if err != nil {
		return fail(fmt.Errorf("failed to read checkpoint directory: %w", err))
	}
	for _, entry := range entries {
		name := entry.Name()
		if name == filesImageFilename || !strings.HasSuffix(name, ".img") {
			continue
		}
		// Restore images are read-only; symlinks avoid copying page images.
		if err := os.Symlink(filepath.Join(checkpointPath, name), filepath.Join(privateDir, name)); err != nil {
			return fail(fmt.Errorf("failed to expose CRIU image %s: %w", name, err))
		}
	}

	privateFilesImage, err := os.OpenFile(filepath.Join(privateDir, filesImageFilename), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return fail(fmt.Errorf("failed to create private %s: %w", filesImageFilename, err))
	}
	if err := crit.New(nil, privateFilesImage, "", false, false).Encode(image); err != nil {
		_ = privateFilesImage.Close()
		return fail(fmt.Errorf("failed to encode private %s: %w", filesImageFilename, err))
	}
	if err := privateFilesImage.Close(); err != nil {
		return fail(fmt.Errorf("failed to close private %s: %w", filesImageFilename, err))
	}

	return privateDir, cleanup, nil
}

func rewriteCUDASocketName(name []byte, restoreSocketScope uint64) ([]byte, bool) {
	if !bytes.HasPrefix(name, []byte(cudaSocketNameStart)) {
		return nil, false
	}

	end := len(name)
	for end > len(cudaSocketNameStart) && name[end-1] == 0 {
		end--
	}
	numericFields := name[len(cudaSocketNameStart):end]
	separator := bytes.IndexByte(numericFields, '-')
	if separator <= 0 ||
		separator == len(numericFields)-1 ||
		!decimalDigits(numericFields[:separator]) ||
		!decimalDigits(numericFields[separator+1:]) {
		return nil, false
	}

	rewritten := make([]byte, 0, len(name)+20)
	rewritten = append(rewritten, cudaSocketNameStart...)
	rewritten = strconv.AppendUint(rewritten, restoreSocketScope, 10)
	rewritten = append(rewritten, numericFields[separator:]...)
	rewritten = append(rewritten, name[end:]...)
	return rewritten, true
}

func decimalDigits(value []byte) bool {
	for _, digit := range value {
		if digit < '0' || digit > '9' {
			return false
		}
	}
	return len(value) > 0
}
