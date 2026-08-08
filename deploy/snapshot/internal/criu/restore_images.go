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
	sk_unix "github.com/checkpoint-restore/go-criu/v8/crit/images/sk-unix"
	"golang.org/x/sys/unix"
)

const (
	filesImageFilename            = "files.img"
	placeholderMountNamespacePath = "/proc/self/ns/mnt"
	cudaSocketNameStart           = "\x00cuda-uvmfd-"
	// CRIU records an unconnected AF_UNIX socket with Linux's CLOSE state value.
	linuxUnixSocketStateClose = 7
)

func prepareRestoreImageDir(checkpointPath string) (string, func(), error) {
	// The placeholder mount namespace remains container-specific with shareProcessNamespace,
	// so its inode uniquely scopes names that would collide in the shared network namespace.
	var stat unix.Stat_t
	if err := unix.Stat(placeholderMountNamespacePath, &stat); err != nil {
		return "", nil, fmt.Errorf("failed to stat placeholder mount namespace at %s: %w", placeholderMountNamespacePath, err)
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
		if rewriteLinuxAutobindDatagramSocketName(fileEntry.Usk, restoreSocketScope) {
			rewritten = true
		}
	}
	if !rewritten {
		return checkpointPath, func() {}, nil
	}

	privateDir, err := os.MkdirTemp(filepath.Dir(checkpointPath), ".dynamo-criu-restore-*")
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
		// CRIU does not modify restore images; hard links keep them visible after it
		// enters the restored mount namespace without copying large page images.
		if err := os.Link(filepath.Join(checkpointPath, name), filepath.Join(privateDir, name)); err != nil {
			return fail(fmt.Errorf("failed to hard-link CRIU image %s: %w", name, err))
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

func rewriteLinuxAutobindDatagramSocketName(entry *sk_unix.UnixSkEntry, restoreSocketScope uint64) bool {
	if entry == nil ||
		entry.Type == nil ||
		entry.State == nil ||
		entry.Peer == nil ||
		entry.Backlog == nil ||
		entry.Uflags == nil ||
		*entry.Type != uint32(unix.SOCK_DGRAM) ||
		*entry.State != linuxUnixSocketStateClose ||
		*entry.Peer != 0 ||
		*entry.Backlog != 0 ||
		*entry.Uflags != 0 ||
		entry.FilePerms != nil ||
		entry.NameDir != nil ||
		(entry.Deleted != nil && *entry.Deleted) ||
		len(entry.Name) != 6 ||
		entry.Name[0] != 0 {
		return false
	}
	for _, digit := range entry.Name[1:] {
		if (digit < '0' || digit > '9') && (digit < 'a' || digit > 'f') {
			return false
		}
	}

	rewritten := make([]byte, 0, len(entry.Name)+1+20)
	rewritten = append(rewritten, entry.Name...)
	rewritten = append(rewritten, '-')
	rewritten = strconv.AppendUint(rewritten, restoreSocketScope, 10)
	entry.Name = rewritten
	return true
}

func decimalDigits(value []byte) bool {
	for _, digit := range value {
		if digit < '0' || digit > '9' {
			return false
		}
	}
	return len(value) > 0
}
