package runtime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

const (
	RootfsDiffFilename   = "rootfs-diff.squashfs"
	deletedFilesFilename = "deleted-files.json"
	// Bounds untrusted metadata while allowing at least ~4,000 PATH_MAX-sized entries and far more normal paths.
	maxDeletedFilesSidecarSize = 16 << 20
)

type deletedFiles struct {
	Whiteouts       []string `json:"whiteouts,omitempty"`
	OpaqueDirectory []string `json:"opaqueDirectories,omitempty"`
}

func PreflightRootfsCapture() error {
	if _, err := exec.LookPath("mksquashfs"); err != nil {
		return fmt.Errorf("mksquashfs is required: %w", err)
	}
	return preflightRootfsMountCapability()
}

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
		var upperDir string
		for _, option := range strings.Split(mount.VFSOptions, ",") {
			switch {
			case strings.HasPrefix(option, "upperdir="):
				upperDir = strings.TrimPrefix(option, "upperdir=")
			case option == "metacopy=on", strings.HasPrefix(option, "redirect_dir="):
				return "", fmt.Errorf("unsupported overlay root option %q", option)
			}
		}
		if upperDir != "" {
			return upperDir, nil
		}
	}
	return "", fmt.Errorf("overlay upperdir not found for pid %d", pid)
}

// CaptureRootfsDiff writes the mandatory SquashFS diff and deletion sidecar.
// It hashes the image once and verifies that the kernel can mount it before
// returning the manifest digest.
func CaptureRootfsDiff(
	ctx context.Context,
	upperDir, checkpointDir string,
	exclusions types.OverlaySettings,
	bindMountDests []string,
) (string, error) {
	if upperDir == "" {
		return "", errors.New("upperdir is empty")
	}
	deletions, overlayEntries, err := scanOverlayUpper(upperDir)
	if err != nil {
		return "", err
	}
	if err := writeDeletedFiles(checkpointDir, deletions); err != nil {
		return "", err
	}

	rootfsPath := filepath.Join(checkpointDir, RootfsDiffFilename)
	if err := runMksquashfs(
		ctx,
		upperDir,
		rootfsPath,
		buildExclusions(exclusions, bindMountDests, overlayEntries),
	); err != nil {
		return "", err
	}
	image, digest, err := openAndValidateRootfs(rootfsPath, "")
	if err != nil {
		return "", err
	}
	defer image.Close()
	mount, err := PrepareDetachedRootfsMount(image)
	if err != nil {
		return "", fmt.Errorf("kernel rejected captured SquashFS: %w", err)
	}
	if err := mount.Close(); err != nil {
		return "", fmt.Errorf("close captured SquashFS verification mount: %w", err)
	}
	return digest, nil
}

func runMksquashfs(
	ctx context.Context,
	source, output string,
	exclusions []string,
) error {
	args := []string{
		source,
		output,
		"-comp", "lz4",
		"-b", "1M",
		"-processors", "8",
		"-noappend",
		"-no-progress",
		"-exit-on-error",
		"-one-file-system-x",
		"-xattrs",
		"-xattrs-exclude", `^(trusted|user)\.overlay\.`,
		"-wildcards",
	}
	if len(exclusions) > 0 {
		args = append(args, "-e")
		args = append(args, exclusions...)
	}
	outputBytes, err := exec.CommandContext(ctx, "mksquashfs", args...).CombinedOutput()
	if err != nil {
		_ = os.Remove(output)
		return fmt.Errorf("mksquashfs failed: %w (output: %s)", err, outputBytes)
	}
	return nil
}

func buildExclusions(
	settings types.OverlaySettings,
	bindMountDests, overlayEntries []string,
) []string {
	var result []string
	for _, exclusion := range settings.Exclusions {
		exclusion = strings.TrimPrefix(filepath.Clean(exclusion), "/")
		exclusion = strings.TrimPrefix(exclusion, "./")
		if exclusion == "" || exclusion == "." {
			continue
		}
		if strings.ContainsAny(exclusion, "*?[") {
			exclusion = strings.TrimPrefix(exclusion, "*/")
			result = append(result, exclusion, "... "+exclusion)
		} else {
			result = append(result, exclusion)
		}
	}
	for _, path := range append(bindMountDests, overlayEntries...) {
		path = strings.TrimPrefix(filepath.Clean(path), "/")
		if path != "" && path != "." {
			result = append(result, escapeSquashfsWildcardLiteral(path))
		}
	}
	return result
}

func escapeSquashfsWildcardLiteral(path string) string {
	var escaped strings.Builder
	escaped.Grow(len(path) * 2)
	for i := range len(path) {
		if path[i] != '/' {
			escaped.WriteByte('\\')
		}
		escaped.WriteByte(path[i])
	}
	return escaped.String()
}

func scanOverlayUpper(upperDir string) (*deletedFiles, []string, error) {
	result := &deletedFiles{}
	var overlayEntries []string
	err := filepath.Walk(upperDir, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(upperDir, path)
		if err != nil {
			return err
		}
		for _, name := range []string{
			"trusted.overlay.metacopy",
			"user.overlay.metacopy",
			"trusted.overlay.redirect",
			"user.overlay.redirect",
		} {
			if value, present, err := readXattr(path, name); err != nil {
				return err
			} else if present {
				return fmt.Errorf("unsupported overlay xattr %s=%q on %s", name, value, relative)
			}
		}
		if info.IsDir() {
			for _, name := range []string{"trusted.overlay.opaque", "user.overlay.opaque"} {
				value, present, err := readXattr(path, name)
				if err != nil {
					return err
				}
				if present && value == "y" {
					result.OpaqueDirectory = append(
						result.OpaqueDirectory,
						filepath.ToSlash(relative),
					)
				} else if present && value != "x" {
					return fmt.Errorf("invalid overlay opaque value %q on %s", value, relative)
				}
			}
		}
		if path == upperDir {
			return nil
		}

		name := info.Name()
		switch {
		case name == ".wh..wh..opq":
			parent := filepath.Dir(relative)
			if parent != "." {
				result.OpaqueDirectory = append(result.OpaqueDirectory, filepath.ToSlash(parent))
			}
			overlayEntries = append(overlayEntries, filepath.ToSlash(relative))
		case strings.HasPrefix(name, ".wh."):
			deleted := filepath.Join(filepath.Dir(relative), strings.TrimPrefix(name, ".wh."))
			result.Whiteouts = append(result.Whiteouts, filepath.ToSlash(deleted))
			overlayEntries = append(overlayEntries, filepath.ToSlash(relative))
		default:
			whiteout, err := isOverlayWhiteout(path, info)
			if err != nil {
				return err
			}
			if whiteout {
				result.Whiteouts = append(result.Whiteouts, filepath.ToSlash(relative))
				overlayEntries = append(overlayEntries, filepath.ToSlash(relative))
			}
		}
		return nil
	})
	if err != nil {
		return nil, nil, fmt.Errorf("scan overlay upperdir: %w", err)
	}
	return result, overlayEntries, nil
}

func isOverlayWhiteout(path string, info os.FileInfo) (bool, error) {
	if stat, ok := info.Sys().(*unix.Stat_t); ok &&
		stat.Mode&unix.S_IFMT == unix.S_IFCHR &&
		unix.Major(uint64(stat.Rdev)) == 0 &&
		unix.Minor(uint64(stat.Rdev)) == 0 {
		return true, nil
	}
	for _, name := range []string{"trusted.overlay.whiteout", "user.overlay.whiteout"} {
		_, present, err := readXattr(path, name)
		if err != nil {
			return false, err
		}
		if present {
			return true, nil
		}
	}
	return false, nil
}

func readXattr(path, name string) (string, bool, error) {
	size, err := unix.Lgetxattr(path, name, nil)
	if errors.Is(err, unix.ENODATA) || errors.Is(err, unix.ENOTSUP) {
		return "", false, nil
	}
	if err != nil {
		return "", false, fmt.Errorf("read xattr %s on %s: %w", name, path, err)
	}
	value := make([]byte, size)
	if _, err := unix.Lgetxattr(path, name, value); err != nil {
		return "", false, fmt.Errorf("read xattr %s on %s: %w", name, path, err)
	}
	return string(value), true, nil
}

func writeDeletedFiles(checkpointDir string, files *deletedFiles) error {
	path := filepath.Join(checkpointDir, deletedFilesFilename)
	if len(files.Whiteouts) == 0 && len(files.OpaqueDirectory) == 0 {
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			return fmt.Errorf("remove empty deletion sidecar: %w", err)
		}
		return nil
	}
	data, err := json.Marshal(files)
	if err != nil {
		return fmt.Errorf("marshal deletion sidecar: %w", err)
	}
	if len(data) > maxDeletedFilesSidecarSize {
		return fmt.Errorf(
			"deletion sidecar exceeds %d-byte limit",
			maxDeletedFilesSidecarSize,
		)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("write deletion sidecar: %w", err)
	}
	return nil
}

// OpenValidatedRootfs opens the fixed artifact without following its final
// component, validates the manifest digest, and returns the same FD for loop
// configuration.
func OpenValidatedRootfs(checkpointDir, checkpointID string) (*os.File, error) {
	manifest, err := types.ReadManifest(checkpointDir)
	if err != nil {
		return nil, err
	}
	if checkpointID != "" && manifest.CheckpointID != checkpointID {
		return nil, fmt.Errorf(
			"checkpoint manifest ID %q does not match %q",
			manifest.CheckpointID,
			checkpointID,
		)
	}
	image, _, err := openAndValidateRootfs(
		filepath.Join(checkpointDir, RootfsDiffFilename),
		manifest.RootFSSHA256,
	)
	return image, err
}

// ValidateCheckpointArtifact validates the fixed artifact files used for
// controller resume.
func ValidateCheckpointArtifact(checkpointDir, checkpointID string) error {
	image, err := OpenValidatedRootfs(checkpointDir, checkpointID)
	if err != nil {
		return err
	}
	if err := image.Close(); err != nil {
		return fmt.Errorf("close validated rootfs image: %w", err)
	}
	sidecar, err := OpenDeletedFiles(checkpointDir)
	if err != nil {
		return err
	}
	if sidecar == nil {
		return nil
	}
	defer sidecar.Close()
	_, err = readDeletedFilesFile(sidecar)
	return err
}

func openAndValidateRootfs(path, expectedDigest string) (*os.File, string, error) {
	if expectedDigest != "" {
		if len(expectedDigest) != sha256.Size*2 {
			return nil, "", errors.New("checkpoint manifest has invalid rootfsSha256")
		}
		if _, err := hex.DecodeString(expectedDigest); err != nil {
			return nil, "", fmt.Errorf("checkpoint manifest has invalid rootfsSha256: %w", err)
		}
	}
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0)
	if err != nil {
		return nil, "", fmt.Errorf("open %s: %w", RootfsDiffFilename, err)
	}
	image := os.NewFile(uintptr(fd), RootfsDiffFilename)
	closeWithError := func(err error) (*os.File, string, error) {
		image.Close()
		return nil, "", err
	}
	var stat unix.Stat_t
	if err := unix.Fstat(fd, &stat); err != nil {
		return closeWithError(fmt.Errorf("stat %s: %w", RootfsDiffFilename, err))
	}
	if stat.Mode&unix.S_IFMT != unix.S_IFREG {
		return closeWithError(fmt.Errorf("%s is not a regular file", RootfsDiffFilename))
	}
	if _, err := image.Seek(0, io.SeekStart); err != nil {
		return closeWithError(fmt.Errorf("seek rootfs image: %w", err))
	}
	sum := sha256.New()
	if _, err := io.Copy(sum, image); err != nil {
		return closeWithError(fmt.Errorf("hash rootfs image: %w", err))
	}
	digest := hex.EncodeToString(sum.Sum(nil))
	if expectedDigest != "" && digest != expectedDigest {
		return closeWithError(fmt.Errorf("rootfs SHA-256 mismatch: got %s", digest))
	}
	if _, err := image.Seek(0, io.SeekStart); err != nil {
		return closeWithError(fmt.Errorf("rewind rootfs image: %w", err))
	}
	return image, digest, nil
}

func OpenDeletedFiles(checkpointDir string) (*os.File, error) {
	fd, err := unix.Open(
		filepath.Join(checkpointDir, deletedFilesFilename),
		unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW|unix.O_NONBLOCK,
		0,
	)
	if errors.Is(err, unix.ENOENT) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("open deletion sidecar: %w", err)
	}
	file := os.NewFile(uintptr(fd), deletedFilesFilename)
	if err := validateDeletedFilesFile(file); err != nil {
		file.Close()
		return nil, err
	}
	return file, nil
}

func readDeletedFilesFile(file *os.File) (*deletedFiles, error) {
	if file == nil {
		return &deletedFiles{}, nil
	}
	if err := validateDeletedFilesFile(file); err != nil {
		return nil, err
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		return nil, fmt.Errorf("seek deletion sidecar: %w", err)
	}
	data, err := io.ReadAll(io.LimitReader(file, maxDeletedFilesSidecarSize+1))
	if err != nil {
		return nil, fmt.Errorf("read deletion sidecar: %w", err)
	}
	if len(data) > maxDeletedFilesSidecarSize {
		return nil, fmt.Errorf(
			"deletion sidecar exceeds %d-byte limit",
			maxDeletedFilesSidecarSize,
		)
	}
	var result deletedFiles
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("decode deletion sidecar: %w", err)
	}
	for _, entry := range result.Whiteouts {
		if err := validateRelativeArtifactPath(entry); err != nil {
			return nil, fmt.Errorf("invalid deletion entry %q: %w", entry, err)
		}
	}
	for _, entry := range result.OpaqueDirectory {
		if entry != "." {
			if err := validateRelativeArtifactPath(entry); err != nil {
				return nil, fmt.Errorf("invalid opaque directory entry %q: %w", entry, err)
			}
		}
	}
	return &result, nil
}

func validateDeletedFilesFile(file *os.File) error {
	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("stat deletion sidecar: %w", err)
	}
	if !info.Mode().IsRegular() {
		return errors.New("deletion sidecar is not a regular file")
	}
	if info.Size() > maxDeletedFilesSidecarSize {
		return fmt.Errorf(
			"deletion sidecar exceeds %d-byte limit",
			maxDeletedFilesSidecarSize,
		)
	}
	return nil
}

func validateRelativeArtifactPath(path string) error {
	if path == "" || filepath.IsAbs(path) || filepath.Clean(path) != path ||
		path == "." || path == ".." ||
		strings.HasPrefix(path, ".."+string(os.PathSeparator)) {
		return errors.New("path must be clean and relative")
	}
	return nil
}
