package runtime

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/google/uuid"
	"golang.org/x/sys/unix"
)

const (
	hostDevicePath  = "/host/dev"
	mountStagingDir = "/var/run/snapshot/mounts"
)

var (
	openLoopControl = unix.Open
	loopIoctlRetInt = unix.IoctlRetInt
	openLoopDevice  = unix.Open
	createLoopNode  = unix.Mknod
	statLoopDevice  = unix.Fstat
	configureLoop   = unix.IoctlLoopConfigure
	clearLoop       = unix.IoctlSetInt
	mountSquashfs   = unix.Mount
	unmountPath     = unix.Unmount
	openMountTree   = unix.OpenTree
)

const loopAllocationAttempts = 8

func preflightRootfsMountCapability() error {
	fd, err := unix.Open(
		filepath.Join(hostDevicePath, "loop-control"),
		unix.O_RDWR|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	if err != nil {
		return fmt.Errorf("open loop-control for rootfs preflight: %w", err)
	}
	if err := unix.Close(fd); err != nil {
		return fmt.Errorf("close loop-control after rootfs preflight: %w", err)
	}
	return nil
}

// PrepareDetachedRootfsMount loop-mounts and detaches an already-validated image.
func PrepareDetachedRootfsMount(image *os.File) (*os.File, error) {
	if image == nil {
		return nil, errors.New("validated rootfs image is required")
	}
	loopFD, loopPath, err := configureAvailableLoop(image)
	if err != nil {
		return nil, err
	}
	defer unix.Close(loopFD)
	loopConfigured := true
	defer func() {
		if loopConfigured {
			_ = clearLoop(loopFD, unix.LOOP_CLR_FD, 0)
		}
	}()

	if err := os.MkdirAll(mountStagingDir, 0o700); err != nil {
		return nil, fmt.Errorf("create SquashFS mount staging root: %w", err)
	}
	mountpoint := filepath.Join(mountStagingDir, uuid.NewString())
	if err := os.Mkdir(mountpoint, 0o700); err != nil {
		return nil, fmt.Errorf("create SquashFS mountpoint: %w", err)
	}
	defer os.Remove(mountpoint)

	if err := mountSquashfs(loopPath, mountpoint, "squashfs", unix.MS_RDONLY, ""); err != nil {
		return nil, fmt.Errorf("mount SquashFS image read-only: %w", err)
	}
	mounted := true
	defer func() {
		if mounted {
			_ = unmountPath(mountpoint, unix.MNT_DETACH)
		}
	}()

	mountFD, err := openMountTree(
		unix.AT_FDCWD,
		mountpoint,
		unix.OPEN_TREE_CLONE|unix.OPEN_TREE_CLOEXEC,
	)
	if err != nil {
		return nil, fmt.Errorf("clone SquashFS mount with open_tree: %w", err)
	}
	detached := os.NewFile(uintptr(mountFD), "rootfs-squashfs-mount")

	if err := unmountPath(mountpoint, unix.MNT_DETACH); err != nil {
		detached.Close()
		return nil, fmt.Errorf("detach SquashFS staging mount: %w", err)
	}
	mounted = false
	loopConfigured = false // Autoclear now owns cleanup when the detached mount closes.

	return detached, nil
}

func configureAvailableLoop(image *os.File) (int, string, error) {
	config := &unix.LoopConfig{
		Fd: uint32(image.Fd()),
		Info: unix.LoopInfo64{
			Flags: unix.LO_FLAGS_READ_ONLY | unix.LO_FLAGS_AUTOCLEAR,
		},
	}
	var lastErr error
	for range loopAllocationAttempts {
		loopControl, err := openLoopControl(
			filepath.Join(hostDevicePath, "loop-control"),
			unix.O_RDWR|unix.O_CLOEXEC,
			0,
		)
		if err != nil {
			return -1, "", fmt.Errorf("open loop-control: %w", err)
		}
		loopNumber, err := loopIoctlRetInt(loopControl, unix.LOOP_CTL_GET_FREE)
		unix.Close(loopControl)
		if err != nil {
			return -1, "", fmt.Errorf("allocate loop device: %w", err)
		}
		if loopNumber < 0 || loopNumber > 1<<20 {
			return -1, "", fmt.Errorf("loop-control returned invalid device number %d", loopNumber)
		}
		loopPath := filepath.Join(hostDevicePath, fmt.Sprintf("loop%d", loopNumber))
		loopFD, err := openLoopDevice(
			loopPath,
			unix.O_RDWR|unix.O_CLOEXEC|unix.O_NOFOLLOW,
			0,
		)
		if errors.Is(err, unix.ENOENT) {
			err = createLoopNode(
				loopPath,
				unix.S_IFBLK|0o600,
				int(unix.Mkdev(7, uint32(loopNumber))),
			)
			if err == nil || errors.Is(err, unix.EEXIST) {
				loopFD, err = openLoopDevice(
					loopPath,
					unix.O_RDWR|unix.O_CLOEXEC|unix.O_NOFOLLOW,
					0,
				)
			}
		}
		if err != nil {
			if errors.Is(err, unix.EBUSY) || errors.Is(err, unix.ENOENT) {
				lastErr = err
				continue
			}
			return -1, "", fmt.Errorf("open allocated loop device %s: %w", loopPath, err)
		}
		var stat unix.Stat_t
		if err := statLoopDevice(loopFD, &stat); err != nil {
			unix.Close(loopFD)
			return -1, "", fmt.Errorf("stat allocated loop device %s: %w", loopPath, err)
		}
		if stat.Mode&unix.S_IFMT != unix.S_IFBLK ||
			unix.Major(uint64(stat.Rdev)) != 7 ||
			unix.Minor(uint64(stat.Rdev)) != uint32(loopNumber) {
			unix.Close(loopFD)
			return -1, "", fmt.Errorf("%s is not loop block device %d", loopPath, loopNumber)
		}
		if err := configureLoop(loopFD, config); err != nil {
			unix.Close(loopFD)
			if errors.Is(err, unix.EBUSY) {
				lastErr = err
				continue
			}
			return -1, "", fmt.Errorf("configure loop device %s: %w", loopPath, err)
		}
		return loopFD, loopPath, nil
	}
	return -1, "", fmt.Errorf(
		"configure a free loop device after %d attempts: %w",
		loopAllocationAttempts,
		lastErr,
	)
}
