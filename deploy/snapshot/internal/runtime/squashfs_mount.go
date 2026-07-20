package runtime

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

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

	loopAllocationMu sync.Mutex
)

const (
	loopAllocationAttempts   = 16
	loopAllocationRetryDelay = 10 * time.Millisecond
)

func preflightRootfsMountCapability() (retErr error) {
	image, err := os.CreateTemp("", "dynamo-snapshot-loop-preflight-*")
	if err != nil {
		return fmt.Errorf("create scratch file for rootfs loop preflight: %w", err)
	}
	imagePath := image.Name()
	loopFD := -1
	defer func() {
		if loopFD >= 0 {
			retErr = errors.Join(
				retErr,
				clearLoop(loopFD, unix.LOOP_CLR_FD, 0),
				unix.Close(loopFD),
			)
		}
		retErr = errors.Join(retErr, image.Close(), os.Remove(imagePath))
	}()

	if err := image.Truncate(4096); err != nil {
		return fmt.Errorf("size scratch file for rootfs loop preflight: %w", err)
	}
	loopFD, _, err = configureAvailableLoop(image)
	if err != nil {
		return fmt.Errorf(
			"LOOP_CONFIGURE preflight failed; enable host loop support and permit loop ioctls: %w",
			err,
		)
	}
	return nil
}

// PrepareDetachedRootfsMount loop-mounts and detaches an already-validated image.
func PrepareDetachedRootfsMount(image *os.File) (detached *os.File, retErr error) {
	return prepareDetachedRootfsMount(image, mountStagingDir)
}

func prepareDetachedRootfsMount(
	image *os.File,
	stagingDir string,
) (detached *os.File, retErr error) {
	if image == nil {
		return nil, errors.New("validated rootfs image is required")
	}
	loopFD, loopPath, err := configureAvailableLoop(image)
	if err != nil {
		return nil, err
	}
	loopConfigured := true
	mounted := false
	mountpoint := ""
	mountpointCreated := false
	defer func() {
		if mounted {
			if err := unmountPath(mountpoint, unix.MNT_DETACH); err != nil {
				retErr = errors.Join(retErr, fmt.Errorf("unmount SquashFS staging mount: %w", err))
			}
		}
		if mountpointCreated {
			if err := os.Remove(mountpoint); err != nil {
				retErr = errors.Join(retErr, fmt.Errorf("remove SquashFS mountpoint: %w", err))
			}
		}
		if loopConfigured {
			if err := clearLoop(loopFD, unix.LOOP_CLR_FD, 0); err != nil {
				retErr = errors.Join(retErr, fmt.Errorf("clear loop device: %w", err))
			}
		}
		if err := unix.Close(loopFD); err != nil {
			retErr = errors.Join(retErr, fmt.Errorf("close loop device: %w", err))
		}
		if retErr != nil && detached != nil {
			if err := detached.Close(); err != nil {
				retErr = errors.Join(retErr, fmt.Errorf("close detached SquashFS mount: %w", err))
			}
			detached = nil
		}
	}()

	if err := os.MkdirAll(stagingDir, 0o700); err != nil {
		return nil, fmt.Errorf("create SquashFS mount staging root: %w", err)
	}
	mountpoint = filepath.Join(stagingDir, uuid.NewString())
	if err := os.Mkdir(mountpoint, 0o700); err != nil {
		return nil, fmt.Errorf("create SquashFS mountpoint: %w", err)
	}
	mountpointCreated = true

	if err := mountSquashfs(loopPath, mountpoint, "squashfs", unix.MS_RDONLY, ""); err != nil {
		return nil, fmt.Errorf("mount SquashFS image read-only: %w", err)
	}
	mounted = true

	mountFD, err := openMountTree(
		unix.AT_FDCWD,
		mountpoint,
		unix.OPEN_TREE_CLONE|unix.OPEN_TREE_CLOEXEC,
	)
	if err != nil {
		return nil, fmt.Errorf("clone SquashFS mount with open_tree: %w", err)
	}
	detached = os.NewFile(uintptr(mountFD), "rootfs-squashfs-mount")

	if err := unmountPath(mountpoint, unix.MNT_DETACH); err != nil {
		retErr = fmt.Errorf("detach SquashFS staging mount: %w", err)
		return
	}
	mounted = false
	loopConfigured = false // Autoclear now owns cleanup when the detached mount closes.

	return detached, nil
}

func configureAvailableLoop(image *os.File) (int, string, error) {
	loopAllocationMu.Lock()
	defer loopAllocationMu.Unlock()

	config := &unix.LoopConfig{
		Fd: uint32(image.Fd()),
		Info: unix.LoopInfo64{
			Flags: unix.LO_FLAGS_READ_ONLY | unix.LO_FLAGS_AUTOCLEAR,
		},
	}
	var lastErr error
	for attempt := range loopAllocationAttempts {
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
			if retryLoopAllocation(attempt, err) {
				lastErr = err
				continue
			}
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
			if retryLoopAllocation(attempt, err) {
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
			if retryLoopAllocation(attempt, err) {
				lastErr = err
				continue
			}
			return -1, "", fmt.Errorf("configure loop device %s with LOOP_CONFIGURE: %w", loopPath, err)
		}
		return loopFD, loopPath, nil
	}
	return -1, "", fmt.Errorf(
		"configure a free loop device after %d attempts: %w",
		loopAllocationAttempts,
		lastErr,
	)
}

func retryLoopAllocation(attempt int, err error) bool {
	if attempt+1 >= loopAllocationAttempts ||
		(!errors.Is(err, unix.EBUSY) &&
			!errors.Is(err, unix.ENOENT) &&
			!errors.Is(err, unix.ENXIO)) {
		return false
	}
	time.Sleep(loopAllocationRetryDelay)
	return true
}
