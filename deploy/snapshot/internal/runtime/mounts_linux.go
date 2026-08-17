//go:build linux

package runtime

import (
	"fmt"
	"os"
	"syscall"

	"github.com/moby/sys/mountinfo"

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// ReadMountInfo reads and parses mountinfo for a container process via /host/proc.
func ReadMountInfo(pid int) ([]types.MountInfo, error) {
	mountinfoPath := fmt.Sprintf("%s/%d/mountinfo", HostProcPath, pid)
	f, err := os.Open(mountinfoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open mountinfo: %w", err)
	}
	defer f.Close()

	infos, err := mountinfo.GetMountsFromReader(f, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to parse mountinfo: %w", err)
	}

	mounts := make([]types.MountInfo, 0, len(infos))
	for _, info := range infos {
		mounts = append(mounts, types.MountInfo{
			MountPoint: info.Mountpoint,
			FSType:     info.FSType,
			VFSOptions: info.VFSOptions,
		})
	}
	return mounts, nil
}

// RemountProcSys remounts /proc/sys read-write or read-only.
func RemountProcSys(rw bool) error {
	flags := uintptr(syscall.MS_BIND | syscall.MS_REMOUNT)
	if !rw {
		flags |= syscall.MS_RDONLY
	}
	if err := syscall.Mount("proc", "/proc/sys", "", flags, ""); err != nil {
		mode := "rw"
		if !rw {
			mode = "ro"
		}
		return fmt.Errorf("failed to remount /proc/sys %s: %w", mode, err)
	}
	return nil
}
