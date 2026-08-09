package runtime

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/moby/sys/mountinfo"
	specs "github.com/opencontainers/runtime-spec/specs-go"

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

// ClassifyMounts sets IsOCIManaged on each mount by matching against the
// container's OCI spec (mounts, masked paths, readonly paths).
// Handles /run/ ↔ /var/run/ aliasing since some images symlink one to the other.
func ClassifyMounts(mounts []types.MountInfo, ociSpec *specs.Spec, rootFS string) []types.MountInfo {
	ociSet := collectOCIManagedPaths(ociSpec, rootFS)

	for i := range mounts {
		mp := mounts[i].MountPoint
		if _, ok := ociSet[mp]; ok {
			mounts[i].IsOCIManaged = true
			continue
		}
		// /run/ ↔ /var/run/ aliasing
		if strings.HasPrefix(mp, "/run/") {
			if _, ok := ociSet["/var"+mp]; ok {
				mounts[i].IsOCIManaged = true
				continue
			}
		}
		if strings.HasPrefix(mp, "/var/run/") {
			if _, ok := ociSet[strings.TrimPrefix(mp, "/var")]; ok {
				mounts[i].IsOCIManaged = true
			}
		}
	}

	return mounts
}

// BuildMountPolicy classifies mounts and masked paths for CRIU dump.
// Mounts must already have IsOCIManaged set by ClassifyMounts.
//
// Policy (evaluated top to bottom):
//  1. Skip: non-OCI /proc/*, /sys/*, /run/* submounts (virtual/runtime, not in placeholder)
//  2. Native: /dev/shm tmpfs (CRIU saves and restores content)
//  3. Masked: OCI masked non-directory paths that exist under rootFS → /dev/null
//  4. Externalize: everything else (OCI mounts the runtime recreates in placeholder)
func BuildMountPolicy(mounts []types.MountInfo, rootFS string, maskedPaths []string) (map[string]string, []string) {
	extMap := make(map[string]string, len(mounts))
	var skipped []string

	for _, m := range mounts {
		if m.MountPoint == "" {
			continue
		}

		// Skip non-OCI virtual/runtime mounts — these won't exist in the placeholder
		if !m.IsOCIManaged && (strings.HasPrefix(m.MountPoint, "/proc/") || strings.HasPrefix(m.MountPoint, "/sys/") || strings.HasPrefix(m.MountPoint, "/run/")) {
			skipped = append(skipped, m.MountPoint)
			continue
		}

		// Let CRIU handle /dev/shm content natively — don't externalize it.
		if m.MountPoint == "/dev/shm" && m.FSType == "tmpfs" {
			continue
		}

		extMap[m.MountPoint] = m.MountPoint
	}

	// Masked paths map to /dev/null. Only non-directory paths that exist under rootFS.
	for _, p := range maskedPaths {
		hostPath := filepath.Join(rootFS, p)
		info, err := os.Lstat(hostPath)
		if err != nil || info.IsDir() {
			continue
		}
		extMap[p] = "/dev/null"
	}

	return extMap, skipped
}

// RemountProcSys remounts /proc/sys read-write or read-only.
func RemountProcSys(rw bool) error {
	const target = "/proc/sys"
	mounted, err := mountinfo.Mounted(target)
	if err != nil {
		return fmt.Errorf("failed to inspect %s mount: %w", target, err)
	}
	var statfs syscall.Statfs_t
	if err := syscall.Statfs(target, &statfs); err != nil {
		return fmt.Errorf("failed to statfs %s: %w", target, err)
	}

	return remountProcSys(rw, mounted, uintptr(statfs.Flags), syscall.Mount)
}

func remountProcSys(rw bool, mounted bool, currentFlags uintptr, mount func(string, string, string, uintptr, string) error) error {
	const target = "/proc/sys"
	// A bind remount requires a mount root, which privileged OCI containers
	// do not create for readonlyPaths.
	if rw && !mounted {
		if err := mount(target, target, "", syscall.MS_BIND, ""); err != nil {
			return fmt.Errorf("failed to bind mount %s for rw remount: %w", target, err)
		}
	}

	// A bind remount replaces per-mount flags, so retain the security flags.
	flags := currentFlags & (syscall.MS_NOSUID | syscall.MS_NODEV | syscall.MS_NOEXEC)
	flags |= syscall.MS_BIND | syscall.MS_REMOUNT
	mode := "rw"
	if !rw {
		flags |= syscall.MS_RDONLY
		mode = "ro"
	}
	if err := mount("", target, "", flags, ""); err != nil {
		return fmt.Errorf("failed to remount %s %s: %w", target, mode, err)
	}
	return nil
}
