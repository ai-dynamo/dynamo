package types

import (
	specs "github.com/opencontainers/runtime-spec/specs-go"
)

// MountInfo holds parsed mount information from /proc/pid/mountinfo.
type MountInfo struct {
	MountPoint string
	FSType     string
	VFSOptions string // superblock options (e.g. "upperdir=...")

	// IsOCIManaged is true when the mount destination matches an OCI spec entry
	// (including /run/ ↔ /var/run/ aliasing). Set by ClassifyMounts.
	IsOCIManaged bool
}

// CheckpointContainerSnapshot holds runtime container state collected during checkpoint inspection.
type CheckpointContainerSnapshot struct {
	PID            int
	RootFS         string
	UpperDir       string
	OCISpec        *specs.Spec
	Mounts         []MountInfo
	NetNSInode     uint64
	StdioFDs       []string // readlink targets for FDs 0, 1, 2 (e.g. "pipe:[12345]")
	HostCgroupPath string   // host filesystem path for CRIU's --freeze-cgroup
	CUDAPIDs       []int    // PIDs with CUDA state in the container
	GPUUUIDs       []string // source GPU UUIDs from kubelet PodResources API
}

// RestoreContainerSnapshot holds inspected state for the restore target.
type RestoreContainerSnapshot struct {
	CheckpointPath string
	PlaceholderPID int
	TargetRoot     string
	CgroupRoot     string
	CUDADeviceMap  string
}
