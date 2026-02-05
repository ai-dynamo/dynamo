// criu provides CRIU-specific configuration and utilities for checkpoint operations.
package checkpoint

import (
	"fmt"
	"os"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"google.golang.org/protobuf/proto"

	checkpointk8s "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint/k8s"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// CRIUConfig holds configuration for CRIU dump operations.
// Most options are always-on with safe defaults for K8s environments.
type CRIUConfig struct {
	PID        int
	ImageDirFD int32
	RootFS     string
	GhostLimit uint32 // From env CRIU_GHOST_LIMIT: max ghost file size (0 = CRIU default)
	Timeout    uint32 // From env CRIU_TIMEOUT: checkpoint timeout in seconds (0 = no timeout)
}

// OpenImageDir opens a checkpoint directory and prepares it for CRIU.
// Returns the opened file and its FD. The caller must close the file when done.
// The file descriptor has CLOEXEC cleared so it can be inherited by CRIU.
func OpenImageDir(checkpointDir string) (*os.File, int32, error) {
	return common.OpenDirForCRIU(checkpointDir)
}

// BuildCRIUOpts creates CRIU options from a config struct.
// This sets up the base options; external mounts and namespaces are added separately.
//
// Always-on options for K8s:
//   - LeaveRunning: always keep process running after checkpoint
//   - ShellJob: containers are often session leaders
//   - TcpClose: pod IPs change on restore/migration
//   - FileLocks: applications use file locks
//   - OrphanPtsMaster: containers with TTYs
//   - ExtUnixSk: containers have external Unix sockets
//   - ManageCgroups (IGNORE): let K8s manage cgroups
//   - LinkRemap: handle deleted-but-open files (safe for all workloads)
//   - ExtMasters: external bind mount masters (safe for all workloads)
func BuildCRIUOpts(cfg CRIUConfig) *criurpc.CriuOpts {
	cgMode := criurpc.CriuCgMode_IGNORE
	criuOpts := &criurpc.CriuOpts{
		Pid:               proto.Int32(int32(cfg.PID)),
		ImagesDirFd:       proto.Int32(cfg.ImageDirFD),
		LogLevel:          proto.Int32(4),
		LogFile:           proto.String("dump.log"),
		Root:              proto.String(cfg.RootFS),
		ManageCgroups:     proto.Bool(true),
		ManageCgroupsMode: &cgMode,
		// Always-on for K8s environments
		LeaveRunning:    proto.Bool(true),
		ShellJob:        proto.Bool(true),
		TcpClose:        proto.Bool(true),
		FileLocks:       proto.Bool(true),
		OrphanPtsMaster: proto.Bool(true),
		ExtUnixSk:       proto.Bool(true),
		LinkRemap:       proto.Bool(true),
		ExtMasters:      proto.Bool(true),
	}

	// Optional: ghost limit from env (0 = use CRIU default)
	if cfg.GhostLimit > 0 {
		criuOpts.GhostLimit = proto.Uint32(cfg.GhostLimit)
	}

	// Optional: timeout from env (0 = no timeout)
	if cfg.Timeout > 0 {
		criuOpts.Timeout = proto.Uint32(cfg.Timeout)
	}

	return criuOpts
}

// AddExternalMounts adds mount points as external mounts to CRIU options.
// CRIU requires all mounts to be marked as external for successful restore.
func AddExternalMounts(criuOpts *criurpc.CriuOpts, mounts []AllMountInfo) {
	addedMounts := make(map[string]bool)

	for _, m := range mounts {
		if addedMounts[m.MountPoint] {
			continue
		}
		criuOpts.ExtMnt = append(criuOpts.ExtMnt, &criurpc.ExtMountMap{
			Key: proto.String(m.MountPoint),
			Val: proto.String(m.MountPoint),
		})
		addedMounts[m.MountPoint] = true
	}
}

// AddExternalPaths adds additional paths (masked/readonly) as external mounts.
// These may not appear in mountinfo but CRIU still needs them marked as external.
func AddExternalPaths(criuOpts *criurpc.CriuOpts, paths []string) {
	// Build set of existing mount points
	existing := make(map[string]bool)
	for _, m := range criuOpts.ExtMnt {
		existing[m.GetKey()] = true
	}

	for _, path := range paths {
		if existing[path] {
			continue
		}
		criuOpts.ExtMnt = append(criuOpts.ExtMnt, &criurpc.ExtMountMap{
			Key: proto.String(path),
			Val: proto.String(path),
		})
		existing[path] = true
	}
}

// AddExternalNamespace adds a namespace as external to CRIU options.
// Format: "<type>[<inode>]:<key>"
func AddExternalNamespace(criuOpts *criurpc.CriuOpts, nsType NamespaceType, inode uint64, key string) {
	extNs := fmt.Sprintf("%s[%d]:%s", nsType, inode, key)
	criuOpts.External = append(criuOpts.External, extNs)
}

// AddExternalStrings adds raw external strings to CRIU options.
// Used for additional external mount mappings (e.g., NVIDIA firmware files).
func AddExternalStrings(criuOpts *criurpc.CriuOpts, externals []string) {
	criuOpts.External = append(criuOpts.External, externals...)
}

// ConfigureExternalMounts adds all required external mounts to CRIU options.
// This includes mounts from /proc/pid/mountinfo plus masked/readonly paths from OCI spec.
func ConfigureExternalMounts(criuOpts *criurpc.CriuOpts, pid int, hostProc string, containerInfo *checkpointk8s.ContainerInfo) error {
	// Get all mounts from mountinfo - CRIU needs every mount marked as external
	allMounts, err := GetAllMountsFromMountinfo(pid, hostProc)
	if err != nil {
		return fmt.Errorf("failed to get all mounts from mountinfo: %w", err)
	}

	// Add mounts from mountinfo
	AddExternalMounts(criuOpts, allMounts)

	// Add masked and readonly paths from OCI spec
	AddExternalPaths(criuOpts, containerInfo.GetMaskedPaths())
	AddExternalPaths(criuOpts, containerInfo.GetReadonlyPaths())

	return nil
}

// ConfigureExternalNamespaces adds external namespaces to CRIU options.
// Returns the network namespace inode if found, for logging purposes.
func ConfigureExternalNamespaces(criuOpts *criurpc.CriuOpts, namespaces map[NamespaceType]*NamespaceInfo, externalMounts []string) uint64 {
	var netNsInode uint64

	// Mark network namespace as external for socket binding preservation
	if netNs, ok := namespaces[NamespaceNet]; ok {
		AddExternalNamespace(criuOpts, NamespaceNet, netNs.Inode, "extNetNs")
		netNsInode = netNs.Inode
	}

	// Add additional external mounts (e.g., for NVIDIA firmware files)
	AddExternalStrings(criuOpts, externalMounts)

	return netNsInode
}

// BuildCRIUOptsFromCheckpointOpts constructs CRIU options from checkpoint Options.
// Returns the configured CriuOpts ready for external mount/namespace configuration.
func BuildCRIUOptsFromCheckpointOpts(opts Options, pid int, imageDirFD int32, rootFS string) *criurpc.CriuOpts {
	cfg := CRIUConfig{
		PID:        pid,
		ImageDirFD: imageDirFD,
		RootFS:     rootFS,
		GhostLimit: opts.GhostLimit,
		Timeout:    opts.Timeout,
	}

	return BuildCRIUOpts(cfg)
}
