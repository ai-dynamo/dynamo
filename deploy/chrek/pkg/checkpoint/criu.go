// criu provides CRIU-specific configuration and utilities for checkpoint operations.
package checkpoint

import (
	"fmt"
	"os"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"google.golang.org/protobuf/proto"

	checkpointk8s "github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint/k8s"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// CRIUDumpParams holds per-checkpoint dynamic parameters for CRIU dump.
// These are values that change per checkpoint operation, not configuration.
type CRIUDumpParams struct {
	PID        int
	ImageDirFD int32
	RootFS     string
}

// OpenImageDir opens a checkpoint directory and prepares it for CRIU.
// Returns the opened file and its FD. The caller must close the file when done.
// The file descriptor has CLOEXEC cleared so it can be inherited by CRIU.
func OpenImageDir(checkpointDir string) (*os.File, int32, error) {
	return common.OpenPathForCRIU(checkpointDir)
}

// BuildCRIUOpts creates CRIU options from config and per-checkpoint parameters.
// This sets up the base options; external mounts and namespaces are added separately.
// All options come from config (values.yaml) - nothing is hardcoded here.
func BuildCRIUOpts(cfg *config.CRIUConfig, params CRIUDumpParams) *criurpc.CriuOpts {
	criuOpts := &criurpc.CriuOpts{
		Pid:         proto.Int32(int32(params.PID)),
		ImagesDirFd: proto.Int32(params.ImageDirFD),
		Root:        proto.String(params.RootFS),
		LogFile:     proto.String(config.DumpLogFilename),
	}

	if cfg == nil {
		return criuOpts
	}

	// RPC options from config
	criuOpts.LogLevel = proto.Int32(cfg.LogLevel)
	criuOpts.LeaveRunning = proto.Bool(cfg.LeaveRunning)
	criuOpts.ShellJob = proto.Bool(cfg.ShellJob)
	criuOpts.TcpClose = proto.Bool(cfg.TcpClose)
	criuOpts.FileLocks = proto.Bool(cfg.FileLocks)
	criuOpts.OrphanPtsMaster = proto.Bool(cfg.OrphanPtsMaster)
	criuOpts.ExtUnixSk = proto.Bool(cfg.ExtUnixSk)
	criuOpts.LinkRemap = proto.Bool(cfg.LinkRemap)
	criuOpts.ExtMasters = proto.Bool(cfg.ExtMasters)
	criuOpts.AutoDedup = proto.Bool(cfg.AutoDedup)
	criuOpts.LazyPages = proto.Bool(cfg.LazyPages)

	// Cgroup management mode
	criuOpts.ManageCgroups = proto.Bool(true)
	var cgMode criurpc.CriuCgMode
	switch cfg.ManageCgroupsMode {
	case "soft":
		cgMode = criurpc.CriuCgMode_SOFT
	case "full":
		cgMode = criurpc.CriuCgMode_FULL
	case "strict":
		cgMode = criurpc.CriuCgMode_STRICT
	case "ignore", "":
		cgMode = criurpc.CriuCgMode_IGNORE
	default:
		cgMode = criurpc.CriuCgMode_IGNORE
	}
	criuOpts.ManageCgroupsMode = &cgMode

	// Optional numeric options
	if cfg.GhostLimit > 0 {
		criuOpts.GhostLimit = proto.Uint32(cfg.GhostLimit)
	}
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
func ConfigureExternalMounts(criuOpts *criurpc.CriuOpts, pid int, containerInfo *checkpointk8s.ContainerInfo) error {
	// Get all mounts from mountinfo - CRIU needs every mount marked as external
	allMounts, err := GetAllMountsFromMountinfo(pid)
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

// ConfigureSkipMounts enumerates mounts under the given prefixes and adds them to CRIU's
// skip mount list. This allows cross-node restore by skipping mounts that may not exist
// on the target node (e.g., NVIDIA runtime mounts like /run/nvidia/driver/...).
// Returns the list of mounts that will be skipped, for logging purposes.
func ConfigureSkipMounts(criuOpts *criurpc.CriuOpts, pid int, prefixes []string) ([]string, error) {
	if len(prefixes) == 0 {
		return nil, nil
	}

	skipMounts, err := GetMountsUnderPrefixes(pid, prefixes)
	if err != nil {
		return nil, fmt.Errorf("failed to enumerate skip mounts from prefixes: %w", err)
	}

	if len(skipMounts) > 0 {
		criuOpts.SkipMnt = skipMounts
	}

	return skipMounts, nil
}

