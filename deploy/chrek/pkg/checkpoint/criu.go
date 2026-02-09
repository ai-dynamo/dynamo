// criu provides CRIU-specific configuration and utilities for checkpoint operations.
package checkpoint

import (
	"fmt"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"
)

// CRIUConfig holds CRIU-specific configuration options.
// Options are categorized by how they are passed to CRIU:
//   - RPC options: Passed via go-criu CriuOpts protobuf
//   - Config file options: Written to criu.conf (NOT available via RPC)
type CRIUConfig struct {
	// === RPC Options (passed via go-criu CriuOpts) ===

	// GhostLimit is the maximum ghost file size in bytes.
	// Ghost files are deleted-but-open files that CRIU needs to checkpoint.
	// 512MB is recommended for GPU workloads with large memory allocations.
	GhostLimit uint32 `yaml:"ghostLimit"`

	// Timeout is the CRIU operation timeout in seconds.
	// 6 hours (21600s) is recommended for large GPU model checkpoints.
	Timeout uint32 `yaml:"timeout"`

	// LogLevel is the CRIU logging verbosity (0-4).
	LogLevel int32 `yaml:"logLevel"`

	// WorkDir is the CRIU work directory for temporary files.
	WorkDir string `yaml:"workDir"`

	// AutoDedup enables auto-deduplication of memory pages.
	AutoDedup bool `yaml:"autoDedup"`

	// LazyPages enables lazy page migration (experimental).
	LazyPages bool `yaml:"lazyPages"`

	// LeaveRunning keeps the process running after checkpoint (dump only).
	LeaveRunning bool `yaml:"leaveRunning"`

	// ShellJob allows checkpointing session leaders (containers are often session leaders).
	ShellJob bool `yaml:"shellJob"`

	// TcpClose closes TCP connections instead of preserving them (pod IPs change on restore).
	TcpClose bool `yaml:"tcpClose"`

	// FileLocks allows checkpointing processes with file locks.
	FileLocks bool `yaml:"fileLocks"`

	// OrphanPtsMaster allows checkpointing containers with TTYs.
	OrphanPtsMaster bool `yaml:"orphanPtsMaster"`

	// ExtUnixSk allows external Unix sockets.
	ExtUnixSk bool `yaml:"extUnixSk"`

	// LinkRemap handles deleted-but-open files.
	LinkRemap bool `yaml:"linkRemap"`

	// ExtMasters allows external bind mount masters.
	ExtMasters bool `yaml:"extMasters"`

	// ManageCgroupsMode controls cgroup handling: "ignore" lets K8s manage cgroups.
	ManageCgroupsMode string `yaml:"manageCgroupsMode"`

	// SkipMounts is a list of mount paths to skip during checkpoint.
	// These are passed to CRIU's --skip-mnt option. This allows cross-node restore
	// when certain mounts (e.g., nvidia runtime mounts) don't exist on the target node.
	// Typically populated dynamically from SkipMountPrefixes in Config.
	SkipMounts []string `yaml:"skipMounts,omitempty"`

	// ExternalMounts are additional external mount mappings (e.g., "mnt[path]:path").
	// Populated dynamically at checkpoint time from container introspection.
	// Serialized to metadata.yaml so restore can use the exact same mappings.
	ExternalMounts []string `yaml:"externalMounts,omitempty"`

	// === Config File Options (NOT available via RPC - written to criu.conf) ===

	// LibDir is the path to CRIU plugin directory (e.g., /usr/local/lib/criu).
	// Required for CUDA checkpoint/restore.
	LibDir string `yaml:"libDir"`

	// AllowUprobes allows user-space probes (required for CUDA checkpoints).
	AllowUprobes bool `yaml:"allowUprobes"`

	// SkipInFlight skips in-flight TCP connections during checkpoint/restore.
	SkipInFlight bool `yaml:"skipInFlight"`
}

// GenerateCRIUConfContent generates the criu.conf file content for options
// that cannot be passed via RPC.
func (c *CRIUConfig) GenerateCRIUConfContent() string {
	var content string

	if c.LibDir != "" {
		content += "libdir " + c.LibDir + "\n"
	}
	if c.AllowUprobes {
		content += "allow-uprobes\n"
	}
	if c.SkipInFlight {
		content += "skip-in-flight\n"
	}

	return content
}

// CRIUDumpParams holds per-checkpoint dynamic parameters for CRIU dump.
// These are values that change per checkpoint operation, not configuration.
type CRIUDumpParams struct {
	PID        int
	ImageDirFD int32
	RootFS     string
}

// BuildCRIUOpts creates CRIU options from config and per-checkpoint parameters.
// This sets up the base options; external mounts and namespaces are added separately.
func BuildCRIUOpts(cfg *CRIUConfig, params CRIUDumpParams) *criurpc.CriuOpts {
	criuOpts := &criurpc.CriuOpts{
		Pid:         proto.Int32(int32(params.PID)),
		ImagesDirFd: proto.Int32(params.ImageDirFD),
		Root:        proto.String(params.RootFS),
		LogFile:     proto.String(DumpLogFilename),
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

// ConfigureExternalMounts adds all required external mounts to CRIU options.
// This includes mounts from /proc/pid/mountinfo plus masked/readonly paths from OCI spec.
func ConfigureExternalMounts(criuOpts *criurpc.CriuOpts, pid int, containerInfo *ContainerInfo) error {
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
		criuOpts.External = append(criuOpts.External, fmt.Sprintf("%s[%d]:%s", NamespaceNet, netNs.Inode, "extNetNs"))
		netNsInode = netNs.Inode
		logrus.WithField("inode", netNsInode).Debug("Marked network namespace as external")
	}

	// Add additional external mounts (e.g., for NVIDIA firmware files)
	criuOpts.External = append(criuOpts.External, externalMounts...)

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
		logrus.WithFields(logrus.Fields{
			"prefixes": prefixes,
			"count":    len(skipMounts),
		}).Debug("Configured mounts to skip for cross-node restore")
	}

	return skipMounts, nil
}
