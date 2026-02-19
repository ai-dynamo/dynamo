// criu provides CRIU-specific configuration and utilities for checkpoint operations.
package checkpoint

import (
	"fmt"
	"time"

	criu "github.com/checkpoint-restore/go-criu/v7"
	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"
)

// CRIUSettings holds CRIU-specific configuration options.
// Options are categorized by how they are passed to CRIU:
//   - RPC options: Passed via go-criu CriuOpts protobuf
//   - CRIU conf file options: Written to criu.conf (NOT available via RPC)
type CRIUSettings struct {
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

	// === CRIU Conf File Options (NOT available via RPC - written to criu.conf) ===

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
func (c *CRIUSettings) GenerateCRIUConfContent() string {
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

// ExternalMountManifestEntry is a serializable CRIU ext-mount entry in checkpoint manifests.
type ExternalMountManifestEntry struct {
	Key string `yaml:"key"`
	Val string `yaml:"val"`
}

// CRIUDumpManifest stores the resolved dump-time CRIU mount plan used for restore.
type CRIUDumpManifest struct {
	CRIU     CRIUSettings                 `yaml:"criu"`
	ExtMnt   []ExternalMountManifestEntry `yaml:"extMnt,omitempty"`
	External []string                     `yaml:"external,omitempty"`
	SkipMnt  []string                     `yaml:"skipMnt,omitempty"`
}

// NewCRIUDumpManifest serializes resolved dump options for restore.
func NewCRIUDumpManifest(criuOpts *criurpc.CriuOpts, settings CRIUSettings) CRIUDumpManifest {
	manifest := CRIUDumpManifest{CRIU: settings}
	if criuOpts == nil {
		return manifest
	}

	for _, mount := range criuOpts.ExtMnt {
		if mount == nil || mount.GetKey() == "" {
			continue
		}
		manifest.ExtMnt = append(manifest.ExtMnt, ExternalMountManifestEntry{
			Key: mount.GetKey(),
			Val: mount.GetVal(),
		})
	}
	manifest.External = append([]string(nil), criuOpts.External...)
	manifest.SkipMnt = append([]string(nil), criuOpts.SkipMnt...)
	return manifest
}

// BuildCRIUDumpOptions creates CRIU options directly from spec settings and runtime state.
func BuildCRIUDumpOptions(
	settings *CRIUSettings,
	pid int,
	imageDirFD int32,
	rootFS string,
	mountInfo []MountInfo,
	ociSpec *specs.Spec,
	namespaces map[NamespaceType]*NamespaceInfo,
) (*criurpc.CriuOpts, error) {
	mountPolicy := BuildMountPolicy(mountInfo, ociSpec, rootFS)

	extMnt := buildExternalMountMaps(mountPolicy.Externalized)
	skipMnt := mountPolicy.Skipped
	external := buildExternalNamespaces(namespaces)
	logrus.WithFields(logrus.Fields{
		"externalized_count": len(mountPolicy.Externalized),
		"skipped_count":      len(mountPolicy.Skipped),
	}).Debug("Resolved mount policy for CRIU dump")

	criuOpts := &criurpc.CriuOpts{
		Pid:         proto.Int32(int32(pid)),
		ImagesDirFd: proto.Int32(imageDirFD),
		Root:        proto.String(rootFS),
		LogFile:     proto.String(DumpLogFilename),
	}
	criuOpts.ExtMnt = extMnt
	criuOpts.External = external
	criuOpts.SkipMnt = skipMnt

	if settings == nil {
		return criuOpts, nil
	}

	// RPC options from spec.
	criuOpts.LogLevel = proto.Int32(settings.LogLevel)
	criuOpts.LeaveRunning = proto.Bool(settings.LeaveRunning)
	criuOpts.ShellJob = proto.Bool(settings.ShellJob)
	criuOpts.TcpClose = proto.Bool(settings.TcpClose)
	criuOpts.FileLocks = proto.Bool(settings.FileLocks)
	criuOpts.OrphanPtsMaster = proto.Bool(settings.OrphanPtsMaster)
	criuOpts.ExtUnixSk = proto.Bool(settings.ExtUnixSk)
	criuOpts.LinkRemap = proto.Bool(settings.LinkRemap)
	criuOpts.ExtMasters = proto.Bool(settings.ExtMasters)
	criuOpts.AutoDedup = proto.Bool(settings.AutoDedup)
	criuOpts.LazyPages = proto.Bool(settings.LazyPages)

	// Cgroup management mode
	criuOpts.ManageCgroups = proto.Bool(true)
	cgMode := criurpc.CriuCgMode_IGNORE
	switch settings.ManageCgroupsMode {
	case "soft":
		cgMode = criurpc.CriuCgMode_SOFT
	case "full":
		cgMode = criurpc.CriuCgMode_FULL
	case "strict":
		cgMode = criurpc.CriuCgMode_STRICT
	}
	criuOpts.ManageCgroupsMode = &cgMode

	// Optional numeric options
	if settings.GhostLimit > 0 {
		criuOpts.GhostLimit = proto.Uint32(settings.GhostLimit)
	}
	if settings.Timeout > 0 {
		criuOpts.Timeout = proto.Uint32(settings.Timeout)
	}

	return criuOpts, nil
}

// buildExternalMountMaps serializes externalized mount paths into CRIU map entries.
func buildExternalMountMaps(paths []string) []*criurpc.ExtMountMap {
	extMnt := make([]*criurpc.ExtMountMap, 0, len(paths))
	existing := make(map[string]struct{}, len(paths))
	for _, path := range paths {
		if path == "" {
			continue
		}
		if _, ok := existing[path]; ok {
			continue
		}
		extMnt = append(extMnt, &criurpc.ExtMountMap{
			Key: proto.String(path),
			Val: proto.String(path),
		})
		existing[path] = struct{}{}
	}

	return extMnt
}

// buildExternalNamespaces builds external namespace/mount references.
func buildExternalNamespaces(namespaces map[NamespaceType]*NamespaceInfo) []string {
	external := make([]string, 0, 1)

	// Mark network namespace as external for socket binding preservation
	if netNs, ok := namespaces[NamespaceNet]; ok {
		external = append(external, fmt.Sprintf("%s[%d]:%s", NamespaceNet, netNs.Inode, "extNetNs"))
		logrus.WithField("inode", netNs.Inode).Debug("Marked network namespace as external")
	}

	return external
}

// ExecuteCRIUDump runs the CRIU dump and logs timing plus dump-log location on failure.
func ExecuteCRIUDump(criuOpts *criurpc.CriuOpts, checkpointDir string, log *logrus.Entry) (time.Duration, error) {
	criuDumpStart := time.Now()
	criuClient := criu.MakeCriu()
	if err := criuClient.Dump(criuOpts, nil); err != nil {
		dumpDuration := time.Since(criuDumpStart)
		log.WithFields(logrus.Fields{
			"duration":       dumpDuration,
			"checkpoint_dir": checkpointDir,
			"dump_log_path":  fmt.Sprintf("%s/%s", checkpointDir, DumpLogFilename),
		}).Error("CRIU dump failed")
		return 0, fmt.Errorf("CRIU dump failed: %w", err)
	}

	criuDumpDuration := time.Since(criuDumpStart)
	log.WithField("duration", criuDumpDuration).Info("CRIU dump completed")
	return criuDumpDuration, nil
}
