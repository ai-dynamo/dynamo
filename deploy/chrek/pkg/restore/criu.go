// criu provides CRIU-specific configuration and utilities for restore operations.
package restore

import (
	"os"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// CRIURestorePlan holds configuration for CRIU restore operations.
// Most fields come from the saved CheckpointManifest.CRIUDump.CRIU settings.
type CRIURestorePlan struct {
	// File descriptors
	ImageDirFD int32
	WorkDirFD  int32
	NetNsFD    int32

	// Paths
	RootPath string
	LogFile  string

	// Options from CheckpointManifest.CRIUDump.CRIU.
	LogLevel          int32
	Timeout           uint32 // CRIU timeout in seconds (0 = no timeout, required for CUDA)
	ShellJob          bool   // Allow session leaders (containers are often session leaders)
	TcpClose          bool   // Close TCP connections (pod IPs change on restore)
	FileLocks         bool   // Allow file locks
	ExtUnixSk         bool   // Allow external Unix sockets
	LinkRemap         bool   // Handle deleted-but-open files via CRIU link remap
	ManageCgroupsMode string // Cgroup handling mode: "ignore" lets K8s manage cgroups

	// External mount mappings (from CheckpointManifest.CRIUDump.ExtMnt).
	ExtMountMaps []*criurpc.ExtMountMap
}

// OpenImageDir opens a checkpoint directory and clears CLOEXEC for CRIU.
// Returns the opened file and its FD. Caller must close the file when done.
func OpenImageDir(checkpointPath string) (*os.File, int32, error) {
	return common.OpenPathForCRIU(checkpointPath)
}

// OpenNetworkNamespace opens the target network namespace for restore.
// Returns the opened file and its FD. Caller must close the file when done.
func OpenNetworkNamespace(nsPath string) (*os.File, int32, error) {
	return common.OpenPathForCRIU(nsPath)
}

// OpenWorkDir opens a work directory for CRIU and clears CLOEXEC.
// Returns the opened file and its FD, or nil/-1 if workDir is empty or fails.
func OpenWorkDir(workDir string, log *logrus.Entry) (*os.File, int32) {
	if workDir == "" {
		return nil, -1
	}

	// Ensure work directory exists
	if err := os.MkdirAll(workDir, 0755); err != nil {
		log.WithError(err).Warn("Failed to create CRIU work directory, using default")
		return nil, -1
	}

	workDirFile, err := os.Open(workDir)
	if err != nil {
		log.WithError(err).Warn("Failed to open CRIU work directory, using default")
		return nil, -1
	}

	if _, err := unix.FcntlInt(workDirFile.Fd(), unix.F_SETFD, 0); err != nil {
		log.WithError(err).Warn("Failed to clear CLOEXEC on work dir")
		workDirFile.Close()
		return nil, -1
	}

	log.WithField("path", workDir).Info("Using custom CRIU work directory")
	return workDirFile, int32(workDirFile.Fd())
}

// BuildCRIURestoreOptions creates CRIU options for restore from a runtime plan.
//
// Options from CheckpointManifest.CRIUDump.CRIU (saved at checkpoint time):
//   - ShellJob, TcpClose, FileLocks, ExtUnixSk, LinkRemap, ManageCgroupsMode
//
// Hardcoded restore-specific options:
//   - RstSibling: restore in detached mode
//   - MntnsCompatMode: cross-container restore
//   - EvasiveDevices, ForceIrmap: device/inode handling
func BuildCRIURestoreOptions(plan CRIURestorePlan) *criurpc.CriuOpts {
	// Map cgroup management mode from plan.
	var cgMode criurpc.CriuCgMode
	switch plan.ManageCgroupsMode {
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

	criuOpts := &criurpc.CriuOpts{
		ImagesDirFd: proto.Int32(plan.ImageDirFD),
		LogLevel:    proto.Int32(plan.LogLevel),
		LogFile:     proto.String(plan.LogFile),

		// Root filesystem - use current container's root
		Root: proto.String(plan.RootPath),

		// Restore in detached mode - process runs in background (restore-specific)
		RstSibling: proto.Bool(true),

		// Mount namespace mode:
		// - MntnsCompatMode=false (default): Uses mount-v2 with MOVE_MOUNT_SET_GROUP (kernel 5.15+)
		//   This is preferred as it doesn't create temp dirs in /tmp
		// - MntnsCompatMode=true: Uses compat mode which creates /tmp/cr-tmpfs.XXX
		//   This can cause "Device or resource busy" errors on cleanup
		// We explicitly set to false to use mount-v2 (requires kernel 5.15+)
		MntnsCompatMode: proto.Bool(false),

		// Options from saved CheckpointManifest.CRIUDump.CRIU.
		ShellJob:  proto.Bool(plan.ShellJob),
		TcpClose:  proto.Bool(plan.TcpClose),
		FileLocks: proto.Bool(plan.FileLocks),
		ExtUnixSk: proto.Bool(plan.ExtUnixSk),
		LinkRemap: proto.Bool(plan.LinkRemap),

		// Cgroup management from saved settings.
		ManageCgroups:     proto.Bool(true),
		ManageCgroupsMode: &cgMode,

		// Device and inode handling (restore-specific)
		EvasiveDevices: proto.Bool(true),
		ForceIrmap:     proto.Bool(true),

		// External mount mappings
		ExtMnt: plan.ExtMountMaps,
	}

	// Add network namespace inheritance if provided
	if plan.NetNsFD >= 0 {
		criuOpts.InheritFd = []*criurpc.InheritFd{
			{
				Key: proto.String("extNetNs"),
				Fd:  proto.Int32(plan.NetNsFD),
			},
		}
	}

	// Add work directory if specified
	if plan.WorkDirFD >= 0 {
		criuOpts.WorkDirFd = proto.Int32(plan.WorkDirFD)
	}

	// Add timeout if specified (required for CUDA restores)
	if plan.Timeout > 0 {
		criuOpts.Timeout = proto.Uint32(plan.Timeout)
	}

	return criuOpts
}
