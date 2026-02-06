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

// CRIURestoreConfig holds configuration for CRIU restore operations.
// Most options are always-on with safe defaults for K8s environments.
type CRIURestoreConfig struct {
	ImageDirFD   int32
	RootPath     string
	LogLevel     int32
	LogFile      string
	WorkDirFD    int32
	NetNsFD      int32
	ExtMountMaps []*criurpc.ExtMountMap
}

// OpenImageDir opens a checkpoint directory and clears CLOEXEC for CRIU.
// Returns the opened file and its FD. Caller must close the file when done.
func OpenImageDir(checkpointPath string) (*os.File, int32, error) {
	return common.OpenDirForCRIU(checkpointPath)
}

// OpenNetworkNamespace opens the target network namespace for restore.
// Returns the opened file and its FD. Caller must close the file when done.
func OpenNetworkNamespace(nsPath string) (*os.File, int32, error) {
	return common.OpenDirForCRIU(nsPath)
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

// BuildRestoreCRIUOpts creates CRIU options for restore from a config struct.
//
// Always-on options for K8s:
//   - ShellJob: containers are often session leaders
//   - TcpClose: pod IPs change on restore/migration
//   - FileLocks: applications use file locks
//   - ExtUnixSk: containers have external Unix sockets
//   - ManageCgroups (IGNORE): let K8s manage cgroups
func BuildRestoreCRIUOpts(cfg CRIURestoreConfig) *criurpc.CriuOpts {
	cgMode := criurpc.CriuCgMode_IGNORE

	criuOpts := &criurpc.CriuOpts{
		ImagesDirFd: proto.Int32(cfg.ImageDirFD),
		LogLevel:    proto.Int32(cfg.LogLevel),
		LogFile:     proto.String(cfg.LogFile),

		// Root filesystem - use current container's root
		Root: proto.String(cfg.RootPath),

		// Restore in detached mode - process runs in background
		RstSibling: proto.Bool(true),

		// Mount namespace compatibility mode for cross-container restore
		MntnsCompatMode: proto.Bool(true),

		// Always-on for K8s environments
		ShellJob:  proto.Bool(true),
		TcpClose:  proto.Bool(true),
		FileLocks: proto.Bool(true),
		ExtUnixSk: proto.Bool(true),

		// Cgroup management - ignore to avoid conflicts
		ManageCgroups:     proto.Bool(true),
		ManageCgroupsMode: &cgMode,

		// Device and inode handling
		EvasiveDevices: proto.Bool(true),
		ForceIrmap:     proto.Bool(true),

		// External mount mappings
		ExtMnt: cfg.ExtMountMaps,
	}

	// Add network namespace inheritance if provided
	if cfg.NetNsFD >= 0 {
		criuOpts.InheritFd = []*criurpc.InheritFd{
			{
				Key: proto.String("extNetNs"),
				Fd:  proto.Int32(cfg.NetNsFD),
			},
		}
	}

	// Add work directory if specified
	if cfg.WorkDirFD >= 0 {
		criuOpts.WorkDirFd = proto.Int32(cfg.WorkDirFD)
	}

	return criuOpts
}
