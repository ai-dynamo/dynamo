package criu

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	criulib "github.com/checkpoint-restore/go-criu/v8"
	criurpc "github.com/checkpoint-restore/go-criu/v8/rpc"
	"github.com/go-logr/logr"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/logging"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

// RestoreLogFilename is the CRIU restore log filename (also used by orchestrate/restore.go).
const RestoreLogFilename = "restore.log"

const (
	netNsPath        = "/proc/1/ns/net"
	placeholderFDDir = "/proc/1/fd"
)

// ExecuteRestore opens the image/work directory FDs, configures inherited
// resources, and calls go-criu Restore. Returns the namespace-relative PID.
func ExecuteRestore(
	criuOpts *criurpc.CriuOpts,
	m *types.CheckpointManifest,
	checkpointPath string,
	log logr.Logger,
) (int32, error) {
	settings := m.CRIUDump.CRIU

	// Open image dir FD
	imageDir, imageDirFD, err := openPathForCRIU(checkpointPath)
	if err != nil {
		return 0, fmt.Errorf("failed to open image directory: %w", err)
	}
	defer imageDir.Close()
	criuOpts.ImagesDirFd = proto.Int32(imageDirFD)

	// Open work dir FD
	if settings.WorkDir != "" {
		if err := os.MkdirAll(settings.WorkDir, 0755); err != nil {
			return 0, fmt.Errorf("failed to create CRIU work directory: %w", err)
		}
		workDirFile, workDirFD, err := openPathForCRIU(settings.WorkDir)
		if err != nil {
			return 0, fmt.Errorf("failed to open CRIU work directory: %w", err)
		}
		defer workDirFile.Close()
		criuOpts.WorkDirFd = proto.Int32(workDirFD)
	}

	c := criulib.MakeCriu()
	if _, err := os.Stat(settings.BinaryPath); err != nil {
		return 0, fmt.Errorf("criu binary not found at %s: %w", settings.BinaryPath, err)
	}
	c.SetCriuPath(settings.BinaryPath)

	netNsFile, err := os.Open(netNsPath)
	if err != nil {
		return 0, fmt.Errorf("failed to open net NS at %s: %w", netNsPath, err)
	}
	defer netNsFile.Close()
	c.AddInheritFd("extNetNs", netNsFile)

	inheritedFiles := registerInheritFDs(c, m.K8s.StdioFDs, log)
	defer closeFiles(inheritedFiles)

	notify := &restoreNotify{log: log}
	log.Info("Executing go-criu Restore call")
	if err := c.Restore(criuOpts, notify); err != nil {
		log.Error(err, "go-criu Restore returned error")
		logging.LogRestoreErrors(checkpointPath, settings.WorkDir, log)
		return 0, fmt.Errorf("CRIU restore failed: %w", err)
	}

	return notify.restoredPID, nil
}

// BuildRestoreOpts assembles CriuOpts for a CRIU restore from the checkpoint manifest.
// ImagesDirFd and WorkDirFd are left unset — ExecuteRestore opens them at restore time.
func BuildRestoreOpts(m *types.CheckpointManifest, cgroupRoot string, log logr.Logger) (*criurpc.CriuOpts, error) {
	extMounts, err := buildRestoreExtMounts(m)
	if err != nil {
		return nil, err
	}
	log.Info("Generated external mount map set", "ext_mount_count", len(extMounts))

	settings := m.CRIUDump.CRIU
	criuOpts := &criurpc.CriuOpts{
		LogFile: proto.String(RestoreLogFilename),
		Root:    proto.String("/"),
		ExtMnt:  extMounts,
	}
	if err := applyCommonSettings(criuOpts, &settings); err != nil {
		return nil, err
	}

	// Restore-only options
	criuOpts.RstSibling = proto.Bool(settings.RstSibling)
	criuOpts.MntnsCompatMode = proto.Bool(settings.MntnsCompatMode)
	criuOpts.EvasiveDevices = proto.Bool(settings.EvasiveDevices)
	criuOpts.ForceIrmap = proto.Bool(settings.ForceIrmap)

	if cgroupRoot != "" && shouldSetCgroupRoot(criuOpts.GetManageCgroupsMode()) {
		criuOpts.CgRoot = []*criurpc.CgroupRoot{
			{Path: proto.String(cgroupRoot)},
		}
	}

	criuConfPath := filepath.Join(settings.WorkDir, "..", criuConfFilename)
	if _, err := os.Stat(criuConfPath); err == nil {
		criuOpts.ConfigFile = proto.String(criuConfPath)
	}

	return criuOpts, nil
}

func buildRestoreExtMounts(m *types.CheckpointManifest) ([]*criurpc.ExtMountMap, error) {
	if len(m.CRIUDump.ExtMnt) == 0 {
		return nil, fmt.Errorf("checkpoint manifest is missing criuDump.extMnt")
	}

	restoreMap := map[string]string{"/": "."}
	for _, val := range m.CRIUDump.ExtMnt {
		if val == "" || val == "/" {
			continue
		}
		restoreMap[val] = val
	}
	return toExtMountMaps(restoreMap), nil
}

func registerInheritFDs(c *criulib.Criu, stdioFDs []string, log logr.Logger) []*os.File {
	if len(stdioFDs) == 0 {
		log.Info("No stdio FD descriptors in manifest, skipping inherit-fd setup")
		return nil
	}

	var openFiles []*os.File
	for i, target := range stdioFDs {
		if !strings.Contains(target, "pipe:") {
			continue
		}
		// stdin (fd 0) is a read-end pipe; stdout/stderr (fd 1, 2) are write-end
		openMode := os.O_WRONLY
		if i == 0 {
			openMode = os.O_RDONLY
		}
		fdPath := fmt.Sprintf("%s/%d", placeholderFDDir, i)
		f, err := os.OpenFile(fdPath, openMode, 0)
		if err != nil {
			log.V(1).Info("Failed to open placeholder stdio FD, skipping", "fd", i, "target", target, "error", err)
			continue
		}
		openFiles = append(openFiles, f)
		c.AddInheritFd(target, f)
	}

	log.Info("Registered inherited stdio pipes", "count", len(openFiles))
	return openFiles
}

func closeFiles(files []*os.File) {
	for _, file := range files {
		if file != nil {
			file.Close()
		}
	}
}

type restoreNotify struct {
	criulib.NoNotify
	restoredPID int32
	log         logr.Logger
}

func (n *restoreNotify) PreRestore() error {
	n.log.V(1).Info("CRIU pre-restore")
	return nil
}

func (n *restoreNotify) PostRestore(pid int32) error {
	n.restoredPID = pid
	n.log.Info("CRIU post-restore: process restored", "pid", pid)
	return nil
}
