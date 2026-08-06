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

	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/logging"
	"github.com/ai-dynamo/dynamo/deploy/snapshot/internal/types"
)

// RestoreLogFilename is the CRIU restore log filename (also used by executor/restore.go).
const RestoreLogFilename = "restore.log"

const (
	netNsPath        = "/proc/1/ns/net"
	placeholderFDDir = "/proc/1/fd"
	// defaultBundleDir is the fallback when the caller does not supply a BundleDir.
	defaultBundleDir = "/tmp/snapshot-binaries"
)

// ExecuteRestore opens the image/work directory FDs, configures inherited
// resources, and calls go-criu Restore. Returns the namespace-relative PID.
// bundleDir is the path where the agent bundle is mounted in this namespace;
// if empty, defaultBundleDir is used.
func ExecuteRestore(
	criuOpts *criurpc.CriuOpts,
	m *types.CheckpointManifest,
	checkpointPath string,
	bundleDir string,
	log logr.Logger,
) (int32, func(), error) {
	if bundleDir == "" {
		bundleDir = defaultBundleDir
	}
	settings := m.CRIUDump.CRIU

	// Return the FD closers as cleanup() rather than deferring them here, so the
	// caller can run them after cuda unlock instead of between the CRIU restore
	// and unlock. That keeps the window where the restored process runs with CUDA
	// still locked as short as possible. cleanup is called on the error paths below.
	var openFiles, inheritedFiles []*os.File
	cleanup := func() {
		closeFiles(inheritedFiles)
		closeFiles(openFiles)
	}

	// Open image dir FD
	imageDir, imageDirFD, err := openPathForCRIU(checkpointPath)
	if err != nil {
		return 0, nil, fmt.Errorf("failed to open image directory: %w", err)
	}
	openFiles = append(openFiles, imageDir)
	criuOpts.ImagesDirFd = proto.Int32(imageDirFD)

	// Open work dir FD
	if settings.WorkDir != "" {
		if err := os.MkdirAll(settings.WorkDir, 0755); err != nil {
			cleanup()
			return 0, nil, fmt.Errorf("failed to create CRIU work directory: %w", err)
		}
		workDirFile, workDirFD, err := openPathForCRIU(settings.WorkDir)
		if err != nil {
			cleanup()
			return 0, nil, fmt.Errorf("failed to open CRIU work directory: %w", err)
		}
		openFiles = append(openFiles, workDirFile)
		criuOpts.WorkDirFd = proto.Int32(workDirFD)
	}

	if err := rewriteCRIULibDir(criuOpts, settings.WorkDir, bundleDir, log); err != nil {
		cleanup()
		return 0, nil, err
	}

	c := criulib.MakeCriu()
	// criu is always sourced from the injected binary bundle — never from the
	// checkpoint-time BinaryPath, which refers to the agent filesystem.
	criuBin := filepath.Join(bundleDir, "criu")
	if _, err := os.Stat(criuBin); err != nil {
		cleanup()
		return 0, nil, fmt.Errorf("criu binary not found at %s (injected from agent): %w", criuBin, err)
	}
	c.SetCriuPath(criuBin)

	netNsFile, err := os.Open(netNsPath)
	if err != nil {
		cleanup()
		return 0, nil, fmt.Errorf("failed to open net NS at %s: %w", netNsPath, err)
	}
	openFiles = append(openFiles, netNsFile)
	c.AddInheritFd("extNetNs", netNsFile)

	inheritedFiles = registerInheritFDs(c, m.K8s.StdioFDs, log)

	notify := &restoreNotify{log: log}
	log.V(1).Info("Executing go-criu Restore call")
	if err := c.Restore(criuOpts, notify); err != nil {
		log.Error(err, "go-criu Restore returned error")
		logging.LogRestoreErrors(checkpointPath, settings.WorkDir, log)
		cleanup()
		return 0, nil, fmt.Errorf("CRIU restore failed: %w", err)
	}

	return notify.restoredPID, cleanup, nil
}

// BuildRestoreOpts assembles CriuOpts for a CRIU restore from the checkpoint manifest.
// ImagesDirFd and WorkDirFd are left unset — ExecuteRestore opens them at restore time.
func BuildRestoreOpts(m *types.CheckpointManifest, checkpointPath string, cgroupRoot string, log logr.Logger) (*criurpc.CriuOpts, error) {
	extMounts, err := buildRestoreExtMounts(m)
	if err != nil {
		return nil, err
	}
	log.V(1).Info("Generated external mount map set", "ext_mount_count", len(extMounts))

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

	criuConfPath := filepath.Join(checkpointPath, criuConfFilename)
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
		log.V(1).Info("No stdio FD descriptors in manifest, skipping inherit-fd setup")
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

	log.V(1).Info("Registered inherited stdio pipes", "count", len(openFiles))
	return openFiles
}

// rewriteCRIULibDir rewrites the libdir line in criu.conf so CRIU loads plugins
// from the injected bundle rather than the dump-time path
// (/usr/local/lib/snapshot/criu-plugins), which only exists on the agent, not in
// the placeholder namespace. The override is written to the work dir (runtime state)
// so the original dump-time config is left intact.
func rewriteCRIULibDir(criuOpts *criurpc.CriuOpts, workDir, criuBundleDir string, log logr.Logger) error {
	if criuOpts.ConfigFile == nil {
		return nil
	}
	if workDir == "" {
		log.Info("criu WorkDir unset; skipping libdir override — criu will use dump-time plugin path")
		return nil
	}
	data, err := os.ReadFile(criuOpts.GetConfigFile())
	if err != nil {
		log.Error(err, "failed to read criu config file; skipping libdir override", "path", criuOpts.GetConfigFile())
		return nil
	}
	overridePath := filepath.Join(workDir, "criu-restore.conf")
	conf := overrideLibDir(string(data), filepath.Join(criuBundleDir, "criu-plugins"))
	if err := os.WriteFile(overridePath, []byte(conf), 0644); err != nil {
		return fmt.Errorf("write criu libdir override to %s: %w", overridePath, err)
	}
	criuOpts.ConfigFile = proto.String(overridePath)
	return nil
}

func overrideLibDir(conf, libDir string) string {
	lines := strings.Split(conf, "\n")
	replaced := false
	for i, line := range lines {
		if isLibDirLine(line) {
			lines[i] = "libdir " + libDir
			replaced = true
		}
	}
	if !replaced {
		lines = append(lines, "libdir "+libDir)
	}
	return strings.Join(lines, "\n")
}

func isLibDirLine(line string) bool {
	return strings.HasPrefix(strings.TrimSpace(line), "libdir ")
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
