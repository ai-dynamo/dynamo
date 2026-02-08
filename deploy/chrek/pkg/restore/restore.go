package restore

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	criu "github.com/checkpoint-restore/go-criu/v7"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/config"
)

// LogGPUDiagnostics logs nvidia-smi and /dev/nvidia* for debugging GPU visibility.
func LogGPUDiagnostics(label string, log *logrus.Entry) {
	log.Infof("=== GPU DIAGNOSTICS [%s] ===", label)
	diagCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(diagCtx, "nvidia-smi", "-L").CombinedOutput(); err != nil {
		log.Infof("nvidia-smi -L: error: %v", err)
	} else {
		log.Infof("nvidia-smi -L:\n%s", string(out))
	}
	// Also log memory usage per GPU to detect OOM conditions
	diagCtx2, cancel2 := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel2()
	if out, err := exec.CommandContext(diagCtx2, "nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total,memory.free", "--format=csv,noheader").CombinedOutput(); err != nil {
		log.Infof("nvidia-smi memory query: error: %v", err)
	} else {
		log.Infof("nvidia-smi memory:\n%s", string(out))
	}
	matches, _ := filepath.Glob("/dev/nvidia*")
	log.Infof("/dev/nvidia* devices: %s", strings.Join(matches, ", "))
	log.Infof("NVIDIA_VISIBLE_DEVICES=%s", os.Getenv("NVIDIA_VISIBLE_DEVICES"))
	log.Infof("=== END GPU DIAGNOSTICS [%s] ===", label)
}

func processSnapshotPIDs(restoredPID int) []int {
	pidSet := map[int]struct{}{
		1:           {},
		os.Getpid(): {},
	}
	if restoredPID > 0 {
		pidSet[restoredPID] = struct{}{}
	}
	pids := make([]int, 0, len(pidSet))
	for pid := range pidSet {
		pids = append(pids, pid)
	}
	sort.Ints(pids)
	return pids
}

func logProcessNamespaces(pid int, log *logrus.Entry) {
	for _, ns := range []string{"mnt", "pid", "ipc", "net", "uts", "cgroup"} {
		nsPath := fmt.Sprintf("/proc/%d/ns/%s", pid, ns)
		link, err := os.Readlink(nsPath)
		if err != nil {
			log.WithError(err).WithFields(logrus.Fields{
				"pid":  pid,
				"path": nsPath,
			}).Warn("Failed to read namespace symlink")
			continue
		}
		log.WithFields(logrus.Fields{
			"pid":       pid,
			"namespace": ns,
			"value":     link,
		}).Info("Namespace snapshot")
	}
}

func logProcessCgroupPath(pid int, log *logrus.Entry) {
	path := fmt.Sprintf("/proc/%d/cgroup", pid)
	data, err := os.ReadFile(path)
	if err != nil {
		log.WithError(err).WithFields(logrus.Fields{
			"pid":  pid,
			"path": path,
		}).Warn("Failed to read cgroup path")
		return
	}
	log.WithFields(logrus.Fields{
		"pid":      pid,
		"path":     path,
		"contents": strings.TrimSpace(string(data)),
	}).Info("Cgroup membership snapshot")
}

func logProcessFilteredMountInfo(pid int, log *logrus.Entry) {
	path := fmt.Sprintf("/proc/%d/mountinfo", pid)
	f, err := os.Open(path)
	if err != nil {
		log.WithError(err).WithFields(logrus.Fields{
			"pid":  pid,
			"path": path,
		}).Warn("Failed to open mountinfo")
		return
	}
	defer f.Close()

	var selected []string
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.Contains(line, " /dev ") ||
			strings.Contains(line, "/dev/") ||
			strings.Contains(line, "nvidia") ||
			strings.Contains(line, "cgroup2") {
			selected = append(selected, line)
		}
	}
	if err := scanner.Err(); err != nil {
		log.WithError(err).WithFields(logrus.Fields{
			"pid":  pid,
			"path": path,
		}).Warn("Failed while scanning mountinfo")
		return
	}

	log.WithFields(logrus.Fields{
		"pid":   pid,
		"path":  path,
		"count": len(selected),
	}).Info("Filtered mountinfo snapshot count")
	if len(selected) > 0 {
		log.Infof("Filtered mountinfo snapshot (pid=%d):\n%s", pid, strings.Join(selected, "\n"))
	}
}

func logNvidiaDeviceNodeMetadata(log *logrus.Entry) {
	devices, err := filepath.Glob("/dev/nvidia*")
	if err != nil {
		log.WithError(err).Warn("Failed to glob /dev/nvidia*")
		return
	}
	if len(devices) == 0 {
		log.Info("No /dev/nvidia* entries found")
		return
	}

	for _, path := range devices {
		fi, err := os.Lstat(path)
		if err != nil {
			log.WithError(err).WithField("path", path).Warn("Failed to stat NVIDIA device entry")
			continue
		}
		stat, ok := fi.Sys().(*syscall.Stat_t)
		if !ok {
			log.WithFields(logrus.Fields{
				"path": path,
				"mode": fi.Mode().String(),
			}).Warn("Unexpected stat type for NVIDIA device entry")
			continue
		}
		log.WithFields(logrus.Fields{
			"path":  path,
			"mode":  fi.Mode().String(),
			"inode": stat.Ino,
			"rdev":  fmt.Sprintf("0x%x", stat.Rdev),
		}).Info("NVIDIA device entry metadata")
	}
}

func logCgroupV2HostInfo(log *logrus.Entry) {
	const controllersPath = "/sys/fs/cgroup/cgroup.controllers"
	data, err := os.ReadFile(controllersPath)
	if err != nil {
		log.WithError(err).WithField("path", controllersPath).Warn("Failed to read cgroup v2 controllers")
		return
	}
	log.WithFields(logrus.Fields{
		"path":        controllersPath,
		"controllers": strings.TrimSpace(string(data)),
	}).Info("cgroup v2 controllers")
}

// LogRestoreBoundaryDiagnostics captures cgroup and namespace state around CRIU restore.
func LogRestoreBoundaryDiagnostics(label string, restoredPID int, log *logrus.Entry) {
	log.Infof("=== RESTORE BOUNDARY DIAGNOSTICS [%s] ===", label)
	for _, pid := range processSnapshotPIDs(restoredPID) {
		logProcessNamespaces(pid, log)
		logProcessCgroupPath(pid, log)
		logProcessFilteredMountInfo(pid, log)
	}
	logCgroupV2HostInfo(log)
	logNvidiaDeviceNodeMetadata(log)
	log.Infof("=== END RESTORE BOUNDARY DIAGNOSTICS [%s] ===", label)
}

// Restore performs the CRIU restore operation using go-criu.
// All CRIU options are read from the saved CheckpointData - no hardcoding.
// Returns the PID of the restored process.
func Restore(ctx context.Context, checkpointPath string, data *config.CheckpointData, log *logrus.Entry) (int, error) {
	if data == nil {
		return 0, fmt.Errorf("checkpoint data is required")
	}

	// Hardcoded restore constants
	const (
		rootPath = "/"
		pidFile  = "/tmp/restored.pid"
		logFile  = "restore.log"
	)

	log.WithField("checkpoint", checkpointPath).Info("Starting CRIU restore")

	// 1. Open checkpoint directory
	imageDir, imageDirFD, err := OpenImageDir(checkpointPath)
	if err != nil {
		return 0, err
	}
	defer imageDir.Close()
	log.WithField("fd", imageDirFD).Debug("Opened checkpoint directory")

	// 2. Generate external mount mappings from saved CheckpointData
	extMounts, err := GenerateExtMountMaps(data)
	if err != nil {
		return 0, fmt.Errorf("failed to generate mount maps: %w", err)
	}
	log.WithField("mount_count", len(extMounts)).Debug("External mount maps ready")

	// 3. Open target network namespace
	netNsFile, netNsFD, err := OpenNetworkNamespace("/proc/1/ns/net")
	if err != nil {
		return 0, err
	}
	defer netNsFile.Close()
	log.WithField("fd", netNsFD).Debug("Opened target network namespace")

	// 4. Open work directory if specified in checkpoint data
	var workDirFile *os.File
	var workDirFD int32 = -1
	if data.CRIU.WorkDir != "" {
		workDirFile, workDirFD = OpenWorkDir(data.CRIU.WorkDir, log)
		if workDirFile != nil {
			defer workDirFile.Close()
		}
	}

	// 5. Build CRIU options from saved checkpoint data
	cfg := CRIURestoreConfig{
		// File descriptors
		ImageDirFD: imageDirFD,
		WorkDirFD:  workDirFD,
		NetNsFD:    netNsFD,
		// Paths
		RootPath: rootPath,
		LogFile:  logFile,
		// Options from CheckpointData.CRIU
		LogLevel:          data.CRIU.LogLevel,
		Timeout:           data.CRIU.Timeout,
		ShellJob:          data.CRIU.ShellJob,
		TcpClose:          data.CRIU.TcpClose,
		FileLocks:         data.CRIU.FileLocks,
		ExtUnixSk:         data.CRIU.ExtUnixSk,
		ManageCgroupsMode: data.CRIU.ManageCgroupsMode,
		// External mounts
		ExtMountMaps: extMounts,
	}
	criuOpts := BuildRestoreCRIUOpts(cfg)

	// 6. Create CRIU config file for CUDA plugin if libdir is specified
	// IMPORTANT: Only these options go in criu.conf (NOT available via RPC):
	// - libdir (plugin directory)
	// - allow-uprobes (required for CUDA)
	// - skip-in-flight (skip in-flight TCP)
	// All other options (timeout, ghost-limit, etc.) should be passed via RPC.
	if data.CRIU.LibDir != "" {
		if data.CRIU.Timeout == 0 {
			return 0, fmt.Errorf("CRIU timeout must be set for CUDA restores (check checkpoint data)")
		}
		configPath := filepath.Join(checkpointPath, "restore-criu.conf")

		// Build config content from saved checkpoint data
		var configLines []string
		configLines = append(configLines, fmt.Sprintf("libdir %s", data.CRIU.LibDir))
		if data.CRIU.AllowUprobes {
			configLines = append(configLines, "allow-uprobes")
		}
		if data.CRIU.SkipInFlight {
			configLines = append(configLines, "skip-in-flight")
		}
		configContent := strings.Join(configLines, "\n") + "\n"

		if err := os.WriteFile(configPath, []byte(configContent), 0644); err != nil {
			return 0, fmt.Errorf("failed to write CRIU config file for CUDA plugin: %w", err)
		}
		criuOpts.ConfigFile = proto.String(configPath)
		log.WithFields(logrus.Fields{
			"config_path": configPath,
			"lib_dir":     data.CRIU.LibDir,
		}).Info("Created CRIU config file with libdir for CUDA plugin")
	}

	// 7. Execute CRIU restore
	c := criu.MakeCriu()
	notify := NewRestoreNotify(log)

	log.Info("Executing CRIU restore")
	criuExecStart := time.Now()
	if err := c.Restore(criuOpts, notify); err != nil {
		log.WithField("duration", time.Since(criuExecStart)).Error("CRIU c.Restore failed")
		logCRIUErrors(checkpointPath, logFile, log)
		return 0, fmt.Errorf("CRIU restore failed: %w", err)
	}

	log.WithFields(logrus.Fields{
		"pid":      notify.RestoredPID,
		"duration": time.Since(criuExecStart),
	}).Info("CRIU c.Restore completed successfully")

	// 8. Get restored PID
	if notify.RestoredPID > 0 {
		return int(notify.RestoredPID), nil
	}

	// Fallback: try to read from PID file
	pid, err := WaitForPidFile(pidFile, 10*time.Second, log)
	if err != nil {
		return 0, fmt.Errorf("failed to get restored PID: %w", err)
	}
	return pid, nil
}

// logCRIUErrors reads CRIU log file and logs errors.
func logCRIUErrors(checkpointPath, logFile string, log *logrus.Entry) {
	logPath := filepath.Join(checkpointPath, logFile)
	data, err := os.ReadFile(logPath)
	if err != nil {
		log.WithError(err).Warn("Could not read CRIU log file")
		return
	}

	log.Error("=== CRIU RESTORE LOG START ===")
	for _, line := range strings.Split(string(data), "\n") {
		if line != "" {
			log.Error(line)
		}
	}
	log.Error("=== CRIU RESTORE LOG END ===")

	// Copy log to shared directory for debugging
	if err := os.MkdirAll(config.CRIULogDir, 0755); err == nil {
		destPath := filepath.Join(config.CRIULogDir, fmt.Sprintf("restore-%d.log", time.Now().Unix()))
		if err := os.WriteFile(destPath, data, 0644); err == nil {
			log.WithField("path", destPath).Info("CRIU log copied to shared directory")
		}
	}
}

// Run is the main entry point for the restore entrypoint.
// It orchestrates the entire restore process.
func Run(ctx context.Context, cfg *config.RestoreConfig, log *logrus.Entry) error {
	log.Info("=== Self-Restoring Placeholder Entrypoint ===")
	log.WithFields(logrus.Fields{
		"checkpoint_location": cfg.CheckpointLocation,
		"checkpoint_hash":     cfg.CheckpointHash,
	}).Info("Configuration")

	// Check CRIU availability
	c := criu.MakeCriu()
	version, err := c.GetCriuVersion()
	if err != nil {
		log.WithError(err).Error("CRIU is not available")
		log.Info("Falling back to default command")
		return RunDefault(cfg, log)
	}
	log.WithField("version", version).Info("CRIU version")

	// Determine checkpoint path
	var checkpointPath string
	var shouldRestore bool

	// Check if we should restore immediately
	checkpointPath, shouldRestore = config.ShouldRestore(cfg, log)

	// If not available yet, wait indefinitely for a checkpoint to appear
	if !shouldRestore {
		log.Info("Waiting for checkpoint...")
		var err error
		checkpointPath, err = config.WaitForCheckpoint(ctx, cfg, log)
		if err != nil {
			log.WithError(err).Info("No checkpoint received, running default command")
			return RunDefault(cfg, log)
		}
		shouldRestore = true
	}

	// Perform restore
	log.WithField("checkpoint", checkpointPath).Info("Checkpoint available, starting restore")
	restoreStart := time.Now()

	// Apply filesystem changes
	rootfsDiffStart := time.Now()
	if err := ApplyRootfsDiff(checkpointPath, "/", log); err != nil {
		log.WithError(err).Error("Failed to apply rootfs diff")
	}
	log.WithField("duration", time.Since(rootfsDiffStart)).Info("ApplyRootfsDiff completed")

	deletedFilesStart := time.Now()
	if err := ApplyDeletedFiles(checkpointPath, "/", log); err != nil {
		log.WithError(err).Error("Failed to apply deleted files")
	}
	log.WithField("duration", time.Since(deletedFilesStart)).Info("ApplyDeletedFiles completed")

	// Load checkpoint data (contains CRIU config + mounts + namespaces)
	// This is required - no fallback to defaults
	loadDataStart := time.Now()
	data, err := config.LoadCheckpointData(checkpointPath)
	if err != nil {
		log.WithError(err).Error("Failed to load checkpoint data")
		return RunDefault(cfg, log)
	}
	log.WithField("duration", time.Since(loadDataStart)).Info("LoadCheckpointData completed")

	// Log CRIU options being used (from checkpoint data)
	log.WithFields(logrus.Fields{
		"lib_dir":   data.CRIU.LibDir,
		"timeout":   data.CRIU.Timeout,
		"log_level": data.CRIU.LogLevel,
	}).Info("Using CRIU options from saved checkpoint data")

	// Write restore marker file before CRIU restore
	// This allows the restored process to detect it's been restored
	// vLLM reads DYN_RESTORE_MARKER_FILE env var which should point to this path
	restoreMarkerFile := config.RestoreMarkerFilePath
	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(restoreMarkerFile), 0755); err != nil {
		log.WithError(err).Warn("Failed to create restore marker directory")
	}
	if err := os.WriteFile(restoreMarkerFile, []byte("restored"), 0644); err != nil {
		log.WithError(err).Warn("Failed to write restore marker file")
	} else {
		log.WithField("path", restoreMarkerFile).Info("Wrote restore marker file")
	}

	// Restore /dev/shm contents before CRIU restore
	// This is critical for processes that use POSIX shared memory (e.g., Python multiprocessing)
	// The files must exist before CRIU tries to restore file descriptors pointing to them
	shmRestoreStart := time.Now()
	if err := RestoreDevShm(checkpointPath, log); err != nil {
		log.WithError(err).Error("Failed to restore /dev/shm contents - CRIU restore may fail with missing FD errors")
	}
	log.WithField("duration", time.Since(shmRestoreStart)).Info("RestoreDevShm completed")

	// Create link_remap stub files for unlinked files referenced in CRIU images
	// This handles files (e.g., semaphores) that were unlink()'d before checkpoint but still had open FDs
	linkRemapStart := time.Now()
	if err := CreateLinkRemapStubs(checkpointPath, log); err != nil {
		log.WithError(err).Warn("Failed to create link_remap stubs")
	}
	log.WithField("duration", time.Since(linkRemapStart)).Info("CreateLinkRemapStubs completed")

	// Log GPU diagnostics right before CRIU restore to track device visibility changes
	LogGPUDiagnostics("PRE-CRIU-RESTORE", log)
	LogRestoreBoundaryDiagnostics("PRE-CRIU-RESTORE", 0, log)

	// Perform CRIU restore (CUDA plugin handles CUDA state automatically)
	criuRestoreStart := time.Now()
	pid, err := Restore(ctx, checkpointPath, data, log)
	if err != nil {
		log.WithField("duration", time.Since(criuRestoreStart)).WithError(err).Error("Restore failed, falling back to default command")
		if cfg.Debug {
			log.Info("DEBUG mode: sleeping 300s to allow log collection...")
			time.Sleep(300 * time.Second)
		}
		return RunDefault(cfg, log)
	}
	criuRestoreDuration := time.Since(criuRestoreStart)
	log.WithField("duration", criuRestoreDuration).Info("CRIU Restore completed (CUDA state restored by plugin)")

	// Log GPU diagnostics AFTER restore to compare with pre-restore
	LogGPUDiagnostics("POST-RESTORE", log)
	LogRestoreBoundaryDiagnostics("POST-RESTORE", pid, log)

	totalDuration := time.Since(restoreStart)
	log.WithFields(logrus.Fields{
		"total_duration":        totalDuration,
		"criu_restore_duration": criuRestoreDuration,
	}).Info("=== Restore operation completed ===")

	// Set up signal forwarding and forward stdout/stderr from restored process
	cleanup := SetupSignalForwarding(pid, log)
	defer cleanup()

	// Use ForwardProcessOutput to ensure restored process logs appear in kubectl logs
	exitCode := ForwardProcessOutput(pid, log)
	os.Exit(exitCode)
	return nil
}
