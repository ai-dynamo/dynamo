package restore

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	criu "github.com/checkpoint-restore/go-criu/v7"
	"github.com/sirupsen/logrus"
	"google.golang.org/protobuf/proto"
)

// Restore performs the CRIU restore operation using go-criu.
// Returns the PID of the restored process.
func Restore(ctx context.Context, opts *RestoreOptions, log *logrus.Entry) (int, error) {
	log.WithField("checkpoint", opts.CheckpointPath).Info("Starting CRIU restore")

	// 1. Open checkpoint directory
	imageDir, imageDirFD, err := OpenImageDir(opts.CheckpointPath)
	if err != nil {
		return 0, err
	}
	defer imageDir.Close()
	log.WithField("fd", imageDirFD).Debug("Opened checkpoint directory")

	// 2. Generate external mount mappings if not already set
	if opts.ExtMountMaps == nil {
		extMounts, err := GenerateExtMountMaps(nil)
		if err != nil {
			return 0, fmt.Errorf("failed to generate mount maps: %w", err)
		}
		opts.ExtMountMaps = extMounts
	}
	log.WithField("mount_count", len(opts.ExtMountMaps)).Debug("External mount maps ready")

	// 3. Open target network namespace
	netNsFile, netNsFD, err := OpenNetworkNamespace("/proc/1/ns/net")
	if err != nil {
		return 0, err
	}
	defer netNsFile.Close()
	log.WithField("fd", netNsFD).Debug("Opened target network namespace")

	// 4. Open work directory if specified
	var workDirFile *os.File
	var workDirFD int32 = -1
	if opts.WorkDir != "" {
		workDirFile, workDirFD = OpenWorkDir(opts.WorkDir, log)
		if workDirFile != nil {
			defer workDirFile.Close()
		}
	}

	// 5. Build CRIU options
	cfg := CRIURestoreConfig{
		ImageDirFD:   imageDirFD,
		RootPath:     opts.RootPath,
		LogLevel:     opts.LogLevel,
		LogFile:      opts.LogFile,
		WorkDirFD:    workDirFD,
		NetNsFD:      netNsFD,
		ExtMountMaps: opts.ExtMountMaps,
	}
	criuOpts := BuildRestoreCRIUOpts(cfg)

	// 6. Create CRIU config file for CUDA plugin if libdir is specified
	if opts.LibDir != "" {
		if opts.Timeout == 0 {
			return 0, fmt.Errorf("CRIU_TIMEOUT environment variable must be set for CUDA restores")
		}
		configPath := filepath.Join(opts.CheckpointPath, "restore-criu.conf")
		configContent := fmt.Sprintf(`enable-external-masters
libdir %s
tcp-close
link-remap
timeout %d
allow-uprobes
skip-in-flight
`, opts.LibDir, opts.Timeout)
		if err := os.WriteFile(configPath, []byte(configContent), 0644); err != nil {
			log.WithError(err).Warn("Failed to write CRIU config file for restore")
		} else {
			criuOpts.ConfigFile = proto.String(configPath)
			log.WithFields(logrus.Fields{
				"config_path": configPath,
				"lib_dir":     opts.LibDir,
			}).Info("Created CRIU config file with libdir for CUDA plugin")
		}
	}

	// 7. Execute CRIU restore
	c := criu.MakeCriu()
	notify := NewRestoreNotify(log)

	log.Info("Executing CRIU restore")
	criuExecStart := time.Now()
	if err := c.Restore(criuOpts, notify); err != nil {
		log.WithField("duration", time.Since(criuExecStart)).Error("CRIU c.Restore failed")
		logCRIUErrors(opts.CheckpointPath, opts.LogFile, log)
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
	if opts.PidFile != "" {
		pid, err := WaitForPidFile(opts.PidFile, 10*time.Second, log)
		if err != nil {
			return 0, fmt.Errorf("failed to get restored PID: %w", err)
		}
		return pid, nil
	}

	return 0, fmt.Errorf("could not determine restored process PID")
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

	// Copy log to shared directory if CRIU_LOG_DIR is set
	if logDir := os.Getenv("CRIU_LOG_DIR"); logDir != "" {
		if err := os.MkdirAll(logDir, 0755); err == nil {
			destPath := filepath.Join(logDir, fmt.Sprintf("restore-%d.log", time.Now().Unix()))
			if err := os.WriteFile(destPath, data, 0644); err == nil {
				log.WithField("path", destPath).Info("CRIU log copied to shared directory")
			}
		}
	}
}

// Run is the main entry point for the restore entrypoint.
// It orchestrates the entire restore process.
func Run(ctx context.Context, cfg *Config, log *logrus.Entry) error {
	log.Info("=== Self-Restoring Placeholder Entrypoint ===")
	log.WithFields(logrus.Fields{
		"checkpoint_path":          cfg.CheckpointPath,
		"checkpoint_hash":          cfg.CheckpointHash,
		"embedded_checkpoint_path": cfg.EmbeddedCheckpointPath,
		"wait_for_checkpoint":      cfg.WaitForCheckpoint,
		"restore_marker_file":      cfg.RestoreMarkerFile,
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
	checkpointPath, shouldRestore = ShouldRestore(cfg, log)

	// If not and we're configured to wait, wait for checkpoint
	if !shouldRestore && cfg.WaitForCheckpoint {
		log.Info("Waiting for checkpoint...")
		var err error
		checkpointPath, err = WaitForCheckpoint(ctx, cfg, log)
		if err != nil {
			log.WithError(err).Info("No checkpoint received, running default command")
			return RunDefault(cfg, log)
		}
		shouldRestore = true
	}

	// If no checkpoint, run default command
	if !shouldRestore {
		log.Info("No checkpoint configured, running default command")
		return RunDefault(cfg, log)
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

	// Load restore options from metadata
	loadOptsStart := time.Now()
	opts, err := LoadRestoreOptions(checkpointPath, cfg.CRIULogLevel)
	if err != nil {
		log.WithError(err).Warn("Could not load restore options from metadata, using defaults")
	}
	log.WithField("duration", time.Since(loadOptsStart)).Info("LoadRestoreOptions completed")

	// Apply additional config options
	if cfg.CRIUWorkDir != "" {
		opts.WorkDir = cfg.CRIUWorkDir
	}

	// Set CUDA plugin directory and timeout for restore config file
	if cfg.CUDAPluginDir != "" {
		if cfg.CRIUTimeout == 0 {
			return fmt.Errorf("CRIU_TIMEOUT environment variable must be set for CUDA restores")
		}
		opts.LibDir = cfg.CUDAPluginDir
		opts.Timeout = cfg.CRIUTimeout
		log.WithFields(logrus.Fields{
			"lib_dir": cfg.CUDAPluginDir,
			"timeout": cfg.CRIUTimeout,
		}).Info("CUDA plugin directory and timeout configured for restore")
	}

	// Write restore marker file before CRIU restore
	// This allows the restored process to detect it's been restored
	if cfg.RestoreMarkerFile != "" {
		if err := os.WriteFile(cfg.RestoreMarkerFile, []byte("restored"), 0644); err != nil {
			log.WithError(err).Warn("Failed to write restore marker file")
		} else {
			log.WithField("path", cfg.RestoreMarkerFile).Info("Wrote restore marker file")
		}
	}

	// Perform CRIU restore (CUDA plugin handles CUDA state automatically)
	criuRestoreStart := time.Now()
	pid, err := Restore(ctx, opts, log)
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
