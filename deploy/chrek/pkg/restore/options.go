// Package restore provides CRIU restore operations for self-restoring placeholder containers.
package restore

import (
	"context"
	"os"
	"strconv"
	"time"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"github.com/sirupsen/logrus"
	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/common"
)

// Config holds the configuration for the restore entrypoint.
// These values are typically set via environment variables.
type Config struct {
	// CheckpointPath is the base directory containing checkpoints (default: /checkpoints)
	// Env: DYN_CHECKPOINT_PATH
	CheckpointPath string

	// CheckpointHash is the ID/hash of the checkpoint to restore
	// Env: DYN_CHECKPOINT_HASH
	CheckpointHash string

	// RestoreTrigger is the path to the trigger file that signals restore should start
	RestoreTrigger string

	// WaitForCheckpoint indicates whether to wait for a checkpoint to appear
	WaitForCheckpoint bool

	// WaitTimeout is the maximum time to wait for a checkpoint to become available
	WaitTimeout time.Duration

	// CRIULogLevel is the CRIU verbosity level (0-4, default: 4)
	CRIULogLevel int32

	// DefaultCmd is the command to run if no checkpoint is available
	DefaultCmd string

	// Debug enables debug logging
	Debug bool

	// EmbeddedCheckpointPath is the path to an embedded checkpoint within the image
	// When set, the checkpoint data is baked into the container image itself
	EmbeddedCheckpointPath string

	// SkipInFlightConnections skips in-flight TCP connections during restore
	SkipInFlightConnections bool

	// AutoDedup enables auto-deduplication of memory pages
	AutoDedup bool

	// LazyPages enables lazy page migration (experimental)
	LazyPages bool

	// CRIUWorkDir is an alternative work directory for CRIU (instead of /tmp)
	// Useful when /tmp has mount issues
	CRIUWorkDir string

	// CUDAPluginDir is the path to CRIU CUDA plugin directory (e.g., /usr/local/lib/criu)
	// When set, a CRIU config file is created with libdir for CUDA plugin discovery during restore.
	CUDAPluginDir string

	// CRIUTimeout is the CRIU timeout in seconds (required for CUDA restores)
	CRIUTimeout uint32

	// RestoreMarkerFile is the path to a marker file created before CRIU restore.
	// The restored process can check for this file to detect it was restored.
	RestoreMarkerFile string
}

// DefaultEmbeddedCheckpointPath is the default path for embedded checkpoints
const DefaultEmbeddedCheckpointPath = "/embedded-checkpoint"

// ConfigFromEnv creates a Config from environment variables.
func ConfigFromEnv() *Config {
	cfg := &Config{
		CheckpointPath:          getEnvOrDefault("DYN_CHECKPOINT_PATH", "/checkpoints"),
		CheckpointHash:          os.Getenv("DYN_CHECKPOINT_HASH"),
		RestoreTrigger:          getEnvOrDefault("RESTORE_TRIGGER", "/tmp/restore-trigger"),
		WaitForCheckpoint:       os.Getenv("WAIT_FOR_CHECKPOINT") == "1",
		WaitTimeout:             parseDurationOrDefault("RESTORE_WAIT_TIMEOUT", 300*time.Second),
		CRIULogLevel:            parseIntOrDefault("CRIU_LOG_LEVEL", 4),
		DefaultCmd:              os.Getenv("DEFAULT_CMD"),
		Debug:                   os.Getenv("DEBUG") == "1",
		EmbeddedCheckpointPath:  getEnvOrDefault("EMBEDDED_CHECKPOINT_PATH", DefaultEmbeddedCheckpointPath),
		SkipInFlightConnections: os.Getenv("CRIU_SKIP_IN_FLIGHT") == "1",
		AutoDedup:               os.Getenv("CRIU_AUTO_DEDUP") == "1",
		LazyPages:               os.Getenv("CRIU_LAZY_PAGES") == "1",
		CRIUWorkDir:             getEnvOrDefault("CRIU_WORK_DIR", ""),
		CUDAPluginDir:           os.Getenv("CUDA_PLUGIN_DIR"), // For CUDA plugin discovery during restore
		CRIUTimeout:             uint32(parseIntOrDefault("CRIU_TIMEOUT", 0)),
		RestoreMarkerFile:       getEnvOrDefault("DYN_RESTORE_MARKER_FILE", "/tmp/dynamo-restored"),
	}
	return cfg
}

// RestoreOptions holds the options for a CRIU restore operation.
// Most CRIU options are hardcoded with safe K8s defaults.
type RestoreOptions struct {
	// CheckpointPath is the path to the checkpoint directory
	CheckpointPath string

	// RootPath is the root filesystem path for restore (typically "/")
	RootPath string

	// PidFile is the path where CRIU writes the restored process PID
	PidFile string

	// LogFile is the name of the CRIU restore log file
	LogFile string

	// LogLevel is the CRIU logging verbosity (0-4)
	LogLevel int32

	// ExtMountMaps contains external mount mappings for CRIU
	ExtMountMaps []*criurpc.ExtMountMap

	// WorkDir is an alternative work directory for CRIU (instead of /tmp)
	WorkDir string

	// LibDir is the path to CRIU plugin directory (e.g., /usr/local/lib/criu)
	// When set, a CRIU config file is created with libdir for CUDA plugin discovery.
	LibDir string

	// Timeout is the CRIU timeout in seconds (required for CUDA restores)
	Timeout uint32
}

// DefaultRestoreOptions returns RestoreOptions with sensible defaults.
func DefaultRestoreOptions(checkpointPath string) *RestoreOptions {
	return &RestoreOptions{
		CheckpointPath: checkpointPath,
		RootPath:       "/",
		PidFile:        "/tmp/restored.pid",
		LogFile:        "restore.log",
		LogLevel:       4,
	}
}

// LoadRestoreOptions creates RestoreOptions from checkpoint metadata.
// CRIU options are hardcoded with safe K8s defaults; metadata is only used for mount mappings.
func LoadRestoreOptions(checkpointPath string, logLevel int32) (*RestoreOptions, error) {
	opts := DefaultRestoreOptions(checkpointPath)
	opts.LogLevel = logLevel

	// Load metadata for OCI-derived paths (masked/readonly paths for external mounts)
	meta, err := common.LoadMetadata(checkpointPath)
	if err != nil {
		// Return defaults if metadata is unavailable
		// GenerateExtMountMaps with nil will use fallback defaults
		return opts, nil
	}

	// Pre-generate external mount maps using OCI-derived paths from metadata
	// This uses masked/readonly paths from the OCI spec instead of hardcoded defaults
	extMounts, err := GenerateExtMountMaps(meta)
	if err != nil {
		// Fall back to defaults if generation fails
		return opts, nil
	}
	opts.ExtMountMaps = extMounts

	return opts, nil
}

// ShouldRestore checks if a restore should be performed.
// Returns the checkpoint path and true if restore should proceed.
// IMPORTANT: We check for checkpoint.done marker (not just metadata.json or inventory.img) because
// checkpoint.done is written LAST in the checkpoint process, after rootfs-diff.tar completes.
// Order: metadata.json -> CRIU dump (*.img files) -> rootfs-diff.tar -> checkpoint.done
func ShouldRestore(cfg *Config, log *logrus.Entry) (string, bool) {
	// Method 0: Embedded checkpoint in image (highest priority)
	// This is for self-contained checkpoint images where data is baked in
	if cfg.EmbeddedCheckpointPath != "" {
		metadataPath := cfg.EmbeddedCheckpointPath + "/" + common.MetadataFilename
		if _, err := os.Stat(metadataPath); err == nil {
			log.WithField("path", cfg.EmbeddedCheckpointPath).Info("Embedded checkpoint found in image")
			return cfg.EmbeddedCheckpointPath, true
		}
	}

	// Method 1: DYN_CHECKPOINT_HASH is set and checkpoint is fully complete
	if cfg.CheckpointHash != "" {
		checkpointPath := cfg.CheckpointPath + "/" + cfg.CheckpointHash
		// Check for checkpoint.done marker (written LAST after rootfs-diff.tar completes)
		donePath := checkpointPath + "/checkpoint.done"

		if _, err := os.Stat(donePath); err == nil {
			log.WithField("path", checkpointPath).Info("Checkpoint found (checkpoint.done marker present)")
			return checkpointPath, true
		}

		// Fallback: check for metadata.json but warn about potential race condition
		metadataPath := checkpointPath + "/" + common.MetadataFilename
		if _, err := os.Stat(metadataPath); err == nil {
			log.WithFields(logrus.Fields{
				"path":    checkpointPath,
				"warning": "checkpoint.done marker not found, checkpoint may be incomplete",
			}).Warn("Checkpoint metadata found but checkpoint.done missing - checkpoint may still be in progress")
			// Don't return true here - wait for checkpoint.done
		}
	}

	// Method 2: Restore trigger file exists with checkpoint path
	if cfg.RestoreTrigger != "" {
		data, err := os.ReadFile(cfg.RestoreTrigger)
		if err == nil {
			checkpointPath := string(data)
			if checkpointPath != "" {
				donePath := checkpointPath + "/checkpoint.done"
				if _, err := os.Stat(donePath); err == nil {
					log.WithField("path", checkpointPath).Info("Restore triggered via file (checkpoint.done marker present)")
					return checkpointPath, true
				}
			}
		}
	}

	return "", false
}

// WaitForCheckpoint waits for a checkpoint to become available.
func WaitForCheckpoint(ctx context.Context, cfg *Config, log *logrus.Entry) (string, error) {
	log.WithField("timeout", cfg.WaitTimeout).Info("Waiting for checkpoint")

	deadline := time.Now().Add(cfg.WaitTimeout)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	lastLog := time.Now()

	for {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-ticker.C:
			if path, ok := ShouldRestore(cfg, log); ok {
				return path, nil
			}

			// Log progress every 30 seconds
			if time.Since(lastLog) >= 30*time.Second {
				elapsed := time.Since(deadline.Add(-cfg.WaitTimeout))
				log.WithField("elapsed", elapsed).Info("Still waiting for checkpoint...")
				lastLog = time.Now()
			}

			if time.Now().After(deadline) {
				return "", context.DeadlineExceeded
			}
		}
	}
}

// Helper functions

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func parseDurationOrDefault(key string, defaultValue time.Duration) time.Duration {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	seconds, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	return time.Duration(seconds) * time.Second
}

func parseIntOrDefault(key string, defaultValue int32) int32 {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	i, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	return int32(i)
}
