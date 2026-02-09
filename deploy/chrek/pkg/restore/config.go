// config.go defines the RestoreConfig struct for CRIU restore operations.
// CRIU options come from the saved CheckpointMetadata, not from this config.
//
// The restore-entrypoint runs in placeholder containers which do NOT mount the
// ConfigMap. Static defaults are hardcoded here; per-pod dynamic values come
// from environment variables injected by the operator.
package restore

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sirupsen/logrus"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

const (
	// RestoreLogFilename is the CRIU restore log filename.
	RestoreLogFilename = "restore.log"

	// CRIULogDir is the directory where CRIU restore logs are copied for debugging.
	CRIULogDir = "/checkpoints/restore-logs"

	// RestoreTriggerPath is the default path to the trigger file for trigger-based restore.
	RestoreTriggerPath = "/tmp/restore-trigger"
)

// RestoreConfig holds the configuration for the restore entrypoint.
// CRIU options are NOT stored here - they come from the saved CheckpointMetadata.
type RestoreConfig struct {
	// === Per-pod dynamic values (from operator-injected env vars) ===

	// CheckpointPath is the base directory containing checkpoints.
	CheckpointPath string

	// CheckpointHash is the ID/hash of the checkpoint to restore.
	CheckpointHash string

	// CheckpointLocation is the full resolved path to the checkpoint directory.
	CheckpointLocation string

	// SkipWaitForCheckpoint controls the entrypoint behavior.
	SkipWaitForCheckpoint bool

	// ColdStartArgs is the command+args to exec if no checkpoint is available.
	ColdStartArgs []string

	// Debug enables debug logging.
	Debug bool

	// === Static defaults (hardcoded) ===

	// RestoreMarkerFilePath is where restore-entrypoint writes a marker before CRIU restore.
	RestoreMarkerFilePath string

	// RestoreTrigger is the path to the trigger file that signals restore should start.
	RestoreTrigger string

	// WaitTimeout is the maximum time to wait for a checkpoint.
	// Zero means wait indefinitely.
	WaitTimeout time.Duration
}

// ConfigError represents a configuration validation error.
type ConfigError struct {
	Field   string
	Message string
}

func (e *ConfigError) Error() string {
	return fmt.Sprintf("config error: %s: %s", e.Field, e.Message)
}

// NewRestoreConfig creates a RestoreConfig with hardcoded defaults and
// operator-injected environment variable values.
func NewRestoreConfig(args []string) (*RestoreConfig, error) {
	cfg := &RestoreConfig{
		RestoreTrigger: RestoreTriggerPath,
		ColdStartArgs:  args,
	}

	if v := os.Getenv("DYN_CHECKPOINT_PATH"); v != "" {
		cfg.CheckpointPath = v
	}
	if v := os.Getenv("DYN_CHECKPOINT_HASH"); v != "" {
		cfg.CheckpointHash = v
	}

	if v := os.Getenv("DYN_CHECKPOINT_LOCATION"); v != "" {
		cfg.CheckpointLocation = v
	} else if cfg.CheckpointPath != "" && cfg.CheckpointHash != "" {
		cfg.CheckpointLocation = cfg.CheckpointPath + "/" + cfg.CheckpointHash
	}

	cfg.SkipWaitForCheckpoint = os.Getenv("SKIP_WAIT_FOR_CHECKPOINT") == "1"
	cfg.Debug = os.Getenv("DEBUG") == "1"
	cfg.RestoreMarkerFilePath = os.Getenv("DYN_RESTORE_MARKER_FILE")
	if cfg.RestoreMarkerFilePath == "" {
		return nil, &ConfigError{
			Field:   "DYN_RESTORE_MARKER_FILE",
			Message: "must be set",
		}
	}

	return cfg, nil
}

// ShouldRestore checks if a restore should be performed.
// Returns the checkpoint path and true if restore should proceed.
func ShouldRestore(cfg *RestoreConfig, log *logrus.Entry) (string, bool) {
	// Method 1: Checkpoint location is set and checkpoint is fully complete
	if cfg.CheckpointLocation != "" {
		donePath := cfg.CheckpointLocation + "/" + checkpoint.CheckpointDoneFilename

		if _, err := os.Stat(donePath); err == nil {
			log.WithField("path", cfg.CheckpointLocation).Info("Checkpoint found (checkpoint.done marker present)")
			return cfg.CheckpointLocation, true
		}

		// Fallback: check for metadata.yaml but warn about potential race condition
		metadataPath := cfg.CheckpointLocation + "/" + checkpoint.CheckpointDataFilename
		if _, err := os.Stat(metadataPath); err == nil {
			log.WithFields(logrus.Fields{
				"path":    cfg.CheckpointLocation,
				"warning": "checkpoint.done marker not found, checkpoint may be incomplete",
			}).Warn("Checkpoint metadata found but checkpoint.done missing - checkpoint may still be in progress")
		}
	}

	// Method 2: Restore trigger file exists with checkpoint path
	if cfg.RestoreTrigger != "" {
		data, err := os.ReadFile(cfg.RestoreTrigger)
		if err == nil {
			checkpointPath := strings.TrimSpace(string(data))
			if checkpointPath != "" {
				donePath := checkpointPath + "/" + checkpoint.CheckpointDoneFilename
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
// If cfg.WaitTimeout is zero, waits indefinitely (until ctx is cancelled).
func WaitForCheckpoint(ctx context.Context, cfg *RestoreConfig, log *logrus.Entry) (string, error) {
	if cfg.WaitTimeout > 0 {
		log.WithField("timeout", cfg.WaitTimeout).Info("Waiting for checkpoint")
	} else {
		log.Info("Waiting for checkpoint indefinitely")
	}

	startTime := time.Now()
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
				elapsed := time.Since(startTime)
				log.WithField("elapsed", elapsed).Info("Still waiting for checkpoint...")
				lastLog = time.Now()
			}

			// Only enforce deadline if WaitTimeout is set (non-zero)
			if cfg.WaitTimeout > 0 && time.Since(startTime) >= cfg.WaitTimeout {
				return "", fmt.Errorf("timed out waiting for checkpoint after %s", cfg.WaitTimeout)
			}
		}
	}
}
