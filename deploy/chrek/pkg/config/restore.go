// restore.go defines the RestoreConfig struct for CRIU restore operations.
// CRIU options come from the saved CheckpointData, not from this config.
//
// The restore-entrypoint runs in placeholder containers which do NOT mount the
// ConfigMap. Static defaults are hardcoded here; per-pod dynamic values come
// from environment variables injected by the operator.
package config

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// RestoreConfig holds the configuration for the restore entrypoint.
// CRIU options are NOT stored here - they come from the saved CheckpointData.
type RestoreConfig struct {
	// === Per-pod dynamic values (from operator-injected env vars) ===

	// CheckpointPath is the base directory containing checkpoints.
	// From DYN_CHECKPOINT_PATH env var (operator-injected).
	CheckpointPath string

	// CheckpointHash is the ID/hash of the checkpoint to restore.
	// From DYN_CHECKPOINT_HASH env var (operator-injected).
	CheckpointHash string

	// CheckpointLocation is the full resolved path to the checkpoint directory.
	// Computed as CheckpointPath + "/" + CheckpointHash for PVC storage.
	// For future S3/OCI support, DYN_CHECKPOINT_LOCATION takes priority if set.
	CheckpointLocation string

	// SkipWaitForCheckpoint controls the entrypoint behavior.
	// When true (operator sets SKIP_WAIT_FOR_CHECKPOINT=1): check once, restore
	// if ready, otherwise exec ColdStartArgs (cold start). No polling.
	// When false (default, standalone): poll/wait for checkpoint.done to appear.
	SkipWaitForCheckpoint bool

	// ColdStartArgs is the command+args to exec if no checkpoint is available
	// and SkipWaitForCheckpoint is true. Populated from os.Args[1:].
	ColdStartArgs []string

	// Debug enables debug logging.
	// From DEBUG env var.
	Debug bool

	// === Static defaults (hardcoded) ===

	// RestoreMarkerFilePath is where restore-entrypoint writes a marker before CRIU restore.
	// From DYN_RESTORE_MARKER_FILE env var.
	RestoreMarkerFilePath string

	// RestoreTrigger is the path to the trigger file that signals restore should start.
	RestoreTrigger string

	// WaitTimeout is the maximum time to wait for a checkpoint.
	// Zero means wait indefinitely.
	WaitTimeout time.Duration
}

// NewRestoreConfig creates a RestoreConfig with hardcoded defaults and
// operator-injected environment variable values.
func NewRestoreConfig(args []string) (*RestoreConfig, error) {
	cfg := &RestoreConfig{
		// Static defaults
		RestoreTrigger: RestoreTriggerPath,
		ColdStartArgs:  args,
	}

	// Per-pod dynamic values from operator-injected env vars
	if v := os.Getenv("DYN_CHECKPOINT_PATH"); v != "" {
		cfg.CheckpointPath = v
	}
	if v := os.Getenv("DYN_CHECKPOINT_HASH"); v != "" {
		cfg.CheckpointHash = v
	}

	// Compute checkpoint location.
	// DYN_CHECKPOINT_LOCATION takes priority (future S3/OCI support).
	// Otherwise construct from PATH + "/" + HASH for PVC storage.
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
// IMPORTANT: We check for checkpoint.done marker (not just metadata.yaml or inventory.img) because
// checkpoint.done is written LAST in the checkpoint process, after rootfs-diff.tar completes.
// Order: metadata.yaml -> CRIU dump (*.img files) -> rootfs-diff.tar -> checkpoint.done
func ShouldRestore(cfg *RestoreConfig, log *logrus.Entry) (string, bool) {
	// Method 1: Checkpoint location is set and checkpoint is fully complete
	if cfg.CheckpointLocation != "" {
		// Check for checkpoint.done marker (written LAST after rootfs-diff.tar completes)
		donePath := cfg.CheckpointLocation + "/" + CheckpointDoneFilename

		if _, err := os.Stat(donePath); err == nil {
			log.WithField("path", cfg.CheckpointLocation).Info("Checkpoint found (checkpoint.done marker present)")
			return cfg.CheckpointLocation, true
		}

		// Fallback: check for metadata.yaml but warn about potential race condition
		metadataPath := cfg.CheckpointLocation + "/" + CheckpointDataFilename
		if _, err := os.Stat(metadataPath); err == nil {
			log.WithFields(logrus.Fields{
				"path":    cfg.CheckpointLocation,
				"warning": "checkpoint.done marker not found, checkpoint may be incomplete",
			}).Warn("Checkpoint metadata found but checkpoint.done missing - checkpoint may still be in progress")
			// Don't return true here - wait for checkpoint.done
		}
	}

	// Method 2: Restore trigger file exists with checkpoint path
	if cfg.RestoreTrigger != "" {
		data, err := os.ReadFile(cfg.RestoreTrigger)
		if err == nil {
			checkpointPath := strings.TrimSpace(string(data))
			if checkpointPath != "" {
				donePath := checkpointPath + "/" + CheckpointDoneFilename
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
