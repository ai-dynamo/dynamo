// restore.go defines the RestoreConfig struct for CRIU restore operations.
// CRIU options come from the saved CheckpointData, not from this config.
package config

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/sirupsen/logrus"
)

// RestoreConfig holds the configuration for the restore entrypoint.
// CRIU options are NOT stored here - they come from the saved CheckpointData.
type RestoreConfig struct {
	// === Environment-only fields (dynamic per pod) ===

	// CheckpointPath is the base directory containing checkpoints (default: /checkpoints)
	// From DYN_CHECKPOINT_PATH env var.
	CheckpointPath string `yaml:"-"`

	// CheckpointHash is the ID/hash of the checkpoint to restore.
	// From DYN_CHECKPOINT_HASH env var.
	CheckpointHash string `yaml:"-"`

	// StorageType is the checkpoint storage type (pvc, s3, oci).
	// From DYN_CHECKPOINT_STORAGE_TYPE env var.
	StorageType string `yaml:"-"`

	// Location is the checkpoint location (varies by storage type).
	// From DYN_CHECKPOINT_LOCATION env var.
	Location string `yaml:"-"`

	// Debug enables debug logging.
	// From DEBUG env var.
	Debug bool `yaml:"-"`

	// WaitForCheckpoint indicates whether to wait for a checkpoint to appear.
	// From WAIT_FOR_CHECKPOINT env var.
	WaitForCheckpoint bool `yaml:"-"`

	// === Static fields (from ConfigMap) ===

	// RestoreTrigger is the path to the trigger file that signals restore should start.
	RestoreTrigger string `yaml:"restoreTrigger"`

	// WaitTimeout is the maximum time to wait for a checkpoint to become available.
	WaitTimeout time.Duration `yaml:"waitTimeout"`

	// DefaultCmd is the command to run if no checkpoint is available.
	DefaultCmd string `yaml:"defaultCmd"`

	// NOTE: CRIU options are NOT stored here.
	// They come from the saved CheckpointData at restore time.
}

// LoadRestoreEnvOverrides applies environment variable overrides to RestoreConfig.
// This is called after loading the base config from YAML.
func (c *RestoreConfig) LoadRestoreEnvOverrides() {
	// Dynamic values from operator/environment
	if v := os.Getenv("DYN_CHECKPOINT_PATH"); v != "" {
		c.CheckpointPath = v
	} else if c.CheckpointPath == "" {
		c.CheckpointPath = "/checkpoints"
	}

	if v := os.Getenv("DYN_CHECKPOINT_HASH"); v != "" {
		c.CheckpointHash = v
	}

	if v := os.Getenv("DYN_CHECKPOINT_STORAGE_TYPE"); v != "" {
		c.StorageType = v
	}

	if v := os.Getenv("DYN_CHECKPOINT_LOCATION"); v != "" {
		c.Location = v
	}

	c.Debug = os.Getenv("DEBUG") == "1"
	c.WaitForCheckpoint = os.Getenv("WAIT_FOR_CHECKPOINT") == "1"
}

// LoadRestoreConfig creates a RestoreConfig from ConfigMap and environment variables.
// If configPath is empty, uses environment variables only (for backwards compatibility).
// Returns an error if the config file exists but cannot be parsed.
func LoadRestoreConfig(configPath string) (*RestoreConfig, error) {
	var cfg *RestoreConfig

	if configPath != "" {
		fullCfg, err := LoadConfig(configPath)
		if err != nil {
			// Check if the error is "file not found" (acceptable) vs parse error (should be surfaced)
			if os.IsNotExist(err) {
				// File not found is acceptable - fall back to zero config
				cfg = &RestoreConfig{}
			} else {
				// Parse errors and other errors should be surfaced
				return nil, fmt.Errorf("failed to load restore config: %w", err)
			}
		} else {
			cfg = &fullCfg.Restore
		}
	} else {
		// No config path - use zero config
		// All defaults should come from ConfigMap (values.yaml)
		cfg = &RestoreConfig{}
	}

	// Apply environment overrides
	cfg.LoadRestoreEnvOverrides()

	// Legacy environment variable overrides for backwards compatibility
	// These override ConfigMap values if set
	if v := os.Getenv("RESTORE_TRIGGER"); v != "" {
		cfg.RestoreTrigger = v
	}
	if v := os.Getenv("RESTORE_WAIT_TIMEOUT"); v != "" {
		if seconds, err := strconv.Atoi(v); err == nil {
			cfg.WaitTimeout = time.Duration(seconds) * time.Second
		}
	}
	if v := os.Getenv("DEFAULT_CMD"); v != "" {
		cfg.DefaultCmd = v
	}

	return cfg, nil
}

// ShouldRestore checks if a restore should be performed.
// Returns the checkpoint path and true if restore should proceed.
// IMPORTANT: We check for checkpoint.done marker (not just metadata.yaml or inventory.img) because
// checkpoint.done is written LAST in the checkpoint process, after rootfs-diff.tar completes.
// Order: metadata.yaml -> CRIU dump (*.img files) -> rootfs-diff.tar -> checkpoint.done
func ShouldRestore(cfg *RestoreConfig, log *logrus.Entry) (string, bool) {
	// Method 1: DYN_CHECKPOINT_HASH is set and checkpoint is fully complete
	if cfg.CheckpointHash != "" {
		checkpointPath := cfg.CheckpointPath + "/" + cfg.CheckpointHash
		// Check for checkpoint.done marker (written LAST after rootfs-diff.tar completes)
		donePath := checkpointPath + "/checkpoint.done"

		if _, err := os.Stat(donePath); err == nil {
			log.WithField("path", checkpointPath).Info("Checkpoint found (checkpoint.done marker present)")
			return checkpointPath, true
		}

		// Fallback: check for metadata.yaml but warn about potential race condition
		metadataPath := checkpointPath + "/" + CheckpointDataFilename
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
func WaitForCheckpoint(ctx context.Context, cfg *RestoreConfig, log *logrus.Entry) (string, error) {
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

// Validate checks that the RestoreConfig has valid values.
func (c *RestoreConfig) Validate() error {
	// RestoreConfig no longer contains CRIU options - nothing to validate
	return nil
}
