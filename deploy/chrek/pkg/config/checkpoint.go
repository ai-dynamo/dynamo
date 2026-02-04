// checkpoint.go defines the CheckpointConfig struct for CRIU checkpoint operations.
package config

import "os"

// CheckpointConfig holds static checkpoint settings that don't change per-checkpoint.
// This includes CRIU options and exclusion lists.
type CheckpointConfig struct {
	// CRIU options for dump operations
	CRIU CRIUConfig `yaml:"criu"`

	// RootfsExclusions defines paths to exclude from rootfs diff capture
	RootfsExclusions RootfsExclusionConfig `yaml:"rootfsExclusions"`

	// SkipMountPrefixes is a list of directory prefixes. All mount points under these
	// directories will be skipped during checkpoint. This allows cross-node restore
	// when certain mounts (e.g., nvidia runtime mounts) don't exist on the target node.
	// Example: ["/run/nvidia/driver"] skips all mounts like:
	//   - /run/nvidia/driver/lib/firmware/nvidia/580.82.07/gsp_tu10x.bin
	//   - /run/nvidia/driver/lib/firmware/nvidia/580.82.07/gsp_ga10x.bin
	// The actual mount paths are enumerated at checkpoint time and passed to CRIU.SkipMounts.
	SkipMountPrefixes []string `yaml:"skipMountPrefixes"`

	// === Environment-only fields (read by vLLM directly) ===
	// These fields are populated from environment variables because vLLM
	// reads these env vars directly to know where to signal checkpoint completion.

	// SignalFile is the path where checkpoint completion signal is written.
	// From DYN_CHECKPOINT_SIGNAL_FILE env var.
	SignalFile string `yaml:"-"`

	// ReadyFile is the path where checkpoint ready signal is written.
	// From DYN_CHECKPOINT_READY_FILE env var.
	ReadyFile string `yaml:"-"`
}

// LoadCheckpointEnvOverrides applies environment variable overrides to CheckpointConfig.
// This is called after loading the base config from YAML.
func (c *CheckpointConfig) LoadCheckpointEnvOverrides() {
	// Signal files - vLLM reads these env vars directly
	if v := os.Getenv("DYN_CHECKPOINT_SIGNAL_FILE"); v != "" {
		c.SignalFile = v
	}
	if v := os.Getenv("DYN_CHECKPOINT_READY_FILE"); v != "" {
		c.ReadyFile = v
	}
}

// Validate checks that the CheckpointConfig has valid values.
func (c *CheckpointConfig) Validate() error {
	if err := c.CRIU.Validate(); err != nil {
		return err
	}
	if err := c.RootfsExclusions.Validate(); err != nil {
		return err
	}
	return nil
}

// GetRootfsExclusions returns all exclusion paths for rootfs diff capture.
// This includes system dirs, cache dirs, and additional exclusions.
func (c *CheckpointConfig) GetRootfsExclusions() []string {
	return c.RootfsExclusions.GetAllExclusions()
}
