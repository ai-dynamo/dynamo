// config.go defines the static checkpoint spec loaded from ConfigMap YAML.
package checkpoint

import "fmt"

// CheckpointSpec is the static checkpoint spec loaded from ConfigMap YAML.
type CheckpointSpec struct {
	// BasePath is the base directory for checkpoint storage (PVC mount point).
	BasePath string `yaml:"basePath"`

	// CRIU options for dump operations
	CRIU CRIUSettings `yaml:"criu"`

	// RootfsExclusions defines paths to exclude from rootfs diff capture
	RootfsExclusions FilesystemConfig `yaml:"rootfsExclusions"`
}

// Validate checks that the CheckpointSpec has valid values.
func (c *CheckpointSpec) Validate() error {
	return c.RootfsExclusions.Validate()
}

// ConfigError represents a configuration validation error.
type ConfigError struct {
	Field   string
	Message string
}

func (e *ConfigError) Error() string {
	return fmt.Sprintf("config error: %s: %s", e.Field, e.Message)
}
