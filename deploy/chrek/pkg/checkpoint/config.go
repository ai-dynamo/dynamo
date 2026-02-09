// config.go defines the static checkpoint configuration loaded from ConfigMap YAML.
package checkpoint

import "fmt"

// Config is the static checkpoint configuration loaded from ConfigMap YAML.
type Config struct {
	// BasePath is the base directory for checkpoint storage (PVC mount point).
	BasePath string `yaml:"basePath"`

	// CRIU options for dump operations
	CRIU CRIUConfig `yaml:"criu"`

	// RootfsExclusions defines paths to exclude from rootfs diff capture
	RootfsExclusions FilesystemConfig `yaml:"rootfsExclusions"`

	// SkipMountPrefixes is a list of directory prefixes. All mount points under these
	// directories will be skipped during checkpoint. This allows cross-node restore
	// when certain mounts (e.g., nvidia runtime mounts) don't exist on the target node.
	SkipMountPrefixes []string `yaml:"skipMountPrefixes"`
}

// Validate checks that the Config has valid values.
func (c *Config) Validate() error {
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
