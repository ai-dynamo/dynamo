// exclusions.go defines rootfs exclusion configuration for checkpoint operations.
package config

import "strings"

// RootfsExclusionConfig defines paths to exclude from the rootfs diff capture.
// These exclusions prevent conflicts during restore and reduce checkpoint size.
type RootfsExclusionConfig struct {
	// SystemDirs are system directories that should be excluded from rootfs diff.
	// These directories are typically injected/bind-mounted by NVIDIA GPU Operator
	// at container start time, so they already exist in the restore target.
	// Excluding them prevents conflicts (especially socket files which cannot be overwritten).
	// Default: ["./usr", "./etc", "./opt", "./var", "./run"]
	SystemDirs []string `yaml:"systemDirs"`

	// CacheDirs are cache directories that can safely be excluded to reduce checkpoint size.
	// Model weights and other cached data are typically re-downloaded if needed.
	// Default: ["./.cache/huggingface", "./.cache/torch"]
	CacheDirs []string `yaml:"cacheDirs"`

	// AdditionalExclusions are custom paths to exclude from the rootfs diff.
	// Use this for application-specific exclusions.
	// Paths should be relative with "./" prefix (e.g., "./data/temp").
	AdditionalExclusions []string `yaml:"additionalExclusions"`
}

// GetAllExclusions returns all exclusion paths combined.
// This is used when building tar arguments for rootfs diff capture.
func (c *RootfsExclusionConfig) GetAllExclusions() []string {
	if c == nil {
		return nil
	}
	total := len(c.SystemDirs) + len(c.CacheDirs) + len(c.AdditionalExclusions)
	exclusions := make([]string, 0, total)
	exclusions = append(exclusions, c.SystemDirs...)
	exclusions = append(exclusions, c.CacheDirs...)
	exclusions = append(exclusions, c.AdditionalExclusions...)
	return exclusions
}

// Validate checks that the RootfsExclusionConfig has valid values.
func (c *RootfsExclusionConfig) Validate() error {
	if c == nil {
		return nil
	}
	// All paths should start with "./" for tar relative path handling
	for _, path := range c.GetAllExclusions() {
		if !strings.HasPrefix(path, "./") {
			return &ConfigError{
				Field:   "rootfsExclusions",
				Message: "all exclusion paths must start with './' (got: " + path + ")",
			}
		}
	}
	return nil
}
