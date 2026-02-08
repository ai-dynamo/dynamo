// loader.go provides shared YAML loading utilities for configuration.
package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// ConfigError represents a configuration validation error.
type ConfigError struct {
	Field   string
	Message string
}

func (e *ConfigError) Error() string {
	return fmt.Sprintf("config error: %s: %s", e.Field, e.Message)
}

// LoadConfig loads the full configuration from a YAML file.
// After loading, it applies environment variable overrides for dynamic values.
// All default values should be defined in the ConfigMap (via values.yaml), not in Go code.
func LoadConfig(path string) (*FullConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", path, err)
	}

	// Start with zero values - all defaults come from ConfigMap YAML
	cfg := &FullConfig{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file %s: %w", path, err)
	}

	// Apply environment variable overrides
	cfg.Agent.LoadAgentEnvOverrides()
	cfg.Checkpoint.LoadCheckpointEnvOverrides()

	return cfg, nil
}

// LoadConfigOrDefault loads configuration from a file, falling back to zero values if the file doesn't exist.
// Returns an error if the config file exists but cannot be parsed (e.g., YAML syntax errors).
// WARNING: When ConfigMap is missing, configuration will have zero/empty values except for env overrides.
// All default values should be defined in the ConfigMap (via values.yaml).
func LoadConfigOrDefault(path string) (*FullConfig, error) {
	cfg, err := LoadConfig(path)
	if err != nil {
		// Check if the error is "file not found" (acceptable) vs parse error (should be surfaced)
		if os.IsNotExist(err) {
			// File not found is acceptable - return zero config with env overrides
			cfg = &FullConfig{}
			cfg.Agent.LoadAgentEnvOverrides()
			cfg.Checkpoint.LoadCheckpointEnvOverrides()
			return cfg, nil
		}
		// Parse errors and other errors should be surfaced
		return nil, err
	}
	return cfg, nil
}

// LoadAgentConfig loads only the agent configuration from a YAML file.
// This is useful when only agent configuration is needed.
func LoadAgentConfig(path string) (*AgentConfig, error) {
	fullCfg, err := LoadConfig(path)
	if err != nil {
		return nil, err
	}
	return &fullCfg.Agent, nil
}

// LoadCheckpointConfig loads only the checkpoint configuration from a YAML file.
func LoadCheckpointConfig(path string) (*CheckpointConfig, error) {
	fullCfg, err := LoadConfig(path)
	if err != nil {
		return nil, err
	}
	return &fullCfg.Checkpoint, nil
}

// Validate validates the full configuration.
func (c *FullConfig) Validate() error {
	if err := c.Agent.Validate(); err != nil {
		return err
	}
	if err := c.Checkpoint.Validate(); err != nil {
		return err
	}
	return nil
}
