// config.go provides configuration loading for the checkpoint agent.
package main

import (
	"errors"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/types"
)

// ConfigMapPath is the default path where the ConfigMap is mounted.
const ConfigMapPath = "/etc/chrek/config.yaml"

// LoadConfig loads the agent configuration from a YAML file.
func LoadConfig(path string) (*types.AgentConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", path, err)
	}

	cfg := &types.AgentConfig{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file %s: %w", path, err)
	}

	cfg.LoadEnvOverrides()
	return cfg, nil
}

// LoadConfigOrDefault loads configuration from a file, falling back to defaults if the file doesn't exist.
func LoadConfigOrDefault(path string) (*types.AgentConfig, error) {
	cfg, err := LoadConfig(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			cfg = &types.AgentConfig{}
			cfg.LoadEnvOverrides()
			return cfg, nil
		}
		return nil, err
	}
	return cfg, nil
}
