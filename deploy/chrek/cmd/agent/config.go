// config.go provides configuration loading for the checkpoint agent.
package main

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

// ConfigMapPath is the default path where the ConfigMap is mounted.
const ConfigMapPath = "/etc/chrek/config.yaml"

// CheckpointSignalSource determines how checkpoint operations are triggered.
type CheckpointSignalSource string

const (
	// SignalFromHTTP triggers checkpoints via HTTP API requests.
	SignalFromHTTP CheckpointSignalSource = "http"
	// SignalFromWatcher triggers checkpoints automatically when pods become Ready.
	SignalFromWatcher CheckpointSignalSource = "watcher"
)

// FullConfig is the root configuration structure loaded from the ConfigMap.
type FullConfig struct {
	Agent      AgentConfig       `yaml:"agent"`
	Checkpoint checkpoint.Config `yaml:"checkpoint"`
}

// AgentConfig holds the runtime configuration for the checkpoint agent daemon.
type AgentConfig struct {
	// SignalSource determines how checkpoints are triggered: "http" or "watcher"
	SignalSource string `yaml:"signalSource"`

	// ListenAddr is the HTTP server address for health checks and API
	ListenAddr string `yaml:"listenAddr"`

	// NodeName is the Kubernetes node name (from NODE_NAME env, downward API)
	NodeName string `yaml:"-"`

	// RestrictedNamespace restricts pod watching to this namespace (optional)
	RestrictedNamespace string `yaml:"-"`
}

// ConfigError represents a configuration validation error.
type ConfigError struct {
	Field   string
	Message string
}

func (e *ConfigError) Error() string {
	return fmt.Sprintf("config error: %s: %s", e.Field, e.Message)
}

// LoadConfig loads the full configuration from a YAML file.
func LoadConfig(path string) (*FullConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", path, err)
	}

	cfg := &FullConfig{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file %s: %w", path, err)
	}

	// Apply environment variable overrides
	cfg.Agent.loadEnvOverrides()

	return cfg, nil
}

// LoadConfigOrDefault loads configuration from a file, falling back to zero values if the file doesn't exist.
func LoadConfigOrDefault(path string) (*FullConfig, error) {
	cfg, err := LoadConfig(path)
	if err != nil {
		if os.IsNotExist(err) {
			cfg = &FullConfig{}
			cfg.Agent.loadEnvOverrides()
			return cfg, nil
		}
		return nil, err
	}
	return cfg, nil
}

// loadEnvOverrides applies environment variable overrides to the AgentConfig.
func (c *AgentConfig) loadEnvOverrides() {
	if v := os.Getenv("NODE_NAME"); v != "" {
		c.NodeName = v
	}
	if v := os.Getenv("RESTRICTED_NAMESPACE"); v != "" {
		c.RestrictedNamespace = v
	}
}

// GetSignalSource returns the signal source as a CheckpointSignalSource type.
func (c *AgentConfig) GetSignalSource() CheckpointSignalSource {
	return CheckpointSignalSource(c.SignalSource)
}

// Validate checks that the AgentConfig has valid values.
func (c *AgentConfig) Validate() error {
	if c.SignalSource != string(SignalFromHTTP) && c.SignalSource != string(SignalFromWatcher) {
		return &ConfigError{
			Field:   "signalSource",
			Message: "must be 'http' or 'watcher'",
		}
	}
	if c.SignalSource == string(SignalFromHTTP) && c.ListenAddr == "" {
		return &ConfigError{
			Field:   "listenAddr",
			Message: "cannot be empty when signalSource is 'http'",
		}
	}
	return nil
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
