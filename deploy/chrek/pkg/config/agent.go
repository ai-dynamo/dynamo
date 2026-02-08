// agent.go defines the AgentConfig struct for checkpoint agent runtime configuration.
package config

import "os"

// CheckpointSignalSource determines how checkpoint operations are triggered.
type CheckpointSignalSource string

const (
	// SignalFromHTTP triggers checkpoints via HTTP API requests.
	SignalFromHTTP CheckpointSignalSource = "http"
	// SignalFromWatcher triggers checkpoints automatically when pods become Ready.
	SignalFromWatcher CheckpointSignalSource = "watcher"
)

// AgentConfig holds the runtime configuration for the checkpoint agent daemon.
type AgentConfig struct {
	// SignalSource determines how checkpoints are triggered: "http" or "watcher"
	SignalSource string `yaml:"signalSource"`

	// ListenAddr is the HTTP server address for health checks and API
	ListenAddr string `yaml:"listenAddr"`

	// NodeName is the Kubernetes node name (from NODE_NAME env, downward API)
	// This is not in the ConfigMap - it's set dynamically from environment.
	NodeName string `yaml:"-"`

	// RestrictedNamespace restricts pod watching to this namespace (optional)
	// This is not in the ConfigMap - it's set dynamically from environment.
	RestrictedNamespace string `yaml:"-"`
}

// LoadAgentEnvOverrides applies environment variable overrides to the AgentConfig.
// This is called after loading the base config from YAML.
func (c *AgentConfig) LoadAgentEnvOverrides() {
	// Dynamic values from Kubernetes downward API
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
