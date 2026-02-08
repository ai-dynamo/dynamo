// Package config provides configuration types and loaders for the chrek checkpoint/restore system.
// Configuration is loaded from a YAML ConfigMap with environment variable overrides for
// dynamic values (e.g., NODE_NAME from Kubernetes downward API).
package config

// FullConfig is the root configuration structure loaded from the ConfigMap.
// It contains configuration for the agent and checkpoint operations.
// Restore runtime configuration (timeouts/triggers) is not here because placeholder
// containers do not mount the ConfigMap. Those values come from hardcoded defaults
// plus operator-injected env vars.
type FullConfig struct {
	// Agent configuration for runtime behavior
	Agent AgentConfig `yaml:"agent"`

	// Checkpoint configuration for CRIU dump operations
	Checkpoint CheckpointConfig `yaml:"checkpoint"`
}

// ConfigMapPath is the default path where the ConfigMap is mounted.
const ConfigMapPath = "/etc/chrek/config.yaml"
