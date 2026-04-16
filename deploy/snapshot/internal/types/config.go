// Package types defines shared data types used across snapshot packages.
package types

import (
	"fmt"
	"os"
	"strings"
	"time"
)

// AgentConfig holds the full agent configuration: static checkpoint settings
// from the ConfigMap YAML, plus runtime fields from environment variables.
type AgentConfig struct {
	NodeName            string          `yaml:"-"`
	RestrictedNamespace string          `yaml:"-"`
	Storage             StorageSpec     `yaml:"storage"`
	Overlay             OverlaySettings `yaml:"overlay"`
	Restore             RestoreSpec     `yaml:"restore"`
	CRIU                CRIUSettings    `yaml:"criu"`
}

func (c *AgentConfig) LoadEnvOverrides() {
	if v := os.Getenv("NODE_NAME"); v != "" {
		c.NodeName = v
	}
	if v := os.Getenv("RESTRICTED_NAMESPACE"); v != "" {
		c.RestrictedNamespace = v
	}
}

func (c *AgentConfig) Validate() error {
	storageType := strings.TrimSpace(c.Storage.Type)
	if storageType == "" {
		storageType = "pvc"
	}
	if storageType != "pvc" {
		return &ConfigError{Field: "storage.type", Message: fmt.Sprintf("unsupported storage type %q; only pvc is implemented today", storageType)}
	}
	if strings.TrimSpace(c.Storage.BasePath) == "" {
		return &ConfigError{Field: "storage.basePath", Message: "storage.basePath is required"}
	}
	if c.CRIU.TcpClose && c.CRIU.TcpEstablished {
		return &ConfigError{
			Field:   "criu",
			Message: "tcpClose and tcpEstablished cannot both be true",
		}
	}
	return c.Restore.Validate()
}

// StorageSpec holds snapshot storage settings that are local to the agent deployment.
type StorageSpec struct {
	Type     string `yaml:"type"`
	BasePath string `yaml:"basePath"`
}

// RestoreSpec holds settings for the CRIU restore process.
type RestoreSpec struct {
	NSRestorePath         string `yaml:"nsRestorePath"`
	RestoreTimeoutSeconds int    `yaml:"restoreTimeoutSeconds"`
}

func (c *RestoreSpec) RestoreTimeout() time.Duration {
	if c.RestoreTimeoutSeconds <= 0 {
		return 0
	}
	return time.Duration(c.RestoreTimeoutSeconds) * time.Second
}

func (c *RestoreSpec) Validate() error {
	if c.NSRestorePath == "" {
		return &ConfigError{Field: "nsRestorePath", Message: "nsRestorePath is required"}
	}
	if c.RestoreTimeoutSeconds <= 0 {
		return &ConfigError{Field: "restoreTimeoutSeconds", Message: "restoreTimeoutSeconds must be greater than zero"}
	}
	return nil
}

// CRIUSettings holds CRIU-specific configuration options.
type CRIUSettings struct {
	GhostLimit        uint32 `yaml:"ghostLimit"`
	LogLevel          int32  `yaml:"logLevel"`
	WorkDir           string `yaml:"workDir"`
	AutoDedup         bool   `yaml:"autoDedup"`
	LazyPages         bool   `yaml:"lazyPages"`
	LeaveRunning      bool   `yaml:"leaveRunning"`
	ShellJob          bool   `yaml:"shellJob"`
	TcpClose          bool   `yaml:"tcpClose"`
	TcpEstablished    bool   `yaml:"tcpEstablished"`
	FileLocks         bool   `yaml:"fileLocks"`
	OrphanPtsMaster   bool   `yaml:"orphanPtsMaster"`
	ExtUnixSk         bool   `yaml:"extUnixSk"`
	LinkRemap         bool   `yaml:"linkRemap"`
	ExtMasters        bool   `yaml:"extMasters"`
	ManageCgroupsMode string `yaml:"manageCgroupsMode"`
	RstSibling        bool   `yaml:"rstSibling"`
	MntnsCompatMode   bool   `yaml:"mntnsCompatMode"`
	EvasiveDevices    bool   `yaml:"evasiveDevices"`
	ForceIrmap        bool   `yaml:"forceIrmap"`
	BinaryPath        string `yaml:"binaryPath"`
	LibDir            string `yaml:"libDir"`
	AllowUprobes      bool   `yaml:"allowUprobes"`
	SkipInFlight      bool   `yaml:"skipInFlight"`
}

// OverlaySettings is the static config for rootfs exclusions.
type OverlaySettings struct {
	Exclusions []string `yaml:"exclusions"`
}

// ConfigError represents a configuration validation error.
type ConfigError struct {
	Field   string
	Message string
}

func (e *ConfigError) Error() string {
	return fmt.Sprintf("config error: %s: %s", e.Field, e.Message)
}
