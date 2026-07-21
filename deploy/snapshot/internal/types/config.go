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

const (
	// StorageAccessModeAgentMount means the snapshot-agent pod mounts the
	// checkpoint store directly at Storage.BasePath.
	StorageAccessModeAgentMount = "agentMount"
	// StorageAccessModePodMount means workload pods mount the checkpoint PVC,
	// and snapshot-agent reaches it through /host/proc/<pid>/root.
	StorageAccessModePodMount = "podMount"

	CRIUCompressionModeOff     = "off"
	CRIUCompressionModePerPage = "per-page"
	CRIUCompressionModeRegion  = "region"

	DefaultCRIUCompressionAcceleration uint32 = 1
	DefaultCRIUCompressionRegionSize   uint32 = 256 * 1024
	DefaultCRIUDecompressThreads       uint32 = 1
	MaxCRIUCompressionAcceleration     uint32 = 65537
	MaxCRIUCompressionRegionSize       uint32 = 4 * 1024 * 1024
	MaxCRIUDecompressThreads           uint32 = 1024
	criuPageSize                       uint32 = 4096
)

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
	basePath := strings.TrimSpace(c.Storage.BasePath)
	if basePath == "" {
		return &ConfigError{Field: "storage.basePath", Message: "storage.basePath is required"}
	}
	if !strings.HasPrefix(basePath, "/") {
		return &ConfigError{Field: "storage.basePath", Message: "storage.basePath must be an absolute path"}
	}
	c.Storage.BasePath = basePath
	accessMode := strings.TrimSpace(c.Storage.AccessMode)
	if accessMode == "" {
		accessMode = StorageAccessModeAgentMount
	}
	switch accessMode {
	case StorageAccessModeAgentMount, StorageAccessModePodMount:
	default:
		return &ConfigError{
			Field:   "storage.accessMode",
			Message: fmt.Sprintf("unsupported access mode %q; expected %q or %q", c.Storage.AccessMode, StorageAccessModeAgentMount, StorageAccessModePodMount),
		}
	}
	c.Storage.AccessMode = accessMode
	if err := c.CRIU.Validate(); err != nil {
		return err
	}
	return c.Restore.Validate()
}

// StorageSpec holds snapshot storage settings that are local to the agent deployment.
type StorageSpec struct {
	Type       string `yaml:"type"`
	BasePath   string `yaml:"basePath"`
	AccessMode string `yaml:"accessMode"`
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
	GhostLimit              uint32  `yaml:"ghostLimit"`
	LogLevel                int32   `yaml:"logLevel"`
	WorkDir                 string  `yaml:"workDir"`
	AutoDedup               bool    `yaml:"autoDedup"`
	LazyPages               bool    `yaml:"lazyPages"`
	LeaveRunning            bool    `yaml:"leaveRunning"`
	ShellJob                bool    `yaml:"shellJob"`
	TcpClose                bool    `yaml:"tcpClose"`
	TcpEstablished          bool    `yaml:"tcpEstablished"`
	FileLocks               bool    `yaml:"fileLocks"`
	OrphanPtsMaster         bool    `yaml:"orphanPtsMaster"`
	ExtUnixSk               bool    `yaml:"extUnixSk"`
	LinkRemap               bool    `yaml:"linkRemap"`
	ExtMasters              bool    `yaml:"extMasters"`
	ManageCgroupsMode       string  `yaml:"manageCgroupsMode"`
	ImageIoMode             string  `yaml:"imageIoMode"`
	CompressionMode         string  `yaml:"compressionMode,omitempty"`
	CompressionAcceleration uint32  `yaml:"compressionAcceleration,omitempty"`
	CompressionRegionSize   uint32  `yaml:"compressionRegionSize,omitempty"`
	DecompressThreads       *uint32 `yaml:"decompressThreads,omitempty"`
	RstSibling              bool    `yaml:"rstSibling"`
	MntnsCompatMode         bool    `yaml:"mntnsCompatMode"`
	EvasiveDevices          bool    `yaml:"evasiveDevices"`
	ForceIrmap              bool    `yaml:"forceIrmap"`
	BinaryPath              string  `yaml:"binaryPath"`
	LibDir                  string  `yaml:"libDir"`
	AllowUprobes            bool    `yaml:"allowUprobes"`
	SkipInFlight            bool    `yaml:"skipInFlight"`
}

// Validate normalizes CRIU settings and rejects combinations CRIU cannot use.
func (c *CRIUSettings) Validate() error {
	if c.TcpClose && c.TcpEstablished {
		return &ConfigError{Field: "criu", Message: "tcpClose and tcpEstablished cannot both be true"}
	}

	switch strings.ToLower(strings.TrimSpace(c.ImageIoMode)) {
	case "", "writeback", "direct":
	default:
		return &ConfigError{
			Field:   "criu.imageIoMode",
			Message: fmt.Sprintf("unsupported imageIoMode %q; expected %q, %q, or empty", c.ImageIoMode, "writeback", "direct"),
		}
	}

	mode := strings.ToLower(strings.TrimSpace(c.CompressionMode))
	if mode == "" {
		mode = CRIUCompressionModeOff
	}
	switch mode {
	case CRIUCompressionModeOff, CRIUCompressionModePerPage, CRIUCompressionModeRegion:
	default:
		return &ConfigError{
			Field:   "criu.compressionMode",
			Message: fmt.Sprintf("unsupported compressionMode %q; expected %q, %q, or %q", c.CompressionMode, CRIUCompressionModeOff, CRIUCompressionModePerPage, CRIUCompressionModeRegion),
		}
	}
	c.CompressionMode = mode

	if mode != CRIUCompressionModeOff {
		if c.CompressionAcceleration == 0 {
			c.CompressionAcceleration = DefaultCRIUCompressionAcceleration
		}
		if c.CompressionAcceleration > MaxCRIUCompressionAcceleration {
			return &ConfigError{
				Field:   "criu.compressionAcceleration",
				Message: fmt.Sprintf("must be between 1 and %d", MaxCRIUCompressionAcceleration),
			}
		}
	}

	if mode == CRIUCompressionModeRegion {
		if c.CompressionRegionSize == 0 {
			c.CompressionRegionSize = DefaultCRIUCompressionRegionSize
		}
		if c.CompressionRegionSize%criuPageSize != 0 || c.CompressionRegionSize > MaxCRIUCompressionRegionSize {
			return &ConfigError{
				Field:   "criu.compressionRegionSize",
				Message: fmt.Sprintf("must be a multiple of %d and no greater than %d", criuPageSize, MaxCRIUCompressionRegionSize),
			}
		}
	}

	if c.DecompressThreads != nil && *c.DecompressThreads > MaxCRIUDecompressThreads {
		return &ConfigError{
			Field:   "criu.decompressThreads",
			Message: fmt.Sprintf("must be between 0 and %d", MaxCRIUDecompressThreads),
		}
	}
	return nil
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
