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

	// StorageTypePVC stores checkpoint artifacts on a shared filesystem volume.
	StorageTypePVC = "pvc"
	// StorageTypeS3 mirrors artifacts to an S3-compatible object store. These
	// mirror snapshotprotocol.StorageType* but are duplicated here so the
	// low-level types package stays free of a protocol dependency.
	StorageTypeS3 = "s3"
)

func (c *AgentConfig) LoadEnvOverrides() {
	if v := os.Getenv("NODE_NAME"); v != "" {
		c.NodeName = v
	}
	if v := os.Getenv("RESTRICTED_NAMESPACE"); v != "" {
		c.RestrictedNamespace = v
	}
	// S3 credentials are sourced from the environment (typically projected from
	// a Secret) so they never live in the agent ConfigMap. Standard AWS_* names
	// are honored for compatibility with existing tooling.
	if v := os.Getenv("AWS_ACCESS_KEY_ID"); v != "" {
		c.Storage.S3.AccessKeyID = v
	}
	if v := os.Getenv("AWS_SECRET_ACCESS_KEY"); v != "" {
		c.Storage.S3.SecretAccessKey = v
	}
	if v := os.Getenv("AWS_SESSION_TOKEN"); v != "" {
		c.Storage.S3.SessionToken = v
	}
}

func (c *AgentConfig) Validate() error {
	storageType := strings.TrimSpace(c.Storage.Type)
	if storageType == "" {
		storageType = StorageTypePVC
	}
	if storageType != StorageTypePVC && storageType != StorageTypeS3 {
		return &ConfigError{Field: "storage.type", Message: fmt.Sprintf("unsupported storage type %q; expected %q or %q", storageType, StorageTypePVC, StorageTypeS3)}
	}
	c.Storage.Type = storageType
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
	if storageType == StorageTypeS3 {
		// The agent stages artifacts on its own filesystem at basePath, then
		// uploads them, so S3 requires agent-local staging rather than reaching
		// into a workload pod's mount namespace.
		if accessMode != StorageAccessModeAgentMount {
			return &ConfigError{Field: "storage.accessMode", Message: fmt.Sprintf("storage type %q requires access mode %q", StorageTypeS3, StorageAccessModeAgentMount)}
		}
		if err := c.Storage.S3.validate(); err != nil {
			return err
		}
	}
	if c.CRIU.TcpClose && c.CRIU.TcpEstablished {
		return &ConfigError{
			Field:   "criu",
			Message: "tcpClose and tcpEstablished cannot both be true",
		}
	}
	switch strings.ToLower(strings.TrimSpace(c.CRIU.ImageIoMode)) {
	case "", "writeback", "direct":
	default:
		return &ConfigError{
			Field:   "criu.imageIoMode",
			Message: fmt.Sprintf("unsupported imageIoMode %q; expected %q, %q, or empty", c.CRIU.ImageIoMode, "writeback", "direct"),
		}
	}
	return c.Restore.Validate()
}

// StorageSpec holds snapshot storage settings that are local to the agent deployment.
type StorageSpec struct {
	Type       string `yaml:"type"`
	BasePath   string `yaml:"basePath"`
	AccessMode string `yaml:"accessMode"`
	// S3 configures the object-store backend when Type is "s3". BasePath then
	// serves as the agent-local staging directory for checkpoint artifacts.
	S3 S3StorageSpec `yaml:"s3"`
}

// S3StorageSpec configures the S3-compatible checkpoint store.
// Connection details come from the agent ConfigMap; credentials are loaded from
// the environment (see AgentConfig.LoadEnvOverrides) so they stay out of config.
type S3StorageSpec struct {
	Endpoint string `yaml:"endpoint"`
	Bucket   string `yaml:"bucket"`
	Region   string `yaml:"region"`
	Prefix   string `yaml:"prefix"`
	UseSSL   bool   `yaml:"useSSL"`
	// ForcePathStyle selects path-style addressing instead of the default
	// virtual-hosted (bucket DNS) style. Enable for endpoints without bucket
	// DNS support.
	ForcePathStyle bool   `yaml:"forcePathStyle"`
	Concurrency    int    `yaml:"concurrency"`
	PartSize       uint64 `yaml:"partSize"`

	// Credentials, populated from the environment rather than the ConfigMap.
	AccessKeyID     string `yaml:"-"`
	SecretAccessKey string `yaml:"-"`
	SessionToken    string `yaml:"-"`
}

func (s *S3StorageSpec) validate() error {
	if strings.TrimSpace(s.Endpoint) == "" {
		return &ConfigError{Field: "storage.s3.endpoint", Message: "storage.s3.endpoint is required for s3 storage"}
	}
	if strings.TrimSpace(s.Bucket) == "" {
		return &ConfigError{Field: "storage.s3.bucket", Message: "storage.s3.bucket is required for s3 storage"}
	}
	if strings.TrimSpace(s.AccessKeyID) == "" || strings.TrimSpace(s.SecretAccessKey) == "" {
		return &ConfigError{Field: "storage.s3", Message: "s3 credentials are required (set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)"}
	}
	return nil
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
	ImageIoMode       string `yaml:"imageIoMode"`
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
