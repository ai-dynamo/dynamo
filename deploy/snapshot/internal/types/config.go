// Package types defines shared data types used across snapshot packages.
package types

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// AgentConfig holds the full agent configuration: static checkpoint settings
// from the ConfigMap YAML, plus runtime fields from environment variables.
type AgentConfig struct {
	NodeName            string                 `yaml:"-"`
	RestrictedNamespace string                 `yaml:"-"`
	Storage             StorageSpec            `yaml:"storage"`
	CUDACheckpoint      CUDACheckpointSettings `yaml:"cudaCheckpoint"`
	Overlay             OverlaySettings        `yaml:"overlay"`
	Restore             RestoreSpec            `yaml:"restore"`
	CRIU                CRIUSettings           `yaml:"criu"`
}

func pathsOverlap(first, second string) bool {
	first = filepath.Clean(first)
	second = filepath.Clean(second)
	return first == string(filepath.Separator) || second == string(filepath.Separator) ||
		first == second ||
		strings.HasPrefix(first, second+string(filepath.Separator)) ||
		strings.HasPrefix(second, first+string(filepath.Separator))
}

// CUDACheckpointSettings holds CUDA checkpoint storage settings.
type CUDACheckpointSettings struct {
	TransferBufferCount *int    `yaml:"transferBufferCount"`
	TransferChunkBytes  *uint64 `yaml:"transferChunkBytes"`
}

// CUDATransferSettings holds validated custom-storage transfer settings.
type CUDATransferSettings struct {
	BufferCount int
	ChunkBytes  uint64
}

const (
	// StorageAccessModeAgentMount means the snapshot-agent pod mounts the
	// checkpoint store directly at Storage.BasePath.
	StorageAccessModeAgentMount = "agentMount"
	// StorageAccessModePodMount means workload pods mount the checkpoint PVC,
	// and snapshot-agent reaches it through /host/proc/<pid>/root.
	StorageAccessModePodMount = "podMount"
	// CUDAStorageModeLegacy uses the CUDA driver's existing host-memory storage.
	CUDAStorageModeLegacy = "legacy"
	// CUDAStorageModePOSIX stores CUDA custom-storage extents in checkpoint files.
	CUDAStorageModePOSIX = "posix"
	// DefaultCUDATransferBufferCount preserves the original single-buffer path.
	DefaultCUDATransferBufferCount = 1
	// DefaultCUDATransferChunkBytes preserves the original 64 MiB chunks.
	DefaultCUDATransferChunkBytes = 64 * 1024 * 1024
	maxCUDATransferBufferCount    = 8
	minCUDATransferChunkBytes     = 1 * 1024 * 1024
	maxCUDATransferChunkBytes     = 256 * 1024 * 1024
	maxCUDAPinnedBytesPerDevice   = 1 * 1024 * 1024 * 1024
	cudaTransferBufferAlignment   = 4096
	CUDAHelperSocketDirectory     = "/run/cuda-checkpoint-helper"
	CUDAHelperSocketPath          = CUDAHelperSocketDirectory + "/helper.sock"
)

func (c CUDACheckpointSettings) TransferSettings() CUDATransferSettings {
	settings := CUDATransferSettings{
		BufferCount: DefaultCUDATransferBufferCount,
		ChunkBytes:  DefaultCUDATransferChunkBytes,
	}
	if c.TransferBufferCount != nil {
		settings.BufferCount = *c.TransferBufferCount
	}
	if c.TransferChunkBytes != nil {
		settings.ChunkBytes = *c.TransferChunkBytes
	}
	return settings
}

func (c CUDATransferSettings) WithDefaults() CUDATransferSettings {
	settings := c
	if settings.BufferCount == 0 {
		settings.BufferCount = DefaultCUDATransferBufferCount
	}
	if settings.ChunkBytes == 0 {
		settings.ChunkBytes = DefaultCUDATransferChunkBytes
	}
	return settings
}

func (c CUDATransferSettings) Validate() error {
	if c.BufferCount < 1 || c.BufferCount > maxCUDATransferBufferCount {
		return fmt.Errorf("buffer count must be between 1 and %d", maxCUDATransferBufferCount)
	}
	if c.ChunkBytes < minCUDATransferChunkBytes || c.ChunkBytes > maxCUDATransferChunkBytes || c.ChunkBytes%cudaTransferBufferAlignment != 0 {
		return fmt.Errorf(
			"chunk bytes must be a %d-byte multiple between %d and %d",
			cudaTransferBufferAlignment,
			minCUDATransferChunkBytes,
			maxCUDATransferChunkBytes,
		)
	}
	if uint64(c.BufferCount) > maxCUDAPinnedBytesPerDevice/c.ChunkBytes {
		return fmt.Errorf("buffers exceed the 1 GiB per-device pinned-memory limit")
	}
	return nil
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
	if accessMode == StorageAccessModeAgentMount &&
		pathsOverlap(basePath, CUDAHelperSocketDirectory) {
		return &ConfigError{
			Field:   "storage.basePath",
			Message: "agent-mounted storage.basePath must not overlap the CUDA helper socket directory",
		}
	}
	if c.CUDACheckpoint.TransferBufferCount == nil {
		value := DefaultCUDATransferBufferCount
		c.CUDACheckpoint.TransferBufferCount = &value
	}
	if c.CUDACheckpoint.TransferChunkBytes == nil {
		value := uint64(DefaultCUDATransferChunkBytes)
		c.CUDACheckpoint.TransferChunkBytes = &value
	}
	if err := c.CUDACheckpoint.TransferSettings().Validate(); err != nil {
		return &ConfigError{
			Field:   "cudaCheckpoint",
			Message: err.Error(),
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
