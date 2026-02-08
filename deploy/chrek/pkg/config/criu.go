// criu.go defines CRIU configuration structures.
package config

// CRIUConfig holds CRIU-specific configuration options.
// Options are categorized by how they are passed to CRIU:
//   - RPC options: Passed via go-criu CriuOpts protobuf
//   - Config file options: Written to criu.conf (NOT available via RPC)
type CRIUConfig struct {
	// === RPC Options (passed via go-criu CriuOpts) ===

	// GhostLimit is the maximum ghost file size in bytes.
	// Ghost files are deleted-but-open files that CRIU needs to checkpoint.
	// 512MB is recommended for GPU workloads with large memory allocations.
	GhostLimit uint32 `yaml:"ghostLimit"`

	// Timeout is the CRIU operation timeout in seconds.
	// 6 hours (21600s) is recommended for large GPU model checkpoints.
	Timeout uint32 `yaml:"timeout"`

	// LogLevel is the CRIU logging verbosity (0-4).
	LogLevel int32 `yaml:"logLevel"`

	// WorkDir is the CRIU work directory for temporary files.
	WorkDir string `yaml:"workDir"`

	// AutoDedup enables auto-deduplication of memory pages.
	AutoDedup bool `yaml:"autoDedup"`

	// LazyPages enables lazy page migration (experimental).
	LazyPages bool `yaml:"lazyPages"`

	// LeaveRunning keeps the process running after checkpoint (dump only).
	LeaveRunning bool `yaml:"leaveRunning"`

	// ShellJob allows checkpointing session leaders (containers are often session leaders).
	ShellJob bool `yaml:"shellJob"`

	// TcpClose closes TCP connections instead of preserving them (pod IPs change on restore).
	TcpClose bool `yaml:"tcpClose"`

	// FileLocks allows checkpointing processes with file locks.
	FileLocks bool `yaml:"fileLocks"`

	// OrphanPtsMaster allows checkpointing containers with TTYs.
	OrphanPtsMaster bool `yaml:"orphanPtsMaster"`

	// ExtUnixSk allows external Unix sockets.
	ExtUnixSk bool `yaml:"extUnixSk"`

	// LinkRemap handles deleted-but-open files.
	LinkRemap bool `yaml:"linkRemap"`

	// ExtMasters allows external bind mount masters.
	ExtMasters bool `yaml:"extMasters"`

	// ManageCgroupsMode controls cgroup handling: "ignore" lets K8s manage cgroups.
	ManageCgroupsMode string `yaml:"manageCgroupsMode"`

	// SkipMounts is a list of mount paths to skip during checkpoint.
	// These are passed to CRIU's --skip-mnt option. This allows cross-node restore
	// when certain mounts (e.g., nvidia runtime mounts) don't exist on the target node.
	// Typically populated dynamically from SkipMountPrefixes in CheckpointConfig.
	SkipMounts []string `yaml:"skipMounts,omitempty"`

	// ExternalMounts are additional external mount mappings (e.g., "mnt[path]:path").
	// Populated dynamically at checkpoint time from container introspection.
	// Serialized to metadata.yaml so restore can use the exact same mappings.
	ExternalMounts []string `yaml:"externalMounts,omitempty"`

	// === Config File Options (NOT available via RPC - written to criu.conf) ===

	// LibDir is the path to CRIU plugin directory (e.g., /usr/local/lib/criu).
	// Required for CUDA checkpoint/restore.
	LibDir string `yaml:"libDir"`

	// AllowUprobes allows user-space probes (required for CUDA checkpoints).
	AllowUprobes bool `yaml:"allowUprobes"`

	// SkipInFlight skips in-flight TCP connections during checkpoint/restore.
	SkipInFlight bool `yaml:"skipInFlight"`
}

// GenerateCRIUConfContent generates the criu.conf file content for options
// that cannot be passed via RPC. This file is created in the checkpoint/restore
// directory and referenced via CriuOpts.ConfigFile.
//
// IMPORTANT: Only these three options go in criu.conf:
//   - libdir (plugin directory)
//   - allow-uprobes (required for CUDA)
//   - skip-in-flight (skip in-flight TCP)
//
// All other options (timeout, ghost-limit, etc.) are passed via RPC.
func (c *CRIUConfig) GenerateCRIUConfContent() string {
	var content string

	// Note: enable-external-masters, tcp-close, link-remap are hardcoded in code
	// via CriuOpts and don't need to be in the config file.

	if c.LibDir != "" {
		content += "libdir " + c.LibDir + "\n"
	}
	if c.AllowUprobes {
		content += "allow-uprobes\n"
	}
	if c.SkipInFlight {
		content += "skip-in-flight\n"
	}

	return content
}

// NeedsCRIUConfFile returns true if a criu.conf file needs to be created.
// This is required when any config-file-only options are set.
func (c *CRIUConfig) NeedsCRIUConfFile() bool {
	return c.LibDir != "" || c.AllowUprobes || c.SkipInFlight
}

// Validate checks that the CRIUConfig has valid values.
func (c *CRIUConfig) Validate() error {
	// Timeout is required when LibDir is set (CUDA checkpoints)
	if c.LibDir != "" && c.Timeout == 0 {
		return &ConfigError{
			Field:   "criu.timeout",
			Message: "timeout must be set when libDir is specified (required for CUDA checkpoints)",
		}
	}
	return nil
}
