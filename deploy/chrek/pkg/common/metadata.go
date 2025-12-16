// metadata.go handles checkpoint metadata for cross-node restore operations.
package common

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const (
	// MetadataFilename is the name of the metadata file in checkpoint directories
	MetadataFilename = "metadata.json"
	// DescriptorsFilename is the name of the file descriptors file
	DescriptorsFilename = "descriptors.json"
)

// CheckpointMetadata stores information needed for cross-node restore
type CheckpointMetadata struct {
	// Checkpoint identification
	CheckpointID string    `json:"checkpoint_id"`
	CreatedAt    time.Time `json:"created_at"`

	// Source information
	SourceNode   string `json:"source_node"`
	SourcePodIP  string `json:"source_pod_ip,omitempty"` // For cross-node TCP detection
	ContainerID  string `json:"container_id"`
	PodName      string `json:"pod_name"`
	PodNamespace string `json:"pod_namespace"`
	Image        string `json:"image"`

	// Process information
	PID int `json:"pid"`

	// Filesystem information
	RootfsDiffPath  string `json:"rootfs_diff_path,omitempty"`   // Path to rootfs-diff.tar
	UpperDir        string `json:"upper_dir,omitempty"`          // Original overlay upperdir
	HasRootfsDiff   bool   `json:"has_rootfs_diff"`              // Whether rootfs diff was captured
	HasDeletedFiles bool   `json:"has_deleted_files"`            // Whether deleted files were tracked

	// Mount mappings from original container
	Mounts []MountMetadata `json:"mounts"`

	// OCI spec derived paths (populated from containerd, used at restore)
	// These replace hardcoded values with runtime-discovered configuration
	MaskedPaths    []string `json:"masked_paths,omitempty"`     // From OCI spec Linux.MaskedPaths
	ReadonlyPaths  []string `json:"readonly_paths,omitempty"`   // From OCI spec Linux.ReadonlyPaths
	BindMountDests []string `json:"bind_mount_dests,omitempty"` // Destinations of bind mounts (for tar exclusions)

	// Namespace information
	Namespaces []NamespaceMetadata `json:"namespaces"`

	// CRIU options used during checkpoint (for restore compatibility)
	CRIUOptions CRIUOptionsMetadata `json:"criu_options"`
}

// CRIUOptionsMetadata stores CRIU options used during checkpoint.
// This allows restore to use compatible options.
// Note: In our implementation, most options are hardcoded as always-on for K8s,
// but we store them for compatibility and debugging purposes.
type CRIUOptionsMetadata struct {
	TcpEstablished bool `json:"tcp_established"`
	TcpClose       bool `json:"tcp_close"`
	ShellJob       bool `json:"shell_job"`
	FileLocks      bool `json:"file_locks"`
	LeaveRunning   bool `json:"leave_running"`
	LinkRemap      bool `json:"link_remap"`
	ExtMasters     bool `json:"ext_masters"`
}

// MountMetadata stores information about a mount for remapping during restore
type MountMetadata struct {
	ContainerPath string   `json:"container_path"`           // Path inside container (e.g., /usr/share/nginx/html)
	HostPath      string   `json:"host_path"`                // Original host path from mountinfo
	OCISource     string   `json:"oci_source,omitempty"`     // Source path from OCI spec (may differ from HostPath)
	OCIType       string   `json:"oci_type,omitempty"`       // Mount type from OCI spec (bind, tmpfs, etc.)
	OCIOptions    []string `json:"oci_options,omitempty"`    // Mount options from OCI spec
	VolumeType    string   `json:"volume_type"`              // emptyDir, pvc, configMap, secret, hostPath (best-effort)
	VolumeName    string   `json:"volume_name"`              // Kubernetes volume name (best-effort from path parsing)
	FSType        string   `json:"fs_type"`                  // Filesystem type from mountinfo
	ReadOnly      bool     `json:"read_only"`                // Whether mount is read-only
}

// NamespaceMetadata stores namespace information
type NamespaceMetadata struct {
	Type       string `json:"type"`        // net, pid, mnt, etc.
	Inode      uint64 `json:"inode"`       // Namespace inode
	IsExternal bool   `json:"is_external"` // Whether namespace is external (shared)
}

// NewCheckpointMetadata creates a new metadata instance
func NewCheckpointMetadata(checkpointID string) *CheckpointMetadata {
	return &CheckpointMetadata{
		CheckpointID: checkpointID,
		CreatedAt:    time.Now().UTC(),
		Mounts:       make([]MountMetadata, 0),
		Namespaces:   make([]NamespaceMetadata, 0),
	}
}

// SaveMetadata writes metadata to a JSON file in the checkpoint directory
func SaveMetadata(checkpointDir string, meta *CheckpointMetadata) error {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	metadataPath := filepath.Join(checkpointDir, MetadataFilename)
	if err := os.WriteFile(metadataPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write metadata file: %w", err)
	}

	return nil
}

// LoadMetadata reads metadata from a checkpoint directory
func LoadMetadata(checkpointDir string) (*CheckpointMetadata, error) {
	metadataPath := filepath.Join(checkpointDir, MetadataFilename)

	data, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata file: %w", err)
	}

	var meta CheckpointMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
	}

	return &meta, nil
}

// SaveDescriptors writes file descriptor information to the checkpoint directory
func SaveDescriptors(checkpointDir string, descriptors []string) error {
	data, err := json.Marshal(descriptors)
	if err != nil {
		return fmt.Errorf("failed to marshal descriptors: %w", err)
	}

	descriptorsPath := filepath.Join(checkpointDir, DescriptorsFilename)
	if err := os.WriteFile(descriptorsPath, data, 0600); err != nil {
		return fmt.Errorf("failed to write descriptors file: %w", err)
	}

	return nil
}

// LoadDescriptors reads file descriptor information from checkpoint directory
func LoadDescriptors(checkpointDir string) ([]string, error) {
	descriptorsPath := filepath.Join(checkpointDir, DescriptorsFilename)

	data, err := os.ReadFile(descriptorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read descriptors file: %w", err)
	}

	var descriptors []string
	if err := json.Unmarshal(data, &descriptors); err != nil {
		return nil, fmt.Errorf("failed to unmarshal descriptors: %w", err)
	}

	return descriptors, nil
}

// GetCheckpointDir returns the path to a checkpoint directory
func GetCheckpointDir(baseDir, checkpointID string) string {
	return filepath.Join(baseDir, checkpointID)
}

// ListCheckpoints returns all checkpoint IDs in the base directory
func ListCheckpoints(baseDir string) ([]string, error) {
	entries, err := os.ReadDir(baseDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint directory: %w", err)
	}

	var checkpoints []string
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		// Check if metadata file exists
		metadataPath := filepath.Join(baseDir, entry.Name(), MetadataFilename)
		if _, err := os.Stat(metadataPath); err == nil {
			checkpoints = append(checkpoints, entry.Name())
		}
	}

	return checkpoints, nil
}

// GetCheckpointInfo returns metadata for a specific checkpoint
func GetCheckpointInfo(baseDir, checkpointID string) (*CheckpointMetadata, error) {
	checkpointDir := GetCheckpointDir(baseDir, checkpointID)
	return LoadMetadata(checkpointDir)
}

// DeleteCheckpoint removes a checkpoint directory
func DeleteCheckpoint(baseDir, checkpointID string) error {
	checkpointDir := GetCheckpointDir(baseDir, checkpointID)
	return os.RemoveAll(checkpointDir)
}
