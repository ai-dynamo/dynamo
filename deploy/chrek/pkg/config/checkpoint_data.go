// checkpoint_data.go defines checkpoint data structures for cross-node restore operations.
// CheckpointData is the unified structure that combines static config and dynamic state.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// CheckpointData combines static config and dynamic state into one struct.
// Saved as metadata.yaml at checkpoint time, loaded at restore time.
// This is the single source of truth for all checkpoint configuration.
type CheckpointData struct {
	// ========== STATIC: From ConfigMap/values.yaml ==========
	// Copied from CheckpointConfig at checkpoint time

	// CRIU options used for checkpoint (and restore)
	CRIU CRIUConfig `yaml:"criu"`

	// Rootfs exclusion config for rootfs diff capture
	RootfsExclusions RootfsExclusionConfig `yaml:"rootfsExclusions"`

	// ========== DYNAMIC: Filled at checkpoint time ==========
	// Populated from container introspection

	// Identification
	CheckpointID string    `yaml:"checkpointId"`
	CreatedAt    time.Time `yaml:"createdAt"`

	// Source info
	SourceNode   string `yaml:"sourceNode"`
	SourcePodIP  string `yaml:"sourcePodIp,omitempty"`
	ContainerID  string `yaml:"containerId"`
	PodName      string `yaml:"podName"`
	PodNamespace string `yaml:"podNamespace"`
	Image        string `yaml:"image"`
	PID          int    `yaml:"pid"`

	// Filesystem
	RootfsDiffPath  string `yaml:"rootfsDiffPath,omitempty"`
	UpperDir        string `yaml:"upperDir,omitempty"`
	HasRootfsDiff   bool   `yaml:"hasRootfsDiff"`
	HasDeletedFiles bool   `yaml:"hasDeletedFiles"`

	// Mounts & Namespaces
	Mounts         []MountMetadata     `yaml:"mounts"`
	MaskedPaths    []string            `yaml:"maskedPaths,omitempty"`
	ReadonlyPaths  []string            `yaml:"readonlyPaths,omitempty"`
	BindMountDests []string            `yaml:"bindMountDests,omitempty"`
	Namespaces     []NamespaceMetadata `yaml:"namespaces"`
}

// MountMetadata stores information about a mount for remapping during restore.
type MountMetadata struct {
	ContainerPath string   `yaml:"containerPath"`           // Path inside container (e.g., /usr/share/nginx/html)
	HostPath      string   `yaml:"hostPath"`                // Original host path from mountinfo
	OCISource     string   `yaml:"ociSource,omitempty"`     // Source path from OCI spec (may differ from HostPath)
	OCIType       string   `yaml:"ociType,omitempty"`       // Mount type from OCI spec (bind, tmpfs, etc.)
	OCIOptions    []string `yaml:"ociOptions,omitempty"`    // Mount options from OCI spec
	VolumeType    string   `yaml:"volumeType"`              // emptyDir, pvc, configMap, secret, hostPath (best-effort)
	VolumeName    string   `yaml:"volumeName"`              // Kubernetes volume name (best-effort from path parsing)
	FSType        string   `yaml:"fsType"`                  // Filesystem type from mountinfo
	ReadOnly      bool     `yaml:"readOnly"`                // Whether mount is read-only
}

// NamespaceMetadata stores namespace information.
type NamespaceMetadata struct {
	Type       string `yaml:"type"`       // net, pid, mnt, etc.
	Inode      uint64 `yaml:"inode"`      // Namespace inode
	IsExternal bool   `yaml:"isExternal"` // Whether namespace is external (shared)
}

// NewCheckpointData creates a new CheckpointData instance with the given ID.
// Static config fields should be populated from CheckpointConfig after creation.
func NewCheckpointData(checkpointID string) *CheckpointData {
	return &CheckpointData{
		CheckpointID: checkpointID,
		CreatedAt:    time.Now().UTC(),
		Mounts:       make([]MountMetadata, 0),
		Namespaces:   make([]NamespaceMetadata, 0),
	}
}

// SaveCheckpointData writes checkpoint data to a YAML file in the checkpoint directory.
func SaveCheckpointData(checkpointDir string, data *CheckpointData) error {
	content, err := yaml.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint data: %w", err)
	}

	metadataPath := filepath.Join(checkpointDir, CheckpointDataFilename)
	if err := os.WriteFile(metadataPath, content, 0600); err != nil {
		return fmt.Errorf("failed to write metadata file: %w", err)
	}

	return nil
}

// LoadCheckpointData reads checkpoint data from a checkpoint directory.
func LoadCheckpointData(checkpointDir string) (*CheckpointData, error) {
	metadataPath := filepath.Join(checkpointDir, CheckpointDataFilename)

	content, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata file: %w", err)
	}

	var data CheckpointData
	if err := yaml.Unmarshal(content, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal checkpoint data: %w", err)
	}

	return &data, nil
}

// SaveDescriptors writes file descriptor information to the checkpoint directory.
func SaveDescriptors(checkpointDir string, descriptors []string) error {
	content, err := yaml.Marshal(descriptors)
	if err != nil {
		return fmt.Errorf("failed to marshal descriptors: %w", err)
	}

	descriptorsPath := filepath.Join(checkpointDir, DescriptorsFilename)
	if err := os.WriteFile(descriptorsPath, content, 0600); err != nil {
		return fmt.Errorf("failed to write descriptors file: %w", err)
	}

	return nil
}

// LoadDescriptors reads file descriptor information from checkpoint directory.
func LoadDescriptors(checkpointDir string) ([]string, error) {
	descriptorsPath := filepath.Join(checkpointDir, DescriptorsFilename)

	content, err := os.ReadFile(descriptorsPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read descriptors file: %w", err)
	}

	var descriptors []string
	if err := yaml.Unmarshal(content, &descriptors); err != nil {
		return nil, fmt.Errorf("failed to unmarshal descriptors: %w", err)
	}

	return descriptors, nil
}

// GetCheckpointDir returns the path to a checkpoint directory.
func GetCheckpointDir(baseDir, checkpointID string) string {
	return filepath.Join(baseDir, checkpointID)
}

// ListCheckpoints returns all checkpoint IDs in the base directory.
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
		metadataPath := filepath.Join(baseDir, entry.Name(), CheckpointDataFilename)
		if _, err := os.Stat(metadataPath); err == nil {
			checkpoints = append(checkpoints, entry.Name())
		}
	}

	return checkpoints, nil
}

// GetCheckpointInfo returns checkpoint data for a specific checkpoint.
func GetCheckpointInfo(baseDir, checkpointID string) (*CheckpointData, error) {
	checkpointDir := GetCheckpointDir(baseDir, checkpointID)
	return LoadCheckpointData(checkpointDir)
}

// DeleteCheckpoint removes a checkpoint directory.
func DeleteCheckpoint(baseDir, checkpointID string) error {
	checkpointDir := GetCheckpointDir(baseDir, checkpointID)
	// Ensure resolved path is within baseDir to prevent path traversal
	absBase, _ := filepath.Abs(baseDir)
	absDir, _ := filepath.Abs(checkpointDir)
	if !strings.HasPrefix(absDir, absBase+string(filepath.Separator)) && absDir != absBase {
		return fmt.Errorf("invalid checkpoint ID: resolved path %s is outside base directory %s", absDir, absBase)
	}
	return os.RemoveAll(checkpointDir)
}
