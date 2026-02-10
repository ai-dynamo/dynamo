// storage.go provides checkpoint storage I/O: write/read manifests, listing, deletion.
package checkpoint

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// WriteCheckpointManifest writes a checkpoint manifest file in the checkpoint directory.
func WriteCheckpointManifest(checkpointDir string, data *CheckpointManifest) error {
	content, err := yaml.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint manifest: %w", err)
	}

	manifestPath := filepath.Join(checkpointDir, CheckpointManifestFilename)
	if err := os.WriteFile(manifestPath, content, 0600); err != nil {
		return fmt.Errorf("failed to write checkpoint manifest: %w", err)
	}

	return nil
}

// ReadCheckpointManifest reads checkpoint manifest from a checkpoint directory.
func ReadCheckpointManifest(checkpointDir string) (*CheckpointManifest, error) {
	manifestPath := filepath.Join(checkpointDir, CheckpointManifestFilename)

	content, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint manifest: %w", err)
	}

	var data CheckpointManifest
	if err := yaml.Unmarshal(content, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal checkpoint manifest: %w", err)
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

		// Check if manifest file exists.
		manifestPath := filepath.Join(baseDir, entry.Name(), CheckpointManifestFilename)
		if _, err := os.Stat(manifestPath); err == nil {
			checkpoints = append(checkpoints, entry.Name())
		}
	}

	return checkpoints, nil
}

// DeleteCheckpoint removes a checkpoint directory.
func DeleteCheckpoint(baseDir, checkpointID string) error {
	checkpointDir := filepath.Join(baseDir, checkpointID)
	// Ensure resolved path is within baseDir to prevent path traversal
	absBase, _ := filepath.Abs(baseDir)
	absDir, _ := filepath.Abs(checkpointDir)
	if !strings.HasPrefix(absDir, absBase+string(filepath.Separator)) && absDir != absBase {
		return fmt.Errorf("invalid checkpoint ID: resolved path %s is outside base directory %s", absDir, absBase)
	}
	return os.RemoveAll(checkpointDir)
}
