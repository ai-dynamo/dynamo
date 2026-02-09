// storage.go provides checkpoint storage I/O: save/load metadata, listing, deletion.
package checkpoint

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// SaveCheckpointMetadata writes checkpoint metadata to a YAML file in the checkpoint directory.
func SaveCheckpointMetadata(checkpointDir string, data *CheckpointMetadata) error {
	content, err := yaml.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint metadata: %w", err)
	}

	metadataPath := filepath.Join(checkpointDir, CheckpointDataFilename)
	if err := os.WriteFile(metadataPath, content, 0600); err != nil {
		return fmt.Errorf("failed to write metadata file: %w", err)
	}

	return nil
}

// LoadCheckpointMetadata reads checkpoint metadata from a checkpoint directory.
func LoadCheckpointMetadata(checkpointDir string) (*CheckpointMetadata, error) {
	metadataPath := filepath.Join(checkpointDir, CheckpointDataFilename)

	content, err := os.ReadFile(metadataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read metadata file: %w", err)
	}

	var data CheckpointMetadata
	if err := yaml.Unmarshal(content, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal checkpoint metadata: %w", err)
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

		// Check if metadata file exists
		metadataPath := filepath.Join(baseDir, entry.Name(), CheckpointDataFilename)
		if _, err := os.Stat(metadataPath); err == nil {
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
