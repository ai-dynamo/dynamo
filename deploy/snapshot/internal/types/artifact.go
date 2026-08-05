// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

const completionFilename = "completion.yaml"

// ArtifactCompletion is written last in the staging directory. Its presence
// proves that capture reached the publication boundary.
type ArtifactCompletion struct {
	CheckpointID string `yaml:"checkpointId"`
}

// WriteArtifactCompletion validates the captured payload and writes its completion record last.
func WriteArtifactCompletion(checkpointDir string, completion *ArtifactCompletion) error {
	if completion == nil {
		return fmt.Errorf("artifact completion is required")
	}
	if strings.TrimSpace(completion.CheckpointID) == "" {
		return fmt.Errorf("artifact completion is missing checkpointId")
	}
	if err := validateRequiredArtifactFiles(checkpointDir); err != nil {
		return err
	}

	content, err := yaml.Marshal(completion)
	if err != nil {
		return fmt.Errorf("marshal artifact completion: %w", err)
	}
	path := filepath.Join(checkpointDir, completionFilename)
	file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
	if err != nil {
		return fmt.Errorf("create artifact completion: %w", err)
	}
	if _, err := file.Write(content); err != nil {
		_ = file.Close()
		return fmt.Errorf("write artifact completion: %w", err)
	}
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return fmt.Errorf("sync artifact completion: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close artifact completion: %w", err)
	}
	return syncDirectory(checkpointDir)
}

// ValidateArtifact verifies the completion record, identity, and required files.
func ValidateArtifact(checkpointDir, checkpointID string) (*ArtifactCompletion, error) {
	content, err := os.ReadFile(filepath.Join(checkpointDir, completionFilename))
	if err != nil {
		return nil, fmt.Errorf("read artifact completion: %w", err)
	}

	var completion ArtifactCompletion
	if err := yaml.Unmarshal(content, &completion); err != nil {
		return nil, fmt.Errorf("unmarshal artifact completion: %w", err)
	}
	if strings.TrimSpace(completion.CheckpointID) == "" {
		return nil, fmt.Errorf("artifact completion is missing checkpointId")
	}
	if checkpointID != "" && completion.CheckpointID != checkpointID {
		return nil, fmt.Errorf("artifact checkpointId %q does not match requested checkpointId %q", completion.CheckpointID, checkpointID)
	}
	if err := validateRequiredArtifactFiles(checkpointDir); err != nil {
		return nil, err
	}

	manifest, err := ReadManifest(checkpointDir)
	if err != nil {
		return nil, fmt.Errorf("validate artifact manifest: %w", err)
	}
	if manifest.CheckpointID != completion.CheckpointID {
		return nil, fmt.Errorf("manifest checkpointId %q does not match completion checkpointId %q", manifest.CheckpointID, completion.CheckpointID)
	}
	return &completion, nil
}

// ValidateArtifactForRestore accepts version 1 and 2 artifacts created before
// completion markers existed. Newer artifacts must carry the marker.
func ValidateArtifactForRestore(checkpointDir, checkpointID string) error {
	if _, err := os.Stat(filepath.Join(checkpointDir, completionFilename)); os.IsNotExist(err) {
		version := filepath.Base(checkpointDir)
		if version == "1" || version == "2" {
			if err := validateRequiredArtifactFiles(checkpointDir); err != nil {
				return err
			}
			manifest, err := ReadManifest(checkpointDir)
			if err != nil {
				return err
			}
			if checkpointID != "" && manifest.CheckpointID != checkpointID {
				return fmt.Errorf("manifest checkpointId %q does not match requested checkpointId %q", manifest.CheckpointID, checkpointID)
			}
			return nil
		}
	}
	_, err := ValidateArtifact(checkpointDir, checkpointID)
	return err
}

func validateRequiredArtifactFiles(checkpointDir string) error {
	for _, name := range []string{manifestFilename, "inventory.img", "pstree.img"} {
		info, err := os.Stat(filepath.Join(checkpointDir, name))
		if err != nil {
			return fmt.Errorf("checkpoint artifact is missing required file %q: %w", name, err)
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("checkpoint artifact file %q is not regular", name)
		}
	}
	return nil
}

func syncDirectory(path string) error {
	dir, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open directory %s for sync: %w", path, err)
	}
	defer dir.Close()
	if err := dir.Sync(); err != nil {
		return fmt.Errorf("sync directory %s: %w", path, err)
	}
	return nil
}
