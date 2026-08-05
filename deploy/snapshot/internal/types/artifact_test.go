// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"os"
	"path/filepath"
	"testing"
)

func TestValidateArtifactRequiresCompletionMarker(t *testing.T) {
	dir := t.TempDir()
	manifest := NewCheckpointManifest("checkpoint-123", CRIUDumpManifest{}, SourcePodManifest{}, OverlayManifest{})
	if err := WriteManifest(dir, manifest); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}
	for _, name := range []string{"inventory.img", "pstree.img"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("payload"), 0o600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}

	if _, err := ValidateArtifact(dir, "checkpoint-123"); err == nil {
		t.Fatal("ValidateArtifact unexpectedly accepted an artifact without a completion marker")
	}
	if err := WriteArtifactCompletion(dir, &ArtifactCompletion{CheckpointID: "checkpoint-123"}); err != nil {
		t.Fatalf("WriteArtifactCompletion: %v", err)
	}
	if _, err := ValidateArtifact(dir, "checkpoint-123"); err != nil {
		t.Fatalf("ValidateArtifact: %v", err)
	}
	if _, err := ValidateArtifact(dir, "another-checkpoint"); err == nil {
		t.Fatal("ValidateArtifact unexpectedly accepted the wrong checkpoint identity")
	}
}

func TestWriteArtifactCompletionRejectsPartialCapture(t *testing.T) {
	dir := t.TempDir()
	manifest := NewCheckpointManifest("checkpoint-123", CRIUDumpManifest{}, SourcePodManifest{}, OverlayManifest{})
	if err := WriteManifest(dir, manifest); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}

	err := WriteArtifactCompletion(dir, &ArtifactCompletion{CheckpointID: "checkpoint-123"})
	if err == nil {
		t.Fatal("WriteArtifactCompletion unexpectedly accepted a partial capture")
	}
	if _, statErr := os.Stat(filepath.Join(dir, completionFilename)); !os.IsNotExist(statErr) {
		t.Fatalf("completion record exists after failed validation: %v", statErr)
	}
}

func TestValidateArtifactForRestoreAcceptsLegacyVersion(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "2")
	if err := os.Mkdir(dir, 0o700); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	manifest := NewCheckpointManifest("checkpoint-123", CRIUDumpManifest{}, SourcePodManifest{}, OverlayManifest{})
	if err := WriteManifest(dir, manifest); err != nil {
		t.Fatalf("WriteManifest: %v", err)
	}
	for _, name := range []string{"inventory.img", "pstree.img"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("payload"), 0o600); err != nil {
			t.Fatalf("write %s: %v", name, err)
		}
	}
	if err := ValidateArtifactForRestore(dir, "checkpoint-123"); err != nil {
		t.Fatalf("ValidateArtifactForRestore: %v", err)
	}
}
