package common

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveAndLoadMetadata(t *testing.T) {
	// Create temp directory
	tempDir := t.TempDir()

	// Create metadata
	meta := NewCheckpointMetadata("test-checkpoint")
	meta.SourceNode = "node-1"
	meta.ContainerID = "container-abc123"
	meta.PodName = "test-pod"
	meta.PodNamespace = "default"
	meta.Image = "nginx:latest"
	meta.PID = 12345
	meta.HasRootfsDiff = true
	meta.Mounts = []MountMetadata{
		{
			ContainerPath: "/data",
			HostPath:      "/var/lib/kubelet/pods/abc/volumes/pvc-123/data",
			VolumeType:    "pvc",
			FSType:        "ext4",
			ReadOnly:      false,
		},
	}
	meta.Namespaces = []NamespaceMetadata{
		{
			Type:       "net",
			Inode:      4026531956,
			IsExternal: true,
		},
	}

	// Save metadata
	err := SaveMetadata(tempDir, meta)
	if err != nil {
		t.Fatalf("SaveMetadata() error = %v", err)
	}

	// Check that file was created
	metadataPath := filepath.Join(tempDir, MetadataFilename)
	if _, err := os.Stat(metadataPath); os.IsNotExist(err) {
		t.Fatalf("Metadata file was not created at %s", metadataPath)
	}

	// Load metadata
	loaded, err := LoadMetadata(tempDir)
	if err != nil {
		t.Fatalf("LoadMetadata() error = %v", err)
	}

	// Compare fields
	if loaded.CheckpointID != meta.CheckpointID {
		t.Errorf("CheckpointID = %v, want %v", loaded.CheckpointID, meta.CheckpointID)
	}
	if loaded.SourceNode != meta.SourceNode {
		t.Errorf("SourceNode = %v, want %v", loaded.SourceNode, meta.SourceNode)
	}
	if loaded.ContainerID != meta.ContainerID {
		t.Errorf("ContainerID = %v, want %v", loaded.ContainerID, meta.ContainerID)
	}
	if loaded.PodName != meta.PodName {
		t.Errorf("PodName = %v, want %v", loaded.PodName, meta.PodName)
	}
	if loaded.PodNamespace != meta.PodNamespace {
		t.Errorf("PodNamespace = %v, want %v", loaded.PodNamespace, meta.PodNamespace)
	}
	if loaded.Image != meta.Image {
		t.Errorf("Image = %v, want %v", loaded.Image, meta.Image)
	}
	if loaded.PID != meta.PID {
		t.Errorf("PID = %v, want %v", loaded.PID, meta.PID)
	}
	if loaded.HasRootfsDiff != meta.HasRootfsDiff {
		t.Errorf("HasRootfsDiff = %v, want %v", loaded.HasRootfsDiff, meta.HasRootfsDiff)
	}
	if len(loaded.Mounts) != len(meta.Mounts) {
		t.Errorf("len(Mounts) = %v, want %v", len(loaded.Mounts), len(meta.Mounts))
	}
	if len(loaded.Namespaces) != len(meta.Namespaces) {
		t.Errorf("len(Namespaces) = %v, want %v", len(loaded.Namespaces), len(meta.Namespaces))
	}
}

func TestSaveAndLoadDescriptors(t *testing.T) {
	// Create temp directory
	tempDir := t.TempDir()

	// Create descriptors
	descriptors := []string{"/dev/null", "/dev/zero", "/dev/random"}

	// Save descriptors
	err := SaveDescriptors(tempDir, descriptors)
	if err != nil {
		t.Fatalf("SaveDescriptors() error = %v", err)
	}

	// Check that file was created
	descriptorsPath := filepath.Join(tempDir, DescriptorsFilename)
	if _, err := os.Stat(descriptorsPath); os.IsNotExist(err) {
		t.Fatalf("Descriptors file was not created at %s", descriptorsPath)
	}

	// Load descriptors
	loaded, err := LoadDescriptors(tempDir)
	if err != nil {
		t.Fatalf("LoadDescriptors() error = %v", err)
	}

	// Compare
	if len(loaded) != len(descriptors) {
		t.Errorf("len(descriptors) = %v, want %v", len(loaded), len(descriptors))
	}

	for i, desc := range descriptors {
		if i >= len(loaded) {
			break
		}
		if loaded[i] != desc {
			t.Errorf("descriptor[%d] = %v, want %v", i, loaded[i], desc)
		}
	}
}

func TestListCheckpoints(t *testing.T) {
	// Create temp directory
	tempDir := t.TempDir()

	// Create some checkpoint directories with metadata
	checkpoints := []string{"ckpt-1", "ckpt-2", "ckpt-3"}
	for _, ckptID := range checkpoints {
		ckptDir := filepath.Join(tempDir, ckptID)
		if err := os.Mkdir(ckptDir, 0755); err != nil {
			t.Fatalf("Failed to create checkpoint dir: %v", err)
		}

		meta := NewCheckpointMetadata(ckptID)
		if err := SaveMetadata(ckptDir, meta); err != nil {
			t.Fatalf("Failed to save metadata: %v", err)
		}
	}

	// Create a directory without metadata (should be ignored)
	invalidDir := filepath.Join(tempDir, "invalid")
	if err := os.Mkdir(invalidDir, 0755); err != nil {
		t.Fatalf("Failed to create invalid dir: %v", err)
	}

	// Create a file (should be ignored)
	invalidFile := filepath.Join(tempDir, "file.txt")
	if err := os.WriteFile(invalidFile, []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create file: %v", err)
	}

	// List checkpoints
	found, err := ListCheckpoints(tempDir)
	if err != nil {
		t.Fatalf("ListCheckpoints() error = %v", err)
	}

	// Should find exactly the 3 valid checkpoints
	if len(found) != len(checkpoints) {
		t.Errorf("ListCheckpoints() found %d checkpoints, want %d", len(found), len(checkpoints))
	}

	// Check that all expected checkpoints are present
	foundMap := make(map[string]bool)
	for _, ckptID := range found {
		foundMap[ckptID] = true
	}

	for _, expected := range checkpoints {
		if !foundMap[expected] {
			t.Errorf("Expected checkpoint %s not found in results", expected)
		}
	}
}
