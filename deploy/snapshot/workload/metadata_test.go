package workload

import "testing"

func TestApplyCheckpointSourceMetadata(t *testing.T) {
	labels := map[string]string{
		RestoreTargetLabel:  "true",
		CheckpointHashLabel: "old",
	}
	annotations := map[string]string{
		CheckpointLocationAnnotation: "old",
		CheckpointStorageAnnotation:  "old",
	}

	applyCheckpointSourceMetadata(labels, annotations, "hash", "/checkpoints/hash", StorageTypePVC)

	if labels[CheckpointSourceLabel] != "true" {
		t.Fatalf("expected checkpoint source label, got %#v", labels)
	}
	if labels[CheckpointHashLabel] != "hash" {
		t.Fatalf("expected checkpoint hash label, got %#v", labels)
	}
	if _, ok := labels[RestoreTargetLabel]; ok {
		t.Fatalf("restore target label was not cleared: %#v", labels)
	}
	if annotations[CheckpointLocationAnnotation] != "/checkpoints/hash" {
		t.Fatalf("expected checkpoint location annotation, got %#v", annotations)
	}
	if annotations[CheckpointStorageAnnotation] != StorageTypePVC {
		t.Fatalf("expected checkpoint storage annotation, got %#v", annotations)
	}
}

func TestApplyRestoreTargetMetadata(t *testing.T) {
	labels := map[string]string{
		CheckpointSourceLabel: "true",
		CheckpointHashLabel:   "old",
	}
	annotations := map[string]string{
		CheckpointLocationAnnotation: "old",
		CheckpointStorageAnnotation:  "old",
	}

	ApplyRestoreTargetMetadata(labels, annotations, true, "hash", "/checkpoints/hash", StorageTypePVC)

	if labels[RestoreTargetLabel] != "true" {
		t.Fatalf("expected restore target label, got %#v", labels)
	}
	if labels[CheckpointHashLabel] != "hash" {
		t.Fatalf("expected checkpoint hash label, got %#v", labels)
	}
	if _, ok := labels[CheckpointSourceLabel]; ok {
		t.Fatalf("checkpoint source label was not cleared: %#v", labels)
	}
	if annotations[CheckpointLocationAnnotation] != "/checkpoints/hash" {
		t.Fatalf("expected checkpoint location annotation, got %#v", annotations)
	}
	if annotations[CheckpointStorageAnnotation] != StorageTypePVC {
		t.Fatalf("expected checkpoint storage annotation, got %#v", annotations)
	}
}

func TestApplyRestoreTargetMetadataDisabledClearsState(t *testing.T) {
	labels := map[string]string{
		RestoreTargetLabel:  "true",
		CheckpointHashLabel: "hash",
	}
	annotations := map[string]string{
		CheckpointLocationAnnotation: "/checkpoints/hash",
		CheckpointStorageAnnotation:  StorageTypePVC,
	}

	ApplyRestoreTargetMetadata(labels, annotations, false, "", "", "")

	if _, ok := labels[RestoreTargetLabel]; ok {
		t.Fatalf("restore target label was not cleared: %#v", labels)
	}
	if _, ok := labels[CheckpointHashLabel]; ok {
		t.Fatalf("checkpoint hash label was not cleared: %#v", labels)
	}
	if _, ok := annotations[CheckpointLocationAnnotation]; ok {
		t.Fatalf("checkpoint location annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[CheckpointStorageAnnotation]; ok {
		t.Fatalf("checkpoint storage annotation was not cleared: %#v", annotations)
	}
}
