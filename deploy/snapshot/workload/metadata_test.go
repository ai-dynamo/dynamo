package workload

import "testing"

func TestApplyCheckpointSourceMetadata(t *testing.T) {
	labels := map[string]string{
		RestoreTargetLabel: "true",
		CheckpointIDLabel:  "old",
	}
	annotations := map[string]string{
		CheckpointArtifactVersionAnnotation: "old",
	}

	applyCheckpointSourceMetadata(labels, annotations, "hash", "2")

	if labels[CheckpointSourceLabel] != "true" {
		t.Fatalf("expected checkpoint source label, got %#v", labels)
	}
	if labels[CheckpointIDLabel] != "hash" {
		t.Fatalf("expected checkpoint hash label, got %#v", labels)
	}
	if _, ok := labels[RestoreTargetLabel]; ok {
		t.Fatalf("restore target label was not cleared: %#v", labels)
	}
	if annotations[CheckpointArtifactVersionAnnotation] != "2" {
		t.Fatalf("expected checkpoint artifact version annotation, got %#v", annotations)
	}
}

func TestApplyRestoreTargetMetadata(t *testing.T) {
	labels := map[string]string{
		CheckpointSourceLabel: "true",
		CheckpointIDLabel:     "old",
	}
	annotations := map[string]string{
		CheckpointArtifactVersionAnnotation: "old",
	}

	ApplyRestoreTargetMetadata(labels, annotations, true, "hash", "2")

	if labels[RestoreTargetLabel] != "true" {
		t.Fatalf("expected restore target label, got %#v", labels)
	}
	if labels[CheckpointIDLabel] != "hash" {
		t.Fatalf("expected checkpoint hash label, got %#v", labels)
	}
	if _, ok := labels[CheckpointSourceLabel]; ok {
		t.Fatalf("checkpoint source label was not cleared: %#v", labels)
	}
	if annotations[CheckpointArtifactVersionAnnotation] != "2" {
		t.Fatalf("expected checkpoint artifact version annotation, got %#v", annotations)
	}
}

func TestApplyRestoreTargetMetadataDisabledClearsState(t *testing.T) {
	labels := map[string]string{
		RestoreTargetLabel: "true",
		CheckpointIDLabel:  "hash",
	}
	annotations := map[string]string{
		CheckpointArtifactVersionAnnotation: "2",
	}

	ApplyRestoreTargetMetadata(labels, annotations, false, "", "")

	if _, ok := labels[RestoreTargetLabel]; ok {
		t.Fatalf("restore target label was not cleared: %#v", labels)
	}
	if _, ok := labels[CheckpointIDLabel]; ok {
		t.Fatalf("checkpoint hash label was not cleared: %#v", labels)
	}
	if _, ok := annotations[CheckpointArtifactVersionAnnotation]; ok {
		t.Fatalf("checkpoint artifact version annotation was not cleared: %#v", annotations)
	}
}
