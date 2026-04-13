// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import "testing"

func TestApplyRestoreTargetMetadata(t *testing.T) {
	labels := map[string]string{
		CheckpointSourceLabel: "true",
		CheckpointIDLabel:     "old",
	}
	annotations := map[string]string{
		CheckpointArtifactVersionAnnotation: "old",
		CheckpointStatusAnnotation:          "completed",
		RestoreStatusAnnotation:             "failed",
		RestoreContainerIDAnnotation:        "dead-container",
		RestoreModeAnnotation:               RestoreModeManual,
		RestoreTriggerAnnotation:            "stale-trigger",
		RestoreProcessedTriggerAnnotation:   "stale-trigger",
		RestoreResumeWaitTokenAnnotation:    "stale-token",
		RestoreResumeReadyTokenAnnotation:   "stale-token",
	}

	ApplyRestoreTargetMetadata(labels, annotations, true, true, "hash", "2")

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
	if _, ok := annotations[CheckpointStatusAnnotation]; ok {
		t.Fatalf("checkpoint status annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreStatusAnnotation]; ok {
		t.Fatalf("restore status annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreContainerIDAnnotation]; ok {
		t.Fatalf("restore container id annotation was not cleared: %#v", annotations)
	}
	if annotations[RestoreModeAnnotation] != RestoreModeManual {
		t.Fatalf("expected manual restore mode annotation, got %#v", annotations)
	}
	if _, ok := annotations[RestoreTriggerAnnotation]; ok {
		t.Fatalf("restore trigger annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreProcessedTriggerAnnotation]; ok {
		t.Fatalf("restore processed trigger annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreResumeWaitTokenAnnotation]; ok {
		t.Fatalf("restore resume wait token annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreResumeReadyTokenAnnotation]; ok {
		t.Fatalf("restore resume ready token annotation was not cleared: %#v", annotations)
	}
}

func TestApplyRestoreTargetMetadataDisabledClearsState(t *testing.T) {
	labels := map[string]string{
		RestoreTargetLabel: "true",
		CheckpointIDLabel:  "hash",
	}
	annotations := map[string]string{
		CheckpointArtifactVersionAnnotation: "2",
		CheckpointStatusAnnotation:          "completed",
		RestoreStatusAnnotation:             "failed",
		RestoreContainerIDAnnotation:        "dead-container",
		RestoreModeAnnotation:               RestoreModeManual,
		RestoreTriggerAnnotation:            "stale-trigger",
		RestoreProcessedTriggerAnnotation:   "stale-trigger",
		RestoreResumeWaitTokenAnnotation:    "stale-token",
		RestoreResumeReadyTokenAnnotation:   "stale-token",
	}

	ApplyRestoreTargetMetadata(labels, annotations, false, false, "", "")

	if _, ok := labels[RestoreTargetLabel]; ok {
		t.Fatalf("restore target label was not cleared: %#v", labels)
	}
	if _, ok := labels[CheckpointIDLabel]; ok {
		t.Fatalf("checkpoint hash label was not cleared: %#v", labels)
	}
	if _, ok := annotations[CheckpointArtifactVersionAnnotation]; ok {
		t.Fatalf("checkpoint artifact version annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[CheckpointStatusAnnotation]; ok {
		t.Fatalf("checkpoint status annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreStatusAnnotation]; ok {
		t.Fatalf("restore status annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreContainerIDAnnotation]; ok {
		t.Fatalf("restore container id annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreModeAnnotation]; ok {
		t.Fatalf("restore mode annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreTriggerAnnotation]; ok {
		t.Fatalf("restore trigger annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreProcessedTriggerAnnotation]; ok {
		t.Fatalf("restore processed trigger annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreResumeWaitTokenAnnotation]; ok {
		t.Fatalf("restore resume wait token annotation was not cleared: %#v", annotations)
	}
	if _, ok := annotations[RestoreResumeReadyTokenAnnotation]; ok {
		t.Fatalf("restore resume ready token annotation was not cleared: %#v", annotations)
	}
}
