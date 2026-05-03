// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package protocol

import (
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

const (
	// CheckpointSourceLabel tags the Job pod template (i.e. the pod whose
	// checkpoint gets written to disk). The snapshot agent's checkpoint
	// informer selects on this label.
	CheckpointSourceLabel = "nvidia.com/snapshot-is-checkpoint-source"

	// CheckpointIDLabel is the 16-character identity hash (or a generated
	// manual ID for snapshotctl flows). Both the checkpoint Job pod and
	// the restore pod carry it; the CheckpointSourceLabel is what
	// distinguishes them.
	//
	// Restore pods are therefore selected as
	//   CheckpointIDLabel exists AND NOT CheckpointSourceLabel exists
	// which is cleaner than carrying a redundant "is restore target"
	// boolean label in parallel with the target-containers annotation.
	CheckpointIDLabel = "nvidia.com/snapshot-checkpoint-id"

	CheckpointArtifactVersionAnnotation = "nvidia.com/snapshot-artifact-version"

	// GMSCheckpointDirAnnotation tells snapshot-agent where the companion
	// GMS saver/loader sidecar writes its lifecycle sentinel on the checkpoint
	// PVC. When absent, the snapshot flow has no GMS barrier.
	GMSCheckpointDirAnnotation = "nvidia.com/snapshot-gms-checkpoint-dir"

	// GMSCompletionFileModeAnnotation controls whether snapshot-agent waits
	// for a pod-UID-scoped GMS completion file or for a shared file written
	// by a separate inter-pod GMS weight-server pod.
	GMSCompletionFileModeAnnotation = "nvidia.com/snapshot-gms-completion-file-mode"
	GMSCompletionFileModePodUID     = "pod-uid"
	GMSCompletionFileModeShared     = "shared"

	// TargetContainersAnnotation names the container(s) a checkpoint or
	// restore operation should act on. It is required — snapshotprotocol /
	// snapshotctl / snapshot-agent all error out when the annotation is
	// missing. Comma-separated list of container names in a single pod:
	//
	//   nvidia.com/snapshot-target-containers = "engine-0,engine-1"
	//
	// Checkpoint Jobs must carry exactly one target container (the snapshot
	// contract captures one workload container per checkpoint). Restore pods
	// may carry one or more target containers; the agent replays the same
	// checkpoint into each of them. The operator stamps the annotation for
	// both user-facing paths (DynamoCheckpoint Jobs and DGD restore pods).
	TargetContainersAnnotation = "nvidia.com/snapshot-target-containers"

	// RestoreModeAnnotation opts a restore target into benchmark-controlled
	// restore triggering. The default path ignores RestoreTriggerAnnotation.
	RestoreModeAnnotation    = "nvidia.com/snapshot-restore-mode"
	RestoreModeManual        = "manual"
	RestoreTriggerAnnotation = "nvidia.com/snapshot-restore-trigger"

	// CheckpointStatusAnnotation is written by snapshot-agent on the
	// checkpoint Job once the (single) target container's checkpoint either
	// completes or fails. Watched by the operator and snapshotctl.
	CheckpointStatusAnnotation = "nvidia.com/snapshot-checkpoint-status"

	// RestoreStatusAnnotationPrefix is the prefix for per-container restore
	// status annotations written by snapshot-agent onto the restore pod.
	// The full key is "nvidia.com/snapshot-restore-status.<containerName>",
	// one per target container. Use RestoreStatusAnnotationFor to build it.
	RestoreStatusAnnotationPrefix = "nvidia.com/snapshot-restore-status."

	// RestoreContainerIDAnnotationPrefix is the prefix for per-container
	// containerd container-ID annotations used to dedupe restore attempts
	// across kubelet container restarts. Full key is
	// "nvidia.com/snapshot-restore-container-id.<containerName>".
	RestoreContainerIDAnnotationPrefix = "nvidia.com/snapshot-restore-container-id."

	// RestoreProcessedTriggerAnnotationPrefix stores the last consumed manual
	// trigger per target container so a pod restart does not replay the same
	// benchmark trigger unless the harness writes a fresh token.
	RestoreProcessedTriggerAnnotationPrefix = "nvidia.com/snapshot-restore-processed-trigger."

	CheckpointVolumeName             = "checkpoint-storage"
	DefaultCheckpointArtifactVersion = "1"
	DefaultCheckpointJobTTLSeconds   = int32(300)
	DefaultSeccompLocalhostProfile   = "profiles/block-iouring.json"
	StorageTypePVC                   = "pvc"

	CheckpointStatusCompleted = "completed"
	CheckpointStatusFailed    = "failed"
	RestoreStatusInProgress   = "in_progress"
	RestoreStatusCompleted    = "completed"
	RestoreStatusFailed       = "failed"
	GMSSaveCompleteFile       = "gms-save-complete"
	GMSLoadCompleteFile       = "gms-load-complete"
)

type Storage struct {
	Type     string
	Location string
	PVCName  string
	BasePath string
}

// findContainerByName returns a pointer to the named container in the slice,
// or nil if not found. Used by the protocol helpers to look up target
// containers declared in the snapshot-target-containers annotation.
func findContainerByName(containers []corev1.Container, name string) *corev1.Container {
	for i := range containers {
		if containers[i].Name == name {
			return &containers[i]
		}
	}
	return nil
}

func ArtifactVersion(version string) string {
	version = strings.TrimSpace(version)
	if version == "" {
		return DefaultCheckpointArtifactVersion
	}
	return version
}

func ResolveCheckpointStorage(checkpointID string, version string, storage Storage) (Storage, error) {
	resolved, err := resolveStorageConfig(storage)
	if err != nil {
		return Storage{}, err
	}
	resolved.Location = strings.TrimRight(resolved.BasePath, "/") + "/" + checkpointID + "/versions/" + ArtifactVersion(version)
	return resolved, nil
}

func ResolveRestoreStorage(checkpointID string, version string, location string, storage Storage) (Storage, error) {
	resolved, err := resolveStorageConfig(storage)
	if err != nil {
		return Storage{}, err
	}
	location = strings.TrimSpace(location)
	if location == "" {
		return ResolveCheckpointStorage(checkpointID, version, storage)
	}
	resolved.Location = location
	return resolved, nil
}

// RestoreStatusAnnotationFor returns the per-container restore status
// annotation key. Safe to call with any container name.
func RestoreStatusAnnotationFor(containerName string) string {
	return RestoreStatusAnnotationPrefix + containerName
}

// RestoreContainerIDAnnotationFor returns the per-container containerd-ID
// annotation key used by the restore dedupe path.
func RestoreContainerIDAnnotationFor(containerName string) string {
	return RestoreContainerIDAnnotationPrefix + containerName
}

func RestoreProcessedTriggerAnnotationFor(containerName string) string {
	return RestoreProcessedTriggerAnnotationPrefix + containerName
}

// FormatTargetContainers renders a target-container list into the canonical
// comma-separated annotation value. Whitespace is trimmed, empty names are
// dropped, and duplicates are preserved in input order (callers are
// responsible for providing a sensible list).
func FormatTargetContainers(names []string) string {
	cleaned := make([]string, 0, len(names))
	for _, name := range names {
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		cleaned = append(cleaned, name)
	}
	return strings.Join(cleaned, ",")
}

// ParseTargetContainers returns the list of target container names encoded
// in the nvidia.com/snapshot-target-containers annotation, in order. A
// missing or empty annotation returns an empty slice; callers decide whether
// that is an error. Whitespace is trimmed and duplicate names are rejected.
func ParseTargetContainers(value string) ([]string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil, nil
	}
	parts := strings.Split(value, ",")
	seen := make(map[string]struct{}, len(parts))
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		name := strings.TrimSpace(part)
		if name == "" {
			return nil, fmt.Errorf("empty container name in %s=%q", TargetContainersAnnotation, value)
		}
		if _, dup := seen[name]; dup {
			return nil, fmt.Errorf("duplicate container name %q in %s=%q", name, TargetContainersAnnotation, value)
		}
		seen[name] = struct{}{}
		out = append(out, name)
	}
	return out, nil
}

// TargetContainersFromAnnotations reads the target-container list from an
// annotation map. It enforces the snapshot contract: the annotation is
// required, and at least minCount / at most maxCount names must be present
// (maxCount == 0 means "no upper bound").
func TargetContainersFromAnnotations(annotations map[string]string, minCount, maxCount int) ([]string, error) {
	raw, ok := annotations[TargetContainersAnnotation]
	if !ok || strings.TrimSpace(raw) == "" {
		return nil, fmt.Errorf("missing required %s annotation", TargetContainersAnnotation)
	}
	names, err := ParseTargetContainers(raw)
	if err != nil {
		return nil, err
	}
	if minCount > 0 && len(names) < minCount {
		return nil, fmt.Errorf("%s must list at least %d container name(s), got %d", TargetContainersAnnotation, minCount, len(names))
	}
	if maxCount > 0 && len(names) > maxCount {
		return nil, fmt.Errorf("%s must list at most %d container name(s), got %d", TargetContainersAnnotation, maxCount, len(names))
	}
	return names, nil
}

// clearRestoreStatusKeys drops every restore status annotation from the map.
// Used when re-applying restore-target metadata before a new restore so stale
// values from a previous run do not leak into observation.
func clearRestoreStatusKeys(annotations map[string]string) {
	delete(annotations, "nvidia.com/snapshot-restore-status")
	delete(annotations, "nvidia.com/snapshot-restore-container-id")
	for key := range annotations {
		if strings.HasPrefix(key, RestoreStatusAnnotationPrefix) ||
			strings.HasPrefix(key, RestoreContainerIDAnnotationPrefix) ||
			strings.HasPrefix(key, RestoreProcessedTriggerAnnotationPrefix) {
			delete(annotations, key)
		}
	}
}

// ApplyRestoreTargetMetadata resets restore-related labels/annotations and
// (when enabled) stamps the checkpoint-id label + artifact version
// annotation. A pod is identified as a restore target by the snapshot agent
// via (CheckpointIDLabel present, CheckpointSourceLabel absent); there is
// no dedicated "is restore target" label. The nvidia.com/snapshot-target-
// containers annotation is the caller's responsibility: the operator stamps
// it based on failover vs non-failover intent, snapshotctl stamps it from
// --containers, etc. This helper never touches it so callers can set it
// before or after with no ordering surprise.
func ApplyRestoreTargetMetadata(labels map[string]string, annotations map[string]string, enabled bool, manualTrigger bool, checkpointID string, artifactVersion string) {
	delete(labels, CheckpointSourceLabel)
	delete(labels, CheckpointIDLabel)
	delete(annotations, CheckpointArtifactVersionAnnotation)
	delete(annotations, CheckpointStatusAnnotation)
	delete(annotations, RestoreModeAnnotation)
	delete(annotations, RestoreTriggerAnnotation)
	clearRestoreStatusKeys(annotations)

	if !enabled {
		return
	}

	if checkpointID != "" {
		labels[CheckpointIDLabel] = checkpointID
	}
	annotations[CheckpointArtifactVersionAnnotation] = ArtifactVersion(artifactVersion)
	if manualTrigger {
		annotations[RestoreModeAnnotation] = RestoreModeManual
	}
}

func applyCheckpointSourceMetadata(labels map[string]string, annotations map[string]string, checkpointID string, artifactVersion string) {
	delete(labels, CheckpointIDLabel)
	delete(annotations, CheckpointArtifactVersionAnnotation)

	labels[CheckpointSourceLabel] = "true"
	if checkpointID != "" {
		labels[CheckpointIDLabel] = checkpointID
	}
	annotations[CheckpointArtifactVersionAnnotation] = ArtifactVersion(artifactVersion)
}

func resolveStorageConfig(storage Storage) (Storage, error) {
	storageType := strings.TrimSpace(storage.Type)
	if storageType == "" {
		storageType = StorageTypePVC
	}
	if storageType != StorageTypePVC {
		return Storage{}, fmt.Errorf("checkpoint storage type %q is not supported", storageType)
	}
	basePath := strings.TrimSpace(storage.BasePath)
	if basePath == "" {
		return Storage{}, fmt.Errorf("checkpoint base path is required")
	}
	return Storage{
		Type:     storageType,
		PVCName:  strings.TrimSpace(storage.PVCName),
		BasePath: strings.TrimRight(basePath, "/"),
	}, nil
}
