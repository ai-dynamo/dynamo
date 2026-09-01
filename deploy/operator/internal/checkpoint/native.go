/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"context"
	"fmt"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// ResolvedPodSnapshot carries the artifact identity and compatibility-sensitive
// portion of a PodSnapshot observation from reconciliation to Pod admission.
type ResolvedPodSnapshot struct {
	UID                  types.UID
	BoundContentName     string
	SourceContainer      string
	CompatibilityVersion string
	GMSMode              string
}

// ResolvePodSnapshotForService resolves an explicit checkpointRef as a native
// PodSnapshot and validates the Dynamo compatibility contract. A compatible
// but not-yet-ready snapshot is returned with Ready=false so callers can gate
// workloads while retaining an admission-time reference to the same object.
// A nil config means checkpointing is disabled. Reader must be non-nil when
// checkpointing is enabled.
func ResolvePodSnapshotForService(
	ctx context.Context,
	reader client.Reader,
	namespace string,
	config *nvidiacomv1alpha1.ServiceCheckpointConfig,
	expectedWorkerHash string,
) (*CheckpointInfo, error) {
	if config == nil || !config.Enabled {
		return &CheckpointInfo{Enabled: false}, nil
	}
	if reader == nil {
		return nil, fmt.Errorf("PodSnapshot client is required")
	}
	if config.CheckpointRef == nil || strings.TrimSpace(*config.CheckpointRef) == "" {
		return nil, fmt.Errorf("checkpointRef is required for native PodSnapshot restore")
	}
	if expectedWorkerHash == "" {
		return nil, fmt.Errorf("worker compatibility hash is required for native PodSnapshot restore")
	}

	// Read the referenced object without falling back to a legacy resource with
	// the same name; the installed Dynamo release defines the API boundary.
	snapshotName := strings.TrimSpace(*config.CheckpointRef)
	snapshot := &snapshotv1alpha1.PodSnapshot{}
	if err := reader.Get(ctx, types.NamespacedName{Namespace: namespace, Name: snapshotName}, snapshot); err != nil {
		return nil, fmt.Errorf("get referenced PodSnapshot %s/%s: %w", namespace, snapshotName, err)
	}
	if snapshot.UID == "" {
		return nil, fmt.Errorf("referenced PodSnapshot %s/%s has no UID", namespace, snapshotName)
	}
	if snapshotv1alpha1.IsPodSnapshotFailed(snapshot) {
		return nil, fmt.Errorf("referenced PodSnapshot %s/%s has failed", namespace, snapshotName)
	}

	// v1alpha1 captures exactly one source container. Validate defensively so a
	// malformed object cannot produce an ambiguous fan-out mapping.
	containers := snapshot.Spec.Source.PodRef.Containers
	if len(containers) != 1 || strings.TrimSpace(containers[0]) == "" {
		return nil, fmt.Errorf("referenced PodSnapshot %s/%s must identify exactly one source container", namespace, snapshotName)
	}

	// Compatibility metadata belongs to Dynamo and is deliberately validated
	// independently of Snapshot's generic capture and restore protocol.
	annotations := snapshot.GetAnnotations()
	version := annotations[consts.SnapshotCompatibilityVersionAnnotation]
	if version != consts.SnapshotCompatibilityVersion {
		return nil, fmt.Errorf(
			"referenced PodSnapshot %s/%s has unsupported Dynamo compatibility version %q",
			namespace,
			snapshotName,
			version,
		)
	}
	workerHash := annotations[consts.SnapshotWorkerHashAnnotation]
	if workerHash != expectedWorkerHash {
		return nil, fmt.Errorf(
			"referenced PodSnapshot %s/%s worker hash %q does not match expected hash %q",
			namespace,
			snapshotName,
			workerHash,
			expectedWorkerHash,
		)
	}
	gmsMode := annotations[consts.SnapshotGMSModeAnnotation]
	gmsSpec, err := gpuMemoryServiceFromSnapshotMode(gmsMode)
	if err != nil {
		return nil, fmt.Errorf("referenced PodSnapshot %s/%s: %w", namespace, snapshotName, err)
	}

	// A Ready snapshot must have a bound immutable content identity. Before it
	// becomes Ready, keep the empty content name and let reconciliation gate Pods.
	ready := snapshotv1alpha1.IsPodSnapshotSucceeded(snapshot)
	contentName := ""
	if snapshot.Status.BoundPodSnapshotContentName != nil {
		contentName = strings.TrimSpace(*snapshot.Status.BoundPodSnapshotContentName)
	}
	if ready && contentName == "" {
		return nil, fmt.Errorf("Ready PodSnapshot %s/%s has no bound PodSnapshotContent", namespace, snapshotName)
	}

	startupPolicy := config.StartupPolicy
	if startupPolicy == "" {
		startupPolicy = nvidiacomv1alpha1.CheckpointStartupPolicyImmediate
	}
	info := &CheckpointInfo{
		Enabled:          true,
		Exists:           true,
		SourceKind:       SourceKindPodSnapshot,
		GPUMemoryService: gmsSpec,
		CheckpointName:   snapshot.Name,
		Ready:            ready,
		StartupPolicy:    startupPolicy,
		NativeSnapshot: &ResolvedPodSnapshot{
			UID:                  snapshot.UID,
			BoundContentName:     contentName,
			SourceContainer:      containers[0],
			CompatibilityVersion: version,
			GMSMode:              gmsMode,
		},
	}
	if config.TargetContainerName != "" {
		info.RestoreTargetContainers = []string{config.TargetContainerName}
	}
	return info, nil
}

func gpuMemoryServiceFromSnapshotMode(mode string) (*nvidiacomv1alpha1.GPUMemoryServiceSpec, error) {
	switch mode {
	case consts.SnapshotGMSModeDisabled:
		return nil, nil
	case string(nvidiacomv1alpha1.GMSModeIntraPod):
		return &nvidiacomv1alpha1.GPUMemoryServiceSpec{
			Enabled: true,
			Mode:    nvidiacomv1alpha1.GMSModeIntraPod,
		}, nil
	case string(nvidiacomv1alpha1.GMSModeInterPod):
		return &nvidiacomv1alpha1.GPUMemoryServiceSpec{
			Enabled: true,
			Mode:    nvidiacomv1alpha1.GMSModeInterPod,
		}, nil
	default:
		return nil, fmt.Errorf("Dynamo GMS mode %q is unsupported", mode)
	}
}
