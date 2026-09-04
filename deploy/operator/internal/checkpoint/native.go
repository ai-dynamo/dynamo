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

type podSnapshotUseKind uint8

const (
	podSnapshotUseInvalid podSnapshotUseKind = iota
	podSnapshotUseExplicitReference
	podSnapshotUseManagedRestore
)

// PodSnapshotUse selects the restore-policy checks for a controller-generated
// path. It is not an authorization boundary; Kubernetes RBAC and namespaced
// Snapshot access control which callers can read or reference an artifact.
type PodSnapshotUse struct {
	kind            podSnapshotUseKind
	managedOwnerUID types.UID
}

// ExplicitPodSnapshotUse applies the public checkpointRef contract.
func ExplicitPodSnapshotUse() PodSnapshotUse {
	return PodSnapshotUse{kind: podSnapshotUseExplicitReference}
}

// ManagedPodSnapshotUse allows an owning DGD to restore its automatic
// checkpoint, including one configured for retention on deletion.
func ManagedPodSnapshotUse(ownerUID types.UID) PodSnapshotUse {
	return PodSnapshotUse{
		kind:            podSnapshotUseManagedRestore,
		managedOwnerUID: ownerUID,
	}
}

// ResolvePodSnapshotForService resolves a native PodSnapshot and validates the
// Dynamo compatibility contract for the requested use. A compatible
// but not-yet-ready snapshot is returned with Ready=false so callers can gate
// workloads while retaining an admission-time reference to the same object.
// A nil config means checkpointing is disabled. Reader must be non-nil when
// checkpointing is enabled. A nil expectedWorkerHash means the target is not a
// worker-class component; a non-nil value must contain its generation hash.
func ResolvePodSnapshotForService(
	ctx context.Context,
	reader client.Reader,
	namespace string,
	config *nvidiacomv1alpha1.ServiceCheckpointConfig,
	expectedWorkerHash *string,
	use PodSnapshotUse,
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
	if expectedWorkerHash != nil && *expectedWorkerHash == "" {
		return nil, fmt.Errorf("worker compatibility hash is required for native PodSnapshot restore")
	}

	// Read the referenced standalone Snapshot object directly.
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
	switch use.kind {
	case podSnapshotUseExplicitReference:
		if annotations[consts.CheckpointAutoAnnotation] == consts.KubeLabelValueTrue &&
			annotations[consts.CheckpointDeletionPolicyAnnotation] == string(nvidiacomv1alpha1.CheckpointDeletionPolicyRetain) {
			return nil, fmt.Errorf(
				"referenced PodSnapshot %s/%s is a retained automatic checkpoint and cannot be used as checkpointRef",
				namespace,
				snapshotName,
			)
		}
	case podSnapshotUseManagedRestore:
		// Managed resolution must name the graph incarnation claiming the artifact.
		if use.managedOwnerUID == "" {
			return nil, fmt.Errorf("managed PodSnapshot restore requires an owning DGD UID")
		}

		// Automatic snapshots remain private to the DGD incarnation that created them.
		if annotations[consts.CheckpointAutoAnnotation] != consts.KubeLabelValueTrue {
			return nil, fmt.Errorf(
				"managed PodSnapshot %s/%s is not marked as a Dynamo automatic checkpoint",
				namespace,
				snapshotName,
			)
		}
		if annotations[consts.CheckpointOwnerUIDAnnotation] != string(use.managedOwnerUID) {
			return nil, fmt.Errorf(
				"automatic PodSnapshot %s/%s belongs to DGD uid %q, not %q",
				namespace,
				snapshotName,
				annotations[consts.CheckpointOwnerUIDAnnotation],
				use.managedOwnerUID,
			)
		}
	default:
		return nil, fmt.Errorf("unsupported PodSnapshot use %d", use.kind)
	}
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
	if expectedWorkerHash != nil && workerHash != *expectedWorkerHash {
		return nil, fmt.Errorf(
			"referenced PodSnapshot %s/%s worker hash %q does not match expected hash %q",
			namespace,
			snapshotName,
			workerHash,
			*expectedWorkerHash,
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
