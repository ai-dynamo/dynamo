/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package checkpoint

import (
	"context"
	"fmt"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// getCheckpointInfoFromCheckpoint extracts CheckpointInfo from a DynamoCheckpoint CR
func getCheckpointInfoFromCheckpoint(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (*CheckpointInfo, error) {
	hash, err := getCheckpointIdentityHash(ckpt)
	if err != nil {
		return nil, err
	}

	info := &CheckpointInfo{
		Enabled:        true,
		Exists:         true,
		Identity:       &ckpt.Spec.Identity,
		Hash:           hash,
		Location:       ckpt.Status.Location,
		StorageType:    ckpt.Status.StorageType,
		CheckpointName: ckpt.Name,
		Ready:          ckpt.Status.Phase == nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
	}

	return info, nil
}

func getCheckpointIdentityHash(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) (string, error) {
	if ckpt.Status.IdentityHash != "" {
		return ckpt.Status.IdentityHash, nil
	}

	computedHash, err := ComputeIdentityHash(ckpt.Spec.Identity)
	if err != nil {
		return "", fmt.Errorf("failed to compute checkpoint hash for %s: %w", ckpt.Name, err)
	}

	return computedHash, nil
}

// FindCheckpointByIdentityHash looks up a checkpoint by identity hash and tolerates
// older or partially reconciled checkpoint objects that do not yet have the hash label
// or status.identityHash backfilled.
func FindCheckpointByIdentityHash(
	ctx context.Context,
	c client.Client,
	namespace string,
	hash string,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	checkpoints := &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err := c.List(
		ctx,
		checkpoints,
		client.InNamespace(namespace),
		client.MatchingLabels{consts.KubeLabelCheckpointHash: hash},
	); err != nil {
		return nil, fmt.Errorf("failed to list checkpoints by hash label: %w", err)
	}

	var existing *nvidiacomv1alpha1.DynamoCheckpoint
	seen := make(map[string]struct{}, len(checkpoints.Items))
	for i := range checkpoints.Items {
		if existing != nil {
			return nil, fmt.Errorf("multiple checkpoints found for identity hash %s", hash)
		}
		existing = checkpoints.Items[i].DeepCopy()
		seen[checkpoints.Items[i].Name] = struct{}{}
	}
	if existing != nil {
		return existing, nil
	}

	checkpoints = &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err := c.List(ctx, checkpoints, client.InNamespace(namespace)); err != nil {
		return nil, fmt.Errorf("failed to list checkpoints: %w", err)
	}
	for i := range checkpoints.Items {
		if _, ok := seen[checkpoints.Items[i].Name]; ok {
			continue
		}

		existingHash, err := getCheckpointIdentityHash(&checkpoints.Items[i])
		if err != nil {
			return nil, err
		}
		if existingHash != hash {
			continue
		}
		if existing != nil {
			return nil, fmt.Errorf("multiple checkpoints found for identity hash %s", hash)
		}
		existing = checkpoints.Items[i].DeepCopy()
	}

	return existing, nil
}

// CreateOrGetAutoCheckpoint returns the canonical checkpoint object for an identity.
// If a checkpoint already exists for the identity hash, it is reused as-is rather than
// mutating its capture pod template.
func CreateOrGetAutoCheckpoint(
	ctx context.Context,
	c client.Client,
	namespace string,
	identity nvidiacomv1alpha1.DynamoCheckpointIdentity,
	podTemplate corev1.PodTemplateSpec,
) (*nvidiacomv1alpha1.DynamoCheckpoint, error) {
	hash, err := ComputeIdentityHash(identity)
	if err != nil {
		return nil, fmt.Errorf("failed to compute identity hash: %w", err)
	}

	existing, err := FindCheckpointByIdentityHash(ctx, c, namespace, hash)
	if err != nil {
		return nil, err
	}
	if existing != nil {
		return existing, nil
	}

	ckptName := fmt.Sprintf("checkpoint-%s", hash)
	ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ckptName,
			Namespace: namespace,
			Labels: map[string]string{
				consts.KubeLabelCheckpointHash: hash,
			},
		},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity: identity,
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: podTemplate,
			},
		},
	}

	if err := c.Create(ctx, ckpt); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return nil, fmt.Errorf("failed to create checkpoint %s: %w", ckptName, err)
		}

		existing = &nvidiacomv1alpha1.DynamoCheckpoint{}
		key := types.NamespacedName{Name: ckptName, Namespace: namespace}
		if getErr := c.Get(ctx, key, existing); getErr != nil {
			return nil, fmt.Errorf("failed to get checkpoint %s after already exists: %w", ckptName, getErr)
		}

		existingHash, err := getCheckpointIdentityHash(existing)
		if err != nil {
			return nil, err
		}
		if existingHash != hash {
			return nil, fmt.Errorf("checkpoint %s already exists with identity hash %s", ckptName, existingHash)
		}
		return existing, nil
	}

	return ckpt, nil
}

// getPVCBasePath returns the PVC base path from storage config.
// Only applicable for PVC storage type
func getPVCBasePath(storageConfig *configv1alpha1.CheckpointStorageConfiguration) string {
	if storageConfig != nil && storageConfig.PVC.BasePath != "" {
		return storageConfig.PVC.BasePath
	}
	return ""
}

// GetPVCBasePath returns the configured PVC base path from controller config.
// This is used by both CheckpointReconciler and DynamoGraphDeploymentReconciler.
// Only applicable for PVC storage type.
func GetPVCBasePath(config *configv1alpha1.CheckpointConfiguration) string {
	if config != nil {
		return getPVCBasePath(&config.Storage)
	}
	return ""
}

// CheckpointInfo contains resolved checkpoint information for a DGD service
type CheckpointInfo struct {
	// Enabled indicates if checkpointing is enabled
	Enabled bool
	// Exists indicates whether a DynamoCheckpoint CR already exists for this service.
	Exists bool
	// Identity is the resolved checkpoint identity (model, framework, etc.)
	Identity *nvidiacomv1alpha1.DynamoCheckpointIdentity
	// Hash is the computed identity hash
	Hash string
	// Location is the full URI/path in the storage backend
	Location string
	// StorageType is the storage backend type (pvc, s3, oci)
	StorageType nvidiacomv1alpha1.DynamoCheckpointStorageType
	// CheckpointName is the name of the Checkpoint CR
	CheckpointName string
	// Ready indicates if the checkpoint is ready for use
	Ready bool
}

// ResolveCheckpointForService resolves checkpoint information for a DGD service.
// It handles both checkpointRef (direct reference) and identity-based lookup.
// Returns CheckpointInfo with the resolved identity populated.
func ResolveCheckpointForService(
	ctx context.Context,
	c client.Client,
	namespace string,
	config *nvidiacomv1alpha1.ServiceCheckpointConfig,
) (*CheckpointInfo, error) {
	if config == nil || !config.Enabled {
		return &CheckpointInfo{Enabled: false}, nil
	}

	// If a direct checkpoint reference is provided, use it
	if config.CheckpointRef != nil && *config.CheckpointRef != "" {
		ckpt := &nvidiacomv1alpha1.DynamoCheckpoint{}
		err := c.Get(ctx, types.NamespacedName{
			Namespace: namespace,
			Name:      *config.CheckpointRef,
		}, ckpt)
		if err != nil {
			return nil, fmt.Errorf("failed to get referenced checkpoint %s: %w", *config.CheckpointRef, err)
		}

		// Extract all checkpoint info including identity from the CR
		return getCheckpointInfoFromCheckpoint(ckpt)
	}

	// Otherwise, compute hash from identity and look up checkpoint
	if config.Identity == nil {
		return nil, fmt.Errorf("checkpoint enabled but no checkpointRef or identity provided")
	}

	hash, err := ComputeIdentityHash(*config.Identity)
	if err != nil {
		return nil, fmt.Errorf("failed to compute identity hash: %w", err)
	}

	info := &CheckpointInfo{
		Enabled:  true,
		Identity: config.Identity,
		Hash:     hash,
	}

	existing, err := FindCheckpointByIdentityHash(ctx, c, namespace, hash)
	if err != nil {
		return nil, err
	}
	if existing != nil {
		foundInfo, err := getCheckpointInfoFromCheckpoint(existing)
		if err != nil {
			return nil, err
		}
		foundInfo.Identity = config.Identity
		return foundInfo, nil
	}

	return info, nil
}

// InjectCheckpointEnvVars adds checkpoint-related environment variables to a restored/DGD container.
// Sets PATH and HASH so the restored process knows its checkpoint identity.
// DYN_CHECKPOINT_LOCATION is reserved for future S3/OCI support.
func InjectCheckpointEnvVars(container *corev1.Container, info *CheckpointInfo, checkpointConfig *configv1alpha1.CheckpointConfiguration) {
	if !info.Enabled {
		return
	}

	var envVars []corev1.EnvVar

	// For PVC storage: inject base path so the restored process knows its checkpoint location.
	// For S3/OCI (future): inject DYN_CHECKPOINT_LOCATION directly.
	storageType := configv1alpha1.CheckpointStorageTypePVC
	if checkpointConfig != nil && checkpointConfig.Storage.Type != "" {
		storageType = checkpointConfig.Storage.Type
	}

	switch storageType {
	case configv1alpha1.CheckpointStorageTypePVC:
		basePath := ""
		if checkpointConfig != nil {
			basePath = getPVCBasePath(&checkpointConfig.Storage)
		}
		envVars = append(envVars, corev1.EnvVar{
			Name:  consts.EnvCheckpointPath,
			Value: basePath,
		})
	default:
		// S3/OCI: inject full location URI directly
		if info.Location != "" {
			envVars = append(envVars, corev1.EnvVar{
				Name:  consts.EnvCheckpointLocation,
				Value: info.Location,
			})
		}
	}

	if info.Hash != "" {
		envVars = append(envVars, corev1.EnvVar{
			Name:  consts.EnvCheckpointHash,
			Value: info.Hash,
		})
	}

	// Prepend checkpoint env vars to ensure they're available
	container.Env = append(envVars, container.Env...)
}

// InjectCheckpointVolume adds the checkpoint PVC volume to a pod spec
func InjectCheckpointVolume(podSpec *corev1.PodSpec, pvcName string) {
	// Check if volume already exists
	for _, v := range podSpec.Volumes {
		if v.Name == consts.CheckpointVolumeName {
			return
		}
	}

	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: consts.CheckpointVolumeName,
		VolumeSource: corev1.VolumeSource{
			PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvcName,
				ReadOnly:  false, // CRIU needs write access during restore
			},
		},
	})
}

// InjectCheckpointVolumeMount adds the checkpoint volume mount to a container
func InjectCheckpointVolumeMount(container *corev1.Container, basePath string) {
	// Check if mount already exists
	for _, m := range container.VolumeMounts {
		if m.Name == consts.CheckpointVolumeName {
			return
		}
	}

	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      consts.CheckpointVolumeName,
		MountPath: basePath,
		ReadOnly:  false, // CRIU needs write access for restore.log and restore-criu.conf
	})
}

// InjectPodInfoVolume adds a Downward API volume for pod identity and restore-time
// namespace.
// This is critical for CRIU checkpoint/restore scenarios where environment variables
// contain stale values from the checkpoint source pod. The Downward API files
// always reflect the current pod's identity and DGD configuration.
func InjectPodInfoVolume(podSpec *corev1.PodSpec) {
	// Check if volume already exists
	for _, v := range podSpec.Volumes {
		if v.Name == consts.PodInfoVolumeName {
			return
		}
	}

	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: consts.PodInfoVolumeName,
		VolumeSource: corev1.VolumeSource{
			DownwardAPI: &corev1.DownwardAPIVolumeSource{
				Items: []corev1.DownwardAPIVolumeFile{
					// Pod identity fields
					{
						Path: "pod_name",
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: consts.PodInfoFieldPodName,
						},
					},
					{
						Path: "pod_uid",
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: consts.PodInfoFieldPodUID,
						},
					},
					{
						Path: "pod_namespace",
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: consts.PodInfoFieldPodNamespace,
						},
					},
					// Restore namespace and worker suffix from current pod labels
					// so the restored process can rebuild its current worker namespace.
					{
						Path: consts.PodInfoFileDynNamespace,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.labels['" + consts.KubeLabelDynamoNamespace + "']",
						},
					},
					{
						Path: consts.PodInfoFileDynNamespaceWorkerSuffix,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.labels['" + consts.KubeLabelDynamoWorkerHash + "']",
						},
					},
				},
			},
		},
	})
}

// InjectPodInfoVolumeMount adds the Downward API volume mount to a container.
func InjectPodInfoVolumeMount(container *corev1.Container) {
	// Check if mount already exists
	for _, m := range container.VolumeMounts {
		if m.Name == consts.PodInfoVolumeName {
			return
		}
	}

	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      consts.PodInfoVolumeName,
		MountPath: consts.PodInfoMountPath,
		ReadOnly:  true,
	})
}

// InjectCheckpointIntoPodSpec injects checkpoint configuration into a pod spec for
// external restore via the snapshot DaemonSet. The pod image is expected to be a
// runtime-compatible restore image (runtime + CRIU tooling). For ready checkpoints,
// the operator overrides command to `sleep infinity` so the watcher can trigger
// external restore via nsenter + nsrestore.
//
// Modifications applied:
//  1. Security context - seccomp profile (io_uring blocking, matches checkpoint environment)
//  2. Environment variables - checkpoint path and hash
//  3. Storage configuration - checkpoint PVC and Downward API (pod identity)
//
// No hostIPC, no privileged mode — those are only needed when CRIU runs inside the
// container. With external restore, all privilege lives in the DaemonSet.
func InjectCheckpointIntoPodSpec(
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
	checkpointConfig *configv1alpha1.CheckpointConfiguration,
) error {
	if checkpointInfo == nil || !checkpointInfo.Enabled {
		return nil
	}

	info := checkpointInfo
	if info.Hash == "" {
		if info.Identity == nil {
			return fmt.Errorf("checkpoint enabled but identity is nil and hash is not set")
		}
		hash, err := ComputeIdentityHash(*info.Identity)
		if err != nil {
			return fmt.Errorf("failed to compute identity hash: %w", err)
		}
		info.Hash = hash
	}

	// Find the main container (needed for volume mounts and env vars)
	var mainContainer *corev1.Container
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == consts.MainContainerName {
			mainContainer = &podSpec.Containers[i]
			break
		}
	}
	if mainContainer == nil && len(podSpec.Containers) > 0 {
		mainContainer = &podSpec.Containers[0]
	}
	if mainContainer == nil {
		return fmt.Errorf("no container found to inject checkpoint config")
	}

	// When a ready checkpoint exists, override the container command to sleep infinity.
	// The DaemonSet watcher detects this pod via the checkpoint-restore label and
	// performs external restore (nsenter + nsrestore). When no checkpoint is ready,
	// the original command runs (cold start).
	if info.Ready {
		mainContainer.Command = []string{"sleep", "infinity"}
		mainContainer.Args = nil
	}

	// Seccomp profile to match checkpoint environment (blocks io_uring syscalls)
	if podSpec.SecurityContext == nil {
		podSpec.SecurityContext = &corev1.PodSecurityContext{}
	}
	podSpec.SecurityContext.SeccompProfile = &corev1.SeccompProfile{
		Type:             corev1.SeccompProfileTypeLocalhost,
		LocalhostProfile: ptr.To(consts.SeccompProfilePath),
	}

	// Determine storage type and compute location/path
	storageType := configv1alpha1.CheckpointStorageTypePVC // default
	var storageConfig *configv1alpha1.CheckpointStorageConfiguration
	if checkpointConfig != nil {
		storageConfig = &checkpointConfig.Storage
		if storageConfig.Type != "" {
			storageType = storageConfig.Type
		}
	}

	switch storageType {
	case configv1alpha1.CheckpointStorageTypeS3:
		info.StorageType = nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType)
		if storageConfig == nil || storageConfig.S3.URI == "" {
			return fmt.Errorf("S3 storage type selected but no S3 URI configured (set checkpoint.storage.s3.uri)")
		}
		info.Location = fmt.Sprintf("%s/%s.tar", storageConfig.S3.URI, info.Hash)

	case configv1alpha1.CheckpointStorageTypeOCI:
		info.StorageType = nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType)
		if storageConfig == nil || storageConfig.OCI.URI == "" {
			return fmt.Errorf("OCI storage type selected but no OCI URI configured (set checkpoint.storage.oci.uri)")
		}
		info.Location = fmt.Sprintf("%s:%s", storageConfig.OCI.URI, info.Hash)

	default: // PVC
		info.StorageType = nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType)
		basePath := getPVCBasePath(storageConfig)
		if storageConfig == nil || storageConfig.PVC.PVCName == "" {
			return fmt.Errorf("PVC storage type selected but no PVC name configured (set checkpoint.storage.pvc.pvcName)")
		}
		pvcName := storageConfig.PVC.PVCName
		if basePath == "" {
			return fmt.Errorf("PVC storage type selected but no PVC base path configured (set checkpoint.storage.pvc.basePath)")
		}
		info.Location = fmt.Sprintf("%s/%s", basePath, info.Hash)

		InjectCheckpointVolume(podSpec, pvcName)
		InjectCheckpointVolumeMount(mainContainer, basePath)
	}

	// Downward API volume for pod identity after CRIU restore
	InjectPodInfoVolume(podSpec)
	InjectPodInfoVolumeMount(mainContainer)

	// Checkpoint environment variables (path, hash)
	InjectCheckpointEnvVars(mainContainer, info, checkpointConfig)

	return nil
}
