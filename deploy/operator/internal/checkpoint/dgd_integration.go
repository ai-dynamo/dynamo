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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controller_common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// getCheckpointInfoFromCheckpoint extracts CheckpointInfo from a DynamoCheckpoint CR
func getCheckpointInfoFromCheckpoint(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) *CheckpointInfo {
	info := &CheckpointInfo{
		Enabled:        true,
		CheckpointName: ckpt.Name,
		Hash:           ckpt.Status.IdentityHash,
		Location:       ckpt.Status.Location,
		StorageType:    ckpt.Status.StorageType,
		Ready:          ckpt.Status.Phase == nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
		Identity:       &ckpt.Spec.Identity,
	}

	return info
}

// getPVCBasePath returns the PVC base path from storage config.
// Only applicable for PVC storage type
func getPVCBasePath(storageConfig *controller_common.CheckpointStorageConfig) string {
	if storageConfig != nil && storageConfig.PVC.BasePath != "" {
		return storageConfig.PVC.BasePath
	}
	return ""
}

// GetPVCBasePath returns the configured PVC base path from controller config.
// This is used by both CheckpointReconciler and DynamoGraphDeploymentReconciler.
// Only applicable for PVC storage type.
func GetPVCBasePath(config *controller_common.CheckpointConfig) string {
	if config != nil {
		return getPVCBasePath(&config.Storage)
	}
	return ""
}

// CheckpointInfo contains resolved checkpoint information for a DGD service
type CheckpointInfo struct {
	// Enabled indicates if checkpointing is enabled
	Enabled bool
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
		return getCheckpointInfoFromCheckpoint(ckpt), nil
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

	// Look for existing checkpoint with matching hash using label selector
	checkpointList := &nvidiacomv1alpha1.DynamoCheckpointList{}
	if err = c.List(ctx, checkpointList,
		client.InNamespace(namespace),
		client.MatchingLabels{consts.KubeLabelCheckpointHash: info.Hash},
	); err != nil {
		return nil, fmt.Errorf("failed to list checkpoints: %w", err)
	}

	// Return the first matching checkpoint (there should be at most one per hash)
	if len(checkpointList.Items) > 0 {
		ckpt := &checkpointList.Items[0]
		// Merge checkpoint info from the CR (overrides the computed values)
		foundInfo := getCheckpointInfoFromCheckpoint(ckpt)
		// Keep the hash and identity we computed from the config
		foundInfo.Hash = info.Hash
		foundInfo.Identity = info.Identity
		return foundInfo, nil
	}

	// No existing checkpoint found
	// In Auto mode, the controller should create one
	return info, nil
}

// InjectCheckpointEnvVars adds checkpoint-related environment variables to a restored/DGD container.
// Sets PATH and HASH so the restored process knows its checkpoint identity.
// DYN_CHECKPOINT_LOCATION is reserved for future S3/OCI support.
func InjectCheckpointEnvVars(container *corev1.Container, info *CheckpointInfo, checkpointConfig *controller_common.CheckpointConfig) {
	if !info.Enabled {
		return
	}

	var envVars []corev1.EnvVar

	// For PVC storage: inject base path so the restored process knows its checkpoint location.
	// For S3/OCI (future): inject DYN_CHECKPOINT_LOCATION directly.
	storageType := controller_common.CheckpointStorageTypePVC
	if checkpointConfig != nil && checkpointConfig.Storage.Type != "" {
		storageType = checkpointConfig.Storage.Type
	}

	switch storageType {
	case controller_common.CheckpointStorageTypePVC:
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

// InjectPodInfoVolume adds a Downward API volume for pod identity and DGD info.
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
					// DGD info from annotations (for CRIU restore)
					{
						Path: consts.PodInfoFileDynNamespace,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.annotations['" + consts.AnnotationDynNamespace + "']",
						},
					},
					{
						Path: consts.PodInfoFileDynComponent,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.annotations['" + consts.AnnotationDynComponent + "']",
						},
					},
					{
						Path: consts.PodInfoFileDynParentDGDName,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.annotations['" + consts.AnnotationDynParentDGDName + "']",
						},
					},
					{
						Path: consts.PodInfoFileDynParentDGDNS,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.annotations['" + consts.AnnotationDynParentDGDNS + "']",
						},
					},
					{
						Path: consts.PodInfoFileDynDiscoveryBackend,
						FieldRef: &corev1.ObjectFieldSelector{
							FieldPath: "metadata.annotations['" + consts.AnnotationDynDiscoveryBackend + "']",
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
// external restore via the chrek DaemonSet. The pod image is expected to be a
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
	checkpointConfig *controller_common.CheckpointConfig,
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
	storageType := controller_common.CheckpointStorageTypePVC // default
	var storageConfig *controller_common.CheckpointStorageConfig
	if checkpointConfig != nil {
		storageConfig = &checkpointConfig.Storage
		if storageConfig.Type != "" {
			storageType = storageConfig.Type
		}
	}

	switch storageType {
	case controller_common.CheckpointStorageTypeS3:
		info.StorageType = nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType)
		if storageConfig == nil || storageConfig.S3.URI == "" {
			return fmt.Errorf("S3 storage type selected but no S3 URI configured (set checkpoint.storage.s3.uri)")
		}
		info.Location = fmt.Sprintf("%s/%s.tar", storageConfig.S3.URI, info.Hash)

	case controller_common.CheckpointStorageTypeOCI:
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

// InjectCheckpointLabelsFromConfig adds checkpoint identity labels to a label map based on config.
// Restore trigger labels are injected only when a concrete restore request is prepared.
func InjectCheckpointLabelsFromConfig(labels map[string]string, config *nvidiacomv1alpha1.ServiceCheckpointConfig) (map[string]string, error) {
	if config == nil || !config.Enabled {
		return labels, nil
	}

	if labels == nil {
		labels = make(map[string]string)
	}

	// Compute hash from identity if provided
	if config.Identity != nil {
		hash, err := ComputeIdentityHash(*config.Identity)
		if err != nil {
			return nil, fmt.Errorf("failed to compute identity hash for labels: %w", err)
		}
		labels[consts.KubeLabelCheckpointHash] = hash
	}

	return labels, nil
}
