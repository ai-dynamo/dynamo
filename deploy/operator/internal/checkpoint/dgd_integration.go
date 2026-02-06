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
	"path/filepath"

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

// DefaultCheckpointPVCName is the default PVC name for checkpoint storage
const DefaultCheckpointPVCName = "checkpoint-storage"

// getPVCBasePath returns the PVC base path from storage config, or the default
// Only applicable for PVC storage type
func getPVCBasePath(storageConfig *controller_common.CheckpointStorageConfig) string {
	if storageConfig != nil && storageConfig.PVC.BasePath != "" {
		return storageConfig.PVC.BasePath
	}
	return consts.CheckpointBasePath
}

// GetPVCBasePath returns the configured PVC base path from controller config,
// or the default if not set. This is used by both CheckpointReconciler and DynamoGraphDeploymentReconciler.
// Only applicable for PVC storage type.
func GetPVCBasePath(config *controller_common.CheckpointConfig) string {
	if config != nil && config.Enabled {
		return getPVCBasePath(&config.Storage)
	}
	return consts.CheckpointBasePath
}

// storageTypeToAPI converts controller_common storage type string to API enum
func storageTypeToAPI(storageType string) nvidiacomv1alpha1.DynamoCheckpointStorageType {
	// Simply cast - the values match between controller constants and API enum
	return nvidiacomv1alpha1.DynamoCheckpointStorageType(storageType)
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

// InjectCheckpointEnvVars adds checkpoint-related environment variables to a container
// Sets STORAGE_TYPE, LOCATION, PATH, HASH, and CRIU-related vars for unified storage backend handling.
func InjectCheckpointEnvVars(container *corev1.Container, info *CheckpointInfo, config *controller_common.CheckpointConfig) {
	if !info.Enabled {
		return
	}

	// Determine storage type (default to PVC if not set)
	storageType := info.StorageType
	if storageType == "" {
		storageType = nvidiacomv1alpha1.DynamoCheckpointStorageType(controller_common.CheckpointStorageTypePVC)
	}

	envVars := []corev1.EnvVar{
		{
			Name:  consts.EnvCheckpointStorageType,
			Value: string(storageType),
		},
	}

	// Location is the source (where to fetch from)
	if info.Location != "" {
		envVars = append(envVars, corev1.EnvVar{
			Name:  consts.EnvCheckpointLocation,
			Value: info.Location,
		})
	}

	// For PVC storage, also inject DYNAMO_CHECKPOINT_PATH (base directory)
	// This is used by k8s-runc-bypass restore entrypoint
	if string(storageType) == controller_common.CheckpointStorageTypePVC && info.Location != "" {
		// Extract base path using filepath.Dir()
		basePath := filepath.Dir(info.Location)
		envVars = append(envVars, corev1.EnvVar{
			Name:  consts.EnvCheckpointPath,
			Value: basePath,
		})
	}

	// Include hash for debugging/observability and for k8s-runc-bypass
	if info.Hash != "" {
		envVars = append(envVars, corev1.EnvVar{
			Name:  consts.EnvCheckpointHash,
			Value: info.Hash,
		})
	}

	// Add CRIU-related env vars for restore operations
	criuTimeout := consts.DefaultCRIUTimeout
	if config != nil && config.CRIUTimeout != "" {
		criuTimeout = config.CRIUTimeout
	}

	envVars = append(envVars,
		corev1.EnvVar{
			Name:  consts.EnvRestoreMarkerFile,
			Value: consts.RestoreMarkerFilePath,
		},
		corev1.EnvVar{
			Name:  consts.EnvCRIUWorkDir,
			Value: consts.CRIUWorkDirPath,
		},
		corev1.EnvVar{
			Name:  consts.EnvCRIULogDir,
			Value: consts.CRIULogDirPath,
		},
		corev1.EnvVar{
			Name:  consts.EnvCUDAPluginDir,
			Value: consts.CUDAPluginDirPath,
		},
		corev1.EnvVar{
			Name:  consts.EnvCRIUTimeout,
			Value: criuTimeout,
		},
	)

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

	if basePath == "" {
		basePath = consts.CheckpointBasePath
	}

	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      consts.CheckpointVolumeName,
		MountPath: basePath,
		ReadOnly:  false, // CRIU needs write access for restore.log and restore-criu.conf
	})
}

// InjectCheckpointSignalVolume adds the checkpoint signal hostPath volume to a pod spec
// This is needed for CRIU mount namespace consistency between checkpoint and restore pods
func InjectCheckpointSignalVolume(podSpec *corev1.PodSpec, checkpointConfig *controller_common.CheckpointConfig) {
	// Check if volume already exists
	for _, v := range podSpec.Volumes {
		if v.Name == consts.CheckpointSignalVolumeName {
			return
		}
	}

	// Get signal host path from config or use default
	signalHostPath := consts.CheckpointSignalHostPath
	if checkpointConfig != nil && checkpointConfig.Storage.SignalHostPath != "" {
		signalHostPath = checkpointConfig.Storage.SignalHostPath
	}

	hostPathType := corev1.HostPathDirectoryOrCreate
	podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
		Name: consts.CheckpointSignalVolumeName,
		VolumeSource: corev1.VolumeSource{
			HostPath: &corev1.HostPathVolumeSource{
				Path: signalHostPath,
				Type: &hostPathType,
			},
		},
	})
}

// InjectCheckpointSignalVolumeMount adds the checkpoint signal volume mount to a container
// This is needed for CRIU mount namespace consistency between checkpoint and restore pods
func InjectCheckpointSignalVolumeMount(container *corev1.Container) {
	// Check if mount already exists
	for _, m := range container.VolumeMounts {
		if m.Name == consts.CheckpointSignalVolumeName {
			return
		}
	}

	container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
		Name:      consts.CheckpointSignalVolumeName,
		MountPath: consts.CheckpointSignalMountPath,
		ReadOnly:  false,
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

// InjectCheckpointIntoPodSpec injects checkpoint configuration into a pod spec.
// This is the single entry point for ALL checkpoint-related pod modifications:
// 1. Command/Args transformation - moves Command to Args to respect image ENTRYPOINT
// 2. Security context - applies hostIPC and privileged mode for CRIU restore
// 3. Environment variables - injects checkpoint path, hash, and CRIU settings
// 4. Storage configuration - adds volumes and mounts based on storage type
//
// Takes CheckpointInfo (resolved by ResolveCheckpointForService) and checkpoint config.
// Returns error if checkpoint is enabled but configuration is invalid.
func InjectCheckpointIntoPodSpec(
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
	checkpointConfig *controller_common.CheckpointConfig,
) error {
	if checkpointInfo == nil || !checkpointInfo.Enabled {
		return nil
	}

	// Use the checkpoint info as-is (already computed by ResolveCheckpointForService)
	// We only need to compute hash if it's not already set
	info := checkpointInfo
	if info.Hash == "" {
		// Identity is required to compute the hash
		if info.Identity == nil {
			return fmt.Errorf("checkpoint enabled but identity is nil and hash is not set")
		}
		hash, err := ComputeIdentityHash(*info.Identity)
		if err != nil {
			return fmt.Errorf("failed to compute identity hash: %w", err)
		}
		info.Hash = hash
	}

	// Find the main container first (needed for all modifications)
	var mainContainer *corev1.Container
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == consts.MainContainerName {
			mainContainer = &podSpec.Containers[i]
			break
		}
	}
	// If no main container found by name, use the first container
	if mainContainer == nil && len(podSpec.Containers) > 0 {
		mainContainer = &podSpec.Containers[0]
	}
	if mainContainer == nil {
		return fmt.Errorf("no container found to inject checkpoint config")
	}

	// 1. Handle command/args for checkpoint-enabled images
	// When checkpoint is enabled, the image has a smart ENTRYPOINT (e.g., /smart-entrypoint.sh)
	// that detects checkpoints and decides between restore and cold start.
	// We need to pass the user's command as arguments to this ENTRYPOINT rather than
	// overriding it with Command.
	if len(mainContainer.Command) > 0 {
		// Combine Command + Args into a single Args array
		// This allows the image's ENTRYPOINT to receive the full command as arguments
		combinedArgs := append(mainContainer.Command, mainContainer.Args...)
		mainContainer.Args = combinedArgs
		mainContainer.Command = nil // Clear Command to use image's ENTRYPOINT
	}
	// If Command is empty but Args exists, keep Args as-is (they'll be passed to ENTRYPOINT)

	// 2. Apply pod-level security context for CRIU restore
	// hostIPC: Required for CRIU to access shared memory segments and IPC resources
	podSpec.HostIPC = true

	// Apply seccomp profile to match checkpoint environment
	// This blocks io_uring syscalls required for CRIU compatibility
	if podSpec.SecurityContext == nil {
		podSpec.SecurityContext = &corev1.PodSecurityContext{}
	}
	podSpec.SecurityContext.SeccompProfile = &corev1.SeccompProfile{
		Type:             corev1.SeccompProfileTypeLocalhost,
		LocalhostProfile: ptr.To("profiles/block-iouring.json"),
	}

	// Apply container-level security context for CRIU restore
	// Privileged mode is required for CRIU restore operations
	if mainContainer.SecurityContext == nil {
		mainContainer.SecurityContext = &corev1.SecurityContext{}
	}
	mainContainer.SecurityContext.Privileged = ptr.To(true)

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
		// S3 storage: location is s3:// URI
		// URI format: s3://[endpoint/]bucket/prefix
		info.StorageType = storageTypeToAPI(storageType)
		s3URI := "s3://checkpoint-storage/checkpoints" // default
		if storageConfig != nil && storageConfig.S3.URI != "" {
			s3URI = storageConfig.S3.URI
		}
		// Append hash to the URI
		info.Location = fmt.Sprintf("%s/%s.tar", s3URI, info.Hash)

	case controller_common.CheckpointStorageTypeOCI:
		// OCI storage: location is oci:// URI
		// URI format: oci://registry/repository
		info.StorageType = storageTypeToAPI(storageType)
		ociURI := "oci://localhost/checkpoints" // default
		if storageConfig != nil && storageConfig.OCI.URI != "" {
			ociURI = storageConfig.OCI.URI
		}
		// Append hash as tag
		info.Location = fmt.Sprintf("%s:%s", ociURI, info.Hash)

	default: // controller_common.CheckpointStorageTypePVC
		// PVC storage: location is the checkpoint directory
		// k8s-runc-bypass expects: /checkpoints/{hash}/ (directory with checkpoint data)
		info.StorageType = storageTypeToAPI(storageType)
		basePath := getPVCBasePath(storageConfig)
		pvcName := DefaultCheckpointPVCName
		if storageConfig != nil && storageConfig.PVC.PVCName != "" {
			pvcName = storageConfig.PVC.PVCName
		}
		info.Location = fmt.Sprintf("%s/%s", basePath, info.Hash)

		// Inject PVC volume and mount (only for PVC storage)
		InjectCheckpointVolume(podSpec, pvcName)
		InjectCheckpointVolumeMount(mainContainer, basePath)
	}

	// Inject signal volume for CRIU mount namespace consistency
	// Even though restore pods don't use the signal file, they need it mounted
	// to match the checkpoint job's mount namespace for CRIU compatibility
	InjectCheckpointSignalVolume(podSpec, checkpointConfig)
	InjectCheckpointSignalVolumeMount(mainContainer)

	// Inject Downward API volume for pod identity after CRIU restore
	// CRIU preserves environment variables from checkpoint time, so pod identity
	// env vars (POD_NAME, POD_UID, POD_NAMESPACE) contain stale values.
	// The Dynamo runtime reads from /etc/podinfo/ files first to get correct identity.
	InjectPodInfoVolume(podSpec)
	InjectPodInfoVolumeMount(mainContainer)

	// Inject checkpoint environment variables (for all storage types)
	InjectCheckpointEnvVars(mainContainer, info, checkpointConfig)

	return nil
}

// InjectCheckpointLabelsFromConfig adds checkpoint labels to a label map based on config
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
