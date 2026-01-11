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
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// getCheckpointInfoFromCheckpoint extracts CheckpointInfo from a DynamoCheckpoint CR
func getCheckpointInfoFromCheckpoint(ckpt *nvidiacomv1alpha1.DynamoCheckpoint) *CheckpointInfo {
	info := &CheckpointInfo{
		Enabled:        true,
		CheckpointName: ckpt.Name,
		Hash:           ckpt.Status.IdentityHash,
		TarPath:        ckpt.Status.TarPath,
		Location:       ckpt.Status.Location,
		StorageType:    ckpt.Status.StorageType,
		Ready:          ckpt.Status.Phase == nvidiacomv1alpha1.DynamoCheckpointPhaseReady,
		Identity: &nvidiacomv1alpha1.DynamoCheckpointIdentity{
			Model:                ckpt.Spec.Identity.Model,
			Framework:            ckpt.Spec.Identity.Framework,
			FrameworkVersion:     ckpt.Spec.Identity.FrameworkVersion,
			TensorParallelSize:   ckpt.Spec.Identity.TensorParallelSize,
			PipelineParallelSize: ckpt.Spec.Identity.PipelineParallelSize,
			Dtype:                ckpt.Spec.Identity.Dtype,
			MaxModelLen:          ckpt.Spec.Identity.MaxModelLen,
			ExtraParameters:      ckpt.Spec.Identity.ExtraParameters,
		},
	}

	// PVCName only relevant for PVC storage type
	if string(info.StorageType) == controller_common.CheckpointStorageTypePVC || info.StorageType == "" {
		info.PVCName = DefaultCheckpointPVCName
	}

	return info
}

// DefaultCheckpointPVCName is the default PVC name for checkpoint storage
const DefaultCheckpointPVCName = "checkpoint-storage"

// getBasePathFromStorage returns the base path from storage config, or the default
func getBasePathFromStorage(storageConfig *controller_common.CheckpointStorageConfig) string {
	if storageConfig != nil && storageConfig.PVC.BasePath != "" {
		return storageConfig.PVC.BasePath
	}
	return consts.CheckpointBasePath
}

// GetCheckpointBasePath returns the configured checkpoint base path from controller config,
// or the default if not set. This is used by both CheckpointReconciler and DynamoGraphDeploymentReconciler.
func GetCheckpointBasePath(config *controller_common.CheckpointConfig) string {
	if config != nil && config.Enabled {
		return getBasePathFromStorage(&config.Storage)
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
	// TarPath is the local path to the checkpoint tar file
	TarPath string
	// Location is the full URI/path in the storage backend
	Location string
	// StorageType is the storage backend type (pvc, s3, oci)
	StorageType nvidiacomv1alpha1.DynamoCheckpointStorageType
	// PVCName is the name of the PVC containing the checkpoint (only for PVC storage)
	PVCName string
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
// Sets STORAGE_TYPE, LOCATION, PATH, and HASH for unified storage backend handling.
func InjectCheckpointEnvVars(container *corev1.Container, info *CheckpointInfo) {
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

	// Path is the local destination (where tar ends up after fetch)
	if info.TarPath != "" {
		envVars = append(envVars, corev1.EnvVar{
			Name:  consts.EnvCheckpointPath,
			Value: info.TarPath,
		})
	}

	// Include hash for debugging/observability
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
				ReadOnly:  true,
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
		ReadOnly:  true,
	})
}

// InjectCheckpointIntoPodSpec injects checkpoint configuration into a pod spec.
// Takes CheckpointInfo (resolved by ResolveCheckpointForService) and adds checkpoint-related
// environment variables and volumes to the pod spec.
// Storage configuration is optional - if not provided, defaults to PVC.
func InjectCheckpointIntoPodSpec(
	podSpec *corev1.PodSpec,
	checkpointInfo *CheckpointInfo,
	storageConfig *controller_common.CheckpointStorageConfig,
) error {
	if checkpointInfo == nil || !checkpointInfo.Enabled {
		return nil
	}

	// Identity is required to compute the checkpoint path
	if checkpointInfo.Identity == nil {
		return fmt.Errorf("checkpoint enabled but identity is nil")
	}

	// Use the checkpoint info as-is (already computed by ResolveCheckpointForService)
	// We only need to compute hash if it's not already set
	info := checkpointInfo
	if info.Hash == "" {
		hash, err := ComputeIdentityHash(*info.Identity)
		if err != nil {
			return fmt.Errorf("failed to compute identity hash: %w", err)
		}
		info.Hash = hash
	}

	// Find the main container first (needed for all storage types)
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

	// Determine storage type and compute location/path
	storageType := controller_common.CheckpointStorageTypePVC // default
	if storageConfig != nil && storageConfig.Type != "" {
		storageType = storageConfig.Type
	}

	switch storageType {
	case controller_common.CheckpointStorageTypeS3:
		// S3 storage: location is s3:// URI, path is local temp
		// URI format: s3://[endpoint/]bucket/prefix
		info.StorageType = storageTypeToAPI(storageType)
		s3URI := "s3://checkpoint-storage/checkpoints" // default
		if storageConfig != nil && storageConfig.S3.URI != "" {
			s3URI = storageConfig.S3.URI
		}
		// Append hash to the URI
		info.Location = fmt.Sprintf("%s/%s.tar", s3URI, info.Hash)
		info.TarPath = fmt.Sprintf("/tmp/%s.tar", info.Hash)

	case controller_common.CheckpointStorageTypeOCI:
		// OCI storage: location is oci:// URI, path is local temp
		// URI format: oci://registry/repository
		info.StorageType = storageTypeToAPI(storageType)
		ociURI := "oci://localhost/checkpoints" // default
		if storageConfig != nil && storageConfig.OCI.URI != "" {
			ociURI = storageConfig.OCI.URI
		}
		// Append hash as tag
		info.Location = fmt.Sprintf("%s:%s", ociURI, info.Hash)
		info.TarPath = fmt.Sprintf("/tmp/%s.tar", info.Hash)

	default: // controller_common.CheckpointStorageTypePVC
		// PVC storage: location and path are the same
		info.StorageType = storageTypeToAPI(storageType)
		basePath := getBasePathFromStorage(storageConfig)
		pvcName := DefaultCheckpointPVCName
		if storageConfig != nil && storageConfig.PVC.PVCName != "" {
			pvcName = storageConfig.PVC.PVCName
		}
		info.TarPath = GetTarPath(basePath, info.Hash)
		info.Location = info.TarPath // Same for PVC
		info.PVCName = pvcName

		// Inject PVC volume and mount (only for PVC storage)
		InjectCheckpointVolume(podSpec, pvcName)
		InjectCheckpointVolumeMount(mainContainer, basePath)
	}

	// Inject checkpoint environment variables (for all storage types)
	InjectCheckpointEnvVars(mainContainer, info)

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
