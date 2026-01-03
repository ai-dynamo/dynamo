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

// DefaultCheckpointPVCName is the default PVC name for checkpoint storage
const DefaultCheckpointPVCName = "checkpoint-storage"

// StorageConfigFromControllerConfig converts controller_common.CheckpointStorageConfig
// to the API type nvidiacomv1alpha1.DynamoCheckpointStorageConfig
func StorageConfigFromControllerConfig(cfg controller_common.CheckpointStorageConfig) *nvidiacomv1alpha1.DynamoCheckpointStorageConfig {
	storageType := nvidiacomv1alpha1.DynamoCheckpointStorageTypePVC
	switch cfg.Type {
	case "s3":
		storageType = nvidiacomv1alpha1.DynamoCheckpointStorageTypeS3
	case "oci":
		storageType = nvidiacomv1alpha1.DynamoCheckpointStorageTypeOCI
	}

	result := &nvidiacomv1alpha1.DynamoCheckpointStorageConfig{
		Type: storageType,
	}

	switch storageType {
	case nvidiacomv1alpha1.DynamoCheckpointStorageTypePVC:
		result.PVC = &nvidiacomv1alpha1.DynamoCheckpointPVCConfig{
			PVCName:  cfg.PVC.PVCName,
			BasePath: cfg.PVC.BasePath,
		}
	case nvidiacomv1alpha1.DynamoCheckpointStorageTypeS3:
		result.S3 = &nvidiacomv1alpha1.DynamoCheckpointS3Config{
			URI:                  cfg.S3.URI,
			CredentialsSecretRef: cfg.S3.CredentialsSecretRef,
		}
	case nvidiacomv1alpha1.DynamoCheckpointStorageTypeOCI:
		result.OCI = &nvidiacomv1alpha1.DynamoCheckpointOCIConfig{
			URI:                  cfg.OCI.URI,
			CredentialsSecretRef: cfg.OCI.CredentialsSecretRef,
		}
	}

	return result
}

// CheckpointInfo contains resolved checkpoint information for a DGD service
type CheckpointInfo struct {
	// Enabled indicates if checkpointing is enabled
	Enabled bool
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

// ResolveCheckpointForService resolves checkpoint information for a DGD service
// It handles both checkpointRef (direct reference) and identity-based lookup.
// For checkpointRef mode, it also populates config.Identity from the Checkpoint CR
// (in-memory only) so that downstream pod spec generation can compute the checkpoint path.
func ResolveCheckpointForService(
	ctx context.Context,
	c client.Client,
	namespace string,
	config *nvidiacomv1alpha1.ServiceCheckpointConfig,
) (*CheckpointInfo, error) {
	if config == nil || !config.Enabled {
		return &CheckpointInfo{Enabled: false}, nil
	}

	info := &CheckpointInfo{Enabled: true}

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

		info.CheckpointName = ckpt.Name
		info.Hash = ckpt.Status.IdentityHash
		info.TarPath = ckpt.Status.TarPath
		info.Location = ckpt.Status.Location
		info.StorageType = ckpt.Status.StorageType
		// PVCName only relevant for PVC storage type
		if info.StorageType == nvidiacomv1alpha1.DynamoCheckpointStorageTypePVC || info.StorageType == "" {
			info.PVCName = DefaultCheckpointPVCName
		}
		info.Ready = ckpt.Status.Phase == nvidiacomv1alpha1.DynamoCheckpointPhaseReady

		// Populate config.Identity from the Checkpoint CR (in-memory only)
		// This allows InjectCheckpointIntoPodSpec to compute the checkpoint path
		// without needing K8s client access
		if config.Identity == nil {
			config.Identity = &nvidiacomv1alpha1.ServiceCheckpointIdentity{
				Model:                ckpt.Spec.Identity.Model,
				Framework:            ckpt.Spec.Identity.Framework,
				FrameworkVersion:     ckpt.Spec.Identity.FrameworkVersion,
				TensorParallelSize:   ckpt.Spec.Identity.TensorParallelSize,
				PipelineParallelSize: ckpt.Spec.Identity.PipelineParallelSize,
				Dtype:                ckpt.Spec.Identity.Dtype,
				MaxModelLen:          ckpt.Spec.Identity.MaxModelLen,
				ExtraParameters:      ckpt.Spec.Identity.ExtraParameters,
			}
		}

		return info, nil
	}

	// Otherwise, compute hash from identity and look up checkpoint
	if config.Identity == nil {
		return nil, fmt.Errorf("checkpoint enabled but no checkpointRef or identity provided")
	}

	info.Hash = ComputeServiceIdentityHash(*config.Identity)

	// Look for existing checkpoint with matching hash using label selector
	checkpointList := &nvidiacomv1alpha1.DynamoCheckpointList{}
	err := c.List(ctx, checkpointList,
		client.InNamespace(namespace),
		client.MatchingLabels{consts.KubeLabelCheckpointHash: info.Hash},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to list checkpoints: %w", err)
	}

	// Return the first matching checkpoint (there should be at most one per hash)
	if len(checkpointList.Items) > 0 {
		ckpt := &checkpointList.Items[0]
		info.CheckpointName = ckpt.Name
		info.TarPath = ckpt.Status.TarPath
		info.Location = ckpt.Status.Location
		info.StorageType = ckpt.Status.StorageType
		// PVCName only relevant for PVC storage type
		if info.StorageType == nvidiacomv1alpha1.DynamoCheckpointStorageTypePVC || info.StorageType == "" {
			info.PVCName = DefaultCheckpointPVCName
		}
		info.Ready = ckpt.Status.Phase == nvidiacomv1alpha1.DynamoCheckpointPhaseReady
		return info, nil
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
		storageType = nvidiacomv1alpha1.DynamoCheckpointStorageTypePVC
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

// InjectCheckpointIntoPodSpec injects checkpoint configuration into a pod spec
// This is the centralized function for checkpoint injection that can be called
// from GenerateBasePodSpec without needing K8s client access.
// It computes the checkpoint hash and path from the identity if provided.
// Storage configuration is optional - if not provided, defaults to PVC.
func InjectCheckpointIntoPodSpec(
	podSpec *corev1.PodSpec,
	config *nvidiacomv1alpha1.ServiceCheckpointConfig,
	mainContainerName string,
	storageConfig *nvidiacomv1alpha1.DynamoCheckpointStorageConfig,
) error {
	if config == nil || !config.Enabled {
		return nil
	}

	// Build checkpoint info from config
	info := &CheckpointInfo{Enabled: true}

	// Identity is required to compute the checkpoint path.
	// This should always be set at this point because:
	// - User provided identity directly, OR
	// - ResolveCheckpointForService populated it from the Checkpoint CR (for checkpointRef mode)
	// This check is a safety net - if triggered, it indicates a bug in the calling code.
	if config.Identity == nil {
		// This shouldn't happen in normal operation - log would be useful here
		return nil
	}

	// Compute hash from identity
	info.Hash = ComputeServiceIdentityHash(*config.Identity)

	// Determine storage type and compute location/path
	storageType := nvidiacomv1alpha1.DynamoCheckpointStorageTypePVC
	if storageConfig != nil && storageConfig.Type != "" {
		storageType = storageConfig.Type
	}
	info.StorageType = storageType

	switch storageType {
	case nvidiacomv1alpha1.DynamoCheckpointStorageTypeS3:
		// S3 storage: location is s3:// URI, path is local temp
		// URI format: s3://[endpoint/]bucket/prefix
		s3URI := "s3://checkpoint-storage/checkpoints" // default
		if storageConfig != nil && storageConfig.S3 != nil && storageConfig.S3.URI != "" {
			s3URI = storageConfig.S3.URI
		}
		// Append hash to the URI
		info.Location = fmt.Sprintf("%s/%s.tar", s3URI, info.Hash)
		info.TarPath = fmt.Sprintf("/tmp/%s.tar", info.Hash)

	case nvidiacomv1alpha1.DynamoCheckpointStorageTypeOCI:
		// OCI storage: location is oci:// URI, path is local temp
		// URI format: oci://registry/repository
		ociURI := "oci://localhost/checkpoints" // default
		if storageConfig != nil && storageConfig.OCI != nil && storageConfig.OCI.URI != "" {
			ociURI = storageConfig.OCI.URI
		}
		// Append hash as tag
		info.Location = fmt.Sprintf("%s:%s", ociURI, info.Hash)
		info.TarPath = fmt.Sprintf("/tmp/%s.tar", info.Hash)

	default: // PVC (default)
		// PVC storage: location and path are the same
		basePath := consts.CheckpointBasePath
		pvcName := DefaultCheckpointPVCName
		if storageConfig != nil && storageConfig.PVC != nil {
			if storageConfig.PVC.BasePath != "" {
				basePath = storageConfig.PVC.BasePath
			}
			if storageConfig.PVC.PVCName != "" {
				pvcName = storageConfig.PVC.PVCName
			}
		}
		info.TarPath = GetTarPath(basePath, info.Hash)
		info.Location = info.TarPath // Same for PVC
		info.PVCName = pvcName
	}

	// Find the main container
	var mainContainer *corev1.Container
	for i := range podSpec.Containers {
		if podSpec.Containers[i].Name == mainContainerName {
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

	// Inject checkpoint environment variables
	InjectCheckpointEnvVars(mainContainer, info)

	// Inject checkpoint volume and mount (only for PVC storage)
	if info.StorageType == nvidiacomv1alpha1.DynamoCheckpointStorageTypePVC && info.PVCName != "" {
		InjectCheckpointVolume(podSpec, info.PVCName)
		InjectCheckpointVolumeMount(mainContainer, consts.CheckpointBasePath)
	}

	return nil
}

// InjectCheckpointLabelsFromConfig adds checkpoint labels to a label map based on config
func InjectCheckpointLabelsFromConfig(labels map[string]string, config *nvidiacomv1alpha1.ServiceCheckpointConfig) map[string]string {
	if config == nil || !config.Enabled {
		return labels
	}

	if labels == nil {
		labels = make(map[string]string)
	}

	// Compute hash from identity if provided
	if config.Identity != nil {
		hash := ComputeServiceIdentityHash(*config.Identity)
		labels[consts.KubeLabelCheckpointHash] = hash
	}

	return labels
}
