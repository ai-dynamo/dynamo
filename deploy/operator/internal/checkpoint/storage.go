/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package checkpoint

import (
	"context"
	"fmt"
	"path"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
)

func StorageFromConfig(config configv1alpha1.CheckpointStorageConfiguration) (snapshotprotocol.Storage, bool, error) {
	storageType := strings.TrimSpace(config.Type)
	if storageType == configv1alpha1.CheckpointStorageTypeS3 {
		return s3StorageFromConfig(config.S3)
	}

	pvcName := strings.TrimSpace(config.PVC.PVCName)
	basePath := strings.TrimSpace(config.PVC.BasePath)
	size := strings.TrimSpace(config.PVC.Size)
	storageClassName := strings.TrimSpace(config.PVC.StorageClassName)
	accessMode := strings.TrimSpace(config.PVC.AccessMode)
	hasPVCConfig := pvcName != "" || basePath != "" || config.PVC.Create || size != "" || storageClassName != "" || accessMode != ""
	if storageType == "" && !hasPVCConfig {
		return snapshotprotocol.Storage{}, false, nil
	}
	if storageType != "" && storageType != snapshotprotocol.StorageTypePVC && !hasPVCConfig {
		switch storageType {
		case configv1alpha1.CheckpointStorageTypeOCI:
			return snapshotprotocol.Storage{}, false, nil
		default:
			return snapshotprotocol.Storage{}, false, fmt.Errorf("checkpoint.storage.type %q is not supported; expected %q or %q", storageType, snapshotprotocol.StorageTypePVC, snapshotprotocol.StorageTypeS3)
		}
	}
	if storageType == "" {
		storageType = snapshotprotocol.StorageTypePVC
	}
	if storageType != snapshotprotocol.StorageTypePVC {
		return snapshotprotocol.Storage{}, false, fmt.Errorf("checkpoint storage type %q is not supported; expected %q or %q", storageType, snapshotprotocol.StorageTypePVC, snapshotprotocol.StorageTypeS3)
	}
	if pvcName == "" || basePath == "" {
		return snapshotprotocol.Storage{}, false, fmt.Errorf("checkpoint.storage.pvc.pvcName and checkpoint.storage.pvc.basePath are required when checkpoint storage is configured")
	}
	basePath, err := normalizeStorageBasePath(basePath, "checkpoint.storage.pvc.basePath")
	if err != nil {
		return snapshotprotocol.Storage{}, false, err
	}
	if config.PVC.Create {
		if _, err := storagePVCAccessMode(accessMode); err != nil {
			return snapshotprotocol.Storage{}, false, err
		}
	}
	return snapshotprotocol.Storage{
		Type:     snapshotprotocol.StorageTypePVC,
		PVCName:  pvcName,
		BasePath: basePath,
	}, true, nil
}

// s3StorageFromConfig resolves an S3 checkpoint storage config.
// The snapshot-agent holds the authoritative S3 connection and credentials; the
// operator only conveys the agent-local staging path and the s3 storage type to
// workload pods (no PVC volume is mounted for s3).
func s3StorageFromConfig(s3 configv1alpha1.CheckpointS3Config) (snapshotprotocol.Storage, bool, error) {
	basePath := strings.TrimSpace(s3.BasePath)
	if basePath == "" {
		return snapshotprotocol.Storage{}, false, fmt.Errorf("checkpoint.storage.s3.basePath is required when checkpoint.storage.type is %q", configv1alpha1.CheckpointStorageTypeS3)
	}
	basePath, err := normalizeStorageBasePath(basePath, "checkpoint.storage.s3.basePath")
	if err != nil {
		return snapshotprotocol.Storage{}, false, err
	}
	return snapshotprotocol.Storage{
		Type:             snapshotprotocol.StorageTypeS3,
		BasePath:         basePath,
		StagingSizeLimit: strings.TrimSpace(s3.Staging.SizeLimit),
		StagingMedium:    strings.TrimSpace(s3.Staging.Medium),
		StagingPVCName:   strings.TrimSpace(s3.Staging.PVCName),
		StagingHostPath:  strings.TrimSpace(s3.Staging.HostPath),
	}, true, nil
}

func EnsureStoragePVC(
	ctx context.Context,
	kubeClient ctrlclient.Client,
	namespace string,
	config configv1alpha1.CheckpointStorageConfiguration,
) error {
	storage, ok, err := StorageFromConfig(config)
	if err != nil {
		return err
	}
	if !ok {
		return nil
	}
	// Object-store (s3) backends need no PVC: the snapshot-agent stages
	// artifacts on its own filesystem and uploads them to the bucket.
	if storage.Type != snapshotprotocol.StorageTypePVC {
		return nil
	}
	if kubeClient == nil {
		return fmt.Errorf("checkpoint storage client is required")
	}

	pvc := &corev1.PersistentVolumeClaim{}
	key := types.NamespacedName{Name: storage.PVCName, Namespace: namespace}
	if err := kubeClient.Get(ctx, key, pvc); err == nil {
		return nil
	} else if !apierrors.IsNotFound(err) {
		return fmt.Errorf("get checkpoint storage PVC %s/%s: %w", namespace, storage.PVCName, err)
	}

	if !config.PVC.Create {
		return fmt.Errorf("checkpoint storage PVC %s/%s does not exist and checkpoint.storage.pvc.create is false", namespace, storage.PVCName)
	}

	pvc, err = buildStoragePVC(namespace, config.PVC)
	if err != nil {
		return err
	}
	if err := kubeClient.Create(ctx, pvc); err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create checkpoint storage PVC %s/%s: %w", namespace, storage.PVCName, err)
	}

	return nil
}

func buildStoragePVC(namespace string, pvcConfig configv1alpha1.CheckpointPVCConfig) (*corev1.PersistentVolumeClaim, error) {
	pvcName := strings.TrimSpace(pvcConfig.PVCName)
	if pvcName == "" {
		return nil, fmt.Errorf("checkpoint.storage.pvc.pvcName is required when checkpoint.storage.pvc.create is true")
	}
	size := strings.TrimSpace(pvcConfig.Size)
	if size == "" {
		return nil, fmt.Errorf("checkpoint.storage.pvc.size is required when checkpoint.storage.pvc.create is true")
	}

	quantity, err := resource.ParseQuantity(size)
	if err != nil {
		return nil, fmt.Errorf("invalid checkpoint.storage.pvc.size %q: %w", size, err)
	}
	if quantity.Sign() <= 0 {
		return nil, fmt.Errorf("checkpoint.storage.pvc.size %q must be greater than zero", size)
	}

	accessMode, err := storagePVCAccessMode(pvcConfig.AccessMode)
	if err != nil {
		return nil, err
	}
	volumeMode := corev1.PersistentVolumeFilesystem

	var storageClassName *string
	if value := strings.TrimSpace(pvcConfig.StorageClassName); value != "" {
		storageClassName = &value
	}

	return &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pvcName,
			Namespace: namespace,
			Labels: map[string]string{
				"app.kubernetes.io/name":       "dynamo",
				"app.kubernetes.io/component":  "checkpoint-storage",
				"app.kubernetes.io/managed-by": "dynamo-operator",
			},
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{accessMode},
			VolumeMode:  &volumeMode,
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: quantity,
				},
			},
			StorageClassName: storageClassName,
		},
	}, nil
}

func normalizeStorageBasePath(basePath, field string) (string, error) {
	basePath = strings.TrimSpace(basePath)
	if basePath == "" {
		return "", fmt.Errorf("%s is required when checkpoint storage is configured", field)
	}
	if !strings.HasPrefix(basePath, "/") {
		return "", fmt.Errorf("%s %q must be absolute", field, basePath)
	}
	basePath = path.Clean(basePath)
	basePath = strings.TrimRight(basePath, "/")
	if basePath == "" {
		basePath = "/"
	}
	return basePath, nil
}

func storagePVCAccessMode(accessMode string) (corev1.PersistentVolumeAccessMode, error) {
	mode := corev1.PersistentVolumeAccessMode(strings.TrimSpace(accessMode))
	if mode == "" {
		return corev1.ReadWriteMany, nil
	}
	switch mode {
	case corev1.ReadWriteOnce, corev1.ReadWriteMany:
		return mode, nil
	default:
		return "", fmt.Errorf("checkpoint.storage.pvc.accessMode %q is not supported; expected %q or %q", accessMode, corev1.ReadWriteOnce, corev1.ReadWriteMany)
	}
}

func ResolveStorage(
	ctx context.Context,
	reader ctrlclient.Reader,
	namespace string,
	checkpointID string,
	artifactVersion string,
	config configv1alpha1.CheckpointStorageConfiguration,
) (snapshotprotocol.Storage, error) {
	storage, ok, err := StorageFromConfig(config)
	if err != nil {
		return snapshotprotocol.Storage{}, err
	}
	if ok {
		return snapshotprotocol.ResolveCheckpointStorage(checkpointID, artifactVersion, storage)
	}
	return snapshotprotocol.DiscoverAndResolveStorage(ctx, reader, namespace, checkpointID, artifactVersion)
}
