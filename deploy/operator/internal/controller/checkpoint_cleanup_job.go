// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"path/filepath"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/utils/ptr"
)

const defaultCheckpointCleanupImage = "busybox:1.36"

func buildCheckpointCleanupJob(
	config *configv1alpha1.OperatorConfiguration,
	ckpt *nvidiacomv1alpha1.DynamoCheckpoint,
	checkpointID string,
	storage snapshotprotocol.Storage,
) (*batchv1.Job, error) {
	if config == nil {
		return nil, fmt.Errorf("operator configuration is required for checkpoint cleanup")
	}
	if err := validateCheckpointIDForCleanup(checkpointID); err != nil {
		return nil, err
	}
	if storage.PVCName == "" || storage.BasePath == "" {
		return nil, fmt.Errorf("checkpoint cleanup requires PVC storage")
	}
	image := strings.TrimSpace(config.Checkpoint.CleanupImage)
	if image == "" {
		image = defaultCheckpointCleanupImage
	}

	backoffLimit := int32(0)
	activeDeadlineSeconds := int64(300)
	ttlSecondsAfterFinished := int32(300)

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "checkpoint-cleanup-" + checkpointID,
			Namespace: ckpt.Namespace,
			Labels: map[string]string{
				snapshotprotocol.CheckpointIDLabel: checkpointID,
			},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: nvidiacomv1alpha1.GroupVersion.String(),
				Kind:       "DynamoCheckpoint",
				Name:       ckpt.Name,
				UID:        ckpt.UID,
				Controller: ptr.To(true),
			}},
		},
		Spec: batchv1.JobSpec{
			BackoffLimit:            &backoffLimit,
			ActiveDeadlineSeconds:   &activeDeadlineSeconds,
			TTLSecondsAfterFinished: &ttlSecondsAfterFinished,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						snapshotprotocol.CheckpointIDLabel: checkpointID,
					},
				},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Volumes: []corev1.Volume{{
						Name: snapshotprotocol.CheckpointVolumeName,
						VolumeSource: corev1.VolumeSource{
							PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
								ClaimName: storage.PVCName,
							},
						},
					}},
					Containers: []corev1.Container{{
						Name:    "cleanup",
						Image:   image,
						Command: []string{"/bin/sh", "-c"},
						Args: []string{`
set -eu
base="${CHECKPOINT_BASE_PATH%/}"
target="${base}/${CHECKPOINT_ID}"
case "$target" in
  "$base"/*) rm -rf -- "$target" ;;
  *) echo "refusing to remove unexpected path: $target" >&2; exit 1 ;;
esac
`},
						Env: []corev1.EnvVar{
							{Name: "CHECKPOINT_BASE_PATH", Value: storage.BasePath},
							{Name: "CHECKPOINT_ID", Value: checkpointID},
						},
						VolumeMounts: []corev1.VolumeMount{{
							Name:      snapshotprotocol.CheckpointVolumeName,
							MountPath: storage.BasePath,
						}},
					}},
				},
			},
		},
	}, nil
}

func validateCheckpointIDForCleanup(checkpointID string) error {
	checkpointID = strings.TrimSpace(checkpointID)
	if checkpointID == "" ||
		checkpointID == "." ||
		strings.Contains(checkpointID, "..") ||
		strings.Contains(checkpointID, "/") ||
		strings.Contains(checkpointID, "\\") ||
		filepath.Clean(checkpointID) != checkpointID {
		return fmt.Errorf("invalid checkpoint ID %q for cleanup", checkpointID)
	}
	if errs := validation.IsDNS1123Label(checkpointID); len(errs) > 0 {
		return fmt.Errorf("invalid checkpoint ID %q for cleanup: %s", checkpointID, strings.Join(errs, "; "))
	}
	if len("checkpoint-cleanup-"+checkpointID) > validation.DNS1123LabelMaxLength {
		return fmt.Errorf("invalid checkpoint ID %q for cleanup: cleanup job name would exceed %d characters", checkpointID, validation.DNS1123LabelMaxLength)
	}
	return nil
}
