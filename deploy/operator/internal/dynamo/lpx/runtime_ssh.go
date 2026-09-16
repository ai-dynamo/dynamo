/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"slices"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

const (
	runtimeSSHVolumeName             = "ssh-secret"
	runtimeSSHSecretMountPath        = "/ssh-pk"
	runtimeSSHSecretPrivateKeyPath   = runtimeSSHSecretMountPath + "/private.key"
	conductorSSHKeyDir               = runtimeTemporaryStorageMountPath + "/dynamo-lpu-ssh"
	conductorSSHPrivateKeyPath       = conductorSSHKeyDir + "/private.key"
	conductorSSHKeyInitContainerName = "prepare-ssh-key"
	conductorSSHKeyVolumeName        = "single-v2-ssh-key"
)

func validateSSHVolume(podSpec *corev1.PodSpec, secretName string) error {
	// Preserve authored storage and key projections instead of replacing their contents.
	for _, volume := range podSpec.Volumes {
		if volume.Name == runtimeSSHVolumeName {
			if volume.Secret == nil || volume.Secret.SecretName != secretName || len(volume.Secret.Items) != 0 {
				return fmt.Errorf("selected LPX podTemplate volume %q is reserved for MPI SSH Secret %q", runtimeSSHVolumeName, secretName)
			}
			return nil
		}
	}
	return fmt.Errorf("selected LPX podTemplate requires volume %q for MPI SSH Secret %q", runtimeSSHVolumeName, secretName)
}

func addConductorSSHKey(podSpec *corev1.PodSpec, conductor *corev1.Container, secretName string) error {
	// Check both container lists before replacing any authored configuration.
	if err := validateRolePodSpecContainerNames(podSpec, field.NewPath("spec"), conductorSSHKeyInitContainerName).ToAggregate(); err != nil {
		return err
	}

	// Require the authored source Secret used by the generated init container.
	if err := validateSSHVolume(podSpec, secretName); err != nil {
		return err
	}

	// The init container and Nova must share the authored Pod-local key directory.
	if !slices.ContainsFunc(podSpec.Volumes, func(volume corev1.Volume) bool {
		return volume.Name == conductorSSHKeyVolumeName && volume.EmptyDir != nil
	}) {
		return fmt.Errorf("LPX conductor requires emptyDir volume %q", conductorSSHKeyVolumeName)
	}
	if !slices.ContainsFunc(conductor.VolumeMounts, func(mount corev1.VolumeMount) bool {
		return mount.Name == conductorSSHKeyVolumeName && mount.MountPath == conductorSSHKeyDir && mount.SubPath == "" && mount.SubPathExpr == ""
	}) {
		return fmt.Errorf("LPX conductor requires volume %q mounted at %q without subPath", conductorSSHKeyVolumeName, conductorSSHKeyDir)
	}

	// Copy the key with OpenSSH's required mode before the conductor starts.
	initContainer := corev1.Container{
		Name:            conductorSSHKeyInitContainerName,
		Image:           conductor.Image,
		ImagePullPolicy: conductor.ImagePullPolicy,
		Command:         []string{"/bin/install"},
		Args:            []string{"-m", "0600", runtimeSSHSecretPrivateKeyPath, conductorSSHPrivateKeyPath},
		SecurityContext: &corev1.SecurityContext{
			RunAsUser:    ptr.To(int64(0)),
			RunAsGroup:   ptr.To(int64(0)),
			RunAsNonRoot: ptr.To(false),
		},
		VolumeMounts: []corev1.VolumeMount{
			{Name: conductorSSHKeyVolumeName, MountPath: conductorSSHKeyDir},
			{Name: runtimeSSHVolumeName, MountPath: runtimeSSHSecretMountPath, ReadOnly: true},
		},
	}
	podSpec.InitContainers = append(podSpec.InitContainers, initContainer)
	return nil
}
