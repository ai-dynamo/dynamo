/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
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
	conductorSSHKeyInitCommand       = "set -euo pipefail; cp " + runtimeSSHSecretPrivateKeyPath + " " +
		conductorSSHPrivateKeyPath + "; chmod 600 " + conductorSSHPrivateKeyPath
)

func addSSHVolume(podSpec *corev1.PodSpec, secretName string, defaultMode int32) error {
	// Preserve authored storage and key projections instead of replacing their contents.
	for _, volume := range podSpec.Volumes {
		if volume.Name == runtimeSSHVolumeName && (volume.Secret == nil ||
			volume.Secret.SecretName != secretName || len(volume.Secret.Items) != 0) {
			return fmt.Errorf("selected LPX podTemplate volume %q is reserved for MPI SSH Secret %q", runtimeSSHVolumeName, secretName)
		}
	}

	// Apply the runtime's key permissions to its dedicated Secret volume.
	sshVolume := corev1.Volume{
		Name: runtimeSSHVolumeName,
		VolumeSource: corev1.VolumeSource{
			Secret: &corev1.SecretVolumeSource{
				SecretName:  secretName,
				DefaultMode: ptr.To(defaultMode),
			},
		},
	}
	podSpec.Volumes = setVolumeByName(podSpec.Volumes, sshVolume)
	return nil
}

func sshVolumeMount() corev1.VolumeMount {
	return corev1.VolumeMount{
		Name:      runtimeSSHVolumeName,
		MountPath: runtimeSSHSecretMountPath,
		ReadOnly:  true,
	}
}

func addConductorSSHKey(podSpec *corev1.PodSpec, conductor *corev1.Container, secretName string) error {
	// Mount the source Secret and the writable destination used by OpenMPI.
	if err := addSSHVolume(podSpec, secretName, 0600); err != nil {
		return err
	}

	// Keep the copied private key in a Pod-local writable volume.
	podSpec.Volumes = setVolumeByName(podSpec.Volumes, corev1.Volume{
		Name: conductorSSHKeyVolumeName,
		VolumeSource: corev1.VolumeSource{
			EmptyDir: &corev1.EmptyDirVolumeSource{},
		},
	})

	// Copy the key with OpenSSH's required mode before the conductor starts.
	initContainer := corev1.Container{
		Name:            conductorSSHKeyInitContainerName,
		Image:           conductor.Image,
		ImagePullPolicy: conductor.ImagePullPolicy,
		Command:         []string{"/bin/bash"},
		Args:            []string{"-c", conductorSSHKeyInitCommand},
		SecurityContext: &corev1.SecurityContext{
			RunAsUser:    ptr.To(int64(0)),
			RunAsGroup:   ptr.To(int64(0)),
			RunAsNonRoot: ptr.To(false),
		},
		VolumeMounts: []corev1.VolumeMount{conductorSSHKeyVolumeMount(), sshVolumeMount()},
	}
	podSpec.InitContainers = setContainerByName(podSpec.InitContainers, initContainer)
	return nil
}

func conductorSSHKeyVolumeMount() corev1.VolumeMount {
	return corev1.VolumeMount{Name: conductorSSHKeyVolumeName, MountPath: conductorSSHKeyDir}
}
