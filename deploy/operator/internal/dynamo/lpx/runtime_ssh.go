/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

const (
	runtimeSSHSecretMountPath        = "/ssh-pk"
	runtimeSSHSecretPrivateKeyPath   = runtimeSSHSecretMountPath + "/private.key"
	conductorSSHKeyDir               = runtimeTemporaryStorageMountPath + "/dynamo-lpu-ssh"
	conductorSSHPrivateKeyPath       = conductorSSHKeyDir + "/private.key"
	conductorSSHKeyInitContainerName = "prepare-ssh-key"
	conductorSSHKeyVolumeName        = "single-v2-ssh-key"
	conductorSSHKeyInitCommand       = "set -euo pipefail; cp " + runtimeSSHSecretPrivateKeyPath + " " +
		conductorSSHPrivateKeyPath + "; chmod 600 " + conductorSSHPrivateKeyPath
)

func addSSHVolumeWithDefaultMode(podSpec *corev1.PodSpec, secretName, volumeName string, defaultMode int32) {
	sshVolume := corev1.Volume{
		Name: volumeName,
		VolumeSource: corev1.VolumeSource{
			Secret: &corev1.SecretVolumeSource{
				SecretName:  secretName,
				DefaultMode: ptr.To(defaultMode),
			},
		},
	}
	podSpec.Volumes = setVolumeByName(podSpec.Volumes, sshVolume)
}

func sshVolumeMount(volumeName string) corev1.VolumeMount {
	return corev1.VolumeMount{
		Name:      volumeName,
		MountPath: runtimeSSHSecretMountPath,
		ReadOnly:  true,
	}
}

func addConductorSSHKey(podSpec *corev1.PodSpec, conductor *corev1.Container, secretName, volumeName string) {
	// Mount the source Secret and the writable destination used by OpenMPI.
	addSSHVolumeWithDefaultMode(podSpec, secretName, volumeName, 0600)
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
		VolumeMounts: []corev1.VolumeMount{conductorSSHKeyVolumeMount()},
	}
	initContainer.VolumeMounts = setVolumeMount(initContainer.VolumeMounts, sshVolumeMount(volumeName))
	podSpec.InitContainers = setContainerByName(podSpec.InitContainers, initContainer)
}

func conductorSSHKeyVolumeMount() corev1.VolumeMount {
	return corev1.VolumeMount{Name: conductorSSHKeyVolumeName, MountPath: conductorSSHKeyDir}
}
