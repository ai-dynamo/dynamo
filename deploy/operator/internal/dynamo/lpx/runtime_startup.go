/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
)

const runtimeConfigExpansion = `: "${GROVE_PCSG_NAME:?missing GROVE_PCSG_NAME}" "${GROVE_PCSG_INDEX:?missing GROVE_PCSG_INDEX}" "${GROVE_HEADLESS_SERVICE:?missing GROVE_HEADLESS_SERVICE}"
sed -e "s|\${GROVE_PCSG_NAME}|${GROVE_PCSG_NAME}|g" -e "s|\${GROVE_PCSG_INDEX}|${GROVE_PCSG_INDEX}|g" -e "s|\${GROVE_HEADLESS_SERVICE}|${GROVE_HEADLESS_SERVICE}|g" "$1" > "$2"
shift 2
`

// wrapRuntimeStartup prepares an optional fixed config and executes the original argument vector.
// configFile and initialization are operator-owned, never user-supplied shell fragments.
func wrapRuntimeStartup(container *corev1.Container, defaultCommand, configFile, initialization string) {
	// Keep paths and the complete caller argument vector out of the shell source.
	args := []string{initialization + "exec \"$@\"\n", "--"}
	if configFile != "" {
		args[0] = runtimeConfigExpansion + args[0]
		args = append(args, lpuConfigMountPath+"/"+configFile, runtimeTemporaryStorageMountPath+"/"+configFile)
	}
	if len(container.Command) == 0 {
		args = append(args, defaultCommand)
	} else {
		args = append(args, container.Command...)
	}
	args = append(args, container.Args...)
	container.Command, container.Args = []string{"/bin/sh", "-ec"}, args
}

// addRuntimeConfigStorage keeps expanded configuration writable and local to this Pod.
func addRuntimeConfigStorage(podSpec *corev1.PodSpec, container *corev1.Container, configFile string) error {
	// Reject incompatible authored bindings instead of replacing their volumes or mounts.
	mount := corev1.VolumeMount{Name: runtimeTemporaryStorageVolumeName, MountPath: runtimeTemporaryStorageMountPath}
	for _, existing := range container.VolumeMounts {
		if existing.MountPath == runtimeTemporaryStorageMountPath+"/"+configFile {
			return fmt.Errorf("LPX runtime configuration cannot overwrite a volume mounted at %q", existing.MountPath)
		}
		if existing.MountPath == mount.MountPath {
			mount = existing
		}
	}
	if mount.ReadOnly {
		return fmt.Errorf("LPX runtime configuration requires writable storage at %q", mount.MountPath)
	}
	for _, volume := range podSpec.Volumes {
		if volume.Name == mount.Name {
			if volume.EmptyDir == nil {
				return fmt.Errorf("LPX runtime configuration requires Pod-local emptyDir storage at %q", mount.MountPath)
			}
			container.VolumeMounts = setVolumeMount(container.VolumeMounts, mount)
			return nil
		}
	}
	if mount.Name != runtimeTemporaryStorageVolumeName {
		return fmt.Errorf("LPX runtime mount at %q references missing volume %q", mount.MountPath, mount.Name)
	}
	addRuntimeTemporaryStorage(podSpec, container, false)
	return nil
}
