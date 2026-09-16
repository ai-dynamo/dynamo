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

// validateRuntimeConfigStorage requires the config mount and a writable, Pod-local runtime workspace.
// expandedConfigFile is empty when the runtime reads configuration directly.
func validateRuntimeConfigStorage(podSpec *corev1.PodSpec, container *corev1.Container, expandedConfigFile string) error {
	// Reject incompatible authored bindings instead of replacing their volumes or mounts.
	hasConfigMount := false
	var mount corev1.VolumeMount
	for _, existing := range container.VolumeMounts {
		hasConfigMount = hasConfigMount || existing.MountPath == lpuConfigMountPath
		if expandedConfigFile != "" && existing.MountPath == runtimeTemporaryStorageMountPath+"/"+expandedConfigFile {
			return fmt.Errorf("LPX runtime configuration cannot overwrite a volume mounted at %q", existing.MountPath)
		}
		if existing.MountPath == runtimeTemporaryStorageMountPath {
			mount = existing
		}
	}
	if !hasConfigMount {
		return fmt.Errorf("LPX runtime configuration requires a volume mounted at %q", lpuConfigMountPath)
	}
	if mount.Name == "" {
		if container.SecurityContext != nil && ptr.Deref(container.SecurityContext.ReadOnlyRootFilesystem, false) {
			return fmt.Errorf("LPX runtime configuration requires writable storage at %q", runtimeTemporaryStorageMountPath)
		}
		return nil
	}
	if mount.ReadOnly {
		return fmt.Errorf("LPX runtime configuration requires writable storage at %q", mount.MountPath)
	}
	for _, volume := range podSpec.Volumes {
		if volume.Name == mount.Name {
			if volume.EmptyDir == nil {
				return fmt.Errorf("LPX runtime configuration requires Pod-local emptyDir storage at %q", mount.MountPath)
			}
			return nil
		}
	}
	return fmt.Errorf("LPX runtime mount at %q references missing volume %q", mount.MountPath, mount.Name)
}
