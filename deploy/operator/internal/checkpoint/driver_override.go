// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Driver-override injection for cross-node/cross-GPU GMS restore on nscale
// B200 (driver 590.48.01). Mounts the patched libcuda.so userspace library
// from a cluster-local PVC into every CUDA-using container (main + gms-*)
// and prepends the mount path to LD_LIBRARY_PATH so the dynamic loader
// picks up the patched library ahead of the stock driver in the image.
//
// Gated by env var DYNAMO_DRIVER_OVERRIDE_PVC on the operator pod so this
// is opt-in per cluster.

package checkpoint

import (
	"os"

	gms "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	corev1 "k8s.io/api/core/v1"
)

const (
	DriverOverrideEnvVar    = "DYNAMO_DRIVER_OVERRIDE_PVC"
	DriverOverrideVolume    = "patched-driver"
	DriverOverrideMountPath = "/opt/patched-driver"
	LDLibraryPathEnvVar     = "LD_LIBRARY_PATH"
)

// DriverOverridePVC returns the PVC name to mount the patched driver from,
// or "" if the override is disabled.
func DriverOverridePVC() string {
	return os.Getenv(DriverOverrideEnvVar)
}

// InjectDriverOverride adds a read-only PVC volume with the patched
// libcuda.so.1 and prepends the mount to LD_LIBRARY_PATH on the main
// container plus any GMS sidecars already present in the pod spec.
func InjectDriverOverride(podSpec *corev1.PodSpec, mainContainer *corev1.Container) {
	pvc := DriverOverridePVC()
	if pvc == "" || podSpec == nil || mainContainer == nil {
		return
	}

	hasVolume := false
	for _, v := range podSpec.Volumes {
		if v.Name == DriverOverrideVolume {
			hasVolume = true
			break
		}
	}
	if !hasVolume {
		podSpec.Volumes = append(podSpec.Volumes, corev1.Volume{
			Name: DriverOverrideVolume,
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvc,
					ReadOnly:  true,
				},
			},
		})
	}

	mountAndEnv(mainContainer)

	// Apply to GMS sidecars that may have already been injected: server
	// (possibly in initContainers or containers), loader (initContainers),
	// saver (initContainers).
	gmsNames := map[string]bool{
		gms.ServerContainerName: true,
		GMSLoaderContainer:      true,
		GMSSaverContainer:       true,
	}
	for i := range podSpec.Containers {
		if gmsNames[podSpec.Containers[i].Name] {
			mountAndEnv(&podSpec.Containers[i])
		}
	}
	for i := range podSpec.InitContainers {
		if gmsNames[podSpec.InitContainers[i].Name] {
			mountAndEnv(&podSpec.InitContainers[i])
		}
	}
}

func mountAndEnv(c *corev1.Container) {
	hasMount := false
	for _, m := range c.VolumeMounts {
		if m.Name == DriverOverrideVolume {
			hasMount = true
			break
		}
	}
	if !hasMount {
		c.VolumeMounts = append(c.VolumeMounts, corev1.VolumeMount{
			Name:      DriverOverrideVolume,
			MountPath: DriverOverrideMountPath,
			ReadOnly:  true,
		})
	}

	// Prepend /opt/patched-driver to LD_LIBRARY_PATH. Keep any existing
	// value; Kubernetes does not evaluate env var references at container
	// start, so we expand $LD_LIBRARY_PATH through the shell by replacing
	// the literal value if one is already present on the container.
	for i := range c.Env {
		if c.Env[i].Name == LDLibraryPathEnvVar {
			existing := c.Env[i].Value
			if existing == "" {
				c.Env[i].Value = DriverOverrideMountPath
			} else {
				c.Env[i].Value = DriverOverrideMountPath + ":" + existing
			}
			return
		}
	}
	c.Env = append(c.Env, corev1.EnvVar{
		Name:  LDLibraryPathEnvVar,
		Value: DriverOverrideMountPath,
	})
}
