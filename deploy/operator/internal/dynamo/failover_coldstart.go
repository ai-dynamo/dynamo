/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
)

const (
	vllmMasterPortFlag   = "--master-port"
	vllmMasterPortStride = 100

	// The CUDA graph archive is an emptyDir so it never outlives the pod or
	// crosses builds. The path is passed as env rather than a launch flag
	// because the engine args may be shell-wrapped.
	gmsGraphArchiveVolumeName = "gms-graph-archive"
	gmsGraphArchiveMountPath  = "/gms-graph-archive"
	gmsGraphArchiveEnvVar     = "DYN_VLLM_GMS_GRAPH_ARCHIVE"
)

// buildColdStartFailoverPod prepares snapshot-less failover, supported only for
// vLLM with GMS V0. Engines initialize concurrently in shadow mode and need
// separate collective/KV ports, unlike the restored snapshot engine pair.
// Mutates podSpec, which must not be nil and must already have GMS resources.
// graphArchive may be nil, in which case no CUDA graph archive is rendered.
func buildColdStartFailoverPod(podSpec *corev1.PodSpec, numberOfNodes int32, backendFramework BackendFramework, graphArchive *v1beta1.GraphArchiveSpec) error {
	// Reject other backends before changing the pod.
	if backendFramework != BackendFrameworkVLLM {
		return fmt.Errorf("cold-start failover is currently supported only for vLLM (detected: %s)", backendFramework)
	}
	if err := buildFailoverEnginePair(podSpec); err != nil {
		return err
	}

	// One archive per pod lets replacement engines load the first shadow's CUDA graphs.
	if graphArchive != nil {
		// Defend direct rendering as well as objects that bypassed admission.
		if numberOfNodes != 1 {
			return fmt.Errorf("failover.graphArchive requires a single node setup")
		}
		ensureGraphArchiveVolume(podSpec, graphArchive)
	}

	// Only the two cloned engines receive legacy shadow initialization settings.
	for engineID := range failoverEngineCount {
		c := &podSpec.Containers[engineID]
		c.Env = append(c.Env, corev1.EnvVar{Name: "DYN_VLLM_GMS_SHADOW_MODE", Value: "true"})
		c.Env = append(c.Env,
			corev1.EnvVar{Name: "VLLM_NIXL_SIDE_CHANNEL_PORT", Value: strconv.Itoa(5600 + engineID)},
			corev1.EnvVar{Name: "DYN_VLLM_KV_EVENT_PORT", Value: strconv.Itoa(20080 + engineID)},
		)

		// Stagger --master-port for TP so each engine group uses a distinct
		// torch.distributed TCP store. engine-0 keeps the default (29500),
		// engine-1 gets 29500 + stride.
		if engineID > 0 {
			if hasMasterPortFlag(c) {
				staggerMasterPort(c, engineID)
			} else {
				c.Args = append(c.Args, vllmMasterPortFlag, strconv.Itoa(29500+engineID*vllmMasterPortStride))
			}
		}

		if numberOfNodes > 1 {
			c.Env = append(c.Env,
				corev1.EnvVar{Name: "NNODES", Value: strconv.Itoa(int(numberOfNodes))},
			)
		}

		// The operator-managed archive path replaces any value inherited from main.
		if graphArchive != nil {
			if err := mountGraphArchive(c); err != nil {
				return err
			}
			removeEnvVar(c, gmsGraphArchiveEnvVar)
			c.Env = append(c.Env, corev1.EnvVar{Name: gmsGraphArchiveEnvVar, Value: gmsGraphArchiveMountPath})
		}
	}
	return nil
}

// ensureGraphArchiveVolume adds the archive emptyDir to the pod. The volume
// name is operator-owned, so an authored volume with that name is replaced.
// Mutates podSpec; podSpec and graphArchive must not be nil.
func ensureGraphArchiveVolume(podSpec *corev1.PodSpec, graphArchive *v1beta1.GraphArchiveSpec) {
	// Copy the limit so the rendered pod never aliases the component spec.
	volume := corev1.Volume{
		Name:     gmsGraphArchiveVolumeName,
		EmptyDir: &corev1.EmptyDirVolumeSource{},
	}
	if graphArchive.SizeLimit != nil {
		sizeLimit := graphArchive.SizeLimit.DeepCopy()
		volume.EmptyDir.SizeLimit = &sizeLimit
	}

	// Replace by name: two volumes with one name make the pod invalid.
	for i := range podSpec.Volumes {
		if podSpec.Volumes[i].Name == gmsGraphArchiveVolumeName {
			podSpec.Volumes[i] = volume
			return
		}
	}
	podSpec.Volumes = append(podSpec.Volumes, volume)
}

// mountGraphArchive mounts the archive volume in an engine container at the
// operator-owned path. Mutates container, which must not be nil.
func mountGraphArchive(container *corev1.Container) error {
	// Drop an inherited mount of the archive volume; refuse another volume at its path.
	mounts := make([]corev1.VolumeMount, 0, len(container.VolumeMounts)+1)
	for _, mount := range container.VolumeMounts {
		if mount.Name == gmsGraphArchiveVolumeName {
			continue
		}
		if mount.MountPath == gmsGraphArchiveMountPath {
			return fmt.Errorf("failover.graphArchive mount path %q is already used by volume %q", gmsGraphArchiveMountPath, mount.Name)
		}
		mounts = append(mounts, mount)
	}
	container.VolumeMounts = append(mounts, corev1.VolumeMount{Name: gmsGraphArchiveVolumeName, MountPath: gmsGraphArchiveMountPath})
	return nil
}

// hasMasterPortFlag checks if --master-port appears in the container args or command.
func hasMasterPortFlag(container *corev1.Container) bool {
	for _, arg := range container.Args {
		if arg == vllmMasterPortFlag || strings.Contains(arg, vllmMasterPortFlag+" ") {
			return true
		}
	}
	for _, cmd := range container.Command {
		if strings.Contains(cmd, vllmMasterPortFlag+" ") {
			return true
		}
	}
	return false
}

func staggerMasterPort(container *corev1.Container, engineID int) {
	offset := engineID * vllmMasterPortStride
	staggerFlagValue(container, vllmMasterPortFlag, offset)
}

// staggerFlagValue finds a --flag VALUE pair in container args and adds offset
// to the integer value. Handles both separate-token args (["--flag", "29500"])
// and shell-wrapped args (["sh", "-c", "... --flag 29500 ..."]).
func staggerFlagValue(container *corev1.Container, flag string, offset int) {
	for i, arg := range container.Args {
		if arg == flag && i+1 < len(container.Args) {
			if port, err := strconv.Atoi(container.Args[i+1]); err == nil {
				container.Args[i+1] = strconv.Itoa(port + offset)
				return
			}
		}
	}

	// Preserve Args precedence when looking inside shell-wrapped launch strings.
	if !staggerEmbeddedFlagValue(container.Args, flag, offset) {
		staggerEmbeddedFlagValue(container.Command, flag, offset)
	}
}

// staggerEmbeddedFlagValue offsets the first embedded flag value it can parse.
func staggerEmbeddedFlagValue(tokens []string, flag string, offset int) bool {
	for i, token := range tokens {
		parts := strings.Split(token, flag+" ")
		if len(parts) < 2 {
			continue
		}

		// Read only the integer prefix, leaving trailing launch arguments intact.
		var portStr strings.Builder
		for _, ch := range parts[1] {
			if ch < '0' || ch > '9' {
				break
			}
			portStr.WriteString(string(ch))
		}
		if port, err := strconv.Atoi(portStr.String()); err == nil {
			tokens[i] = strings.Replace(token, flag+" "+portStr.String(), flag+" "+strconv.Itoa(port+offset), 1)
			return true
		}
	}
	return false
}
