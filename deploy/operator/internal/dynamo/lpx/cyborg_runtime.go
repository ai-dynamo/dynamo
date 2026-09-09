/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	corev1 "k8s.io/api/core/v1"
)

const (
	// CyborgBatchSizeEnv configures the batch handled by each Cyborg replica.
	CyborgBatchSizeEnv           = "CYBORG_BATCH_SIZE"
	cyborgFpgaGpiIOFPGACountEnv  = "CYBORG_FPGA_GPI_IO_FPGA_COUNT"
	cyborgFpgaGpiReplicaIndexEnv = "CYBORG_FPGA_GPI_REPLICA_INDEX"
	cyborgSwaCacheIDsEnv         = "CYBORG_SWA_CACHE_IDS"
	gbuildManifestPathEnv        = "GBUILD_MANIFEST_PATH"
	grovePodCliquePodIndexPath   = "metadata.labels['grove.io/podclique-pod-index']"
)

// applyCyborgManifestPath projects an authoritative manifest location into one Cyborg container.
func applyCyborgManifestPath(container *corev1.Container, projection *ModelProjection, modelStoragePath string) error {
	buildRoot, err := buildRuntimePath(lpuRuntimeBuildRef(projection, modelStoragePath), modelStoragePath)
	if err != nil {
		return fmt.Errorf("resolve GBuild manifest path: %w", err)
	}
	setContainerEnv(container, false, corev1.EnvVar{
		Name:  gbuildManifestPathEnv,
		Value: filepath.Join(buildRoot, gbuildManifestV2CapnpFile),
	})
	return nil
}

// applyCyborgRuntimeIO projects the resolved split-I/O fanout into one Cyborg container.
func applyCyborgRuntimeIO(container *corev1.Container, cyborgBatchSize int, ioFPGACount int32) {
	setContainerEnv(container, false,
		corev1.EnvVar{Name: cyborgFpgaGpiIOFPGACountEnv, Value: strconv.FormatInt(int64(ioFPGACount), 10)},
		corev1.EnvVar{
			Name: cyborgFpgaGpiReplicaIndexEnv,
			ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{
				FieldPath: grovePodCliquePodIndexPath,
			}},
		},
		corev1.EnvVar{Name: CyborgBatchSizeEnv, Value: strconv.Itoa(cyborgBatchSize)},
	)
}

// applyCyborgSWACacheIDs binds the replica index for an unbatched Cyborg container.
func applyCyborgSWACacheIDs(container *corev1.Container, cyborgBatchSize int) {
	if cyborgBatchSize == 1 {
		setContainerEnv(container, false, corev1.EnvVar{
			Name: cyborgSwaCacheIDsEnv,
			ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{
				FieldPath: grovePodCliquePodIndexPath,
			}},
		})
	}
}

const cyborgSwaBatchIDsEntrypointScript = `if [ -z "${CYBORG_SWA_CACHE_IDS+x}" ]; then
	: "${CYBORG_FPGA_GPI_REPLICA_INDEX:?CYBORG_FPGA_GPI_REPLICA_INDEX is required}"
	: "${CYBORG_BATCH_SIZE:?CYBORG_BATCH_SIZE is required}"
	base=$((CYBORG_FPGA_GPI_REPLICA_INDEX * CYBORG_BATCH_SIZE))
	ids="${base}"
	i=1
	while [ "${i}" -lt "${CYBORG_BATCH_SIZE}" ]; do
		ids="${ids},$((base + i))"
		i=$((i + 1))
	done
	export CYBORG_SWA_CACHE_IDS="${ids}"
fi
exec "$@"
`

// wrapCyborgDecodeForSwaBatch derives each batched replica's contiguous SWA cache IDs.
func wrapCyborgDecodeForSwaBatch(container *corev1.Container, cyborgBatchSize int) error {
	if cyborgBatchSize == 1 {
		return nil
	}
	if len(container.Command) == 0 {
		return fmt.Errorf("cyborg command is empty; cannot wrap image entrypoint for SWA batch ID generation")
	}

	// Preserve the complete image command while installing the batch-ID wrapper.
	args := slices.Grow(
		[]string{cyborgSwaBatchIDsEntrypointScript, "--"},
		len(container.Command)+len(container.Args),
	)
	args = append(args, container.Command...)
	args = append(args, container.Args...)
	container.Command = []string{"/bin/sh", "-ec"}
	container.Args = args
	return nil
}

// cyborgRuntimeIO requires a nonnil normalized build and validates its Cyborg replica domain.
func cyborgRuntimeIO(build *Build, replicas int32) (int, int32, error) {
	// Require every physical endpoint and client-owned transaction in this replica domain.
	ioFPGACount := build.IOFPGACount
	ioFanoutFactor := build.IOFanoutFactor
	if replicas%ioFPGACount != 0 {
		return 0, 0, fmt.Errorf("decode service Cyborg replicas %d must be divisible by ioFpgaCount %d", replicas, ioFPGACount)
	}
	if replicas/ioFPGACount%ioFanoutFactor != 0 {
		return 0, 0, fmt.Errorf(
			"decode service Cyborg replicas %d must provide fanoutFactor %d clients for each of %d I/O FPGA endpoints",
			replicas,
			ioFanoutFactor,
			ioFPGACount,
		)
	}

	// Derive the normalized batch carried by one fanout transaction on one endpoint.
	return build.BatchSize / int(ioFPGACount) / int(ioFanoutFactor), ioFPGACount, nil
}

func configuredCyborgBatchSize(container *corev1.Container, derived int) (int, error) {
	for _, variable := range container.Env {
		if variable.Name != CyborgBatchSizeEnv {
			continue
		}
		if variable.ValueFrom != nil {
			return 0, fmt.Errorf("Cyborg %s must use a literal value", CyborgBatchSizeEnv)
		}
		configured, err := strconv.Atoi(strings.TrimSpace(variable.Value))
		if err != nil || configured < 1 {
			return 0, fmt.Errorf("Cyborg %s must be an integer greater than zero, got %q", CyborgBatchSizeEnv, variable.Value)
		}
		return configured, nil
	}
	return derived, nil
}

// setContainerEnv updates variables in order while optionally preserving the first existing value.
func setContainerEnv(container *corev1.Container, preserveExisting bool, variables ...corev1.EnvVar) {
variables:
	for _, variable := range variables {
		for index := range container.Env {
			if container.Env[index].Name == variable.Name {
				if !preserveExisting {
					container.Env[index] = variable
				}
				continue variables
			}
		}
		container.Env = append(container.Env, variable)
	}
}
