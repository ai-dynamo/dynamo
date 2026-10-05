/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"

	dynamov1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
)

const localPartitionIDsEnv = "LPX_LOCAL_PARTITION_IDS"

// selectRemotePartitions filters partitions, the scheduler reservations derived
// from build, to those that remain on LPUs and returns the sorted runtime
// partition IDs that run on the Cyborg GPU. build is a nonnil physical build that
// has not collapsed its selected prop-sync chains. A nil selection returns
// partitions unchanged. No input is mutated.
func selectRemotePartitions(
	selection *dynamov1beta1.LPXLocalPartitions,
	pipeline Pipeline,
	build *Build,
	partitions []BuildPartition,
) ([]BuildPartition, []int, error) {
	if selection == nil {
		return partitions, nil, nil
	}
	if pipeline != PipelineLPX {
		return nil, nil, fmt.Errorf("%w: localPartitions requires a hybrid build with a Cyborg conductor", ErrUnsupportedRuntime)
	}

	// A selected prop-sync chain is one runtime partition owned by its first member.
	runtimeOwner := make(map[int]int, len(build.Partitions))
	for _, partition := range build.Partitions {
		runtimeOwner[partition.SourcePartitionID] = partition.SourcePartitionID
	}
	for _, chain := range build.SelectedPropSyncChains {
		for _, member := range chain {
			runtimeOwner[member] = chain[0]
		}
	}

	local := make(map[int]bool)
	switch selection.Mode {
	case dynamov1beta1.LPXLocalPartitionsModeAll:
		for _, owner := range runtimeOwner {
			local[owner] = true
		}
	case dynamov1beta1.LPXLocalPartitionsModeIDs:
		for _, requested := range selection.IDs {
			id := int(requested)
			owner, exists := runtimeOwner[id]
			if !exists {
				return nil, nil, fmt.Errorf("localPartitions references partition %d, which the build does not contain", id)
			}
			if owner != id {
				return nil, nil, fmt.Errorf(
					"localPartitions references partition %d of the prop-sync chain that starts at partition %d; select %d",
					id, owner, owner,
				)
			}
			local[id] = true
		}
	default:
		return nil, nil, fmt.Errorf("localPartitions has unsupported mode %q", selection.Mode)
	}

	remote := make([]BuildPartition, 0, len(partitions))
	for _, partition := range partitions {
		if !local[runtimeOwner[partition.SourcePartitionID]] {
			remote = append(remote, partition)
		}
	}
	return remote, slices.Sorted(maps.Keys(local)), nil
}

// applyLocalPartitionIDs publishes the projection's GPU-local partitions into one
// Cyborg container. The operator owns the variable: it removes any authored value
// and sets it when the projection has local partitions. Without local partitions,
// it sets an empty value only when envFrom sources could otherwise supply one;
// Cyborg treats an empty value as no selection.
func applyLocalPartitionIDs(container *corev1.Container, projection *ModelProjection) {
	ids := make([]string, len(projection.localPartitionIDs))
	for index, id := range projection.localPartitionIDs {
		ids[index] = strconv.Itoa(id)
	}
	publish := len(ids) > 0 || len(container.EnvFrom) > 0

	// Kubernetes expands environment references in order; publish the selection before authored bindings.
	env := make([]corev1.EnvVar, 0, len(container.Env)+1)
	if publish {
		env = append(env, corev1.EnvVar{Name: localPartitionIDsEnv, Value: strings.Join(ids, ",")})
	}
	for _, variable := range container.Env {
		if variable.Name != localPartitionIDsEnv {
			env = append(env, variable)
		}
	}
	container.Env = env
}
