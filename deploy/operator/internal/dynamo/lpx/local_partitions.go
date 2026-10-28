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

// selectRemoteUnits returns the units that remain on LPUs, in Agent order, and
// the sorted roots of the units that run on the Cyborg GPU. A unit is one runtime
// partition named by its root. A nil selection returns units unchanged. No input
// is mutated.
func selectRemoteUnits(
	selection *dynamov1beta1.LPXLocalPartitions,
	pipeline Pipeline,
	units []lpuUnit,
) ([]lpuUnit, []int, error) {
	if selection == nil {
		return units, nil, nil
	}

	// Every unit member resolves to the unit root that names its runtime partition.
	runtimeOwner := make(map[int]int)
	for _, unit := range units {
		for _, id := range unit.partitionIDs() {
			runtimeOwner[id] = unit.root()
		}
	}
	local, err := selectLocalRuntimePartitions(selection, pipeline, runtimeOwner)
	if err != nil {
		return nil, nil, err
	}

	remote := make([]lpuUnit, 0, len(units))
	for _, unit := range units {
		if !local[unit.root()] {
			remote = append(remote, unit)
		}
	}
	return remote, slices.Sorted(maps.Keys(local)), nil
}

// selectLocalRuntimePartitions resolves a nonnil selection against runtimeOwner,
// which maps every LPU partition ID to the ID of the runtime partition that owns
// it, and returns the selected runtime partition IDs. Only a hybrid pipeline has
// a Cyborg GPU to run them.
func selectLocalRuntimePartitions(
	selection *dynamov1beta1.LPXLocalPartitions,
	pipeline Pipeline,
	runtimeOwner map[int]int,
) (map[int]bool, error) {
	if pipeline != PipelineHybrid {
		return nil, fmt.Errorf("%w: localPartitions requires a hybrid build with a Cyborg conductor", ErrUnsupportedRuntime)
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
				return nil, fmt.Errorf("localPartitions references partition %d, which the build does not contain", id)
			}
			if owner != id {
				return nil, fmt.Errorf(
					"localPartitions references partition %d of the prop-sync chain that starts at partition %d; select %d",
					id, owner, owner,
				)
			}
			local[id] = true
		}
	default:
		return nil, fmt.Errorf("localPartitions has unsupported mode %q", selection.Mode)
	}
	return local, nil
}

// applyLocalPartitionIDs publishes the projection's GPU-local partitions into one
// Cyborg or hybrid Agent container. The operator owns the variable: it removes any
// authored value and always sets it, empty when every unit stays on LPUs, so
// runtimes can tell which units run on the GPU.
func applyLocalPartitionIDs(container *corev1.Container, projection *ModelProjection) {
	ids := make([]string, len(projection.localPartitionIDs))
	for index, id := range projection.localPartitionIDs {
		ids[index] = strconv.Itoa(id)
	}

	// Kubernetes expands environment references in order; publish the selection before authored bindings.
	env := make([]corev1.EnvVar, 0, len(container.Env)+1)
	env = append(env, corev1.EnvVar{Name: localPartitionIDsEnv, Value: strings.Join(ids, ",")})
	for _, variable := range container.Env {
		if variable.Name != localPartitionIDsEnv {
			env = append(env, variable)
		}
	}
	container.Env = env
}
