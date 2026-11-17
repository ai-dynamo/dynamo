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
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
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
	partitions []buildPartition,
) ([]buildPartition, []int, error) {
	if selection == nil {
		return partitions, nil, nil
	}

	// A selected prop-sync chain is one runtime partition owned by its first member.
	runtimeOwner := make(map[int]int, len(build.partitions))
	for _, partition := range build.partitions {
		runtimeOwner[partition.sourcePartitionID] = partition.sourcePartitionID
	}
	for _, chain := range build.selectedPropSyncChains {
		for _, member := range chain {
			runtimeOwner[member] = chain[0]
		}
	}
	local, err := selectLocalRuntimePartitions(selection, pipeline, runtimeOwner)
	if err != nil {
		return nil, nil, err
	}

	remote := make([]buildPartition, 0, len(partitions))
	for _, partition := range partitions {
		if !local[runtimeOwner[partition.sourcePartitionID]] {
			remote = append(remote, partition)
		}
	}
	return remote, slices.Sorted(maps.Keys(local)), nil
}

func selectRemoteUnits(
	selection *dynamov1beta1.LPXLocalPartitions,
	pipeline Pipeline,
	units deploymentcapnpv1.LpuUnit_List,
) ([]deploymentcapnpv1.LpuUnit, []int, error) {
	remote := make([]deploymentcapnpv1.LpuUnit, 0, units.Len())
	if selection == nil {
		for i := range units.Len() {
			remote = append(remote, units.At(i))
		}
		return remote, nil, nil
	}

	runtimeOwner := make(map[int]int)
	roots := make([]int, units.Len())
	for i := range units.Len() {
		unit := units.At(i)
		root, err := unitRoot(unit)
		if err != nil {
			return nil, nil, err
		}
		roots[i] = root
		reservations, err := unit.Reservations()
		if err != nil {
			return nil, nil, fmt.Errorf("reading unit reservations: %w", err)
		}
		for j := range reservations.Len() {
			ids, err := reservations.At(j).PartitionIds()
			if err != nil {
				return nil, nil, fmt.Errorf("reading unit members: %w", err)
			}
			for k := range ids.Len() {
				runtimeOwner[int(ids.At(k))] = root
			}
		}
	}
	local, err := selectLocalRuntimePartitions(selection, pipeline, runtimeOwner)
	if err != nil {
		return nil, nil, err
	}
	for i, root := range roots {
		if !local[root] {
			remote = append(remote, units.At(i))
		}
	}
	return remote, slices.Sorted(maps.Keys(local)), nil
}

// Only a hybrid pipeline has a Cyborg GPU to run local units.
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

// applyLocalPartitionIDs publishes the operator-owned selection. An empty value
// clears envFrom defaults; units runtimes require it even with no local roots.
func applyLocalPartitionIDs(container *corev1.Container, projection *ModelProjection) {
	ids := make([]string, len(projection.localPartitionIDs))
	for index, id := range projection.localPartitionIDs {
		ids[index] = strconv.Itoa(id)
	}
	publish := len(ids) > 0 || len(container.EnvFrom) > 0 || projection.placesLPUUnits()

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
