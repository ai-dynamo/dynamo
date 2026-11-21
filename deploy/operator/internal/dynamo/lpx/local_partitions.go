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

	// A unit's root is the first partition of its first reservation; the root owns every member.
	runtimeOwner := make(map[int]int)
	roots := make([]int, units.Len())
	for i := range units.Len() {
		reservations, err := units.At(i).Reservations()
		if err != nil {
			return nil, nil, fmt.Errorf("reading unit reservations: %w", err)
		}
		if reservations.Len() == 0 {
			return nil, nil, fmt.Errorf("lpuUnit has no root partition")
		}
		for j := range reservations.Len() {
			ids, err := reservations.At(j).PartitionIds()
			if err != nil {
				return nil, nil, fmt.Errorf("reading unit members: %w", err)
			}
			if j == 0 {
				if ids.Len() == 0 {
					return nil, nil, fmt.Errorf("lpuUnits reservation has no root partition")
				}
				roots[i] = int(ids.At(0))
			}
			for k := range ids.Len() {
				runtimeOwner[int(ids.At(k))] = roots[i]
			}
		}
	}

	// Only a hybrid pipeline has a Cyborg GPU to run local units.
	if pipeline != PipelineHybrid {
		return nil, nil, fmt.Errorf("%w: localPartitions requires a hybrid build with a Cyborg conductor", ErrUnsupportedRuntime)
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
					"localPartitions references partition %d of the unit that starts at partition %d; select %d",
					id, owner, owner,
				)
			}
			local[id] = true
		}
	}
	for i, root := range roots {
		if !local[root] {
			remote = append(remote, units.At(i))
		}
	}
	return remote, slices.Sorted(maps.Keys(local)), nil
}

// applyLocalPartitionIDs publishes the operator-owned selection. An empty value
// clears envFrom defaults; units runtimes require it even with no local roots.
func applyLocalPartitionIDs(container *corev1.Container, projection *ModelProjection) {
	ids := make([]string, len(projection.localPartitionIDs))
	for index, id := range projection.localPartitionIDs {
		ids[index] = strconv.Itoa(id)
	}

	prependEnv(container, corev1.EnvVar{
		Name:  localPartitionIDsEnv,
		Value: strings.Join(ids, ","),
	})
}
