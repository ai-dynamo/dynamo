/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package enginegroup

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

const (
	profileGeometryFingerprintVersion = "engine-group-profile-geometry/v1"
	// MainCapacityRoleName is the primary Dynamo workload role supported by the narrow resolver.
	MainCapacityRoleName = "main"
)

// ErrUnsupportedProfile classifies declarative geometry that this narrow resolver cannot prove safe.
var ErrUnsupportedProfile = errors.New("unsupported Engine Group profile geometry")

// UnsupportedProfileReason is a stable machine-readable explanation for unsupported geometry.
type UnsupportedProfileReason string

const (
	// UnsupportedProfileReasonCapacityLayout means there is no unambiguous single main capacity role.
	UnsupportedProfileReasonCapacityLayout UnsupportedProfileReason = "capacity-layout"
	// UnsupportedProfileReasonGPUAllocation means the engine GPU allocation is not dedicated to its capacity pod.
	UnsupportedProfileReasonGPUAllocation UnsupportedProfileReason = "gpu-allocation"
	// UnsupportedProfileReasonPackedReplicas means one pod contains more than one independently scalable replica.
	UnsupportedProfileReasonPackedReplicas UnsupportedProfileReason = "packed-replicas"
	// UnsupportedProfileReasonMultiPodReplica means one replica spans multiple pods.
	UnsupportedProfileReasonMultiPodReplica UnsupportedProfileReason = "multi-pod-replica"
)

// UnsupportedProfileError describes why valid typed geometry is outside this resolver's scope.
type UnsupportedProfileError struct {
	Reason UnsupportedProfileReason
	Detail string
}

// Error returns the stable unsupported-profile class and reason with human-readable detail.
func (e *UnsupportedProfileError) Error() string {
	if e.Detail == "" {
		return fmt.Sprintf("%s: %s", ErrUnsupportedProfile, e.Reason)
	}

	return fmt.Sprintf("%s: %s: %s", ErrUnsupportedProfile, e.Reason, e.Detail)
}

// Unwrap makes every UnsupportedProfileError match ErrUnsupportedProfile with errors.Is.
func (e *UnsupportedProfileError) Unwrap() error {
	return ErrUnsupportedProfile
}

// CapacityRoleClass states whether a pod-producing role contributes engine membership.
type CapacityRoleClass string

const (
	// CapacityRoleClassRankOwningCapacity means each role instance owns native engine membership.
	CapacityRoleClassRankOwningCapacity CapacityRoleClass = "RankOwningCapacity"
	// CapacityRoleClassControlOnly means role instances coordinate capacity without owning membership.
	CapacityRoleClassControlOnly CapacityRoleClass = "ControlOnly"
)

// CapacityRoleGeometry is the provider-resolved physical geometry of one pod-producing role.
type CapacityRoleGeometry struct {
	Name             string
	Class            CapacityRoleClass
	EngineGPUsPerPod int64
	DedicatedGPUs    bool
}

// ProfileGeometryInput contains typed, immutable facts resolved at a provider boundary.
// EngineGeometryDigest and WorkloadRevisionDigest must exclude initial and live replica targets.
type ProfileGeometryInput struct {
	Backend                string
	GPUsPerReplica         int64
	CapacityRoles          []CapacityRoleGeometry
	EngineGeometryDigest   string
	WorkloadRevisionDigest string
}

// ResolvedProfileGeometry is the immutable physical geometry proven by this resolver.
// It does not prove bootstrap behavior, native-rank placement, safe bounds, topology,
// membership capabilities, or overall Engine Group eligibility.
type ResolvedProfileGeometry struct {
	Backend        string
	GPUsPerReplica int64
	PodsPerReplica int32
	CapacityRole   CapacityRoleGeometry
	Fingerprint    string
}

type profileGeometryFingerprintProjection struct {
	Version          string                        `json:"version"`
	Backend          string                        `json:"backend"`
	GPUsPerReplica   int64                         `json:"gpusPerReplica"`
	PodsPerReplica   int32                         `json:"podsPerReplica"`
	CapacityRole     profileCapacityRoleProjection `json:"capacityRole"`
	EngineGeometry   string                        `json:"engineGeometry"`
	WorkloadRevision string                        `json:"workloadRevision"`
}

type profileCapacityRoleProjection struct {
	Name             string            `json:"name"`
	Class            CapacityRoleClass `json:"class"`
	EngineGPUsPerPod int64             `json:"engineGPUsPerPod"`
	DedicatedGPUs    bool              `json:"dedicatedGPUs"`
}

// ResolveProfileGeometry validates typed geometry for the narrow single-main,
// one-pod-per-replica profile. A successful result does not by itself enable an
// Engine Group; provider extraction and runtime conformance are separate contracts.
// The function does not mutate input or its CapacityRoles slice.
func ResolveProfileGeometry(input ProfileGeometryInput) (ResolvedProfileGeometry, error) {
	// Reject malformed scalar facts before reasoning about profile support.
	if input.Backend == "" {
		return ResolvedProfileGeometry{}, fmt.Errorf("backend is required")
	}
	if input.GPUsPerReplica <= 0 {
		return ResolvedProfileGeometry{}, fmt.Errorf("GPUs per replica must be positive, got %d", input.GPUsPerReplica)
	}
	if input.EngineGeometryDigest == "" {
		return ResolvedProfileGeometry{}, fmt.Errorf("engine geometry digest is required")
	}
	if input.WorkloadRevisionDigest == "" {
		return ResolvedProfileGeometry{}, fmt.Errorf("workload revision digest is required")
	}

	// Require the only currently proven layout: one rank-owning main capacity role.
	if len(input.CapacityRoles) != 1 ||
		input.CapacityRoles[0].Name != MainCapacityRoleName ||
		input.CapacityRoles[0].Class != CapacityRoleClassRankOwningCapacity {
		return ResolvedProfileGeometry{}, &UnsupportedProfileError{
			Reason: UnsupportedProfileReasonCapacityLayout,
			Detail: "exactly one rank-owning capacity role named main is required",
		}
	}
	role := input.CapacityRoles[0]
	if !role.DedicatedGPUs {
		return ResolvedProfileGeometry{}, &UnsupportedProfileError{
			Reason: UnsupportedProfileReasonGPUAllocation,
			Detail: "the main engine GPU allocation must be dedicated",
		}
	}
	if role.EngineGPUsPerPod <= 0 {
		return ResolvedProfileGeometry{}, fmt.Errorf("engine GPUs per pod must be positive, got %d", role.EngineGPUsPerPod)
	}

	// Reject shapes that cannot release exactly one logical replica as one whole pod.
	if input.GPUsPerReplica < role.EngineGPUsPerPod {
		return ResolvedProfileGeometry{}, &UnsupportedProfileError{
			Reason: UnsupportedProfileReasonPackedReplicas,
			Detail: fmt.Sprintf("one replica needs %d GPUs but each pod allocates %d", input.GPUsPerReplica, role.EngineGPUsPerPod),
		}
	}
	if input.GPUsPerReplica > role.EngineGPUsPerPod {
		return ResolvedProfileGeometry{}, &UnsupportedProfileError{
			Reason: UnsupportedProfileReasonMultiPodReplica,
			Detail: fmt.Sprintf("one replica needs %d GPUs but each pod allocates %d", input.GPUsPerReplica, role.EngineGPUsPerPod),
		}
	}

	// Materialize only the immutable physical facts established by this calculation.
	resolved := ResolvedProfileGeometry{
		Backend:        input.Backend,
		GPUsPerReplica: input.GPUsPerReplica,
		PodsPerReplica: 1,
		CapacityRole:   role,
	}

	// Bind the fingerprint to canonical immutable semantics, excluding replica targets.
	fingerprint, err := fingerprintProfileGeometry(
		resolved,
		input.EngineGeometryDigest,
		input.WorkloadRevisionDigest,
	)
	if err != nil {
		return ResolvedProfileGeometry{}, err
	}
	resolved.Fingerprint = fingerprint

	return resolved, nil
}

func fingerprintProfileGeometry(
	resolved ResolvedProfileGeometry,
	engineGeometryDigest string,
	workloadRevisionDigest string,
) (string, error) {
	// Canonicalize a versioned projection so future semantic changes can change the identity safely.
	projection := profileGeometryFingerprintProjection{
		Version:        profileGeometryFingerprintVersion,
		Backend:        resolved.Backend,
		GPUsPerReplica: resolved.GPUsPerReplica,
		PodsPerReplica: resolved.PodsPerReplica,
		CapacityRole: profileCapacityRoleProjection{
			Name:             resolved.CapacityRole.Name,
			Class:            resolved.CapacityRole.Class,
			EngineGPUsPerPod: resolved.CapacityRole.EngineGPUsPerPod,
			DedicatedGPUs:    resolved.CapacityRole.DedicatedGPUs,
		},
		EngineGeometry:   engineGeometryDigest,
		WorkloadRevision: workloadRevisionDigest,
	}
	encoded, err := json.Marshal(projection)
	if err != nil {
		return "", fmt.Errorf("marshal profile geometry fingerprint projection: %w", err)
	}

	// Prefix the digest so the fingerprint encoding is self-describing.
	digest := sha256.Sum256(encoded)
	return "sha256:" + hex.EncodeToString(digest[:]), nil
}
