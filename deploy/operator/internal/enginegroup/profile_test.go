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
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var profileFingerprintPattern = regexp.MustCompile("^sha256:[0-9a-f]{64}$")

type testProfileGeometryOverrides struct {
	backend                    string
	gpusPerReplica             int64
	capacityRoles              []CapacityRoleGeometry
	engineGeometryDigest       string
	workloadRevisionDigest     string
	omitBackend                bool
	omitGPUsPerReplica         bool
	omitEngineGeometryDigest   bool
	omitWorkloadRevisionDigest bool
}

func TestResolveProfileGeometry(t *testing.T) {
	tests := []struct {
		name      string
		overrides testProfileGeometryOverrides
		want      ResolvedProfileGeometry
	}{
		{
			name: "vLLM typed geometry",
			want: ResolvedProfileGeometry{
				Backend:        "vllm",
				GPUsPerReplica: 4,
				PodsPerReplica: 1,
				CapacityRole: CapacityRoleGeometry{
					Name:             MainCapacityRoleName,
					Class:            CapacityRoleClassRankOwningCapacity,
					EngineGPUsPerPod: 4,
					DedicatedGPUs:    true,
				},
			},
		},
		{
			name: "backend-neutral typed geometry",
			overrides: testProfileGeometryOverrides{
				backend: "future-engine",
			},
			want: ResolvedProfileGeometry{
				Backend:        "future-engine",
				GPUsPerReplica: 4,
				PodsPerReplica: 1,
				CapacityRole: CapacityRoleGeometry{
					Name:             MainCapacityRoleName,
					Class:            CapacityRoleClassRankOwningCapacity,
					EngineGPUsPerPod: 4,
					DedicatedGPUs:    true,
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build the provider-resolved geometry input")
			input := newTestProfileGeometryInput(test.overrides)

			t.Log("resolve the immutable physical geometry")
			got, err := ResolveProfileGeometry(input)
			require.NoError(t, err)

			t.Log("verify every proven geometry field")
			fingerprint := got.Fingerprint
			got.Fingerprint = ""
			assert.Equal(t, test.want, got)
			assert.Regexp(t, profileFingerprintPattern, fingerprint)
		})
	}
}

func TestResolveProfileGeometryUnsupported(t *testing.T) {
	noRoles := newTestProfileGeometryInput(testProfileGeometryOverrides{})
	noRoles.CapacityRoles = []CapacityRoleGeometry{}
	twoRoles := newTestProfileGeometryInput(testProfileGeometryOverrides{})
	twoRoles.CapacityRoles = append(twoRoles.CapacityRoles, CapacityRoleGeometry{Name: "worker"})

	tests := []struct {
		name   string
		input  ProfileGeometryInput
		reason UnsupportedProfileReason
	}{
		{
			name:   "no capacity role",
			input:  noRoles,
			reason: UnsupportedProfileReasonCapacityLayout,
		},
		{
			name:   "multiple capacity roles",
			input:  twoRoles,
			reason: UnsupportedProfileReasonCapacityLayout,
		},
		{
			name: "non-main capacity role",
			input: newTestProfileGeometryInput(testProfileGeometryOverrides{
				capacityRoles: []CapacityRoleGeometry{
					{Name: "worker", Class: CapacityRoleClassRankOwningCapacity, EngineGPUsPerPod: 4, DedicatedGPUs: true},
				},
			}),
			reason: UnsupportedProfileReasonCapacityLayout,
		},
		{
			name: "control-only main role",
			input: newTestProfileGeometryInput(testProfileGeometryOverrides{
				capacityRoles: []CapacityRoleGeometry{
					{Name: MainCapacityRoleName, Class: CapacityRoleClassControlOnly, EngineGPUsPerPod: 4, DedicatedGPUs: true},
				},
			}),
			reason: UnsupportedProfileReasonCapacityLayout,
		},
		{
			name: "unclassified main role",
			input: newTestProfileGeometryInput(testProfileGeometryOverrides{
				capacityRoles: []CapacityRoleGeometry{
					{Name: MainCapacityRoleName, EngineGPUsPerPod: 4, DedicatedGPUs: true},
				},
			}),
			reason: UnsupportedProfileReasonCapacityLayout,
		},
		{
			name: "shared engine GPUs",
			input: newTestProfileGeometryInput(testProfileGeometryOverrides{
				capacityRoles: []CapacityRoleGeometry{
					{Name: MainCapacityRoleName, Class: CapacityRoleClassRankOwningCapacity, EngineGPUsPerPod: 4},
				},
			}),
			reason: UnsupportedProfileReasonGPUAllocation,
		},
		{
			name: "packed replicas",
			input: newTestProfileGeometryInput(testProfileGeometryOverrides{
				capacityRoles: []CapacityRoleGeometry{
					{Name: MainCapacityRoleName, Class: CapacityRoleClassRankOwningCapacity, EngineGPUsPerPod: 8, DedicatedGPUs: true},
				},
			}),
			reason: UnsupportedProfileReasonPackedReplicas,
		},
		{
			name: "multi-pod replica",
			input: newTestProfileGeometryInput(testProfileGeometryOverrides{
				capacityRoles: []CapacityRoleGeometry{
					{Name: MainCapacityRoleName, Class: CapacityRoleClassRankOwningCapacity, EngineGPUsPerPod: 2, DedicatedGPUs: true},
				},
			}),
			reason: UnsupportedProfileReasonMultiPodReplica,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("attempt to resolve unsupported typed geometry")
			got, err := ResolveProfileGeometry(test.input)
			require.Error(t, err)

			t.Log("verify fail-closed output and the stable unsupported reason")
			assert.Zero(t, got)
			assert.ErrorIs(t, err, ErrUnsupportedProfile)
			var unsupported *UnsupportedProfileError
			require.ErrorAs(t, err, &unsupported)
			assert.Equal(t, test.reason, unsupported.Reason)
			assert.NotEmpty(t, unsupported.Detail)
		})
	}
}

func TestResolveProfileGeometryInvalid(t *testing.T) {
	tests := []struct {
		name        string
		overrides   testProfileGeometryOverrides
		wantMessage string
	}{
		{
			name:        "missing backend",
			overrides:   testProfileGeometryOverrides{omitBackend: true},
			wantMessage: "backend is required",
		},
		{
			name:        "zero GPUs per replica",
			overrides:   testProfileGeometryOverrides{omitGPUsPerReplica: true},
			wantMessage: "GPUs per replica must be positive",
		},
		{
			name:        "negative GPUs per replica",
			overrides:   testProfileGeometryOverrides{gpusPerReplica: -1},
			wantMessage: "GPUs per replica must be positive",
		},
		{
			name:        "missing engine geometry digest",
			overrides:   testProfileGeometryOverrides{omitEngineGeometryDigest: true},
			wantMessage: "engine geometry digest is required",
		},
		{
			name:        "missing workload revision digest",
			overrides:   testProfileGeometryOverrides{omitWorkloadRevisionDigest: true},
			wantMessage: "workload revision digest is required",
		},
		{
			name: "zero engine GPUs per pod",
			overrides: testProfileGeometryOverrides{
				capacityRoles: []CapacityRoleGeometry{
					{Name: MainCapacityRoleName, Class: CapacityRoleClassRankOwningCapacity, DedicatedGPUs: true},
				},
			},
			wantMessage: "engine GPUs per pod must be positive",
		},
		{
			name: "negative engine GPUs per pod",
			overrides: testProfileGeometryOverrides{
				capacityRoles: []CapacityRoleGeometry{
					{Name: MainCapacityRoleName, Class: CapacityRoleClassRankOwningCapacity, EngineGPUsPerPod: -1, DedicatedGPUs: true},
				},
			},
			wantMessage: "engine GPUs per pod must be positive",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("build malformed provider-resolved facts")
			input := newTestProfileGeometryInput(test.overrides)

			t.Log("attempt to resolve invalid geometry")
			got, err := ResolveProfileGeometry(input)
			require.Error(t, err)

			t.Log("verify invalid input remains distinct from unsupported geometry")
			assert.Zero(t, got)
			assert.NotErrorIs(t, err, ErrUnsupportedProfile)
			assert.ErrorContains(t, err, test.wantMessage)
		})
	}
}

func TestResolveProfileGeometryFingerprint(t *testing.T) {
	t.Log("resolve the baseline immutable geometry")
	baselineInput := newTestProfileGeometryInput(testProfileGeometryOverrides{})
	baseline, err := ResolveProfileGeometry(baselineInput)
	require.NoError(t, err)

	tests := []struct {
		name      string
		overrides testProfileGeometryOverrides
		wantSame  bool
	}{
		{
			name:     "same semantic facts",
			wantSame: true,
		},
		{
			name: "backend changes",
			overrides: testProfileGeometryOverrides{
				backend: "sglang",
			},
			wantSame: false,
		},
		{
			name: "engine geometry changes",
			overrides: testProfileGeometryOverrides{
				engineGeometryDigest: "sha256:geometry-b",
			},
			wantSame: false,
		},
		{
			name: "workload revision changes",
			overrides: testProfileGeometryOverrides{
				workloadRevisionDigest: "sha256:workload-b",
			},
			wantSame: false,
		},
		{
			name: "physical GPU geometry changes",
			overrides: testProfileGeometryOverrides{
				gpusPerReplica: 8,
				capacityRoles: []CapacityRoleGeometry{
					{Name: MainCapacityRoleName, Class: CapacityRoleClassRankOwningCapacity, EngineGPUsPerPod: 8, DedicatedGPUs: true},
				},
			},
			wantSame: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("resolve the comparison geometry")
			input := newTestProfileGeometryInput(test.overrides)
			resolved, err := ResolveProfileGeometry(input)
			require.NoError(t, err)

			t.Log("compare the canonical immutable fingerprints")
			if test.wantSame {
				assert.Equal(t, baseline.Fingerprint, resolved.Fingerprint)
			} else {
				assert.NotEqual(t, baseline.Fingerprint, resolved.Fingerprint)
			}
		})
	}
}

func TestResolveProfileGeometryDoesNotMutateInput(t *testing.T) {
	t.Log("capture typed provider facts before resolution")
	input := newTestProfileGeometryInput(testProfileGeometryOverrides{})
	before := cloneTestProfileGeometryInput(input)

	t.Log("resolve the profile geometry")
	_, err := ResolveProfileGeometry(input)
	require.NoError(t, err)

	t.Log("verify the input and its role slice retain their original content")
	assert.Equal(t, before, input)
}

func newTestProfileGeometryInput(overrides testProfileGeometryOverrides) ProfileGeometryInput {
	input := ProfileGeometryInput{
		Backend:                "vllm",
		GPUsPerReplica:         4,
		EngineGeometryDigest:   "sha256:geometry-a",
		WorkloadRevisionDigest: "sha256:workload-a",
		CapacityRoles: []CapacityRoleGeometry{
			{
				Name:             MainCapacityRoleName,
				Class:            CapacityRoleClassRankOwningCapacity,
				EngineGPUsPerPod: 4,
				DedicatedGPUs:    true,
			},
		},
	}

	// Apply scalar overrides while retaining concise defaults for table cases.
	if overrides.omitBackend {
		input.Backend = ""
	} else if overrides.backend != "" {
		input.Backend = overrides.backend
	}
	if overrides.omitGPUsPerReplica {
		input.GPUsPerReplica = 0
	} else if overrides.gpusPerReplica != 0 {
		input.GPUsPerReplica = overrides.gpusPerReplica
	}
	if overrides.engineGeometryDigest != "" {
		input.EngineGeometryDigest = overrides.engineGeometryDigest
	}
	if overrides.workloadRevisionDigest != "" {
		input.WorkloadRevisionDigest = overrides.workloadRevisionDigest
	}
	if overrides.omitEngineGeometryDigest {
		input.EngineGeometryDigest = ""
	}
	if overrides.omitWorkloadRevisionDigest {
		input.WorkloadRevisionDigest = ""
	}

	// Replace the entire role classification when a case needs a different layout.
	if overrides.capacityRoles != nil {
		input.CapacityRoles = append([]CapacityRoleGeometry(nil), overrides.capacityRoles...)
	}

	return input
}

func cloneTestProfileGeometryInput(input ProfileGeometryInput) ProfileGeometryInput {
	cloned := input
	cloned.CapacityRoles = append([]CapacityRoleGeometry(nil), input.CapacityRoles...)
	return cloned
}
