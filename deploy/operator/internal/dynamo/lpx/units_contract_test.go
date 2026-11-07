/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
)

// These fixtures are shared with Sonata gbuild-contract. They are hand-written
// from capnp v12.8.0, not derived from either reader's output.
// Source: lpu-monorepo c4253aaabed, Sonata gbuild-contract/testdata/lpu-units-cases.json.
type unitsContractCase struct {
	Name      string
	Valid     bool
	Rule      string
	Consumers map[string]string
	Agents    []struct {
		PodIndex   int
		LocalRoots []int64
		Rule       string
	}
	Deployment struct {
		CompilationMode                                  string
		SupportsCpuEmbeddings, StandaloneTokenEmbeddings bool
		Partitions                                       []struct {
			ID  uint32
			LPU *struct {
				NumChips, DevicesPerNode uint32
				Architecture             string
			}
			CUDA *struct{}
			CPU  *struct{}
		}
		LpuUnits []struct {
			Reservations []struct {
				PartitionIDs []uint32
				Size         struct {
					Nodes uint32
					HX    *struct{ TraysPerHemisphere, Hemispheres, Racks uint8 }
				}
			}
			Connectors []struct {
				Xt8888Gap    *struct{}
				HxPropSyncV1 *struct {
					Connections []struct{ FromDevice, ToDevice uint32 }
				}
			}
		}
	}
}

func encodeUnitsContractCase(t *testing.T, tc unitsContractCase) deploymentcapnpv1.Deployment {
	t.Helper()
	deployment := newDeploymentV1ContractFixture(t)
	config, err := deployment.Config()
	require.NoError(t, err)
	switch tc.Deployment.CompilationMode {
	case "lpuOnly":
		config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpuOnly)
	case "lpx":
		config.SetCompilationMode(deploymentcapnpv1.CompilationMode_lpx)
	default:
		t.Fatalf("unknown fixture compilation mode %q", tc.Deployment.CompilationMode)
	}
	program, err := config.Program()
	require.NoError(t, err)
	program.SetSupportsCpuEmbeddings(tc.Deployment.SupportsCpuEmbeddings)
	program.SetStandaloneTokenEmbeddings(tc.Deployment.StandaloneTokenEmbeddings)
	partitions, err := deployment.NewPartitions(int32(len(tc.Deployment.Partitions)))
	require.NoError(t, err)
	for i, p := range tc.Deployment.Partitions {
		ref, err := partitions.At(i).NewPartition()
		require.NoError(t, err)
		ref.SetPartitionId(p.ID)
		if p.CPU != nil {
			ref.SetDeviceType(commoncapnpv1.DeviceType_cpu)
			_, err = partitions.At(i).Detail().NewCpu()
			require.NoError(t, err)
			continue
		}
		if p.CUDA != nil {
			ref.SetDeviceType(commoncapnpv1.DeviceType_cuda)
			_, err = partitions.At(i).Detail().NewCuda()
			require.NoError(t, err)
			continue
		}
		require.NotNil(t, p.LPU)
		ref.SetDeviceType(commoncapnpv1.DeviceType_lpu)
		detail, err := partitions.At(i).Detail().NewLpu()
		require.NoError(t, err)
		require.NoError(t, detail.SetPath(fmt.Sprintf("part-%d", p.ID)))
		require.NoError(t, detail.SetTopology("test-topology"))
		detail.SetNumChips(p.LPU.NumChips)
		detail.SetDevicesPerNode(p.LPU.DevicesPerNode)
		switch p.LPU.Architecture {
		case "lp20":
			detail.SetArchitecture(deploymentcapnpv1.LpuArchitecture_lp20)
		case "lp30":
			detail.SetArchitecture(deploymentcapnpv1.LpuArchitecture_lp30)
		default:
			t.Fatalf("unknown fixture architecture %q", p.LPU.Architecture)
		}
	}
	units, err := config.NewLpuUnits(int32(len(tc.Deployment.LpuUnits)))
	require.NoError(t, err)
	for i, u := range tc.Deployment.LpuUnits {
		reservations, err := units.At(i).NewReservations(int32(len(u.Reservations)))
		require.NoError(t, err)
		for j, r := range u.Reservations {
			ids, err := reservations.At(j).NewPartitionIds(int32(len(r.PartitionIDs)))
			require.NoError(t, err)
			for k, id := range r.PartitionIDs {
				ids.Set(k, id)
			}
			if r.Size.HX == nil {
				reservations.At(j).Size().SetNodes(r.Size.Nodes)
			} else {
				hx, err := reservations.At(j).Size().NewHx()
				require.NoError(t, err)
				hx.SetTraysPerHemisphere(r.Size.HX.TraysPerHemisphere)
				hx.SetHemispheres(r.Size.HX.Hemispheres)
				hx.SetRacks(r.Size.HX.Racks)
			}
		}
		connectors, err := units.At(i).NewConnectors(int32(len(u.Connectors)))
		require.NoError(t, err)
		for j, c := range u.Connectors {
			if c.Xt8888Gap != nil {
				connectors.At(j).SetXt8888Gap()
				continue
			}
			require.NotNil(t, c.HxPropSyncV1)
			hx, err := connectors.At(j).NewHxPropSyncV1()
			require.NoError(t, err)
			pairs, err := hx.NewConnections(int32(len(c.HxPropSyncV1.Connections)))
			require.NoError(t, err)
			for k, pair := range c.HxPropSyncV1.Connections {
				pairs.At(k).SetFromDevice(pair.FromDevice)
				pairs.At(k).SetToDevice(pair.ToDevice)
			}
		}
	}
	bytes, err := deployment.Message().Marshal()
	require.NoError(t, err)
	decoded, err := decodeGbuildDeploymentV1(bytes)
	require.NoError(t, err)
	return decoded
}

func TestSharedUnitsContract(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile("testdata/lpu-units-cases.json")
	require.NoError(t, err)
	var fixture struct {
		Version int
		Cases   []unitsContractCase
	}
	require.NoError(t, json.Unmarshal(data, &fixture))
	require.Equal(t, 1, fixture.Version)
	ruleMessages := map[string]string{
		"coverage": "omits LPU partition", "duplicate-partition": "repeats partition", "undeployed-partition": "not a deployed LPU partition",
		"empty-reservation": "partitionIds must not be empty", "empty-unit": "reservations must not be empty", "host-count": "reserves",
		"zero-size": ".size.", "hx-not-lp30": "LP20 reservations use size.nodes", "connector-count": "connectors for",
		"connector-family": "units use", "empty-device-pairs": "connections must not be empty", "device-pair-range": "outside the joined reservations",
		"repeated-device-pair": "repeats device pair", "mixed-architecture": "mixes XT and HX", "mixed-reservation-shape": "devicesPerNode",
		"packed-not-sole": "only reservation", "packed-with-cuda": "requires a bundle without CUDA", "zero-partition-geometry": "numChips must be",
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			t.Log("Decode the same serialized contract used by the runtime reader")
			deployment := encodeUnitsContractCase(t, tc)
			build, err := buildFromGbuildDeploymentV1("fixture", deployment)
			if tc.Consumers["operator"] == "legacy path" {
				require.Equal(t, "empty-units", tc.Rule)
				require.NoError(t, err)
				require.Empty(t, build.lpuUnits)
				return
			}
			if tc.Consumers["operator"] == "restriction: one devicesPerNode for the whole build (one Agent template)" {
				require.True(t, tc.Valid)
				require.Equal(t, "restriction: one devicesPerNode for the whole build (one Agent template)", tc.Consumers["operator"])
				require.ErrorContains(t, err, "all LPU partitions must use the same devicesPerNode")
				return
			}
			if tc.Valid {
				require.NoError(t, err)
				assertSharedDecodedUnits(t, tc, build)
				if tc.Consumers["operator"] != "" && tc.Name != "lp30-nodes-size-schema-valid" {
					projections, projectionErr := projectTestUnits(build, nil)
					require.Nil(t, projections)
					require.ErrorIs(t, projectionErr, ErrUnsupportedRuntime)
					switch tc.Consumers["operator"] {
					case "restriction: LP20 projects nodes n -> C(8n) only with devicesPerNode 8":
						require.ErrorContains(t, projectionErr, "LP20 nodes projection requires devicesPerNode=8")
					case "restriction: at most 512 pairs per connector (scheduler/CRD)":
						require.ErrorContains(t, projectionErr, "513 device pairs, limit is 512")
					case "restriction: an LPU-only model projects exactly one unit":
						require.ErrorContains(t, projectionErr, "exactly one lpuUnits entry")
					default:
						t.Fatalf("unmapped consumer restriction %q", tc.Name)
					}
				}
				if tc.Name == "lp30-nodes-size-schema-valid" {
					projections, err := projectTestUnits(build, nil)
					require.NoError(t, err)
					require.Equal(t, &[]int64{16, 1, 1, 1}, projections[0].RequestSpec(&MaterializationPlan{}, "agents").Partitions[0].Extent)
				}
				for _, agent := range tc.Agents {
					var selection *v1beta1.LPXLocalPartitions
					if len(agent.LocalRoots) > 0 {
						selection = &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeIDs, IDs: agent.LocalRoots}
					}
					projections, projectionErr := projectTestUnits(build, selection)
					if agent.Rule == "local-root-not-unit-root" {
						require.Error(t, projectionErr)
						continue
					}
					if tc.Consumers["operator"] == "restriction: an LPU-only model projects exactly one unit" {
						require.ErrorIs(t, projectionErr, ErrUnsupportedRuntime)
						require.ErrorContains(t, projectionErr, "exactly one lpuUnits entry")
						continue
					}
					require.NoError(t, projectionErr)
					var wantIDs []int64
					wantHosts := 0
					for _, unit := range tc.Deployment.LpuUnits {
						if slices.Contains(agent.LocalRoots, int64(unit.Reservations[0].PartitionIDs[0])) {
							continue
						}
						for _, reservation := range unit.Reservations {
							wantIDs = append(wantIDs, int64(reservation.PartitionIDs[0]))
							hosts := int(reservation.Size.Nodes)
							if hx := reservation.Size.HX; hx != nil {
								hosts = int(hx.TraysPerHemisphere) * int(hx.Hemispheres) * int(hx.Racks)
							}
							wantHosts += hosts
						}
					}
					projection := projections[0]
					require.Equal(t, wantHosts, projection.AgentReplicas())
					require.Equal(t, agent.Rule == "pod-index-out-of-range", agent.PodIndex >= projection.AgentReplicas())
					request := projection.RequestSpec(&MaterializationPlan{}, "agents")
					var gotIDs []int64
					for _, partition := range request.Partitions {
						gotIDs = append(gotIDs, partition.CompilerPartitionID)
					}
					require.Equal(t, wantIDs, gotIDs)
				}
				return
			}
			message, ok := ruleMessages[tc.Rule]
			require.True(t, ok, "unmapped shared rule %q", tc.Rule)
			require.ErrorContains(t, err, message)
		})
	}
}

// assertSharedDecodedUnits checks every placement value against the shared input,
// including encoded order and device pairs, before consumer restrictions apply.
func assertSharedDecodedUnits(t *testing.T, tc unitsContractCase, build *Build) {
	t.Helper()
	require.Len(t, build.lpuUnits, len(tc.Deployment.LpuUnits))
	widths := make(map[uint32]int)
	for _, p := range tc.Deployment.Partitions {
		if p.LPU != nil {
			widths[p.ID] = int(p.LPU.DevicesPerNode)
		}
	}
	for i, want := range tc.Deployment.LpuUnits {
		got := build.lpuUnits[i]
		require.Len(t, got.Reservations, len(want.Reservations))
		for j, r := range want.Reservations {
			var ids []int
			for _, id := range r.PartitionIDs {
				ids = append(ids, int(id))
			}
			require.Equal(t, ids, got.Reservations[j].PartitionIDs)
			if hx := r.Size.HX; hx != nil {
				require.Equal(t, int(hx.TraysPerHemisphere)*int(hx.Hemispheres)*int(hx.Racks), got.Reservations[j].Hosts)
				require.Equal(t, []int64{int64(widths[r.PartitionIDs[0]]), int64(hx.TraysPerHemisphere), int64(hx.Hemispheres), int64(hx.Racks)}, got.Reservations[j].HXExtent)
			} else {
				require.EqualValues(t, r.Size.Nodes, got.Reservations[j].Hosts)
				require.Nil(t, got.Reservations[j].HXExtent)
			}
		}
		require.Len(t, got.Connectors, len(want.Connectors))
		for j, c := range want.Connectors {
			if c.HxPropSyncV1 != nil {
				require.Equal(t, lpxv1alpha1.PropSyncConnectorKindHxPropSyncV1, got.Connectors[j].Kind)
				pairs := make([]lpxv1alpha1.HxLogicalConnection, 0, len(c.HxPropSyncV1.Connections))
				for _, p := range c.HxPropSyncV1.Connections {
					pairs = append(pairs, lpxv1alpha1.HxLogicalConnection{FromLogicalDevice: int64(p.FromDevice), ToLogicalDevice: int64(p.ToDevice)})
				}
				require.Equal(t, &pairs, got.Connectors[j].Connections)
			} else {
				require.Equal(t, lpxv1alpha1.PropSyncConnectorKindXt8888Gap, got.Connectors[j].Kind)
				require.Nil(t, got.Connectors[j].Connections)
				require.Equal(t, new(int64(0)), got.Connectors[j].MaxInterPartitionOffset)
			}
		}
	}
}
