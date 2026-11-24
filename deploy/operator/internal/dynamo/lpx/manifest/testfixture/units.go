// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package testfixture supplies compiler contract builders for operator tests.
package testfixture

import (
	"testing"

	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/stretchr/testify/require"
)

type Unit struct {
	Reservations []Reservation
	Connectors   []Connector
}

type Reservation struct {
	IDs   []uint32
	Nodes uint32
	// HX is trays per hemisphere, hemispheres, racks; nil selects Nodes.
	HX []uint8
}

type Connector struct {
	HX    bool
	Pairs [][2]uint32
}

// WriteUnits encodes the authored reservations and connectors without inferring placement.
func WriteUnits(t *testing.T, config deploymentcapnpv1.DeploymentInfo, specs []Unit) {
	t.Helper()
	units, err := config.NewLpuUnits(int32(len(specs)))
	require.NoError(t, err)
	for unitIndex, unit := range specs {
		reservations, err := units.At(unitIndex).NewReservations(int32(len(unit.Reservations)))
		require.NoError(t, err)
		for index, reservation := range unit.Reservations {
			ids, err := reservations.At(index).NewPartitionIds(int32(len(reservation.IDs)))
			require.NoError(t, err)
			for idIndex, id := range reservation.IDs {
				ids.Set(idIndex, id)
			}
			size := reservations.At(index).Size()
			if reservation.HX == nil {
				size.SetNodes(reservation.Nodes)
				continue
			}
			extent, err := size.NewHx()
			require.NoError(t, err)
			extent.SetTraysPerHemisphere(reservation.HX[0])
			extent.SetHemispheres(reservation.HX[1])
			extent.SetRacks(reservation.HX[2])
		}
		connectors, err := units.At(unitIndex).NewConnectors(int32(len(unit.Connectors)))
		require.NoError(t, err)
		for index, connector := range unit.Connectors {
			if !connector.HX {
				connectors.At(index).SetXt8888Gap()
				continue
			}
			hx, err := connectors.At(index).NewHxPropSyncV1()
			require.NoError(t, err)
			pairs, err := hx.NewConnections(int32(len(connector.Pairs)))
			require.NoError(t, err)
			for pairIndex, pair := range connector.Pairs {
				pairs.At(pairIndex).SetFromDevice(pair[0])
				pairs.At(pairIndex).SetToDevice(pair[1])
			}
		}
	}
}
