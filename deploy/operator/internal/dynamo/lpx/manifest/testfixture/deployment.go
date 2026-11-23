// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package testfixture

import (
	"testing"

	"capnproto.org/go/capnp/v3"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deployment "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/stretchr/testify/require"
)

type Partition struct {
	ID             uint32
	DeviceType     common.DeviceType
	Architecture   deployment.LpuArchitecture
	DevicesPerNode uint32
}

// NewDeployment encodes the units, partition facts and runtime I/O used by the operator.
func NewDeployment(t *testing.T, partitions []Partition, units []Unit, ioFPGACount, fanoutFactor uint32) deployment.Deployment {
	t.Helper()
	_, segment := capnp.NewSingleSegmentMessage(nil)
	manifest, err := deployment.NewRootDeployment(segment)
	require.NoError(t, err)
	manifest.SetContractRevision(deployment.CurrentContractRevision)
	config, err := manifest.NewConfig()
	require.NoError(t, err)
	_, err = config.NewProgram()
	require.NoError(t, err)
	runtimeIO, err := config.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetProtocol(deployment.RuntimeIoProtocol_fpgaRoce)
	runtimeIO.SetIoFpgaCount(ioFPGACount)
	runtimeIO.SetFanoutFactor(fanoutFactor)
	if units != nil {
		WriteUnits(t, config, units)
	}
	encoded, err := manifest.NewPartitions(int32(len(partitions)))
	require.NoError(t, err)
	for index, spec := range partitions {
		ref, err := encoded.At(index).NewPartition()
		require.NoError(t, err)
		ref.SetPartitionId(spec.ID)
		ref.SetDeviceType(spec.DeviceType)
		switch spec.DeviceType {
		case common.DeviceType_lpu:
			detail, err := encoded.At(index).Detail().NewLpu()
			require.NoError(t, err)
			detail.SetArchitecture(spec.Architecture)
			detail.SetDevicesPerNode(spec.DevicesPerNode)
		case common.DeviceType_cuda:
			_, err = encoded.At(index).Detail().NewCuda()
			require.NoError(t, err)
		case common.DeviceType_cpu:
			_, err = encoded.At(index).Detail().NewCpu()
			require.NoError(t, err)
		default:
			t.Fatalf("unsupported test device type %q", spec.DeviceType)
		}
	}
	return manifest
}
