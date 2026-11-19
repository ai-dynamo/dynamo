/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
)

// lpuArchitecture is the compiler's name for the physical LPU target.
type lpuArchitecture = deploymentcapnpv1.LpuArchitecture

const (
	lp20 = deploymentcapnpv1.LpuArchitecture_lp20
	lp30 = deploymentcapnpv1.LpuArchitecture_lp30
)

// Build is the registry's source-independent view of an LPU build.
//
// GBuild establishes placement validity. Consumers decode the contract and
// check the execution and scheduling shapes they support.
type Build struct {
	// contentID identifies the exact compiler contract used to acquire this build.
	contentID string
	// path is the absolute file or GCS reference of the build payload.
	path string

	// architecture is the physical LPU target
	architecture lpuArchitecture
	// devicesPerNode is the architecture's number of hardware devices per Kubernetes node
	devicesPerNode int

	// lpuUnits is the list of deployable LPU artifacts
	lpuUnits deploymentcapnpv1.LpuUnit_List

	// hasCUDAPartitions retains GPU requirements excluded from the LPU partition projection.
	hasCUDAPartitions bool
	// ioFPGACount is the number of I/O FPGA endpoints described by the build.
	ioFPGACount int32
	// ioFanoutFactor is the number of clients assigned to each I/O FPGA transaction.
	ioFanoutFactor int32

	// checkpoint is the validated source checkpoint that the deployment contract records, or nil.
	checkpoint *Checkpoint
}
