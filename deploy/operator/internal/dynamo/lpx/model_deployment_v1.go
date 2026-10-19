/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"
	"slices"

	"capnproto.org/go/capnp/v3"
	commoncapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/common/v1"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
)

const gbuildDeploymentV1CapnpFile = "deployment.v1.capnp.bin"

// buildContractFormat selects a supported compiler contract and runtime entry file.
type buildContractFormat uint8

const (
	buildContractUnknown buildContractFormat = iota
	buildContractManifestV2
	buildContractDeploymentV1
)

// selectBuildContract selects exactly one entry point from a normalized inventory.
func selectBuildContract(inventory []string) (buildContractFormat, error) {
	legacy := slices.Contains(inventory, gbuildManifestV2CapnpFile)
	split := slices.Contains(inventory, gbuildDeploymentV1CapnpFile)
	switch {
	case legacy && split:
		return buildContractUnknown, fmt.Errorf("ambiguous build contracts: both %s and %s", gbuildManifestV2CapnpFile, gbuildDeploymentV1CapnpFile)
	case legacy:
		return buildContractManifestV2, nil
	case split:
		return buildContractDeploymentV1, nil
	default:
		return buildContractUnknown, fmt.Errorf("missing build contract: require %s or %s", gbuildManifestV2CapnpFile, gbuildDeploymentV1CapnpFile)
	}
}

// filename returns the selected runtime entry point; unknown formats are invalid.
func (format buildContractFormat) filename() (string, error) {
	switch format {
	case buildContractManifestV2:
		return gbuildManifestV2CapnpFile, nil
	case buildContractDeploymentV1:
		return gbuildDeploymentV1CapnpFile, nil
	default:
		return "", fmt.Errorf("unsupported build contract format %d", format)
	}
}

// decodeGbuildDeploymentV1 reads the scheduler contract without execution/report schemas.
func decodeGbuildDeploymentV1(data []byte) (deploymentcapnpv1.Deployment, error) {
	msg, err := capnp.Unmarshal(data)
	if err != nil {
		return deploymentcapnpv1.Deployment{}, fmt.Errorf("parsing %s: %w", gbuildDeploymentV1CapnpFile, err)
	}
	deployment, err := deploymentcapnpv1.ReadRootDeployment(msg)
	if err != nil {
		return deploymentcapnpv1.Deployment{}, fmt.Errorf("reading %s root: %w", gbuildDeploymentV1CapnpFile, err)
	}
	return deployment, nil
}

// buildFromGbuildDeploymentV1 validates compiler input using the acquired canonical build reference.
func buildFromGbuildDeploymentV1(buildRef string, manifest deploymentcapnpv1.Deployment) (*Build, error) {
	if manifest.ContractRevision() != deploymentcapnpv1.CurrentContractRevision {
		return nil, fmt.Errorf(
			"%s contractRevision = %d, want %d",
			gbuildDeploymentV1CapnpFile,
			manifest.ContractRevision(),
			deploymentcapnpv1.CurrentContractRevision,
		)
	}
	if !manifest.HasConfig() {
		return nil, fmt.Errorf("%s is missing config", gbuildDeploymentV1CapnpFile)
	}
	deployment, err := manifest.Config()
	if err != nil {
		return nil, fmt.Errorf("reading %s config: %w", gbuildDeploymentV1CapnpFile, err)
	}
	if !deployment.HasProgram() {
		return nil, fmt.Errorf("%s config.program is missing", gbuildDeploymentV1CapnpFile)
	}
	program, err := deployment.Program()
	if err != nil {
		return nil, fmt.Errorf("reading %s config.program: %w", gbuildDeploymentV1CapnpFile, err)
	}
	chains, err := selectedPropSyncChainsFromDeploymentV1(deployment)
	if err != nil {
		return nil, err
	}
	ioFPGACount, ioFanoutFactor, err := runtimeIOFromDeploymentV1(deployment)
	if err != nil {
		return nil, err
	}

	build := &Build{
		path:                      buildRef,
		selectedPropSyncChains:    chains,
		standaloneTokenEmbeddings: program.StandaloneTokenEmbeddings(),
		supportsCPUEmbeddings:     program.SupportsCpuEmbeddings(),
		ioFPGACount:               ioFPGACount,
		ioFanoutFactor:            ioFanoutFactor,
	}

	// Complete the normalized build with scheduler-facing LPU artifacts.
	if err := addLPUArtifactsFromDeploymentV1(manifest, build); err != nil {
		return nil, err
	}
	return build, nil
}

func runtimeIOFromDeploymentV1(deployment deploymentcapnpv1.DeploymentInfo) (int32, int32, error) {
	if !deployment.HasRuntimeIo() {
		return 0, 0, fmt.Errorf("%s config.runtimeIo is missing", gbuildDeploymentV1CapnpFile)
	}
	runtimeIO, err := deployment.RuntimeIo()
	if err != nil {
		return 0, 0, fmt.Errorf("reading %s config.runtimeIo: %w", gbuildDeploymentV1CapnpFile, err)
	}
	count := runtimeIO.IoFpgaCount()
	if count == 0 || count > math.MaxInt32 {
		return 0, 0, fmt.Errorf("%s config.runtimeIo.ioFpgaCount must be in [1, %d], got %d", gbuildDeploymentV1CapnpFile, math.MaxInt32, count)
	}

	// Fanout is a separate positive client count for each physical endpoint.
	fanoutFactor := runtimeIO.FanoutFactor()
	if fanoutFactor == 0 || fanoutFactor > math.MaxInt32 {
		return 0, 0, fmt.Errorf("%s config.runtimeIo.fanoutFactor must be in [1, %d], got %d", gbuildDeploymentV1CapnpFile, math.MaxInt32, fanoutFactor)
	}
	return int32(count), int32(fanoutFactor), nil
}

func selectedPropSyncChainsFromDeploymentV1(deployment deploymentcapnpv1.DeploymentInfo) ([][]int, error) {
	if !deployment.HasSelectedPropSyncChains() {
		return nil, nil
	}
	rawChains, err := deployment.SelectedPropSyncChains()
	if err != nil {
		return nil, fmt.Errorf("reading %s config.selectedPropSyncChains: %w", gbuildDeploymentV1CapnpFile, err)
	}
	chains := make([][]int, 0, rawChains.Len())
	for chainIndex := 0; chainIndex < rawChains.Len(); chainIndex++ {
		rawChain := rawChains.At(chainIndex)
		if !rawChain.HasPartitionIds() {
			return nil, fmt.Errorf("%s config.selectedPropSyncChains[%d].partitionIds is missing", gbuildDeploymentV1CapnpFile, chainIndex)
		}
		rawIDs, err := rawChain.PartitionIds()
		if err != nil {
			return nil, fmt.Errorf("reading %s config.selectedPropSyncChains[%d].partitionIds: %w", gbuildDeploymentV1CapnpFile, chainIndex, err)
		}

		chainPath := fmt.Sprintf("%s config.selectedPropSyncChains[%d]", gbuildDeploymentV1CapnpFile, chainIndex)
		chain, err := decodePropSyncChain(rawIDs, chainPath)
		if err != nil {
			return nil, err
		}
		chains = append(chains, chain)
	}
	return chains, nil
}

func addLPUArtifactsFromDeploymentV1(
	artifacts deploymentcapnpv1.Deployment,
	build *Build,
) error {
	rawPartitions, err := artifacts.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s partitions: %w", gbuildDeploymentV1CapnpFile, err)
	}
	partitions := make([]buildPartition, 0, rawPartitions.Len())
	for index := 0; index < rawPartitions.Len(); index++ {
		partition, err := buildPartitionFromDeploymentV1(rawPartitions.At(index))
		if err != nil {
			return err
		}
		// Project LPU artifacts while retaining whether the build also requires CUDA.
		if partition.partPath != "" {
			partitions = append(partitions, partition)
		} else {
			ref, err := rawPartitions.At(index).Partition()
			if err != nil {
				return fmt.Errorf("reading %s artifact partition ref: %w", gbuildDeploymentV1CapnpFile, err)
			}
			build.hasCUDAPartitions = build.hasCUDAPartitions || ref.DeviceType() == commoncapnpv1.DeviceType_cuda
		}
	}
	if len(partitions) == 0 {
		return fmt.Errorf("%s artifacts contain no LPU partitions", gbuildDeploymentV1CapnpFile)
	}
	partialSelection := artifacts.HasPartSelect()
	family, err := classifyManifestPartitions(gbuildDeploymentV1CapnpFile, partitions, partialSelection)
	if err != nil {
		return err
	}

	// Publish the scheduler-facing artifact projection.
	build.partitions = partitions
	build.family = family

	return nil
}

func buildPartitionFromDeploymentV1(raw deploymentcapnpv1.PartitionDeployment) (buildPartition, error) {
	if !raw.HasPartition() {
		return buildPartition{}, fmt.Errorf("%s artifact partition is missing partition ref", gbuildDeploymentV1CapnpFile)
	}
	ref, err := raw.Partition()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading %s artifact partition ref: %w", gbuildDeploymentV1CapnpFile, err)
	}
	switch ref.DeviceType() {
	case commoncapnpv1.DeviceType_cuda, commoncapnpv1.DeviceType_cpu:
		return buildPartition{}, nil
	case commoncapnpv1.DeviceType_lpu:
		if raw.Detail().Which() != deploymentcapnpv1.PartitionDeployment_detail_Which_lpu || !raw.Detail().HasLpu() {
			return buildPartition{}, fmt.Errorf("LPU partition %d has mismatched detail", ref.PartitionId())
		}
	default:
		return buildPartition{}, fmt.Errorf("partition %d has unsupported deviceType %d", ref.PartitionId(), ref.DeviceType())
	}
	detail, err := raw.Detail().Lpu()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading %s LPU partition %d detail: %w", gbuildDeploymentV1CapnpFile, ref.PartitionId(), err)
	}
	switch detail.Architecture() {
	case deploymentcapnpv1.LpuArchitecture_lp20:
		return buildLPUArtifact(gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail, BuildFamilyXT)
	case deploymentcapnpv1.LpuArchitecture_lp30:
		return buildLPUArtifact(gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail, BuildFamilyHX)
	default:
		return buildPartition{}, fmt.Errorf("%s LPU partition %d unsupported chip architecture %d", gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail.Architecture())
	}
}
