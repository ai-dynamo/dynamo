/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"

	"capnproto.org/go/capnp/v3"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
)

const (
	gbuildManifestV2CapnpFile = "manifest.v2.capnp.bin"
	chipArchXT                = "polaris"
	chipArchHX                = "polarisB0"
)

func decodeGbuildManifestV2(data []byte) (manifestcapnpv2.Manifest, error) {
	msg, err := capnp.Unmarshal(data)
	if err != nil {
		return manifestcapnpv2.Manifest{}, fmt.Errorf("parsing %s: %w", gbuildManifestV2CapnpFile, err)
	}
	manifest, err := manifestcapnpv2.ReadRootManifest(msg)
	if err != nil {
		return manifestcapnpv2.Manifest{}, fmt.Errorf("reading %s root: %w", gbuildManifestV2CapnpFile, err)
	}
	return manifest, nil
}

// buildFromGbuildManifestV2 validates compiler input using the acquired canonical build reference.
func buildFromGbuildManifestV2(buildRef string, manifest manifestcapnpv2.Manifest) (*Build, error) {
	if manifest.ContractRevision() != manifestcapnpv2.CurrentContractRevision {
		return nil, fmt.Errorf(
			"%s contractRevision = %d, want %d",
			gbuildManifestV2CapnpFile,
			manifest.ContractRevision(),
			manifestcapnpv2.CurrentContractRevision,
		)
	}
	if !manifest.HasDeployment() {
		return nil, fmt.Errorf("%s is missing deployment", gbuildManifestV2CapnpFile)
	}
	deployment, err := manifest.Deployment()
	if err != nil {
		return nil, fmt.Errorf("reading %s deployment: %w", gbuildManifestV2CapnpFile, err)
	}
	if !deployment.HasProgram() {
		return nil, fmt.Errorf("%s deployment.program is missing", gbuildManifestV2CapnpFile)
	}
	program, err := deployment.Program()
	if err != nil {
		return nil, fmt.Errorf("reading %s deployment.program: %w", gbuildManifestV2CapnpFile, err)
	}
	chains, err := selectedPropSyncChainsFromManifestV2(deployment)
	if err != nil {
		return nil, err
	}
	ioFPGACount, ioFanoutFactor, err := runtimeIOFromManifestV2(deployment)
	if err != nil {
		return nil, err
	}

	if !manifest.HasArtifacts() {
		return nil, fmt.Errorf("%s is missing artifacts", gbuildManifestV2CapnpFile)
	}
	artifacts, err := manifest.Artifacts()
	if err != nil {
		return nil, fmt.Errorf("reading %s artifacts: %w", gbuildManifestV2CapnpFile, err)
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
	if err := addLPUArtifactsFromManifestV2(artifacts, build); err != nil {
		return nil, err
	}
	return build, nil
}

func runtimeIOFromManifestV2(deployment manifestcapnpv2.DeploymentInfo) (int32, int32, error) {
	if !deployment.HasRuntimeIo() {
		return 0, 0, fmt.Errorf("%s deployment.runtimeIo is missing", gbuildManifestV2CapnpFile)
	}
	runtimeIO, err := deployment.RuntimeIo()
	if err != nil {
		return 0, 0, fmt.Errorf("reading %s deployment.runtimeIo: %w", gbuildManifestV2CapnpFile, err)
	}
	count := runtimeIO.IoFpgaCount()
	if count == 0 || count > math.MaxInt32 {
		return 0, 0, fmt.Errorf("%s deployment.runtimeIo.ioFpgaCount must be in [1, %d], got %d", gbuildManifestV2CapnpFile, math.MaxInt32, count)
	}

	// Fanout is a separate positive client count for each physical endpoint.
	fanoutFactor := runtimeIO.FanoutFactor()
	if fanoutFactor == 0 || fanoutFactor > math.MaxInt32 {
		return 0, 0, fmt.Errorf("%s deployment.runtimeIo.fanoutFactor must be in [1, %d], got %d", gbuildManifestV2CapnpFile, math.MaxInt32, fanoutFactor)
	}
	return int32(count), int32(fanoutFactor), nil
}

func selectedPropSyncChainsFromManifestV2(deployment manifestcapnpv2.DeploymentInfo) ([][]int, error) {
	if !deployment.HasSelectedPropSyncChains() {
		return nil, nil
	}
	rawChains, err := deployment.SelectedPropSyncChains()
	if err != nil {
		return nil, fmt.Errorf("reading %s deployment.selectedPropSyncChains: %w", gbuildManifestV2CapnpFile, err)
	}
	chains := make([][]int, 0, rawChains.Len())
	for chainIndex := 0; chainIndex < rawChains.Len(); chainIndex++ {
		rawChain := rawChains.At(chainIndex)
		if !rawChain.HasPartitionIds() {
			return nil, fmt.Errorf("%s deployment.selectedPropSyncChains[%d].partitionIds is missing", gbuildManifestV2CapnpFile, chainIndex)
		}
		rawIDs, err := rawChain.PartitionIds()
		if err != nil {
			return nil, fmt.Errorf("reading %s deployment.selectedPropSyncChains[%d].partitionIds: %w", gbuildManifestV2CapnpFile, chainIndex, err)
		}

		chainPath := fmt.Sprintf("%s deployment.selectedPropSyncChains[%d]", gbuildManifestV2CapnpFile, chainIndex)
		chain, err := decodePropSyncChain(rawIDs, chainPath)
		if err != nil {
			return nil, err
		}
		chains = append(chains, chain)
	}
	return chains, nil
}

func addLPUArtifactsFromManifestV2(
	artifacts manifestcapnpv2.ArtifactInfo,
	build *Build,
) error {
	rawPartitions, err := artifacts.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s artifacts.partitions: %w", gbuildManifestV2CapnpFile, err)
	}
	partitions := make([]buildPartition, 0, rawPartitions.Len())
	for index := 0; index < rawPartitions.Len(); index++ {
		partition, err := buildPartitionFromManifestV2(rawPartitions.At(index))
		if err != nil {
			return err
		}
		// Project LPU artifacts while retaining whether the build also requires CUDA.
		if partition.partPath != "" {
			partitions = append(partitions, partition)
		} else {
			ref, err := rawPartitions.At(index).Partition()
			if err != nil {
				return fmt.Errorf("reading %s artifact partition ref: %w", gbuildManifestV2CapnpFile, err)
			}
			build.hasCUDAPartitions = build.hasCUDAPartitions || ref.DeviceType() == manifestcapnpv2.DeviceType_cuda
		}
	}
	if len(partitions) == 0 {
		return fmt.Errorf("%s artifacts contain no LPU partitions", gbuildManifestV2CapnpFile)
	}
	partialSelection := artifacts.HasPartSelect()
	family, err := classifyManifestPartitions(gbuildManifestV2CapnpFile, partitions, partialSelection)
	if err != nil {
		return err
	}

	// Publish the scheduler-facing artifact projection.
	build.partitions = partitions
	build.topology = family

	return nil
}

func buildPartitionFromManifestV2(raw manifestcapnpv2.PartitionInfo) (buildPartition, error) {
	if !raw.HasPartition() {
		return buildPartition{}, fmt.Errorf("%s artifact partition is missing partition ref", gbuildManifestV2CapnpFile)
	}
	ref, err := raw.Partition()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading %s artifact partition ref: %w", gbuildManifestV2CapnpFile, err)
	}
	if ref.DeviceType() != manifestcapnpv2.DeviceType_lpu {
		return buildPartition{}, nil
	}
	if raw.Detail().Which() != manifestcapnpv2.PartitionInfo_detail_Which_lpu || !raw.Detail().HasLpu() {
		return buildPartition{}, fmt.Errorf("%s artifact partition %d has deviceType lpu without LPU detail", gbuildManifestV2CapnpFile, ref.PartitionId())
	}
	detail, err := raw.Detail().Lpu()
	if err != nil {
		return buildPartition{}, fmt.Errorf("reading %s LPU partition %d detail: %w", gbuildManifestV2CapnpFile, ref.PartitionId(), err)
	}
	subject := fmt.Sprintf("%s LPU partition %d", gbuildManifestV2CapnpFile, ref.PartitionId())
	programs, err := detail.Reserved4()
	if err != nil || programs.List().Len() == 0 {
		return buildPartition{}, fmt.Errorf("%s programs are missing or invalid", subject)
	}

	chips, err := programs.List().Struct(0).Ptr(1)
	if err != nil || chips.List().Len() == 0 {
		return buildPartition{}, fmt.Errorf("%s program chips are missing or invalid", subject)
	}
	architecture, err := chips.List().Struct(0).Ptr(1)
	if err != nil {
		return buildPartition{}, fmt.Errorf("%s reading chip architecture: %w", subject, err)
	}

	switch architecture.Text() {
	case chipArchHX:
		return buildLPUArtifact(gbuildManifestV2CapnpFile, ref.PartitionId(), detail, hxBuildTopology)
	case chipArchXT:
		return buildLPUArtifact(gbuildManifestV2CapnpFile, ref.PartitionId(), detail, xtBuildTopology)
	default:
		return buildPartition{}, fmt.Errorf("%s unsupported chip architecture %q", subject, architecture.Text())
	}
}
