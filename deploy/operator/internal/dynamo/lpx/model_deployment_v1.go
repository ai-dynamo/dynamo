/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"fmt"
	"math"
	"slices"
	"strings"

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
	if len(data) > maxBuildSnapshotMetadataBytes {
		return deploymentcapnpv1.Deployment{}, fmt.Errorf("%s: %w", gbuildDeploymentV1CapnpFile, errBuildFileTooLarge)
	}
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

// validateExecutionReference checks association metadata without reading runtime bytes.
func validateExecutionReference(deployment deploymentcapnpv1.Deployment, inventory []string) error {
	if !deployment.HasExecution() {
		return fmt.Errorf("%s is missing execution", gbuildDeploymentV1CapnpFile)
	}
	execution, err := deployment.Execution()
	if err != nil {
		return fmt.Errorf("reading execution: %w", err)
	}
	rawPath, err := execution.Path()
	if err != nil {
		return fmt.Errorf("reading execution.path: %w", err)
	}
	path, err := cleanManifestRelativeBuildPath("execution.path", rawPath)
	if err != nil {
		return err
	}
	if path != rawPath || strings.Contains(rawPath, "\\") || strings.Contains(rawPath, ":") {
		return fmt.Errorf("execution.path %q must be a canonical bundle-relative path", rawPath)
	}
	if path == gbuildDeploymentV1CapnpFile || path == gbuildManifestV2CapnpFile || path == "build-report.v1.capnp.bin" {
		return fmt.Errorf("execution.path %q must name the execution contract", path)
	}
	digest, err := execution.Sha256()
	if err != nil {
		return fmt.Errorf("reading execution.sha256: %w", err)
	}
	if len(digest) != sha256.Size {
		return fmt.Errorf("execution.sha256 must contain %d bytes, got %d", sha256.Size, len(digest))
	}
	if !slices.Contains(inventory, path) {
		return fmt.Errorf("execution.path %q is missing from build inventory", path)
	}
	return nil
}

// buildFromGbuildDeploymentV1 validates compiler input using the acquired canonical build reference.
func buildFromGbuildDeploymentV1(buildRef string, manifest deploymentcapnpv1.Deployment, inventory []string) (*Build, error) {
	if manifest.ContractRevision() != deploymentcapnpv1.CurrentContractRevision {
		return nil, fmt.Errorf(
			"%s contractRevision = %d, want %d",
			gbuildDeploymentV1CapnpFile,
			manifest.ContractRevision(),
			deploymentcapnpv1.CurrentContractRevision,
		)
	}
	// A split deployment must identify its build before scheduling.
	identity, err := manifest.Identity()
	if err != nil {
		return nil, fmt.Errorf("reading %s identity: %w", gbuildDeploymentV1CapnpFile, err)
	}
	buildID, err := identity.BuildId()
	if err != nil {
		return nil, fmt.Errorf("reading %s identity.buildId: %w", gbuildDeploymentV1CapnpFile, err)
	}
	if strings.TrimSpace(buildID) == "" {
		return nil, fmt.Errorf("%s identity.buildId is required", gbuildDeploymentV1CapnpFile)
	}
	if !manifest.HasConfig() {
		return nil, fmt.Errorf("%s is missing config", gbuildDeploymentV1CapnpFile)
	}
	deployment, err := manifest.Config()
	if err != nil {
		return nil, fmt.Errorf("reading %s config: %w", gbuildDeploymentV1CapnpFile, err)
	}
	compilationMode, err := buildCompilationMode(gbuildDeploymentV1CapnpFile, deployment.CompilationMode().String())
	if err != nil {
		return nil, err
	}
	if !deployment.HasProgram() {
		return nil, fmt.Errorf("%s config.program is missing", gbuildDeploymentV1CapnpFile)
	}
	program, err := deployment.Program()
	if err != nil {
		return nil, fmt.Errorf("reading %s config.program: %w", gbuildDeploymentV1CapnpFile, err)
	}
	batchSize, err := positiveManifestUInt32ToInt(
		fmt.Sprintf("%s config.program.batchSize", gbuildDeploymentV1CapnpFile),
		program.BatchSize(),
	)
	if err != nil {
		return nil, err
	}
	chains, err := selectedPropSyncChainsFromDeploymentV1(deployment)
	if err != nil {
		return nil, err
	}
	ioFPGACount, ioFanoutFactor, err := runtimeIOFromDeploymentV1(deployment)
	if err != nil {
		return nil, err
	}
	if err := validateManifestBatchSize(gbuildDeploymentV1CapnpFile, batchSize, ioFPGACount, ioFanoutFactor); err != nil {
		return nil, err
	}
	if err := validateExecutionReference(manifest, inventory); err != nil {
		return nil, err
	}
	runtimeTokenEmbeddingsPath, err := runtimeTokenEmbeddingsPathFromDeploymentV1(manifest)
	if err != nil {
		return nil, err
	}
	// Validate runtime embedding assets before projecting scheduler-facing artifacts.
	if runtimeTokenEmbeddingsPath != "" && !program.SupportsCpuEmbeddings() {
		return nil, fmt.Errorf("%s runtimeAssets.tokenEmbeddingsPath requires supportsCpuEmbeddings=true", gbuildDeploymentV1CapnpFile)
	}
	if program.SupportsCpuEmbeddings() && program.StandaloneTokenEmbeddings() && runtimeTokenEmbeddingsPath == "" {
		return nil, fmt.Errorf("%s runtimeAssets.tokenEmbeddingsPath is required when standaloneTokenEmbeddings=true", gbuildDeploymentV1CapnpFile)
	}

	build := &Build{
		Path:                      buildRef,
		CompilationMode:           compilationMode,
		SelectedPropSyncChains:    chains,
		StandaloneTokenEmbeddings: program.StandaloneTokenEmbeddings(),
		SupportsCPUEmbeddings:     program.SupportsCpuEmbeddings(),
		IOFPGACount:               ioFPGACount,
		IOFanoutFactor:            ioFanoutFactor,
	}

	// Complete the normalized build with scheduler-facing LPU artifacts.
	if err := addLPUArtifactsFromDeploymentV1(manifest, deployment, build); err != nil {
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
	switch runtimeIO.Protocol() {
	case deploymentcapnpv1.RuntimeIoProtocol_host:
		if count != 1 {
			return 0, 0, fmt.Errorf("%s host runtime I/O requires ioFpgaCount 1, got %d", gbuildDeploymentV1CapnpFile, count)
		}
	case deploymentcapnpv1.RuntimeIoProtocol_fpgaRoce:
	default:
		return 0, 0, fmt.Errorf("%s config.runtimeIo.protocol %d is not supported", gbuildDeploymentV1CapnpFile, runtimeIO.Protocol())
	}
	if mode := runtimeIO.FpgaMode(); mode > deploymentcapnpv1.FpgaIoMode_dibDeb {
		return 0, 0, fmt.Errorf("%s config.runtimeIo.fpgaMode %d is not supported", gbuildDeploymentV1CapnpFile, mode)
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
	deployment deploymentcapnpv1.DeploymentInfo,
	build *Build,
) error {
	rawPartitions, err := artifacts.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s partitions: %w", gbuildDeploymentV1CapnpFile, err)
	}
	partitions := make([]BuildPartition, 0, rawPartitions.Len())
	hxDoubleNodeCount := true
	for index := 0; index < rawPartitions.Len(); index++ {
		partition, compatible, err := buildPartitionFromDeploymentV1(rawPartitions.At(index))
		if err != nil {
			return err
		}
		// Nonempty paths mark LPU artifacts; only metadata-less HX partitions permit the historical doubled node count.
		if partition.PartPath != "" {
			partitions = append(partitions, partition)
			hxDoubleNodeCount = hxDoubleNodeCount && compatible
		}
	}
	// An LPU-only build cannot silently discard packaged CUDA or CPU partitions.
	if build.CompilationMode == BuildCompilationModeLPUOnly && len(partitions) != rawPartitions.Len() {
		return fmt.Errorf(
			"%s config.compilationMode lpuOnly requires every artifact partition to use deviceType lpu",
			gbuildDeploymentV1CapnpFile,
		)
	}
	if len(partitions) == 0 {
		return fmt.Errorf("%s artifacts contain no LPU partitions", gbuildDeploymentV1CapnpFile)
	}
	partialSelection := artifacts.HasPartSelect()
	family, packagedNodes, partitionZeroNodes, err := classifyManifestPartitions(gbuildDeploymentV1CapnpFile, partitions, partialSelection)
	if err == nil && family == BuildFamilyXT {
		err = validateDeploymentV1PartSelect(artifacts, partitions)
	}
	if err != nil {
		return err
	}

	// Publish the artifact projection before validating the complete deployment geometry.
	build.Partitions = partitions
	build.Family = family

	want, err := positiveManifestUInt32ToInt(
		fmt.Sprintf("%s config.numLpuNodes", gbuildDeploymentV1CapnpFile),
		deployment.NumLpuNodes(),
	)
	if err != nil {
		return err
	}
	return validateManifestPartitionNodeCount(gbuildDeploymentV1CapnpFile, want, build, partialSelection, hxDoubleNodeCount, packagedNodes, partitionZeroNodes)
}

func buildPartitionFromDeploymentV1(raw deploymentcapnpv1.PartitionDeployment) (BuildPartition, bool, error) {
	if !raw.HasPartition() {
		return BuildPartition{}, false, fmt.Errorf("%s artifact partition is missing partition ref", gbuildDeploymentV1CapnpFile)
	}
	ref, err := raw.Partition()
	if err != nil {
		return BuildPartition{}, false, fmt.Errorf("reading %s artifact partition ref: %w", gbuildDeploymentV1CapnpFile, err)
	}
	// Require the device kind and payload discriminator to agree even for non-LPU partitions.
	switch ref.DeviceType() {
	case commoncapnpv1.DeviceType_cuda:
		if raw.Detail().Which() != deploymentcapnpv1.PartitionDeployment_detail_Which_cuda || !raw.Detail().HasCuda() {
			return BuildPartition{}, false, fmt.Errorf("CUDA partition %d has mismatched detail", ref.PartitionId())
		}
		return BuildPartition{}, false, nil
	case commoncapnpv1.DeviceType_cpu:
		if raw.Detail().Which() != deploymentcapnpv1.PartitionDeployment_detail_Which_cpu || !raw.Detail().HasCpu() {
			return BuildPartition{}, false, fmt.Errorf("CPU partition %d has mismatched detail", ref.PartitionId())
		}
		return BuildPartition{}, false, nil
	case commoncapnpv1.DeviceType_lpu:
		if raw.Detail().Which() != deploymentcapnpv1.PartitionDeployment_detail_Which_lpu || !raw.Detail().HasLpu() {
			return BuildPartition{}, false, fmt.Errorf("LPU partition %d has mismatched detail", ref.PartitionId())
		}
	default:
		return BuildPartition{}, false, fmt.Errorf("partition %d has unsupported deviceType %d", ref.PartitionId(), ref.DeviceType())
	}
	detail, err := raw.Detail().Lpu()
	if err != nil {
		return BuildPartition{}, false, fmt.Errorf("reading %s LPU partition %d detail: %w", gbuildDeploymentV1CapnpFile, ref.PartitionId(), err)
	}
	switch detail.Architecture() {
	case deploymentcapnpv1.LpuArchitecture_lp20:
		return buildLPUArtifact(gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail, BuildFamilyXT)
	case deploymentcapnpv1.LpuArchitecture_lp30:
		return buildLPUArtifact(gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail, BuildFamilyHX)
	default:
		return BuildPartition{}, false, fmt.Errorf("%s LPU partition %d unsupported chip architecture %s", gbuildDeploymentV1CapnpFile, ref.PartitionId(), detail.Architecture())
	}
}

func validateDeploymentV1PartSelect(artifacts deploymentcapnpv1.Deployment, partitions []BuildPartition) error {
	if !artifacts.HasPartSelect() {
		return nil
	}
	partSelect, err := artifacts.PartSelect()
	if err != nil {
		return fmt.Errorf("reading %s partSelect: %w", gbuildDeploymentV1CapnpFile, err)
	}
	selected, err := partSelect.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s partSelect.partitions: %w", gbuildDeploymentV1CapnpFile, err)
	}
	if selected.Len() == 0 {
		return fmt.Errorf("%s partSelect.partitions is empty", gbuildDeploymentV1CapnpFile)
	}
	selectedIDs := make([]int, 0, selected.Len())
	for index := 0; index < selected.Len(); index++ {
		ref := selected.At(index)
		if ref.DeviceType() != commoncapnpv1.DeviceType_lpu {
			continue
		}
		selectedIDs = append(selectedIDs, int(ref.PartitionId()))
	}
	return validateSelectedLPUPartitions(gbuildDeploymentV1CapnpFile, "partSelect", selectedIDs, partitions)
}

func runtimeTokenEmbeddingsPathFromDeploymentV1(artifacts deploymentcapnpv1.Deployment) (string, error) {
	if !artifacts.HasRuntimeAssets() {
		return "", nil
	}
	runtimeAssets, err := artifacts.RuntimeAssets()
	if err != nil {
		return "", fmt.Errorf("reading %s runtimeAssets: %w", gbuildDeploymentV1CapnpFile, err)
	}
	if !runtimeAssets.HasTokenEmbeddingsPath() {
		return "", nil
	}
	rawPath, err := runtimeAssets.TokenEmbeddingsPath()
	if err != nil {
		return "", fmt.Errorf("reading %s runtimeAssets.tokenEmbeddingsPath: %w", gbuildDeploymentV1CapnpFile, err)
	}
	return cleanManifestRelativeBuildPath(
		fmt.Sprintf("%s runtimeAssets.tokenEmbeddingsPath", gbuildDeploymentV1CapnpFile),
		rawPath,
	)
}
