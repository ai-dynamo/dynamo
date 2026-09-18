/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"capnproto.org/go/capnp/v3"
	manifestcapnpv2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
)

const (
	gbuildManifestV2CapnpFile             = "manifest.v2.capnp.bin"
	runtimeIOProtocolHost          uint16 = 0
	runtimeIOProtocolMultiEndpoint uint16 = 1
	runtimeIOMaxMode               uint16 = 2
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
	compilationMode, err := buildCompilationMode(gbuildManifestV2CapnpFile, deployment.CompilationMode().String())
	if err != nil {
		return nil, err
	}
	if !deployment.HasProgram() {
		return nil, fmt.Errorf("%s deployment.program is missing", gbuildManifestV2CapnpFile)
	}
	program, err := deployment.Program()
	if err != nil {
		return nil, fmt.Errorf("reading %s deployment.program: %w", gbuildManifestV2CapnpFile, err)
	}
	batchSize, err := positiveManifestUInt32ToInt(
		fmt.Sprintf("%s deployment.program.batchSize", gbuildManifestV2CapnpFile),
		program.BatchSize(),
	)
	if err != nil {
		return nil, err
	}
	chains, err := selectedPropSyncChainsFromManifestV2(deployment)
	if err != nil {
		return nil, err
	}
	ioFPGACount, ioFanoutFactor, err := runtimeIOFromManifestV2(deployment)
	if err != nil {
		return nil, err
	}
	// Reject an incomplete split-I/O batch before any runtime path consumes it.
	if batchSize%int(ioFPGACount) != 0 {
		return nil, fmt.Errorf(
			"%s deployment.program.batchSize %d must be divisible by deployment.runtimeIo.ioFpgaCount %d",
			gbuildManifestV2CapnpFile,
			batchSize,
			ioFPGACount,
		)
	}

	// Reject incomplete client-owned transaction regions after endpoint splitting.
	perEndpointBatchSize := batchSize / int(ioFPGACount)
	if perEndpointBatchSize%int(ioFanoutFactor) != 0 {
		return nil, fmt.Errorf(
			"%s deployment.program.batchSize per endpoint %d must be divisible by deployment.runtimeIo.fanoutFactor %d",
			gbuildManifestV2CapnpFile,
			perEndpointBatchSize,
			ioFanoutFactor,
		)
	}
	var settings map[string]any
	if compilationMode == BuildCompilationModeLPUOnly {
		settings, err = runtimeSettingsFromManifestV2(manifest, program)
		if err != nil {
			return nil, err
		}
	}
	if !manifest.HasArtifacts() {
		return nil, fmt.Errorf("%s is missing artifacts", gbuildManifestV2CapnpFile)
	}
	artifacts, err := manifest.Artifacts()
	if err != nil {
		return nil, fmt.Errorf("reading %s artifacts: %w", gbuildManifestV2CapnpFile, err)
	}
	runtimeTokenEmbeddingsPath, err := runtimeTokenEmbeddingsPathFromManifestV2(artifacts)
	if err != nil {
		return nil, err
	}
	// Validate runtime embedding assets before projecting scheduler-facing artifacts.
	if runtimeTokenEmbeddingsPath != "" && !program.SupportsCpuEmbeddings() {
		return nil, fmt.Errorf("%s artifacts.runtimeAssets.tokenEmbeddingsPath requires supportsCpuEmbeddings=true", gbuildManifestV2CapnpFile)
	}
	if program.SupportsCpuEmbeddings() && program.StandaloneTokenEmbeddings() && runtimeTokenEmbeddingsPath == "" {
		return nil, fmt.Errorf("%s artifacts.runtimeAssets.tokenEmbeddingsPath is required when standaloneTokenEmbeddings=true", gbuildManifestV2CapnpFile)
	}

	build := &Build{
		Path:                      buildRef,
		CompilationMode:           compilationMode,
		SelectedPropSyncChains:    chains,
		StandaloneTokenEmbeddings: program.StandaloneTokenEmbeddings(),
		SupportsCPUEmbeddings:     program.SupportsCpuEmbeddings(),
		IOFPGACount:               ioFPGACount,
		IOFanoutFactor:            ioFanoutFactor,
		runtimeSettings:           settings,
	}

	// Complete the normalized build with scheduler-facing LPU artifacts.
	if err := addLPUArtifactsFromManifestV2(artifacts, deployment, build); err != nil {
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
	switch runtimeIO.Protocol() {
	case runtimeIOProtocolHost:
		if count != 1 {
			return 0, 0, fmt.Errorf("%s host runtime I/O requires ioFpgaCount 1, got %d", gbuildManifestV2CapnpFile, count)
		}
	case runtimeIOProtocolMultiEndpoint:
	default:
		return 0, 0, fmt.Errorf("%s deployment.runtimeIo.protocol %d is not supported", gbuildManifestV2CapnpFile, runtimeIO.Protocol())
	}
	if mode := runtimeIO.Reserved1(); mode > runtimeIOMaxMode {
		return 0, 0, fmt.Errorf("%s deployment.runtimeIo mode %d is not supported", gbuildManifestV2CapnpFile, mode)
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

func runtimeSettingsFromManifestV2(manifest manifestcapnpv2.Manifest, program manifestcapnpv2.ProgramConfig) (map[string]any, error) {
	// Retain only settings whose omission would change the runtime's interpretation.
	settings := map[string]any{
		"batch_folding":             program.BatchFolding(),
		"num_batch_split_divisions": int64(program.NumBatchSplitDivisions()),
	}
	var archVocabulary uint32
	hasSWA := false
	if manifest.HasModel() {
		model, err := manifest.Model()
		if err != nil {
			return nil, fmt.Errorf("reading %s model: %w", gbuildManifestV2CapnpFile, err)
		}
		if model.HasArch() {
			arch, err := model.Arch()
			if err != nil {
				return nil, fmt.Errorf("reading %s model.arch: %w", gbuildManifestV2CapnpFile, err)
			}
			archVocabulary = arch.VocabSize()
			hasSWA = arch.HasSwa()
		}

		// Tokenizer vocabulary takes precedence over architecture padding in the deployment contract.
		if model.HasTokenizer() {
			tokenizer, err := model.Tokenizer()
			if err != nil {
				return nil, fmt.Errorf("reading %s model.tokenizer: %w", gbuildManifestV2CapnpFile, err)
			}
			if vocabulary := tokenizer.VocabSize(); vocabulary != 0 && vocabulary != archVocabulary {
				settings["vocab_size"] = int64(vocabulary)
			}
		}
	}

	// Program-only SWA settings are not inferred when the architecture has no SWA descriptor.
	if !hasSWA && (program.SwaChunked() || program.NumSwaDkvcBlocks() != 0) {
		settings["swa"] = map[string]any{
			"chunked":             program.SwaChunked(),
			"num_swa_dkvc_blocks": int64(program.NumSwaDkvcBlocks()),
		}
	}
	return settings, nil
}

func addLPUArtifactsFromManifestV2(
	artifacts manifestcapnpv2.ArtifactInfo,
	deployment manifestcapnpv2.DeploymentInfo,
	build *Build,
) error {
	rawPartitions, err := artifacts.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s artifacts.partitions: %w", gbuildManifestV2CapnpFile, err)
	}
	partitions := make([]BuildPartition, 0, rawPartitions.Len())
	hxDoubleNodeCount := true
	for index := 0; index < rawPartitions.Len(); index++ {
		partition, compatible, err := buildPartitionFromManifestV2(rawPartitions.At(index))
		if err != nil {
			return err
		}
		// Nonempty paths mark LPU artifacts; only metadata-less HX partitions permit the historical doubled node count.
		if partition.PartPath != "" {
			partitions = append(partitions, partition)
			hxDoubleNodeCount = hxDoubleNodeCount && (len(partition.HXExtent) == 0 || compatible)
		}
	}
	// An LPU-only build cannot silently discard packaged CUDA or CPU partitions.
	if build.CompilationMode == BuildCompilationModeLPUOnly && len(partitions) != rawPartitions.Len() {
		return fmt.Errorf(
			"%s deployment.compilationMode lpuOnly requires every artifact partition to use deviceType lpu",
			gbuildManifestV2CapnpFile,
		)
	}
	if len(partitions) == 0 {
		return fmt.Errorf("%s artifacts contain no LPU partitions", gbuildManifestV2CapnpFile)
	}
	partialSelection := artifacts.HasPartSelect()
	family, packagedNodes, partitionZeroNodes, err := classifyManifestPartitions(gbuildManifestV2CapnpFile, partitions, partialSelection)
	if err == nil && family == BuildFamilyXT {
		err = validateManifestV2PartSelect(artifacts, partitions)
	}
	if err != nil {
		return err
	}

	// Publish the artifact projection before validating the complete deployment geometry.
	build.Partitions = partitions
	build.Family = family

	want, err := positiveManifestUInt32ToInt(
		fmt.Sprintf("%s deployment.numLpuNodes", gbuildManifestV2CapnpFile),
		deployment.NumLpuNodes(),
	)
	if err != nil {
		return err
	}
	return validateManifestPartitionNodeCount(gbuildManifestV2CapnpFile, want, build, partialSelection, hxDoubleNodeCount, packagedNodes, partitionZeroNodes)
}

func buildPartitionFromManifestV2(raw manifestcapnpv2.PartitionInfo) (BuildPartition, bool, error) {
	if !raw.HasPartition() {
		return BuildPartition{}, false, fmt.Errorf("%s artifact partition is missing partition ref", gbuildManifestV2CapnpFile)
	}
	ref, err := raw.Partition()
	if err != nil {
		return BuildPartition{}, false, fmt.Errorf("reading %s artifact partition ref: %w", gbuildManifestV2CapnpFile, err)
	}
	if ref.DeviceType() != manifestcapnpv2.DeviceType_lpu {
		return BuildPartition{}, false, nil
	}
	if raw.Detail().Which() != manifestcapnpv2.PartitionInfo_detail_Which_lpu || !raw.Detail().HasLpu() {
		return BuildPartition{}, false, fmt.Errorf("%s artifact partition %d has deviceType lpu without LPU detail", gbuildManifestV2CapnpFile, ref.PartitionId())
	}
	detail, err := raw.Detail().Lpu()
	if err != nil {
		return BuildPartition{}, false, fmt.Errorf("reading %s LPU partition %d detail: %w", gbuildManifestV2CapnpFile, ref.PartitionId(), err)
	}
	topology, err := detail.Topology()
	if err != nil {
		return BuildPartition{}, false, fmt.Errorf("reading %s LPU partition %d topology: %w", gbuildManifestV2CapnpFile, ref.PartitionId(), err)
	}
	path, err := detail.Path()
	if err != nil {
		return BuildPartition{}, false, fmt.Errorf("reading %s LPU partition %d path: %w", gbuildManifestV2CapnpFile, ref.PartitionId(), err)
	}

	// Both families require a safe topology string, even when HX treats it as opaque.
	subject := fmt.Sprintf("%s LPU partition %d", gbuildManifestV2CapnpFile, ref.PartitionId())
	if strings.ContainsAny(topology, "\x00\r\n") {
		return BuildPartition{}, false, fmt.Errorf("%s topology must not contain NUL bytes or line breaks: %q", subject, topology)
	}

	// Metadata selects HX; only the 16-chip, 16-device HX shape can omit it.
	var partition BuildPartition
	var compatible bool
	if detail.HasTopologyMetadata() || (detail.NumChips() == 16 && detail.DevicesPerNode() == 16) {
		partition, compatible, err = buildHXPartition(subject, topology, detail)
	} else {
		partition, err = buildXTPartition(subject, topology, detail)
	}
	if err != nil {
		return BuildPartition{}, false, err
	}

	// Validate shared artifact fields once, after the selected geometry is accepted.
	partition.PartPath, err = cleanManifestRelativeBuildPath(subject+" path", path, false)
	if err != nil {
		return BuildPartition{}, false, err
	}
	partition.SourcePartitionID = int(ref.PartitionId())
	return partition, compatible, nil
}

func validateManifestV2PartSelect(artifacts manifestcapnpv2.ArtifactInfo, partitions []BuildPartition) error {
	if !artifacts.HasPartSelect() {
		return nil
	}
	partSelect, err := artifacts.PartSelect()
	if err != nil {
		return fmt.Errorf("reading %s artifacts.partSelect: %w", gbuildManifestV2CapnpFile, err)
	}
	selected, err := partSelect.Partitions()
	if err != nil {
		return fmt.Errorf("reading %s artifacts.partSelect.partitions: %w", gbuildManifestV2CapnpFile, err)
	}
	if selected.Len() == 0 {
		return fmt.Errorf("%s artifacts.partSelect.partitions is empty", gbuildManifestV2CapnpFile)
	}
	selectedLPU := make(map[int]struct{}, len(partitions))
	for index := 0; index < selected.Len(); index++ {
		ref := selected.At(index)
		if ref.DeviceType() != manifestcapnpv2.DeviceType_lpu {
			continue
		}
		id := int(ref.PartitionId())
		if _, duplicate := selectedLPU[id]; duplicate {
			return fmt.Errorf("%s artifacts.partSelect has duplicate LPU partition id %d", gbuildManifestV2CapnpFile, id)
		}
		partitionIndex := sort.Search(len(partitions), func(i int) bool {
			return partitions[i].SourcePartitionID >= id
		})
		if partitionIndex == len(partitions) || partitions[partitionIndex].SourcePartitionID != id {
			return fmt.Errorf("%s artifacts.partSelect references unpackaged LPU partition id %d", gbuildManifestV2CapnpFile, id)
		}
		selectedLPU[id] = struct{}{}
	}
	if len(selectedLPU) != len(partitions) {
		return fmt.Errorf("%s artifacts.partSelect selects %d LPU partitions, but artifacts package %d", gbuildManifestV2CapnpFile, len(selectedLPU), len(partitions))
	}
	return nil
}

func runtimeTokenEmbeddingsPathFromManifestV2(artifacts manifestcapnpv2.ArtifactInfo) (string, error) {
	if !artifacts.HasRuntimeAssets() {
		return "", nil
	}
	runtimeAssets, err := artifacts.RuntimeAssets()
	if err != nil {
		return "", fmt.Errorf("reading %s artifacts.runtimeAssets: %w", gbuildManifestV2CapnpFile, err)
	}
	if !runtimeAssets.HasTokenEmbeddingsPath() {
		return "", nil
	}
	rawPath, err := runtimeAssets.TokenEmbeddingsPath()
	if err != nil {
		return "", fmt.Errorf("reading %s artifacts.runtimeAssets.tokenEmbeddingsPath: %w", gbuildManifestV2CapnpFile, err)
	}
	return cleanManifestRelativeBuildPath(
		fmt.Sprintf("%s artifacts.runtimeAssets.tokenEmbeddingsPath", gbuildManifestV2CapnpFile),
		rawPath,
		false,
	)
}
