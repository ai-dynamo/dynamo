/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"capnproto.org/go/capnp/v3"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	otherFixtureValue  = "other"
	v2TestBuildName    = "v2-test-build"
	v3HXTopologyFamily = "16x8x2x3"
	v3OpaqueTopology   = "opaque-v3-topology"
)

type unreachableBuildSnapshotSource struct{}

func (unreachableBuildSnapshotSource) AcquireBuildSnapshot(context.Context, string) (*BuildSnapshot, error) {
	return nil, fmt.Errorf("unexpected build snapshot acquisition")
}

type staticBuildSnapshotSource map[string]*BuildSnapshot

func (source staticBuildSnapshotSource) AcquireBuildSnapshot(_ context.Context, buildID string) (*BuildSnapshot, error) {
	snapshot := source[buildID]
	if snapshot == nil {
		return nil, fmt.Errorf("unknown test build %q", buildID)
	}
	return snapshot, nil
}

func newSelectedTestDGD(t *testing.T, name string, components ...v1beta1.DynamoComponentDeploymentSharedSpec) *v1beta1.DynamoGraphDeployment {
	t.Helper()

	t.Log("Construct one fresh selected-LPX projection fixture around the caller-owned configuration")
	return &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name: name, Namespace: "test", UID: "uid", Generation: 1,
			Annotations: map[string]string{
				commonconsts.KubeAnnotationLPXSchedulerBackend: commonconsts.LPXSchedulerBackend,
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: components,
		},
	}
}

// testLPXComponent constructs an independent component with the caller's role membership.
func testLPXComponent(name, buildID string, roles ...v1beta1.ComponentRoleSpec) v1beta1.DynamoComponentDeploymentSharedSpec {
	return v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: name,
		ComponentType: v1beta1.ComponentTypeLPX,
		LPX:           &v1beta1.LPXConfig{BuildID: buildID},
		Roles:         roles,
	}
}

// testLPXPodTemplate constructs an independent role template for each caller.
func testLPXPodTemplate(image string) *corev1.PodTemplateSpec {
	return &corev1.PodTemplateSpec{
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Image: image}}},
	}
}

func acquireTestSnapshot(t *testing.T, buildDir string) *BuildSnapshot {
	t.Helper()
	registry, err := NewModelRegistry(buildDir, nil)
	require.NoError(t, err)
	snapshot, err := registry.AcquireBuildSnapshot(t.Context(), buildDir)
	require.NoError(t, err)
	return snapshot
}

func normalizeTestSnapshot(t *testing.T, snapshot *BuildSnapshot) NormalizedBuildSnapshot {
	t.Helper()
	normalized, err := normalizeBuildSnapshot(snapshot)
	require.NoError(t, err)
	return normalized
}

func projectTestBuild(t *testing.T, snapshot NormalizedBuildSnapshot, pipeline Pipeline) *ModelProjection {
	t.Helper()

	t.Log("Project the caller-owned normalized build for the selected pipeline")
	projectionBatch, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline: pipeline, Models: []string{"default"}, BuildSnapshot: snapshot,
	})
	require.NoError(t, err)
	return projectionBatch[0]
}

func writeV2CompilerFixture(t *testing.T) string {
	t.Helper()
	return writeCompilerFixture(t, newV2CompilerFixture())
}

func newV2CompilerFixture() testV3CapnpFixture {
	topology := "URSA_V2__Q8__16C__G_96_25__KP_FEC__GHZ_1_0__NO_FPGA"
	fixture := newV3CompilerFixture()
	fixture.buildDirectoryName = v2TestBuildName
	fixture.numLPUNodes = 4
	fixture.selectedPropSyncChains = [][]uint32{{7, 8}}
	fixture.partitions = []testV3CapnpPartition{
		{id: 7, deviceType: manifestcapnp.DeviceType_lpu, topology: topology, numChips: 16, devicesPerNode: 8},
		{id: 8, deviceType: manifestcapnp.DeviceType_lpu, topology: topology, numChips: 16, devicesPerNode: 8},
	}
	return fixture
}

func writeV3CompilerFixture(t *testing.T) string {
	t.Helper()
	return writeCompilerFixture(t, newV3CompilerFixture())
}

type testV3CapnpFixture struct {
	buildDirectoryName     string
	contractRevision       uint32
	pipelineName           string
	compilationMode        manifestcapnp.CompilationMode
	numLPUNodes            uint32
	selectedPropSyncChains [][]uint32
	partSelect             bool
	partitions             []testV3CapnpPartition
}

type testV3CapnpPartition struct {
	id             uint32
	deviceType     manifestcapnp.DeviceType
	topology       string
	numChips       uint32
	devicesPerNode uint32
	topologyFamily string
	partitionShape []uint32
}

func newV3CompilerFixture() testV3CapnpFixture {
	return testV3CapnpFixture{
		contractRevision: manifestcapnp.CurrentContractRevision,
		pipelineName:     "default",
		compilationMode:  manifestcapnp.CompilationMode_lpuOnly,
		numLPUNodes:      2,
		partitions: []testV3CapnpPartition{{
			id:             1,
			deviceType:     manifestcapnp.DeviceType_lpu,
			topology:       v3OpaqueTopology,
			numChips:       16,
			devicesPerNode: 16,
		}},
	}
}

func writeCompilerFixture(t *testing.T, fixture testV3CapnpFixture) string {
	t.Helper()

	t.Log("Create the selected compiler fixture directory")
	buildDir := t.TempDir()
	if fixture.buildDirectoryName != "" {
		buildDir = filepath.Join(buildDir, fixture.buildDirectoryName)
		require.NoError(t, os.Mkdir(buildDir, 0o700))
	}

	t.Log("Write the revision-2 compiler manifest fixture")
	writeTestV3CapnpManifest(t, buildDir, fixture)
	return buildDir
}

func writeTestV3CapnpManifest(t *testing.T, buildDir string, fixture testV3CapnpFixture) {
	t.Helper()

	t.Log("Encode the revision-2 manifest header and model metadata")
	message, segment := capnp.NewSingleSegmentMessage(nil)
	manifest, err := manifestcapnp.NewRootManifest(segment)
	require.NoError(t, err)
	manifest.SetContractRevision(fixture.contractRevision)
	model, err := manifest.NewModel()
	require.NoError(t, err)
	tokenizer, err := model.NewTokenizer()
	require.NoError(t, err)
	require.NoError(t, tokenizer.SetPath("tokenizer"))
	stopTokens, err := tokenizer.NewStopTokens(1)
	require.NoError(t, err)
	stopTokens.Set(0, 1)

	t.Log("Encode the deployment program, runtime I/O, and selected prop-sync chains")
	deployment, err := manifest.NewDeployment()
	require.NoError(t, err)
	deployment.SetCompilationMode(fixture.compilationMode)
	deployment.SetNumLpuNodes(fixture.numLPUNodes)
	program, err := deployment.NewProgram()
	require.NoError(t, err)
	require.NoError(t, program.SetProgramName(fixture.pipelineName))
	program.SetBatchSize(1)
	program.SetSequenceLength(8192)
	program.SetInputSize(1)
	program.SetOutputSize(1)
	program.SetNumKvCaches(1)
	runtimeIO, err := deployment.NewRuntimeIo()
	require.NoError(t, err)
	runtimeIO.SetProtocol(runtimeIOProtocolHost)
	runtimeIO.SetReserved1(0)
	runtimeIO.SetIoFpgaCount(1)
	runtimeIO.SetFanoutFactor(1)
	if fixture.selectedPropSyncChains != nil {
		chains, err := deployment.NewSelectedPropSyncChains(int32(len(fixture.selectedPropSyncChains)))
		require.NoError(t, err)
		for chainIndex, partitionIDs := range fixture.selectedPropSyncChains {
			chain := chains.At(chainIndex)
			ids, err := chain.NewPartitionIds(int32(len(partitionIDs)))
			require.NoError(t, err)
			for idIndex, partitionID := range partitionIDs {
				ids.Set(idIndex, partitionID)
			}
		}
	}

	t.Log("Encode the flat revision-2 artifact inventory")
	artifacts, err := manifest.NewArtifacts()
	require.NoError(t, err)
	if fixture.partSelect {
		partSelect, err := artifacts.NewPartSelect()
		require.NoError(t, err)
		selected, err := partSelect.NewPartitions(1)
		require.NoError(t, err)
		selected.At(0).SetDeviceType(manifestcapnp.DeviceType_lpu)
		selected.At(0).SetPartitionId(fixture.partitions[0].id)
	}
	partitions, err := artifacts.NewPartitions(int32(len(fixture.partitions)))
	require.NoError(t, err)
	for index, fixturePartition := range fixture.partitions {
		partition := partitions.At(index)
		ref, err := partition.NewPartition()
		require.NoError(t, err)
		ref.SetDeviceType(fixturePartition.deviceType)
		ref.SetPartitionId(fixturePartition.id)
		switch fixturePartition.deviceType {
		case manifestcapnp.DeviceType_lpu:
			detail, err := partition.Detail().NewLpu()
			require.NoError(t, err)
			require.NoError(t, detail.SetPath(fmt.Sprintf("part-%d", fixturePartition.id)))
			require.NoError(t, detail.SetTopology(fixturePartition.topology))
			detail.SetNumChips(fixturePartition.numChips)
			detail.SetDevicesPerNode(fixturePartition.devicesPerNode)
			if fixturePartition.topologyFamily != "" || fixturePartition.partitionShape != nil {
				metadata, err := detail.NewTopologyMetadata()
				require.NoError(t, err)
				require.NoError(t, metadata.SetTopologyFamily(fixturePartition.topologyFamily))
				shape, err := metadata.NewPartitionShape(int32(len(fixturePartition.partitionShape)))
				require.NoError(t, err)
				for dimension, extent := range fixturePartition.partitionShape {
					shape.Set(dimension, extent)
				}
			}
		case manifestcapnp.DeviceType_cuda:
			_, err := partition.Detail().NewCuda()
			require.NoError(t, err)
		case manifestcapnp.DeviceType_cpu:
			_, err := partition.Detail().NewCpu()
			require.NoError(t, err)
		default:
			t.Fatalf("unsupported test device type %q", fixturePartition.deviceType)
		}
	}

	t.Log("Publish the revision-2 compiler manifest")
	data, err := message.Marshal()
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, gbuildManifestV2CapnpFile), data, 0o600))
}
