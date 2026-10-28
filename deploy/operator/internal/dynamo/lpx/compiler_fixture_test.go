/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	deploymentcapnpv1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/deployment/v1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

const (
	otherFixtureValue = "other"
	v2TestBuildName   = "v2-test-build"
)

type staticModelRegistry map[string]*Build

func (staticModelRegistry) BuildURL(id string) (*url.URL, error) {
	return nil, fmt.Errorf("unexpected build URL lookup for %q", id)
}
func (staticModelRegistry) EnsureDownloaded(_ context.Context, buildURL url.URL) (bool, error) {
	return false, fmt.Errorf("unexpected build download for %q", buildURL.String())
}
func (staticModelRegistry) EnsureCheckpointDownloaded(_ context.Context, checkpoint *Checkpoint) (bool, error) {
	return false, fmt.Errorf("unexpected checkpoint download for %q", CheckpointKey(checkpoint))
}

func (source staticModelRegistry) AcquireBuild(_ context.Context, buildID string) (*Build, error) {
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
		Name: name, Namespace: "test", UID: "uid", Generation: 1,
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

func acquireTestSnapshot(t *testing.T, buildDir string) *Build {
	t.Helper()
	registry, err := NewModelRegistry("", nil)
	require.NoError(t, err)
	snapshot, err := registry.AcquireBuild(t.Context(), buildDir)
	require.NoError(t, err)
	return snapshot
}

func projectTestBuild(t *testing.T, snapshot *Build, pipeline Pipeline) *ModelProjection {
	t.Helper()

	t.Log("Project the caller-owned normalized build for the selected pipeline")
	projectionBatch, err := appendModelProjections(nil, ModelProjectionInput{
		Pipeline: pipeline, Models: []string{"default"}, Build: snapshot,
	})
	require.NoError(t, err)
	return projectionBatch[0]
}

// testCompilerFixture describes a published deployment.v1 build directory.
type testCompilerFixture struct {
	// buildDirectoryName names a subdirectory for the build; empty publishes
	// the build in the temporary directory itself.
	buildDirectoryName string
	testUnitBundle
}

func writeV2CompilerFixture(t *testing.T) string {
	t.Helper()
	return writeCompilerFixture(t, newV2CompilerFixture())
}

// newV2CompilerFixture returns an LPU-only LP20 build whose two 16-chip
// partitions form one prop-sync chain on two hosts each.
func newV2CompilerFixture() testCompilerFixture {
	return testCompilerFixture{
		buildDirectoryName: v2TestBuildName,
		architecture:       deploymentcapnpv1.LpuArchitecture_lp20,
		devicesPerNode:     8,
		partitions:         testUnitPartitions(16, 7, 8),
		units: []testLPUUnit{{
			reservations: []testLPUReservation{testNodesReservation(2, 7), testNodesReservation(2, 8)},
			connectors:   []testLPUConnector{{}},
		}},
	}
}

func writeV3CompilerFixture(t *testing.T) string {
	t.Helper()
	return writeCompilerFixture(t, newV3CompilerFixture())
}

// newV3CompilerFixture returns an LPU-only LP30 build with one 16-chip
// partition on one tray.
func newV3CompilerFixture() testCompilerFixture {
	return testCompilerFixture{
		architecture:   deploymentcapnpv1.LpuArchitecture_lp30,
		devicesPerNode: 16,
		partitions:     testUnitPartitions(16, 1),
		units:          testSingleUnits(testHXReservation(1, 1, 1, 1)),
	}
}

func writeCompilerFixture(t *testing.T, fixture testCompilerFixture) string {
	t.Helper()

	t.Log("Create the selected compiler fixture directory")
	buildDir := t.TempDir()
	if fixture.buildDirectoryName != "" {
		buildDir = filepath.Join(buildDir, fixture.buildDirectoryName)
		require.NoError(t, os.Mkdir(buildDir, 0o700))
	}

	t.Log("Publish the deployment contract")
	require.NoError(t, os.WriteFile(filepath.Join(buildDir, gbuildDeploymentV1CapnpFile), encodeTestUnitBundle(t, fixture.testUnitBundle), 0o600))
	return buildDir
}
