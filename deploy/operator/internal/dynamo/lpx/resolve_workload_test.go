/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

func TestResolveWorkloadClassifiesSnapshotFailures(t *testing.T) {
	t.Parallel()

	t.Log("Distinguish missing manifests from malformed bytes and invalid compiler contracts")
	manifest := newManifestV2ContractFixture(t)
	manifest.SetContractRevision(0)
	payload, err := manifest.Message().Marshal()
	require.NoError(t, err)
	tests := []struct {
		name        string
		manifest    []byte
		wantInvalid bool
		wantErr     string
	}{
		{name: "missing", wantErr: gbuildManifestV2CapnpFile},
		{name: "malformed", manifest: []byte("invalid"), wantInvalid: true, wantErr: "parsing manifest.v2.capnp.bin"},
		{name: "invalid contract", manifest: payload, wantInvalid: true, wantErr: "contractRevision"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve one component through the production snapshot loader")
			registryDir := t.TempDir()
			buildDir := filepath.Join(registryDir, "build")
			require.NoError(t, os.MkdirAll(buildDir, 0o700))
			if test.manifest != nil {
				writeManifestV2Payload(t, buildDir, test.manifest)
			}
			registry, err := NewModelRegistry(registryDir, nil)
			require.NoError(t, err)
			dgd := newSelectedTestDGD(t, "graph", testLPXComponent("LPX", "build"))
			workload, err := resolveTestWorkload(t, dgd, registry)

			t.Log("Preserve failure classification without returning a partial workload")
			require.Nil(t, workload)
			require.ErrorContains(t, err, test.wantErr)
			if test.wantInvalid {
				require.ErrorIs(t, err, errInvalidBuildManifest)
				require.NotErrorIs(t, err, ErrBuildSnapshotAcquisition)
			} else {
				require.ErrorIs(t, err, ErrBuildSnapshotAcquisition)
				require.ErrorIs(t, err, os.ErrNotExist)
			}
		})
	}
}

func TestResolveWorkloadDerivesRuntimeShapeFromCompilationMode(t *testing.T) {
	t.Log("Create one LPX component beside an unrelated conventional decode")
	dgd := newSelectedTestDGD(t, "graph", testLPXComponent("LPX", "build", v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("lpu-runtime")}, v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor-runtime")}))
	dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "ordinary-decode", ComponentType: v1beta1.ComponentTypeDecode,
		Replicas:    ptr.To(int32(0)),
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "ordinary"}}}},
	})

	t.Log("Resolve an LPU-only workload without selecting the conventional decode")
	hxSnapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	hx, err := resolveTestWorkload(t, dgd, staticModelRegistry{"build": hxSnapshot})
	require.NoError(t, err)
	require.Equal(t, PipelineSingle, hx.Pipeline())
	require.Equal(t, hxFamily, hx.models[0].component.configuredBuild.family)
	require.Equal(t, lpxv1alpha1.WorkloadModeV3HxLPUOnly, hx.models[0].requestSpec().WorkloadMode)
	require.Equal(t, "LPX", hx.Name())
	require.Equal(t, "test-pcs-0-lpx", hx.ScalingGroup())
	require.Equal(t, "cond", hx.ConductorTemplate())

	t.Log("Scale Nova workloads without changing their model or workload digest")
	for _, replicas := range []int32{2, 10, 12, 123} {
		dgd.Spec.Components[0].Replicas = ptr.To(replicas)
		scaled, err := resolveTestWorkload(t, dgd, staticModelRegistry{"build": hxSnapshot})
		require.NoError(t, err)
		require.Equal(t, hx.Digest(), scaled.Digest())
		require.Equal(t, replicas, scaled.scalingGroupReplicas)
	}
	dgd.Spec.Components[0].Replicas = nil

	t.Log("Require resources on the independently authored hybrid conductor")
	fixture := newV2CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	fixture.selectedPropSyncChains = nil
	fixture.partitions = append(fixture.partitions, testV3CapnpPartition{id: 11, deviceType: manifestcapnp.DeviceType_cuda})
	snapshot := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
	source := staticModelRegistry{"build": snapshot}
	_, err = resolveTestWorkload(t, dgd, source)
	require.ErrorContains(t, err, "requires a declared resourceClaim or a positive nvidia.com/gpu request")
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate = testLPXPodTemplate("cyborg-runtime")
	_, err = resolveTestWorkload(t, dgd, source)
	require.ErrorContains(t, err, "requires a declared resourceClaim or a positive nvidia.com/gpu request")

	t.Log("Validate claim consumption by the Cyborg main container")
	for _, test := range []struct {
		name                         string
		claims                       []corev1.ResourceClaim
		declared, sidecar, scalarGPU bool
		wantErr                      bool
	}{
		{name: "unused claim", declared: true, wantErr: true},
		{name: "sidecar-only claim", declared: true, sidecar: true, wantErr: true},
		{name: "undeclared claim", claims: []corev1.ResourceClaim{{Name: "gpu"}}, wantErr: true},
		{name: "mismatched claim", declared: true, claims: []corev1.ResourceClaim{{Name: "missing"}}, wantErr: true},
		{name: "external name is not the alias", declared: true, claims: []corev1.ResourceClaim{{Name: "external-gpu"}}, wantErr: true},
		{name: "consumed claim", declared: true, claims: []corev1.ResourceClaim{{Name: "gpu"}}},
		{name: "scalar GPU with unused claim", declared: true, scalarGPU: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Author independent Pod and main-container claim references")
			candidate := dgd.DeepCopy()
			pod := &candidate.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Spec
			pod.Containers[0].Resources.Claims = test.claims
			if test.declared {
				pod.ResourceClaims = []corev1.PodResourceClaim{
					{Name: "other", ResourceClaimName: ptr.To("external-other")},
					{Name: "gpu", ResourceClaimName: ptr.To("external-gpu")},
				}
			}
			if test.sidecar {
				pod.Containers = append(pod.Containers, corev1.Container{
					Name: "sidecar", Image: "sidecar", Resources: corev1.ResourceRequirements{Claims: []corev1.ResourceClaim{{Name: "gpu"}}},
				})
			}
			if test.scalarGPU {
				pod.Containers[0].Resources.Limits = corev1.ResourceList{corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1")}
			}
			before := candidate.DeepCopy()

			t.Log("Require scalar GPUs or a claim consumed by main without rewriting the template")
			_, err := resolveTestWorkload(t, candidate, source)
			if test.wantErr {
				require.ErrorContains(t, err, "conductor main container requires a declared resourceClaim")
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, before, candidate)
		})
	}

	conductor.PodTemplate.Spec.Containers[0].Resources.Limits = corev1.ResourceList{
		corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
	}
	dgd.Spec.Components[0].Replicas = ptr.To(int32(2))

	t.Log("Project two complete hybrid replicas")
	xt, err := resolveTestWorkload(t, dgd, source)
	require.NoError(t, err)
	require.Equal(t, PipelineHybrid, xt.Pipeline())
	require.Equal(t, xtFamily, xt.models[0].component.configuredBuild.family)
	require.Equal(t, lpxv1alpha1.WorkloadModeV2StrictHybrid, xt.models[0].requestSpec().WorkloadMode)
	require.Len(t, xt.models[0].component.configuredBuild.partitions, 2)
	require.Equal(t, "cond", xt.ConductorTemplate())
	require.EqualValues(t, 2, xt.scalingGroupReplicas)
	first, second := xt.RequestSpec(xt.models[0], 0), xt.RequestSpec(xt.models[0], 1)
	require.NotEqual(t, first.NodeLocal.AgentPodCliqueRef, second.NodeLocal.AgentPodCliqueRef)
	require.NotEqual(t, first.CyborgPodCliqueRef, second.CyborgPodCliqueRef)

	t.Log("A scheduling deadline does not change hybrid launch")
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To(int64(30))}
	scheduled, err := resolveTestWorkload(t, dgd, source)
	require.NoError(t, err)
	require.Equal(t, xt, scheduled)
}

func TestResolveWorkloadSpecDecodeXTAndHX(t *testing.T) {
	t.Log("Define revision-specific SpecDecode compiler snapshots")
	tests := []struct {
		name     string
		family   *family
		wantMode lpxv1alpha1.WorkloadMode
	}{
		{
			name: "XT", family: xtFamily,
			wantMode: lpxv1alpha1.WorkloadModeV2LPUOnly,
		},
		{
			name: "HX", family: hxFamily,
			wantMode: lpxv1alpha1.WorkloadModeV3HxLPUOnly,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Acquire the selected family fixtures: shared XT build or distinct HX snapshots")
			var draftSnapshot, targetSnapshot *Build
			if test.family == hxFamily {
				draftSnapshot = acquireTestSnapshot(t, writeV3CompilerFixture(t))
				targetSnapshot = acquireTestSnapshot(t, writeV3CompilerFixture(t))
			} else {
				draftSnapshot = acquireTestSnapshot(t, writeV2CompilerFixture(t))
				targetSnapshot = draftSnapshot
			}

			t.Log("Build a selected SpecDecode DGD for the fixture's manifest generation")
			dgd := newSelectedTestDGD(t, "specdecode",
				testLPXComponent("lpx", "target-build", v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor-runtime")}, v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("lpu-runtime")}),
				testLPXComponent("small", "draft-build", v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("lpu-runtime")}),
			)
			dgd.Spec.Components[1].Replicas = ptr.To(int32(2))
			compiledAgentCount := int32(1)
			if test.family == xtFamily {
				compiledAgentCount = 4
			}
			dgd.Spec.Components[1].ComponentRole(v1beta1.ComponentRoleLPXAgent).Replicas = ptr.To(compiledAgentCount)
			source := staticModelRegistry{
				"draft-build":  draftSnapshot,
				"target-build": targetSnapshot,
			}

			t.Log("Project the selected SpecDecode workload")
			before := dgd.DeepCopy()
			selected, err := resolveTestWorkload(t, dgd, source)

			t.Log("Project the selected family, workload mode, and pipeline")
			require.NoError(t, err)
			require.Equal(t, before, dgd, "canonical ordering must not rewrite the authored target-first list")
			require.Equal(t, test.family, selected.models[0].component.configuredBuild.family)
			require.Equal(t, test.wantMode, selected.models[0].requestSpec().WorkloadMode)
			require.Equal(t, PipelineSpecDecode, selected.Pipeline())
			require.Equal(t, "lpx", selected.Name())

			t.Log("Expand draft fanout while preserving model and template identity order")
			projections := selected.Models()
			require.Len(t, projections, 3)
			require.Equal(
				t,
				[]string{"draft0", "draft1", "target"},
				[]string{projections[0].Name(), projections[1].Name(), projections[2].Name()},
			)
			require.EqualValues(t, 1, selected.scalingGroupReplicas, "draft fanout must not become the shared scaling-group axis")
			require.Equal(t, "small", projections[0].component.name)
			require.Equal(t, "lpx", projections[2].component.name)
			require.Equal(
				t,
				[]string{"agt0", "agt1", "agt2"},
				[]string{projections[0].agentTemplate, projections[1].agentTemplate, projections[2].agentTemplate},
				"template identities follow canonical stage and draft-instance order",
			)

			t.Log("Reordering authored components does not change build identities or resources")
			reordered := dgd.DeepCopy()
			slices.Reverse(reordered.Spec.Components)
			reselected, err := resolveTestWorkload(t, reordered, source)
			require.NoError(t, err)
			require.Equal(t, selected, reselected)
			require.Equal(t, []string{"small", "lpx"}, reselected.ComponentNames())

			t.Log("Agent replica assertions count one compiled model instance, not draft fanout")
			invalidCount := dgd.DeepCopy()
			invalidCount.Spec.Components[1].ComponentRole(v1beta1.ComponentRoleLPXAgent).Replicas = ptr.To(compiledAgentCount * 2)
			_, err = resolveTestWorkload(t, invalidCount, source)
			require.ErrorContains(t, err, "must match the compiled count")

			t.Log("Derive an aggregate digest and reject mixed-family aggregation")
			require.NotEqual(t, WorkloadDigest{}, selected.Digest())
			require.NotEqual(t, projections[0].Digest(), selected.Digest())
			mixedComponent := *projections[2].component
			mixedComponent.configuredBuild.family = &family{target: "other"}
			mixedFamily := &Model{name: projections[2].name, component: &mixedComponent}
			_, err = workloadSetDigest([]*Model{projections[0], mixedFamily})
			require.ErrorContains(t, err, "mixed target families")

			t.Log("Preserve compiled placement without repeating runtime-derived model settings")
			for _, projection := range projections {
				require.EqualValues(t, compiledAgentCount, projection.component.agentReplicas)
			}

			for _, expansion := range []struct {
				name   string
				count  int32
				models []string
			}{
				{name: "default", count: 1, models: []string{"draft0", "target"}},
				{
					name:   "maximum",
					count:  8,
					models: []string{"draft0", "draft1", "draft2", "draft3", "draft4", "draft5", "draft6", "draft7", "target"},
				},
			} {
				t.Run(expansion.name, func(t *testing.T) {
					t.Logf("Project SpecDecode draft fanout %d", expansion.count)
					draft := &dgd.Spec.Components[1]
					draft.Replicas = nil
					if expansion.count > 1 {
						draft.Replicas = ptr.To(expansion.count)
					}
					expanded, err := resolveTestWorkload(t, dgd, source)
					require.NoError(t, err)
					models := make([]string, 0, len(expansion.models))
					for _, projection := range expanded.Models() {
						models = append(models, projection.Name())
					}

					t.Log("Preserve the expected logical model ordering")
					require.Equal(t, expansion.models, models)
				})
			}

			if test.family == hxFamily {
				t.Log("Project separate draft and target roles from the same immutable HX build")
				draft := &dgd.Spec.Components[1]
				draft.Replicas = nil
				draft.LPX.BuildID = "target-build"
				shared, err := resolveTestWorkload(t, dgd, source)
				require.NoError(t, err)
				projections := shared.Models()
				require.Len(t, projections, 2)
				require.Equal(t, []string{"draft0", "target"}, []string{projections[0].Name(), projections[1].Name()})
				require.Equal(t, "target-build", projections[0].component.runtimeBuildRef)
				require.Equal(t, "target-build", projections[1].component.runtimeBuildRef)
				require.NotEqual(t, projections[0].Digest(), projections[1].Digest())
			}
		})
	}
}

func TestResolveWorkloadIsolatesComponentGroups(t *testing.T) {
	t.Log("Author two LPU-only workloads with different builds and replica counts")
	dgd := newSelectedTestDGD(t, "graph", v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
	})
	for index, name := range []string{"first", "second"} {
		component := testLPXComponent(name, name+"-build",
			v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("agent")},
			v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor")},
		)
		component.Replicas = ptr.To(int32(index + 2))
		dgd.Spec.Components = append(dgd.Spec.Components, component)
	}
	snapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	before := dgd.DeepCopy()

	t.Log("Resolve each group from its own build, replicas and component identity, named within the shared PCS")
	source := staticModelRegistry{"first-build": snapshot, "second-build": snapshot}
	workloads, err := ResolveWorkloads(t.Context(), dgd, "pcs", source)
	require.NoError(t, err)
	require.Len(t, workloads, 2)
	for index, name := range []string{"first", "second"} {
		workload := workloads[index]
		require.Equal(t, PipelineSingle, workload.Pipeline())
		require.Equal(t, name, workload.Name())
		require.Equal(t, []string{name}, workload.ComponentNames())
		require.EqualValues(t, index+2, workload.scalingGroupReplicas)
		require.Equal(t, "pcs-0-"+name, workload.ScalingGroup())
		require.Equal(t, "pcs-"+name, workload.ResourcePrefix())
		require.Equal(t, name+"-cond", workload.ConductorTemplate())
	}
	require.Equal(t, before, dgd)

	t.Log("Require a singleton conductor independently of component replica counts")
	conductor := dgd.GetComponentByName("second").ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.Replicas = ptr.To(int32(2))
	_, err = ResolveWorkloads(t.Context(), dgd, "pcs", source)
	require.ErrorContains(t, err, `component "second" conductor replicas must be one`)
}
