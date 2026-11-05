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
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestResolveWorkloadClassifiesSnapshotFailures(t *testing.T) {
	t.Parallel()

	t.Log("Distinguish missing manifests from malformed bytes and invalid compiler contracts")
	reader := fake.NewClientBuilder().Build()
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
			workload, err := ResolveWorkload(t.Context(), reader, dgd, []string{"LPX"}, registry)

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

func TestResolveWorkloadDerivesRuntimeShapeFromExecutionDevices(t *testing.T) {
	t.Log("Create one LPX component beside an unrelated conventional decode")
	reader := fake.NewClientBuilder().Build()
	dgd := newSelectedTestDGD(t, "graph", testLPXComponent("LPX", "build", v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("lpu-runtime")}, v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor-runtime")}))
	dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "ordinary-decode", ComponentType: v1beta1.ComponentTypeDecode,
		Replicas:    new(int32(0)),
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "ordinary"}}}},
	})

	t.Log("Resolve an LPU-only workload without selecting the conventional decode")
	hxSnapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	hx, err := ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), staticModelRegistry{"build": hxSnapshot})
	require.NoError(t, err)
	require.Equal(t, PipelineSingle, hx.Pipeline())
	hxRequest := hx.modelProjections[0].RequestSpec(&MaterializationPlan{}, "agents")
	require.Equal(t, lpxv1alpha1.TargetFamilyHx16x8x2x3, hxRequest.TargetFamily)
	require.Equal(t, lpxv1alpha1.WorkloadModeV3HxLPUOnly, hxRequest.WorkloadMode)
	require.Equal(t, "LPX", hx.ServingComponentName())
	plan, err := hx.PlanNodeLocalMaterialization("test-pcs", dgd.GetComponentByName(hx.ServingComponentName()))
	require.NoError(t, err)
	require.Equal(t, "test-pcs-0-lpx", plan.LPXScalingGroup)
	require.Equal(t, "cond", plan.ConductorTemplate)
	require.Empty(t, plan.CyborgTemplate)

	t.Log("Scale Nova workloads without changing their model or workload digest")
	for _, replicas := range []int32{2, 10, 12, 123} {
		dgd.Spec.Components[0].Replicas = new(replicas)
		scaled, err := ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), staticModelRegistry{"build": hxSnapshot})
		require.NoError(t, err)
		require.Equal(t, hx.Digest(), scaled.Digest())
		require.Equal(t, replicas, scaled.scalingGroupReplicas)
	}
	dgd.Spec.Components[0].Replicas = nil

	t.Log("Run LPX-ingested LP20 and LP30 builds with CPU preparation without a GPU")
	for _, fixture := range []testV3CapnpFixture{newV2CompilerFixture(), newV3CompilerFixture()} {
		fixture.compilationMode = manifestcapnp.CompilationMode_lpx
		fixture.selectedPropSyncChains = nil
		fixture.partitions = append(fixture.partitions, testV3CapnpPartition{id: 0, deviceType: manifestcapnp.DeviceType_cpu})
		snapshot := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
		source := staticModelRegistry{"build": snapshot}
		workload, err := ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), source)
		require.NoError(t, err)
		require.Equal(t, PipelineSingle, workload.Pipeline())
		require.False(t, workload.modelProjections[0].configuredBuild.hasCUDAPartitions)

		t.Log("Authored GPU resources do not make an LPU-only build hybrid")
		candidate := dgd.DeepCopy()
		candidate.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Spec.Containers[0].Resources.Limits = corev1.ResourceList{
			corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
		}
		workload, err = ResolveWorkload(t.Context(), reader, candidate, singleGroupComponents(t, candidate), source)
		require.NoError(t, err)
		require.Equal(t, PipelineSingle, workload.Pipeline())

		t.Log("Reject GPU-local partition execution for LPU-only builds")
		candidate.Spec.Components[0].LPX.Experimental = &v1beta1.LPXExperimentalSpec{
			LocalPartitions: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeAll},
		}
		_, err = ResolveWorkload(t.Context(), reader, candidate, singleGroupComponents(t, candidate), source)
		require.ErrorContains(t, err, "localPartitions requires a hybrid build with a Cyborg conductor")
	}

	t.Log("Require resources on the independently authored hybrid conductor")
	source := staticModelRegistry{"build": acquireHybridTestSnapshot(t)}
	_, err = ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), source)
	require.ErrorContains(t, err, "requires a positive GPU count")
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate = testLPXPodTemplate("cyborg-runtime")
	_, err = ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), source)
	require.ErrorContains(t, err, "requires a positive GPU count")

	t.Log("Validate GPU resources and claim consumption by the Cyborg main container")
	const missingResources = "conductor main container requires a positive GPU count"
	for _, test := range []struct {
		name, gpuResource, request, limit string
		claimObject                       string
		claims                            []corev1.ResourceClaim
		declared, sidecar, template       bool
		wantErr                           string
	}{
		{name: "unused claim", declared: true, wantErr: missingResources},
		{name: "sidecar-only claim", declared: true, sidecar: true, wantErr: missingResources},
		{name: "undeclared claim", claims: []corev1.ResourceClaim{{Name: "gpu"}}, wantErr: "has no matching pod resourceClaim"},
		{name: "mismatched claim", declared: true, claims: []corev1.ResourceClaim{{Name: "missing"}}, wantErr: "has no matching pod resourceClaim"},
		{name: "external name is not the alias", declared: true, claims: []corev1.ResourceClaim{{Name: "external-gpu"}}, wantErr: "has no matching pod resourceClaim"},
		{name: "consumed claim", declared: true, claims: []corev1.ResourceClaim{{Name: "gpu"}}},
		{name: "consumed claim template", declared: true, template: true, claims: []corev1.ResourceClaim{{Name: "gpu"}}},
		{name: "missing claim", declared: true, claimObject: "missing", claims: []corev1.ResourceClaim{{Name: "gpu"}}, wantErr: "failed to get ResourceClaim test/missing"},
		{name: "missing claim template", declared: true, template: true, claimObject: "missing", claims: []corev1.ResourceClaim{{Name: "gpu"}}, wantErr: "failed to get ResourceClaimTemplate test/missing"},
		{name: "non-GPU claim", declared: true, claimObject: "network", claims: []corev1.ResourceClaim{{Name: "gpu"}}, wantErr: missingResources},
		{name: "scalar GPU with unused claim", declared: true, limit: "1"},
		{name: "scalar GPU with missing claim", declared: true, claimObject: "missing", claims: []corev1.ResourceClaim{{Name: "gpu"}}, limit: "1"},
		{name: "GPU request without limit", request: "1"},
		{name: "MIG GPU", gpuResource: "nvidia.com/mig-3g.20gb", limit: "1"},
		{name: "zero GPUs", limit: "0", wantErr: missingResources},
		{name: "fractional GPU", limit: "500m", wantErr: `component "LPX" conductor resources: GPU resource "nvidia.com/gpu" quantity "500m" must be a whole number`},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Provide GPU and network claims in the deployment namespace")
			gpuClass := &resourcev1.DeviceClass{
				Name: "gpu.nvidia.com",
				Spec: resourcev1.DeviceClassSpec{ExtendedResourceName: ptr.To(commonconsts.KubeResourceGPUNvidia)},
			}
			claim := &resourcev1.ResourceClaim{
				Name: "external-gpu", Namespace: dgd.Namespace,
				Spec: resourcev1.ResourceClaimSpec{Devices: resourcev1.DeviceClaim{Requests: []resourcev1.DeviceRequest{{
					Name: "gpus", Exactly: &resourcev1.ExactDeviceRequest{DeviceClassName: gpuClass.Name, Count: 1},
				}}}},
			}
			claimTemplate := &resourcev1.ResourceClaimTemplate{
				ObjectMeta: claim.ObjectMeta,
				Spec:       resourcev1.ResourceClaimTemplateSpec{Spec: *claim.Spec.DeepCopy()},
			}
			networkClass := &resourcev1.DeviceClass{
				Name: "network.example.com",
				Spec: resourcev1.DeviceClassSpec{ExtendedResourceName: new("rdma/hca")},
			}
			networkClaim := claim.DeepCopy()
			networkClaim.Name = "network"
			networkClaim.Spec.Devices.Requests[0].Exactly.DeviceClassName = networkClass.Name
			reader := fake.NewClientBuilder().WithObjects(gpuClass, claim, claimTemplate, networkClass, networkClaim).Build()

			t.Log("Author GPU resources and independent Pod and main-container claim references")
			candidate := dgd.DeepCopy()
			pod := &candidate.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).PodTemplate.Spec
			pod.Containers[0].Resources.Claims = test.claims
			if test.declared {
				podClaim := corev1.PodResourceClaim{Name: "gpu"}
				objectName := defaultTestString(test.claimObject, "external-gpu")
				if test.template {
					podClaim.ResourceClaimTemplateName = new(objectName)
				} else {
					podClaim.ResourceClaimName = new(objectName)
				}
				pod.ResourceClaims = []corev1.PodResourceClaim{
					{Name: "other", ResourceClaimName: new("external-other")},
					podClaim,
				}
			}
			if test.sidecar {
				pod.Containers = append(pod.Containers, corev1.Container{
					Name: "sidecar", Image: "sidecar", Resources: corev1.ResourceRequirements{Claims: []corev1.ResourceClaim{{Name: "gpu"}}},
				})
			}
			gpuResource := corev1.ResourceName(defaultTestString(test.gpuResource, commonconsts.KubeResourceGPUNvidia))
			if test.request != "" {
				pod.Containers[0].Resources.Requests = corev1.ResourceList{gpuResource: resource.MustParse(test.request)}
			}
			if test.limit != "" {
				pod.Containers[0].Resources.Limits = corev1.ResourceList{gpuResource: resource.MustParse(test.limit)}
			}
			before := candidate.DeepCopy()

			t.Log("Require scalar GPUs or a claim consumed by main without rewriting the template")
			_, err := ResolveWorkload(t.Context(), reader, candidate, singleGroupComponents(t, candidate), source)
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
			} else {
				require.NoError(t, err)
			}
			require.Equal(t, before, candidate)

			if test.claimObject == "missing" && test.wantErr != "" {
				t.Log("Preserve lookup failures for retries and resolve again after the missing object appears")
				require.True(t, apierrors.IsNotFound(err))
				if test.template {
					claimTemplate.ObjectMeta = metav1.ObjectMeta{Name: "missing", Namespace: dgd.Namespace}
					require.NoError(t, reader.Create(t.Context(), claimTemplate))
				} else {
					claim.ObjectMeta = metav1.ObjectMeta{Name: "missing", Namespace: dgd.Namespace}
					require.NoError(t, reader.Create(t.Context(), claim))
				}
				_, err = ResolveWorkload(t.Context(), reader, candidate, singleGroupComponents(t, candidate), source)
				require.NoError(t, err)
				require.Equal(t, before, candidate)
			}
		})
	}

	conductor.PodTemplate.Spec.Containers[0].Resources.Limits = corev1.ResourceList{
		corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
	}
	dgd.Spec.Components[0].Replicas = new(int32(2))

	t.Log("Project two complete hybrid replicas")
	xt, err := ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), source)
	require.NoError(t, err)
	require.Equal(t, PipelineHybrid, xt.Pipeline())
	xtRequest := xt.modelProjections[0].RequestSpec(&MaterializationPlan{}, "agents")
	require.Equal(t, lpxv1alpha1.TargetFamilyXt8888, xtRequest.TargetFamily)
	require.Equal(t, lpxv1alpha1.WorkloadModeV2StrictHybrid, xtRequest.WorkloadMode)
	require.Len(t, xt.modelProjections[0].configuredBuild.partitions, 2)
	plan, err = xt.PlanNodeLocalMaterialization("test-pcs", dgd.GetComponentByName(xt.ServingComponentName()))
	require.NoError(t, err)
	require.Equal(t, "cond", plan.CyborgTemplate)
	require.EqualValues(t, 2, plan.Replicas)
	replica := plan.ForReplica(1)
	require.NotEqual(t, plan.ForReplica(0).Agents[0].CliqueName, replica.Agents[0].CliqueName)

	t.Log("CUDA-containing hybrid builds retain GPU-local partition execution")
	local := dgd.DeepCopy()
	local.Spec.Components[0].LPX.Experimental = &v1beta1.LPXExperimentalSpec{
		LocalPartitions: &v1beta1.LPXLocalPartitions{Mode: v1beta1.LPXLocalPartitionsModeAll},
	}
	localWorkload, err := ResolveWorkload(t.Context(), reader, local, singleGroupComponents(t, local), source)
	require.NoError(t, err)
	require.Equal(t, PipelineHybrid, localWorkload.Pipeline())
	require.Zero(t, localWorkload.modelProjections[0].AgentReplicas())

	t.Log("A scheduling deadline does not change hybrid launch")
	dgd.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: new(int64(30))}
	scheduled, err := ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), source)
	require.NoError(t, err)
	require.Equal(t, xt, scheduled)
	scheduledPlan, err := scheduled.PlanNodeLocalMaterialization("test-pcs", dgd.GetComponentByName(scheduled.ServingComponentName()))
	require.NoError(t, err)
	require.Equal(t, plan, scheduledPlan)
	require.Empty(t, scheduledPlan.ConductorTemplate)

}

func TestResolveWorkloadSpecDecodeV2AndV3(t *testing.T) {
	t.Log("Define revision-specific SpecDecode compiler snapshots")
	reader := fake.NewClientBuilder().Build()
	tests := []struct {
		name     string
		family   BuildFamily
		wantMode lpxv1alpha1.WorkloadMode
		packed   [2]bool
	}{
		{name: "v2", family: BuildFamilyXT, wantMode: lpxv1alpha1.WorkloadModeV2LPUOnly},
		{name: "v2 packed draft", family: BuildFamilyXT, wantMode: lpxv1alpha1.WorkloadModeV2LPUOnly, packed: [2]bool{true, false}},
		{name: "v2 packed target", family: BuildFamilyXT, wantMode: lpxv1alpha1.WorkloadModeV2LPUOnly, packed: [2]bool{false, true}},
		{name: "v3", family: BuildFamilyHX, wantMode: lpxv1alpha1.WorkloadModeV3HxLPUOnly},
		{name: "v3 packed draft", family: BuildFamilyHX, wantMode: lpxv1alpha1.WorkloadModeV3HxLPUOnly, packed: [2]bool{true, false}},
		{name: "v3 packed target", family: BuildFamilyHX, wantMode: lpxv1alpha1.WorkloadModeV3HxLPUOnly, packed: [2]bool{false, true}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Acquire draft and target fixtures with independently selected packing")
			var snapshots [2]*Build
			agentCounts, requestCounts := [2]int32{4, 4}, [2]int{2, 2}
			for index, packed := range test.packed {
				fixture := newV2CompilerFixture()
				if test.family == BuildFamilyHX {
					fixture = newV3CompilerFixture()
					agentCounts[index], requestCounts[index] = 1, 1
				}
				if packed {
					fixture.partitions[0].numChips = fixture.partitions[0].devicesPerNode / 2
					fixture.partitions = []testV3CapnpPartition{fixture.partitions[0], fixture.partitions[0]}
					fixture.partitions[1].id++
					fixture.selectedPropSyncChains = [][]uint32{{fixture.partitions[0].id, fixture.partitions[1].id}}
					agentCounts[index], requestCounts[index] = 1, 1
				}
				snapshots[index] = acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
			}
			if test.family == BuildFamilyXT && test.packed[0] == test.packed[1] {
				snapshots[1] = snapshots[0]
			}

			t.Log("Build a selected SpecDecode DGD for the fixture's manifest generation")
			dgd := newSelectedTestDGD(t, "specdecode",
				testLPXComponent("lpx", "target-build", v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor-runtime")}, v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("lpu-runtime")}),
				testLPXComponent("small", "draft-build", v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("lpu-runtime")}),
			)
			dgd.Spec.Components[1].Replicas = new(int32(2))
			dgd.Spec.Components[1].ComponentRole(v1beta1.ComponentRoleLPXAgent).Replicas = new(agentCounts[0])
			source := staticModelRegistry{
				"draft-build":  snapshots[0],
				"target-build": snapshots[1],
			}

			t.Log("Project the selected SpecDecode workload")
			before := dgd.DeepCopy()
			selected, err := ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), source)

			t.Log("Project the selected family, workload mode, and pipeline")
			require.NoError(t, err)
			require.Equal(t, before, dgd, "canonical ordering must not rewrite the authored target-first list")
			request := selected.modelProjections[0].RequestSpec(&MaterializationPlan{}, "agents")
			require.Equal(t, lpxv1alpha1.TargetFamily(test.family), request.TargetFamily)
			require.Equal(t, test.wantMode, request.WorkloadMode)
			require.Equal(t, PipelineSpecDecode, selected.Pipeline())
			require.Equal(t, "lpx", selected.ServingComponentName())

			t.Log("Expand draft fanout while preserving model and template identity order")
			projections := selected.ModelProjections()
			require.Len(t, projections, 3)
			require.Equal(
				t,
				[]string{"draft0", "draft1", "target"},
				[]string{projections[0].Model(), projections[1].Model(), projections[2].Model()},
			)
			plan, err := selected.PlanNodeLocalMaterialization("test-pcs", dgd.GetComponentByName(selected.ServingComponentName()))
			require.NoError(t, err)
			require.EqualValues(t, 1, plan.Replicas, "draft fanout must not become the shared scaling-group axis")
			require.Equal(t, "small", projections[0].stage)
			require.Equal(t, "lpx", projections[2].stage)
			require.Equal(
				t,
				[]string{"agt0", "agt1", "agt2"},
				[]string{
					plan.Agents[0].TemplateName,
					plan.Agents[1].TemplateName,
					plan.Agents[2].TemplateName,
				},
				"template identities follow canonical stage and draft-instance order",
			)

			t.Log("Reordering authored components does not change build identities or resources")
			reordered := dgd.DeepCopy()
			slices.Reverse(reordered.Spec.Components)
			members := singleGroupComponents(t, reordered)
			slices.Reverse(members)
			reselected, err := ResolveWorkload(t.Context(), reader, reordered, members, source)
			require.NoError(t, err)
			require.Equal(t, selected.Digest(), reselected.Digest())
			reorderedPlan, err := reselected.PlanNodeLocalMaterialization("test-pcs", reordered.GetComponentByName(reselected.ServingComponentName()))
			require.NoError(t, err)
			require.Equal(t, plan, reorderedPlan)
			require.Equal(t, []string{"small", "lpx"}, reselected.ComponentNames())

			t.Log("Agent replica assertions count one compiled model instance, not draft fanout")
			invalidCount := dgd.DeepCopy()
			invalidCount.Spec.Components[1].ComponentRole(v1beta1.ComponentRoleLPXAgent).Replicas = new(agentCounts[0] * 2)
			_, err = ResolveWorkload(t.Context(), reader, invalidCount, singleGroupComponents(t, invalidCount), source)
			require.ErrorContains(t, err, "must match the compiled count")

			t.Log("Derive an aggregate digest and reject mixed-family aggregation")
			require.NotEqual(t, WorkloadDigest{}, selected.Digest())
			require.NotEqual(t, projections[0].Digest(), selected.Digest())
			mixedFamily := *projections[2]
			mixedFamily.configuredBuild.family = BuildFamily("other")
			_, err = workloadSetDigest([]*ModelProjection{projections[0], &mixedFamily})
			require.ErrorContains(t, err, "mixed target families")

			t.Log("Preserve compiled placement without repeating runtime-derived model settings")
			for index, projection := range projections {
				original := snapshots[index/2].partitions
				require.EqualValues(t, agentCounts[index/2], projection.agentReplicas)
				require.EqualValues(t, agentCounts[index/2], plan.Agents[index].Replicas)
				request := projection.RequestSpec(plan, plan.Agents[index].TemplateName)
				require.Len(t, request.Partitions, requestCounts[index/2])
				require.EqualValues(t, original[0].sourcePartitionID, request.Partitions[0].CompilerPartitionID)
				require.Equal(t, projection.Model(), request.NodeLocal.Model)
				if test.packed[index/2] {
					require.Empty(t, request.PropSyncConnectors)
				}
				require.Equal(t, original, projection.configuredBuild.partitions)
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
						draft.Replicas = new(expansion.count)
					}
					expanded, err := ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), source)
					require.NoError(t, err)
					models := make([]string, 0, len(expansion.models))
					for _, projection := range expanded.ModelProjections() {
						models = append(models, projection.Model())
					}

					t.Log("Preserve the expected logical model ordering")
					require.Equal(t, expansion.models, models)
				})
			}

			if test.family == BuildFamilyHX {
				t.Log("Project separate draft and target roles from the same immutable HX build")
				draft := &dgd.Spec.Components[1]
				draft.Replicas = nil
				draft.LPX.BuildID = "target-build"
				shared, err := ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), source)
				require.NoError(t, err)
				projections := shared.ModelProjections()
				require.Len(t, projections, 2)
				require.Equal(t, []string{"draft0", "target"}, []string{projections[0].Model(), projections[1].Model()})
				require.Equal(t, "target-build", projections[0].runtimeBuildRef)
				require.Equal(t, "target-build", projections[1].runtimeBuildRef)
				require.NotEqual(t, projections[0].Digest(), projections[1].Digest())
			}
		})
	}
}

func TestResolveWorkloadPreservesAuthoredLaunch(t *testing.T) {
	t.Log("Author independent wrappers without exposing launch syntax to the operator")
	reader := fake.NewClientBuilder().Build()
	conductor := testLPXPodTemplate("conductor-runtime")
	conductor.Spec.Containers[0].Command = []string{"/bin/sh", "-c"}
	conductor.Spec.Containers[0].Args = []string{"exec custom-conductor --workers \"$LPX_ALLOCATION\"", "--"}
	agent := testLPXPodTemplate("agent-runtime")
	agent.Spec.Containers[0].Command = []string{"/custom-worker", "--"}
	agent.Spec.Containers[0].Args = []string{"--allocation=application-owned"}
	dgd := newSelectedTestDGD(t, "selected", testLPXComponent("lpx", "build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: agent},
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: conductor},
	))
	snapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	before := dgd.DeepCopy()

	t.Log("Validate placement and role ownership without parsing either command line")
	_, err := ResolveWorkload(t.Context(), reader, dgd, singleGroupComponents(t, dgd), staticModelRegistry{"build": snapshot})
	require.NoError(t, err)
	require.Equal(t, before, dgd)
}

func TestResolveWorkloadPlacesRuntimeSidecarWithTheEngine(t *testing.T) {
	t.Log("Acquire single-Pod HX, two-Pod HX, four-Pod XT and hybrid builds")
	multiPodHX := newV3CompilerFixture()
	multiPodHX.selectedPropSyncChains = [][]uint32{{1, 2}}
	multiPodHX.partitions = append(multiPodHX.partitions, testV3CapnpPartition{
		id: 2, deviceType: manifestcapnp.DeviceType_lpu, topology: v3OpaqueTopology, numChips: 16, devicesPerNode: 16,
	})
	snapshots := map[string]*Build{
		"hx":           acquireTestSnapshot(t, writeV3CompilerFixture(t)),
		"multi-pod-hx": acquireTestSnapshot(t, writeCompilerFixture(t, multiPodHX)),
		"multi-pod-xt": acquireTestSnapshot(t, writeV2CompilerFixture(t)),
		"hybrid":       acquireHybridTestSnapshot(t),
	}

	t.Log("Cover the multi-Pod agent placement and the builds whose engine runs elsewhere")
	for _, test := range []struct {
		name        string
		build       string
		sidecarRole string
		wantAgents  int
		wantErr     string
	}{
		{name: "agent sidecar serves a multi-Pod XT build", build: "multi-pod-xt", sidecarRole: v1beta1.ComponentRoleLPXAgent, wantAgents: 4},
		{name: "agent sidecar serves a multi-Pod HX build", build: "multi-pod-hx", sidecarRole: v1beta1.ComponentRoleLPXAgent, wantAgents: 2},
		{name: "agent sidecar rejects a hybrid build", build: "hybrid", sidecarRole: v1beta1.ComponentRoleLPXAgent,
			wantErr: `component "lpx" agent "runtime" sidecar requires an LPU-only single-model build; declare the conductor role instead`},
		{name: "conductor sidecar rejects an LPU-only build", build: "hx", sidecarRole: v1beta1.ComponentRoleLPXConductor,
			wantErr: `component "lpx" conductor Dynamo sidecar mode requires a hybrid build whose conductor runs the GPU engine`},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Author the runtime sidecar, behind a setup init container, in the selected role")
			agent, conductor := testLPXPodTemplate("agent"), testLPXPodTemplate("conductor")
			roles := []v1beta1.ComponentRoleSpec{{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: agent}}
			sidecarTemplate := agent
			if test.sidecarRole == v1beta1.ComponentRoleLPXConductor {
				roles = append(roles, v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: conductor})
				sidecarTemplate = conductor
			}
			sidecarTemplate.Spec.InitContainers = []corev1.Container{
				{Name: "setup", Image: "setup"},
				{Name: commonconsts.RuntimeContainerName, Image: "sidecar", RestartPolicy: ptr.To(corev1.ContainerRestartPolicyAlways)},
			}
			dgd := newSelectedTestDGD(t, "selected", testLPXComponent("lpx", "build", roles...))
			before := dgd.DeepCopy()

			t.Log("Resolve the build without mutating the graph")
			workload, err := ResolveWorkload(t.Context(), fake.NewClientBuilder().Build(), dgd, singleGroupComponents(t, dgd), staticModelRegistry{"build": snapshots[test.build]})
			require.Equal(t, before, dgd)
			if test.wantErr != "" {
				require.ErrorIs(t, err, ErrUnsupportedRuntime)
				require.EqualError(t, err, "unsupported LPX runtime: "+test.wantErr)
				return
			}
			require.NoError(t, err)

			t.Log("Serve from every Agent Pod the build places")
			plan, err := workload.PlanNodeLocalMaterialization("pcs", dgd.GetComponentByName(workload.ServingComponentName()))
			require.NoError(t, err)
			require.Empty(t, plan.ConductorTemplate)
			require.Equal(t, []ExpectedAgent{{TemplateName: "agt", Replicas: test.wantAgents}}, plan.Agents)
		})
	}
}

func TestResolveWorkloadIsolatesComponentGroups(t *testing.T) {
	t.Log("Author two LPU-only workloads with different builds and replica counts")
	reader := fake.NewClientBuilder().Build()
	dgd := newSelectedTestDGD(t, "graph", v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend,
	})
	for index, name := range []string{"first", "second"} {
		component := testLPXComponent(name, name+"-build",
			v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("agent")},
			v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor")},
		)
		component.Replicas = new(int32(index + 2))
		dgd.Spec.Components = append(dgd.Spec.Components, component)
	}
	snapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	before := dgd.DeepCopy()

	t.Log("Resolve each group using only its own build, replicas and component identity")
	groups := ComponentGroups(dgd)
	for index, name := range []string{"first", "second"} {
		workload, err := ResolveWorkload(t.Context(), reader, dgd, groups[name], staticModelRegistry{name + "-build": snapshot})
		require.NoError(t, err)
		require.Equal(t, PipelineSingle, workload.Pipeline())
		require.Equal(t, name, workload.ServingComponentName())
		require.Equal(t, []string{name}, workload.ComponentNames())
		plan, err := workload.PlanNodeLocalMaterialization("pcs", dgd.GetComponentByName(workload.ServingComponentName()))
		require.NoError(t, err)
		require.EqualValues(t, index+2, plan.Replicas)
	}
	require.Equal(t, before, dgd)

	t.Log("Require a singleton conductor independently of component replica counts")
	conductor := dgd.GetComponentByName("second").ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.Replicas = new(int32(2))
	_, err := ResolveWorkload(t.Context(), reader, dgd, groups["second"], staticModelRegistry{"second-build": snapshot})
	require.ErrorContains(t, err, `component "second" conductor replicas must be one`)
}

func TestExpandedModelNames(t *testing.T) {
	t.Log("Expand admitted component replicas into runtime model names")
	require.Equal(t, []string{"draft0", "draft1", "draft2"}, expandedModelNames(2, false, 3))
	require.Equal(t, []string{"draft0"}, expandedModelNames(2, false, 1))
	require.Equal(t, []string{"target"}, expandedModelNames(2, true, 1))
	require.Equal(t, []string{"default"}, expandedModelNames(1, true, 2))
}

// acquireHybridTestSnapshot acquires a revision-2 build whose CUDA partition selects a Cyborg conductor.
func acquireHybridTestSnapshot(t *testing.T) *Build {
	t.Helper()
	fixture := newV2CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	fixture.selectedPropSyncChains = nil
	fixture.partitions = append(fixture.partitions, testV3CapnpPartition{id: 11, deviceType: manifestcapnp.DeviceType_cuda})
	return acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
}
