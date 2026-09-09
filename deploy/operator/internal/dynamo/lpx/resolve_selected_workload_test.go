/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"slices"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/capnp/gbuild_manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/utils/ptr"
)

func TestResolveSelectedWorkloadDerivesRuntimeShapeFromCompilationMode(t *testing.T) {
	t.Log("Create one LPX component beside an unrelated conventional decode")
	dgd := newSelectedTestDGD(t, "graph", testLPXComponent("LPX", "build", v1beta1.ComponentRoleSpec{Name: "worker", PodTemplate: testLPXPodTemplate("lpu-runtime")}))
	dgd.Annotations = nil
	dgd.Spec.Components = append(dgd.Spec.Components, v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "ordinary-decode", ComponentType: v1beta1.ComponentTypeDecode,
		Replicas:    ptr.To(int32(0)),
		PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "ordinary"}}}},
	})

	t.Log("Resolve an LPU-only engine without selecting the conventional decode")
	hxSnapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	hx, err := ResolveSelectedWorkload(t.Context(), dgd, staticBuildSnapshotSource{"build": hxSnapshot})
	require.NoError(t, err)
	require.Equal(t, PipelineSingle, hx.Pipeline())
	require.Equal(t, BuildFamilyHX, hx.BuildFamily())
	require.Equal(t, lpxv1alpha1.WorkloadModeV3HxLPUOnly, hx.modelProjections[0].RequestSpec("test", "agents", nil).WorkloadMode)
	require.Empty(t, hx.CyborgTemplateName())
	require.Equal(t, "LPX", hx.LPXComponentName())
	plan, err := hx.PlanNodeLocalMaterialization("test-pcs")
	require.NoError(t, err)
	require.Equal(t, "lpx", plan.LPXScalingGroupTemplate)
	require.Equal(t, "lpx-ldr", plan.ConductorTemplate)
	require.NotEmpty(t, plan.ConductorClique)

	t.Log("Require hybrid conductor resources in either the explicit or inherited template")
	fixture := newV2CompilerFixture()
	fixture.compilationMode = manifestcapnp.CompilationMode_lpx
	fixture.selectedPropSyncChains = nil
	fixture.partitions = append(fixture.partitions, testV3CapnpPartition{id: 11, deviceType: manifestcapnp.DeviceType_cuda})
	snapshot := acquireTestSnapshot(t, writeCompilerFixture(t, fixture))
	source := staticBuildSnapshotSource{"build": snapshot}
	_, err = ResolveSelectedWorkload(t.Context(), dgd, source)
	require.ErrorContains(t, err, "requires resourceClaims or a positive nvidia.com/gpu request")
	dgd.Spec.Components[0].Roles = append(dgd.Spec.Components[0].Roles, v1beta1.ComponentRoleSpec{Name: "leader"})
	conductor := dgd.Spec.Components[0].ComponentRole("leader")
	conductor.PodTemplate = testLPXPodTemplate("cyborg-runtime")
	_, err = ResolveSelectedWorkload(t.Context(), dgd, source)
	require.ErrorContains(t, err, "requires resourceClaims or a positive nvidia.com/gpu request")
	conductor.PodTemplate.Spec.Containers[0].Resources.Limits = corev1.ResourceList{
		corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
	}
	dgd.Spec.Components[0].Replicas = ptr.To(int32(2))

	t.Log("Project two complete hybrid replicas")
	xt, err := ResolveSelectedWorkload(t.Context(), dgd, source)
	require.NoError(t, err)
	require.Equal(t, PipelineLPX, xt.Pipeline())
	require.Equal(t, BuildFamilyXT, xt.BuildFamily())
	require.Equal(t, lpxv1alpha1.WorkloadModeV2StrictHybrid, xt.modelProjections[0].RequestSpec("test", "agents", nil).WorkloadMode)
	require.Equal(t, "lpx-engine-gpu", xt.CyborgTemplateName())
	require.Len(t, xt.modelProjections[0].configuredBuild.Partitions, 2)
	plan, err = xt.PlanNodeLocalMaterialization("test-pcs")
	require.NoError(t, err)
	require.EqualValues(t, 2, plan.Replicas)
	replica := plan.ForReplica(1)
	require.NotEqual(t, plan.Agents[0].CliqueName, replica.Agents[0].CliqueName)

	t.Log("A scheduling deadline and annotation do not change hybrid launch")
	dgd.Spec.Scheduling = &v1beta1.SchedulingSpec{AttemptDeadlineSeconds: ptr.To(int64(30))}
	dgd.Annotations = map[string]string{commonconsts.KubeAnnotationLPXSchedulerBackend: commonconsts.LPXSchedulerBackend}
	scheduled, err := ResolveSelectedWorkload(t.Context(), dgd, source)
	require.NoError(t, err)
	require.Equal(t, xt, scheduled)
	scheduledPlan, err := scheduled.PlanNodeLocalMaterialization("test-pcs")
	require.NoError(t, err)
	require.Equal(t, plan, scheduledPlan)
	require.Empty(t, scheduledPlan.ConductorTemplate)

	t.Log("Use the same agent template when the standalone conductor role is omitted")
	agent := *dgd.Spec.Components[0].ComponentRole("worker")
	agent.PodTemplate = conductor.PodTemplate.DeepCopy()
	dgd.Spec.Components[0].Roles = []v1beta1.ComponentRoleSpec{agent}
	before := dgd.DeepCopy()
	implicit, err := ResolveSelectedWorkload(t.Context(), dgd, source)
	require.NoError(t, err)
	require.Equal(t, xt, implicit)
	require.Same(t, &dgd.Spec.Components[0], ServingComponent(dgd))
	require.Equal(t, before, dgd)
}

func TestResolveSelectedWorkloadSpecDecodeV2AndV3(t *testing.T) {
	t.Log("Define revision-specific SpecDecode compiler snapshots")
	tests := []struct {
		name     string
		family   BuildFamily
		wantMode lpxv1alpha1.WorkloadMode
	}{
		{
			name: "v2", family: BuildFamilyXT,
			wantMode: lpxv1alpha1.WorkloadModeV2LPUOnly,
		},
		{
			name: "v3", family: BuildFamilyHX,
			wantMode: lpxv1alpha1.WorkloadModeV3HxLPUOnly,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Acquire the selected family fixtures: shared XT build or distinct HX snapshots")
			var draftSnapshot, targetSnapshot *BuildSnapshot
			var settings *apiextensionsv1.JSON
			if test.family == BuildFamilyHX {
				draftSnapshot = acquireTestSnapshot(t, writeV3CompilerFixture(t))
				targetSnapshot = acquireTestSnapshot(t, writeV3CompilerFixture(t))
			} else {
				draftSnapshot = acquireTestSnapshot(t, writeV2CompilerFixture(t))
				targetSnapshot = draftSnapshot
				settings = &apiextensionsv1.JSON{Raw: []byte(`{ "prop_sync": true }`)}
			}

			t.Log("Build a selected SpecDecode DGD for the fixture's manifest generation")
			dgd := newSelectedTestDGD(t, "specdecode",
				testLPXComponent("lpx", "target-build", v1beta1.ComponentRoleSpec{Name: "leader"}, v1beta1.ComponentRoleSpec{Name: "worker", PodTemplate: testLPXPodTemplate("lpu-runtime")}),
				testLPXComponent("small", "draft-build", v1beta1.ComponentRoleSpec{Name: "worker", PodTemplate: testLPXPodTemplate("lpu-runtime")}),
			)
			dgd.Spec.Components[0].LPX.Settings = settings
			dgd.Spec.Components[1].LPX.Settings = settings.DeepCopy()
			dgd.Spec.Components[1].Replicas = ptr.To(int32(2))
			compiledAgentCount := int32(1)
			if test.family == BuildFamilyXT {
				compiledAgentCount = 4
			}
			dgd.Spec.Components[1].ComponentRole("worker").Replicas = ptr.To(compiledAgentCount)
			source := staticBuildSnapshotSource{
				"draft-build":  draftSnapshot,
				"target-build": targetSnapshot,
			}

			t.Log("Project the selected SpecDecode workload")
			before := dgd.DeepCopy()
			selected, err := ResolveSelectedWorkload(t.Context(), dgd, source)

			t.Log("Project the selected family, workload mode, and pipeline")
			require.NoError(t, err)
			require.Equal(t, before, dgd, "canonical ordering must not rewrite the authored target-first list")
			require.Equal(t, test.family, selected.BuildFamily())
			require.Equal(t, test.wantMode, selected.modelProjections[0].RequestSpec("test", "agents", nil).WorkloadMode)
			require.Equal(t, PipelineSpecDecode, selected.Pipeline())
			require.Equal(t, "lpx", selected.LPXComponentName())

			t.Log("Expand draft fanout while preserving model and template identity order")
			projections := selected.ModelProjections()
			require.Len(t, projections, 3)
			require.Equal(
				t,
				[]string{"draft0", "draft1", "target"},
				[]string{projections[0].Model(), projections[1].Model(), projections[2].Model()},
			)
			plan, err := selected.PlanNodeLocalMaterialization("test-pcs")
			require.NoError(t, err)
			require.Equal(t, "small", projections[0].stage)
			require.Equal(t, "lpx", projections[2].stage)
			require.Equal(
				t,
				[]string{"lpx-wkr-m-0", "lpx-wkr-m-1", "lpx-wkr-m-2"},
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
			reselected, err := ResolveSelectedWorkload(t.Context(), reordered, source)
			require.NoError(t, err)
			require.Equal(t, selected.Digest(), reselected.Digest())
			reorderedPlan, err := reselected.PlanNodeLocalMaterialization("test-pcs")
			require.NoError(t, err)
			require.Equal(t, plan, reorderedPlan)

			t.Log("Agent replica assertions count one compiled model instance, not draft fanout")
			invalidCount := dgd.DeepCopy()
			invalidCount.Spec.Components[1].ComponentRole("worker").Replicas = ptr.To(compiledAgentCount * 2)
			_, err = ResolveSelectedWorkload(t.Context(), invalidCount, source)
			require.ErrorContains(t, err, "must match the compiled count")

			t.Log("Derive an aggregate digest and reject mixed-family aggregation")
			require.NotEqual(t, WorkloadDigest{}, selected.Digest())
			require.NotEqual(t, projections[0].Digest(), selected.Digest())
			mixedFamily := *projections[2]
			mixedFamily.configuredBuild.Family = BuildFamily("other")
			_, err = workloadSetDigest([]*ModelProjection{projections[0], &mixedFamily})
			require.ErrorContains(t, err, "mixed target families")

			t.Log("Preserve family-specific runtime settings in every logical projection")
			for _, projection := range projections {
				require.EqualValues(t, compiledAgentCount, projection.agentReplicas)
				if test.family == BuildFamilyHX {
					require.EqualValues(t, 8192, projection.configuredBuild.runtimeSettings["sequence_length"])
				}
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
					expanded, err := ResolveSelectedWorkload(t.Context(), dgd, source)
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
				shared, err := ResolveSelectedWorkload(t.Context(), dgd, source)
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

func TestResolveConductorLaunchErrorUsesAuthoredRolePath(t *testing.T) {
	t.Log("Place the target first and author a conductor argument owned by the operator")
	conductor := testLPXPodTemplate("conductor-runtime")
	conductor.Spec.Containers[0].Args = []string{"--allocation=forged"}
	dgd := newSelectedTestDGD(t, "specdecode",
		testLPXComponent("large", "build", v1beta1.ComponentRoleSpec{Name: "worker", PodTemplate: testLPXPodTemplate("target-runtime")}, v1beta1.ComponentRoleSpec{Name: "leader", PodTemplate: conductor}),
		testLPXComponent("small", "build", v1beta1.ComponentRoleSpec{Name: "worker", PodTemplate: testLPXPodTemplate("draft-runtime")}),
	)
	snapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	before := dgd.DeepCopy()

	t.Log("Report the authored target and conductor indices after canonical build selection")
	_, err := ResolveSelectedWorkload(t.Context(), dgd, staticBuildSnapshotSource{"build": snapshot})
	require.ErrorContains(t, err, "spec.components[0].roles[1].podTemplate.spec.containers[0].args")
	require.ErrorContains(t, err, "must not set --allocation")
	require.Equal(t, before, dgd)
}

func TestResolveSelectedWorkloadRejectsInvalidRolesBeforeBuildAcquisition(t *testing.T) {
	t.Log("Author independent launch and placement errors on both roles")
	agent, conductor := testLPXPodTemplate("agent"), testLPXPodTemplate("conductor")
	agent.Spec.Containers[0].Command = []string{"/bin/sh", "-c"}
	agent.Spec.Containers[0].Args = []string{"--allocation=forged", "--"}
	agent.Spec.Hostname = "custom-host"
	conductor.Spec.NodeName = "chosen-node"
	conductor.Spec.TopologySpreadConstraints = []corev1.TopologySpreadConstraint{{
		MaxSkew: 1, TopologyKey: "zone", WhenUnsatisfiable: corev1.DoNotSchedule,
	}}
	dgd := newSelectedTestDGD(t, "selected", testLPXComponent("lpx", "build",
		v1beta1.ComponentRoleSpec{Name: "worker", PodTemplate: agent},
		v1beta1.ComponentRoleSpec{Name: "leader", PodTemplate: conductor},
	))

	t.Log("Aggregate every actionable error at its authored path before acquiring a build")
	_, err := ResolveSelectedWorkload(t.Context(), dgd, unreachableBuildSnapshotSource{})
	for _, message := range []string{
		"spec.components[0].roles[0].podTemplate.spec.containers[0].args: Forbidden: selected LPX main container must not set --allocation; Dynamo renders the immutable Agent clique list",
		"spec.components[0].roles[0].podTemplate.spec.containers[0].args: Forbidden: selected LPX main container must not terminate arguments before Dynamo appends --allocation",
		"spec.components[0].roles[0].podTemplate.spec.containers[0].command: Forbidden: selected LPX main container cannot use a shell because Dynamo appends --allocation",
		"spec.components[0].roles[0].podTemplate.spec.hostname: Forbidden: LPX owns role addressing and placement",
		"spec.components[0].roles[1].podTemplate.spec.nodeName: Forbidden: LPX owns role addressing and placement",
		"spec.components[0].roles[1].podTemplate.spec.topologySpreadConstraints: Forbidden: LPX owns role placement",
	} {
		require.ErrorContains(t, err, message)
	}

	t.Log("Reject missing main containers for both roles before acquiring a build")
	agent.Spec.Containers, conductor.Spec.Containers = nil, nil
	_, err = ResolveSelectedWorkload(t.Context(), dgd, unreachableBuildSnapshotSource{})
	require.ErrorContains(t, err, `spec.components[0].roles[0].podTemplate.spec.containers: Required value: LPX worker component requires a "main" runtime container`)
	require.ErrorContains(t, err, `spec.components[0].roles[1].podTemplate.spec.containers: Required value: LPX leader component requires a "main" runtime container`)
}
