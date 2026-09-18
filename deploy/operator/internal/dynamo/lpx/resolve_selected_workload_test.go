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
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

func TestResolveSelectedWorkloadDerivesRuntimeShapeFromCompilationMode(t *testing.T) {
	t.Log("Create one LPX component beside an unrelated conventional decode")
	dgd := newSelectedTestDGD(t, "graph", testLPXComponent("LPX", "build", v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("lpu-runtime")}, v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor-runtime")}))
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
	require.Equal(t, lpxv1alpha1.WorkloadModeV3HxLPUOnly, hx.modelProjections[0].RequestSpec(&MaterializationPlan{}, "agents").WorkloadMode)
	require.Equal(t, "LPX", hx.LPXComponentName())
	plan, err := hx.PlanNodeLocalMaterialization("test-pcs")
	require.NoError(t, err)
	require.Equal(t, "test-pcs-0-lpx", plan.LPXScalingGroup)
	require.Equal(t, "cond", plan.ConductorTemplate)
	require.Empty(t, plan.CyborgTemplate)
	require.NotEmpty(t, plan.ConductorClique)

	t.Log("Scale Nova engines without changing their model or workload digest")
	for _, replicas := range []int32{2, 10, 12, 123} {
		dgd.Spec.Components[0].Replicas = ptr.To(replicas)
		scaled, err := ResolveSelectedWorkload(t.Context(), dgd, staticBuildSnapshotSource{"build": hxSnapshot})
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
	source := staticBuildSnapshotSource{"build": snapshot}
	_, err = ResolveSelectedWorkload(t.Context(), dgd, source)
	require.ErrorContains(t, err, "requires a declared resourceClaim or a positive nvidia.com/gpu request")
	conductor := dgd.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor)
	conductor.PodTemplate = testLPXPodTemplate("cyborg-runtime")
	_, err = ResolveSelectedWorkload(t.Context(), dgd, source)
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
			_, err := ResolveSelectedWorkload(t.Context(), candidate, source)
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
	xt, err := ResolveSelectedWorkload(t.Context(), dgd, source)
	require.NoError(t, err)
	require.Equal(t, PipelineLPX, xt.Pipeline())
	require.Equal(t, BuildFamilyXT, xt.BuildFamily())
	require.Equal(t, lpxv1alpha1.WorkloadModeV2StrictHybrid, xt.modelProjections[0].RequestSpec(&MaterializationPlan{}, "agents").WorkloadMode)
	require.Len(t, xt.modelProjections[0].configuredBuild.Partitions, 2)
	plan, err = xt.PlanNodeLocalMaterialization("test-pcs")
	require.NoError(t, err)
	require.Equal(t, "cond", plan.CyborgTemplate)
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
			if test.family == BuildFamilyHX {
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
			if test.family == BuildFamilyXT {
				compiledAgentCount = 4
			}
			dgd.Spec.Components[1].ComponentRole(v1beta1.ComponentRoleLPXAgent).Replicas = ptr.To(compiledAgentCount)
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
			require.Equal(t, test.wantMode, selected.modelProjections[0].RequestSpec(&MaterializationPlan{}, "agents").WorkloadMode)
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
			reselected, err := ResolveSelectedWorkload(t.Context(), reordered, source)
			require.NoError(t, err)
			require.Equal(t, selected.Digest(), reselected.Digest())
			reorderedPlan, err := reselected.PlanNodeLocalMaterialization("test-pcs")
			require.NoError(t, err)
			require.Equal(t, plan, reorderedPlan)

			t.Log("Agent replica assertions count one compiled model instance, not draft fanout")
			invalidCount := dgd.DeepCopy()
			invalidCount.Spec.Components[1].ComponentRole(v1beta1.ComponentRoleLPXAgent).Replicas = ptr.To(compiledAgentCount * 2)
			_, err = ResolveSelectedWorkload(t.Context(), invalidCount, source)
			require.ErrorContains(t, err, "must match the compiled count")

			t.Log("Derive an aggregate digest and reject mixed-family aggregation")
			require.NotEqual(t, WorkloadDigest{}, selected.Digest())
			require.NotEqual(t, projections[0].Digest(), selected.Digest())
			mixedFamily := *projections[2]
			mixedFamily.configuredBuild.Family = BuildFamily("other")
			_, err = workloadSetDigest([]*ModelProjection{projections[0], &mixedFamily})
			require.ErrorContains(t, err, "mixed target families")

			t.Log("Preserve compiled placement without repeating runtime-derived model settings")
			for _, projection := range projections {
				require.EqualValues(t, compiledAgentCount, projection.agentReplicas)
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

func TestResolveSelectedWorkloadPreservesAuthoredLaunch(t *testing.T) {
	t.Log("Author independent wrappers without exposing launch syntax to the operator")
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
	_, err := ResolveSelectedWorkload(t.Context(), dgd, staticBuildSnapshotSource{"build": snapshot})
	require.NoError(t, err)
	require.Equal(t, before, dgd)
}

func TestResolveSelectedWorkloadRejectsInvalidRolesBeforeBuildAcquisition(t *testing.T) {
	t.Log("Author placement and container identity errors on both roles")
	agent, conductor := testLPXPodTemplate("agent"), testLPXPodTemplate("conductor")
	agent.Spec.Containers[0].Command = []string{"/bin/sh", "-c"}
	agent.Spec.Containers[0].Args = []string{"--allocation=forged", "--"}
	agent.Spec.Hostname = "custom-host"
	agent.Spec.Containers = append(agent.Spec.Containers, corev1.Container{Name: "agent", Image: "sidecar"})
	agent.Spec.InitContainers = append(agent.Spec.InitContainers, corev1.Container{Name: "agent", Image: "setup"})
	conductor.Spec.NodeName = "chosen-node"
	conductor.Spec.TopologySpreadConstraints = []corev1.TopologySpreadConstraint{{
		MaxSkew: 1, TopologyKey: "zone", WhenUnsatisfiable: corev1.DoNotSchedule,
	}}
	dgd := newSelectedTestDGD(t, "selected", testLPXComponent("lpx", "build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: agent},
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: conductor},
	))

	t.Log("Aggregate every actionable error at its authored path before acquiring a build")
	_, err := ResolveSelectedWorkload(t.Context(), dgd, unreachableBuildSnapshotSource{})
	for _, message := range []string{
		"spec.components[0].roles[0].podTemplate.spec.hostname: Forbidden: LPX owns role addressing and placement",
		`spec.components[0].roles[0].podTemplate.spec.containers[1].name: Forbidden: LPX reserves "agent" for the materialized role container`,
		`spec.components[0].roles[0].podTemplate.spec.initContainers[0].name: Forbidden: LPX reserves "agent" for the materialized role container`,
		"spec.components[0].roles[1].podTemplate.spec.nodeName: Forbidden: LPX owns role addressing and placement",
		"spec.components[0].roles[1].podTemplate.spec.topologySpreadConstraints: Forbidden: LPX owns role placement",
	} {
		require.ErrorContains(t, err, message)
	}

	t.Log("Reject missing main containers for both roles before acquiring a build")
	agent.Spec.Containers, conductor.Spec.Containers = nil, nil
	_, err = ResolveSelectedWorkload(t.Context(), dgd, unreachableBuildSnapshotSource{})
	require.ErrorContains(t, err, `spec.components[0].roles[0].podTemplate.spec.containers: Required value: LPX agent component requires a "main" runtime container`)
	require.ErrorContains(t, err, `spec.components[0].roles[1].podTemplate.spec.containers: Required value: LPX conductor component requires a "main" runtime container`)
}

func TestResolveSelectedWorkloadChecksConductorContainerNamesForSelectedBuild(t *testing.T) {
	t.Log("Acquire LPU-only and hybrid builds that select different runtime container identities")
	lpuSnapshot := acquireTestSnapshot(t, writeV3CompilerFixture(t))
	hybridFixture := newV2CompilerFixture()
	hybridFixture.compilationMode = manifestcapnp.CompilationMode_lpx
	hybridFixture.selectedPropSyncChains = nil
	hybridFixture.partitions = append(hybridFixture.partitions, testV3CapnpPartition{id: 11, deviceType: manifestcapnp.DeviceType_cuda})
	hybridSnapshot := acquireTestSnapshot(t, writeCompilerFixture(t, hybridFixture))

	t.Log("Cover independent serving, draft and hybrid template ownership")
	tests := []struct {
		name                 string
		conductor            *v1beta1.ComponentRoleSpec
		addDraft             bool
		containerInDraft     bool
		containerInConductor bool
		hybrid               bool
		wantForbidden        bool
	}{
		{
			name: "independent conductor template",
			conductor: &v1beta1.ComponentRoleSpec{
				Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor"),
			},
		},
		{
			name: "explicit LPU-only conductor collision",
			conductor: &v1beta1.ComponentRoleSpec{
				Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor"),
			},
			containerInConductor: true,
			wantForbidden:        true,
		},
		{
			name:             "draft does not supply the conductor template",
			conductor:        &v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("conductor")},
			addDraft:         true,
			containerInDraft: true,
		},
		{
			name: "hybrid with explicit conductor template",
			conductor: &v1beta1.ComponentRoleSpec{
				Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: testLPXPodTemplate("cyborg"),
			},
			containerInConductor: true,
			hybrid:               true,
		},
	}

	for _, test := range tests {
		for _, containerList := range []string{"containers", "initContainers"} {
			t.Run(test.name+"/"+containerList, func(t *testing.T) {
				t.Log("Author the serving component and its optional independent draft")
				target := testLPXComponent("target", "build",
					v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("agent")},
				)
				if test.conductor != nil {
					target.Roles = append(target.Roles, *test.conductor.DeepCopy())
				}
				dgd := newSelectedTestDGD(t, "selected", target)
				if test.addDraft {
					dgd.Spec.Components = append(dgd.Spec.Components, testLPXComponent("draft", "build",
						v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("draft")},
					))
				}

				t.Log("Provide GPU resources only when selecting a hybrid runtime")
				snapshot := lpuSnapshot
				if test.hybrid {
					snapshot = hybridSnapshot
					for _, role := range dgd.Spec.Components[0].Roles {
						if role.PodTemplate != nil {
							role.PodTemplate.Spec.Containers[0].Resources.Limits = corev1.ResourceList{
								corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
							}
						}
					}
				}

				t.Log("Add one conductor-named container to the selected authored role template")
				componentIndex := 0
				if test.containerInDraft {
					componentIndex = 1
				}
				roleIndex := 0
				if test.containerInConductor {
					roleIndex = 1
				}
				podSpec := &dgd.Spec.Components[componentIndex].Roles[roleIndex].PodTemplate.Spec
				container := corev1.Container{Name: "conductor", Image: "sidecar"}
				containerIndex := 0
				if containerList == "initContainers" {
					podSpec.InitContainers = append(podSpec.InitContainers, container)
				} else {
					containerIndex = len(podSpec.Containers)
					podSpec.Containers = append(podSpec.Containers, container)
				}
				before := dgd.DeepCopy()

				t.Log("Defer conductor-name checks until the immutable build has been acquired")
				require.Empty(t, ValidateAgentContainerNames(dgd))
				require.Empty(t, ValidateSelectedIntent(dgd))
				_, err := ResolveSelectedWorkload(t.Context(), dgd, unreachableBuildSnapshotSource{})
				require.ErrorIs(t, err, ErrBuildSnapshotAcquisition)

				t.Log("Reject only actual conductor collisions for the selected runtime")
				selected, err := ResolveSelectedWorkload(t.Context(), dgd, staticBuildSnapshotSource{"build": snapshot})
				if test.wantForbidden {
					namePath := field.NewPath("spec", "components").Index(componentIndex).Child("roles").Index(roleIndex).Child("podTemplate", "spec", containerList).Index(containerIndex).Child("name")
					want := field.Forbidden(namePath, `LPX reserves "conductor" for the materialized role container`)
					require.ErrorContains(t, err, want.Error())
					require.NotErrorIs(t, err, ErrBuildSnapshotAcquisition)
				} else {
					require.NoError(t, err)
					if test.hybrid {
						require.Equal(t, PipelineLPX, selected.Pipeline())
					}
				}
				require.Equal(t, before, dgd)
			})
		}
	}
}
