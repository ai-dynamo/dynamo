/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// lastWorkloadReplica is the highest workload replica index that LPX supports.
const lastWorkloadReplica = 2495

// testServingLPXComponent constructs a component of the build "build" that serves its endpoint:
// from the conductor of a hybrid build, or from the Agent Pods of an LPU-only build.
func testServingLPXComponent(name string, hybrid bool) v1beta1.DynamoComponentDeploymentSharedSpec {
	if !hybrid {
		return testLPXComponent(name, "build",
			v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXSidecarTemplate("lpu-runtime")})
	}
	conductor := testLPXSidecarTemplate("gpu-runtime")
	conductor.Spec.Containers[0].Resources.Limits = corev1.ResourceList{
		corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
	}
	return testLPXComponent(name, "build",
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("lpu-runtime")},
		v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXConductor, PodTemplate: conductor})
}

// lastPodHostname returns the hostname that Grove gives the last Pod of a clique.
func lastPodHostname(cliqueName string, pods int) string {
	return fmt.Sprintf("%s-%d", cliqueName, pods-1)
}

// groveNameLength returns the length that Grove counts against its combined name limit for
// a clique of one workload replica: the PodCliqueSet, scaling-group and role names.
func groveNameLength(cliqueName string, replica int32) int {
	return len(cliqueName) - len(fmt.Sprintf("-0--%d-", replica))
}

func TestSoleWorkloadNames(t *testing.T) {
	t.Parallel()

	t.Log("Acquire a hybrid build and an LPU-only build")
	registries := map[bool]staticModelRegistry{
		true:  {"build": acquireHybridTestSnapshot(t)},
		false: {"build": acquireTestSnapshot(t, writeV3CompilerFixture(t))},
	}
	pcsName := strings.Repeat("a", MaxPodCliqueSetNameLength)
	for _, test := range []struct {
		name          string
		hybrid        bool
		drafts        int32
		wantConductor string
		wantAgents    []string
	}{
		{name: "hybrid", hybrid: true, wantConductor: "cond", wantAgents: []string{"agt"}},
		{name: "LPU-only", wantAgents: []string{"agt"}},
		{name: "SpecDecode", drafts: 1, wantAgents: []string{"agt0", "agt1"}},
		{name: "maximum draft fanout", drafts: MaxSpecDecodeNumDrafts,
			wantAgents: []string{"agt0", "agt1", "agt2", "agt3", "agt4", "agt5", "agt6", "agt7", "agt8"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve the only workload of a PodCliqueSet whose name has the maximum length")
			reader, registry := fake.NewClientBuilder().Build(), registries[test.hybrid]
			components := []v1beta1.DynamoComponentDeploymentSharedSpec{
				testServingLPXComponent(strings.Repeat("component", 7), test.hybrid),
			}
			if test.drafts > 0 {
				draft := testLPXComponent(strings.Repeat("draftname", 7), "build",
					v1beta1.ComponentRoleSpec{Name: v1beta1.ComponentRoleLPXAgent, PodTemplate: testLPXPodTemplate("lpu-runtime")})
				draft.Replicas = new(test.drafts)
				components = append(components, draft)
			}
			dgd := newSelectedTestDGD(t, "names", components...)
			workloads, err := ResolveWorkloads(t.Context(), reader, dgd, pcsName, registry)
			require.NoError(t, err)
			require.Len(t, workloads, 1)
			workload := workloads[0]

			t.Log("Name the workload after its PodCliqueSet only")
			require.Empty(t, workload.Group())
			require.Equal(t, pcsName, workload.ResourcePrefix())
			require.Equal(t, pcsName+"-0-lpx", workload.ScalingGroup())
			require.Equal(t, test.wantConductor, workload.ConductorTemplate())
			wantServing := test.wantAgents[len(test.wantAgents)-1]
			if test.hybrid {
				wantServing = test.wantConductor
			}
			require.Equal(t, wantServing, workload.ServingTemplate())

			t.Log("Address each role of the last replica by a name within the Grove and DNS limits")
			require.Len(t, workload.ModelProjections(), len(test.wantAgents))
			for model, projection := range workload.ModelProjections() {
				clique := workload.RequestSpec(model, lastWorkloadReplica).NodeLocal.AgentPodCliqueRef.Name
				require.Equal(t, pcsName+"-0-lpx-2495-"+test.wantAgents[model], clique)
				require.LessOrEqual(t, groveNameLength(clique, lastWorkloadReplica), commonconsts.MaxCombinedGroveResourceNameLength)
				require.Empty(t, validation.IsDNS1123Label(lastPodHostname(clique, projection.AgentReplicas())))
			}
			if test.hybrid {
				clique := workload.ConductorCliqueName(lastWorkloadReplica)
				require.Equal(t, pcsName+"-0-lpx-2495-cond", clique)
				require.Equal(t, clique, workload.RequestSpec(0, lastWorkloadReplica).CyborgPodCliqueRef.Name)
				require.LessOrEqual(t, groveNameLength(clique, lastWorkloadReplica), commonconsts.MaxCombinedGroveResourceNameLength)
				require.Empty(t, validation.IsDNS1123Label(lastPodHostname(clique, int(workload.ConductorReplicas()))))
			}

			t.Log("Give the same names when the components have other names")
			renamed := dgd.DeepCopy()
			for index := range renamed.Spec.Components {
				renamed.Spec.Components[index].ComponentName = fmt.Sprintf("short%d", index)
			}
			renamedWorkloads, err := ResolveWorkloads(t.Context(), reader, renamed, pcsName, registry)
			require.NoError(t, err)
			require.Equal(t, workload.ResourcePrefix(), renamedWorkloads[0].ResourcePrefix())
			for model := range workload.ModelProjections() {
				require.Equal(t, workload.RequestSpec(model, lastWorkloadReplica), renamedWorkloads[0].RequestSpec(model, lastWorkloadReplica))
			}

			t.Log("Reject a PodCliqueSet name that is one character too long")
			_, err = ResolveWorkloads(t.Context(), reader, dgd, pcsName+"a", registry)
			require.ErrorContains(t, err, "exceeds the LPX maximum of 38 characters")

			t.Log("Accept an authored capacity only in the supported range")
			for _, replicas := range []int32{-1, 0, 1, 2496, 2497, math.MaxInt32} {
				dgd.Spec.Components[0].Replicas = new(replicas)
				_, err := ResolveWorkloads(t.Context(), reader, dgd, pcsName, registry)
				require.Equal(t, replicas < 0 || replicas > 2496, err != nil, "replicas=%d: %v", replicas, err)
			}
		})
	}
}

func TestSharedWorkloadNames(t *testing.T) {
	t.Parallel()

	t.Log("Acquire a hybrid build, whose conductor has the longest role name, and an LPU-only build")
	registries := map[bool]staticModelRegistry{
		true:  {"build": acquireHybridTestSnapshot(t)},
		false: {"build": acquireTestSnapshot(t, writeV3CompilerFixture(t))},
	}
	reader := fake.NewClientBuilder().Build()
	for _, pcsName := range []string{"model", strings.Repeat("p", 26)} {
		t.Run(pcsName, func(t *testing.T) {
			t.Log("Resolve components whose names are readable, differ only in case, or are too long")
			names := []string{
				"draft", "target", "Draft", "DRAFT",
				strings.Repeat("long-", 12) + "one", strings.Repeat("long-", 12) + "two",
			}
			components := make([]v1beta1.DynamoComponentDeploymentSharedSpec, len(names))
			for index, name := range names {
				components[index] = testServingLPXComponent(name, true)
			}
			workloads, err := ResolveWorkloads(t.Context(), reader, newSelectedTestDGD(t, "names", components...), pcsName, registries[true])
			require.NoError(t, err)
			require.Len(t, workloads, len(names))

			scalingGroups, prefixes := make(map[string]bool), make(map[string]bool)
			for _, workload := range workloads {
				t.Log("Scope each workload by a readable name, or by a hashed name when it must change")
				component := workload.ServingComponentName()
				group := strings.TrimPrefix(workload.ScalingGroup(), pcsName+"-0-")
				agentClique := workload.RequestSpec(0, lastWorkloadReplica).NodeLocal.AgentPodCliqueRef.Name
				require.Equal(t, component, workload.Group())
				if component == "draft" || component == "target" {
					require.Equal(t, pcsName+"-"+component, workload.ResourcePrefix())
					require.Equal(t, component, group)
					require.Equal(t, component+"-cond", workload.ConductorTemplate())
					require.Equal(t, pcsName+"-0-"+component+"-2495-"+component+"-agt", agentClique)
				} else {
					require.Regexp(t, "^"+pcsName+"-(draft|long).*-[a-f0-9]{8}$", workload.ResourcePrefix())
					require.Regexp(t, "^[a-z].*-[a-f0-9]{4,8}$", group)
				}
				scalingGroups[workload.ScalingGroup()], prefixes[workload.ResourcePrefix()] = true, true

				t.Log("Keep the Service name and the last replica's names within the Grove and DNS limits")
				require.Empty(t, validation.IsDNS1035Label(workload.ResourcePrefix()+"-serve"))
				conductorClique := workload.ConductorCliqueName(lastWorkloadReplica)
				require.LessOrEqual(t, groveNameLength(conductorClique, lastWorkloadReplica), commonconsts.MaxCombinedGroveResourceNameLength)
				require.Empty(t, validation.IsDNS1123Label(lastPodHostname(conductorClique, int(workload.ConductorReplicas()))))
				require.LessOrEqual(t, groveNameLength(agentClique, lastWorkloadReplica), commonconsts.MaxCombinedGroveResourceNameLength)
				require.Empty(t, validation.IsDNS1123Label(lastPodHostname(agentClique, workload.ModelProjections()[0].AgentReplicas())))
			}
			require.Len(t, scalingGroups, len(names))
			require.Len(t, prefixes, len(names))
		})
	}

	t.Log("Use the remaining Grove budget for named workloads")
	for _, test := range []struct {
		name      string
		hybrid    bool
		pcsLength int
		component string
		wantError bool
	}{
		{name: "readable at limit", hybrid: true, pcsLength: 28, component: "target"},
		{name: "hashed at limit", hybrid: true, pcsLength: 28, component: "TARGET"},
		{name: "sole-workload name leaves too little room", hybrid: true, pcsLength: 38, component: "target", wantError: true},
		{name: "LPU-only readable at limit", pcsLength: 29, component: "target"},
		{name: "LPU-only hashed at limit", pcsLength: 29, component: "TARGET"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve the component beside a second workload, which sorts after it")
			dgd := newSelectedTestDGD(t, "names",
				testServingLPXComponent(test.component, test.hybrid),
				testServingLPXComponent("z", test.hybrid))
			workloads, err := ResolveWorkloads(t.Context(), reader, dgd, strings.Repeat("p", test.pcsLength), registries[test.hybrid])
			if test.wantError {
				require.ErrorContains(t, err, "shorten the deployment name")
				return
			}
			require.NoError(t, err)

			t.Log("Use the complete Grove budget for the serving role")
			servingClique := fmt.Sprintf("%s-0-%s", workloads[0].ScalingGroup(), workloads[0].ServingTemplate())
			require.Equal(t, commonconsts.MaxCombinedGroveResourceNameLength, groveNameLength(servingClique, 0))
		})
	}
}

func TestWorkloadBoundsConductorPodHostnames(t *testing.T) {
	t.Parallel()

	t.Log("Author a hybrid component in a shared PodCliqueSet at the combined name limit")
	reader, registry := fake.NewClientBuilder().Build(), staticModelRegistry{"build": acquireHybridTestSnapshot(t)}
	pcsName := strings.Repeat("a", 28)
	dgd := newSelectedTestDGD(t, "names", testServingLPXComponent("target", true), testServingLPXComponent("z", true))
	for _, test := range []struct {
		name      string
		replicas  int32
		wantError bool
	}{
		{name: "one conductor Pod", replicas: 1},
		{name: "hostname at DNS limit", replicas: 100_000_000},
		{name: "hostname over DNS limit", replicas: 100_000_001, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Bound the authored width at the highest replica while the group has one")
			authored := dgd.DeepCopy()
			authored.Spec.Components[0].ComponentRole(v1beta1.ComponentRoleLPXConductor).Replicas = new(test.replicas)
			_, err := ResolveWorkloads(t.Context(), reader, authored, pcsName, registry)
			if test.wantError {
				require.ErrorContains(t, err, "materialized conductor Pod hostname")
				require.ErrorContains(t, err, "must be no more than 63 characters")
			} else {
				require.NoError(t, err)
			}

			t.Log("Apply the same bound to a width that an external scaler set")
			workloads, err := ResolveWorkloads(t.Context(), reader, dgd, pcsName, registry)
			require.NoError(t, err)
			err = workloads[0].ValidateConductorReplicas(test.replicas)
			if test.wantError {
				require.ErrorContains(t, err, "materialized conductor Pod hostname")
				require.ErrorContains(t, err, "must be no more than 63 characters")
				return
			}
			require.NoError(t, err)

			t.Log("Confirm that the accepted width gives a valid name in the highest replica")
			hostname := lastPodHostname(workloads[0].ConductorCliqueName(lastWorkloadReplica), int(test.replicas))
			require.Empty(t, validation.IsDNS1123Label(hostname))
			if test.replicas > 1 {
				require.Len(t, hostname, validation.DNS1123LabelMaxLength)
			}
		})
	}
}

func TestWorkloadBoundsAgentPodHostnames(t *testing.T) {
	t.Parallel()

	reader, pcsName := fake.NewClientBuilder().Build(), strings.Repeat("a", 29)
	for _, test := range []struct {
		name      string
		nodes     uint32
		wantError bool
	}{
		{name: "one Agent Pod", nodes: 1},
		{name: "hostname at DNS limit", nodes: 100_000_000},
		{name: "hostname over DNS limit", nodes: 100_000_001, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Compile an LPU-only build whose one partition has the Agent count under test")
			fixture := newV3CompilerFixture()
			fixture.architecture = lp20
			fixture.partitions[0].devicesPerNode = 8
			fixture.units = testSingleUnits(testNodesReservation(test.nodes, 1))
			registry := staticModelRegistry{"build": acquireTestSnapshot(t, writeCompilerFixture(t, fixture))}

			t.Log("Bound the compiled Agent count in a shared PodCliqueSet at the combined name limit")
			dgd := newSelectedTestDGD(t, "names", testServingLPXComponent("target", false), testServingLPXComponent("z", false))
			workloads, err := ResolveWorkloads(t.Context(), reader, dgd, pcsName, registry)
			if test.wantError {
				require.ErrorContains(t, err, "materialized Agent Pod hostname")
				require.ErrorContains(t, err, "must be no more than 63 characters")
				return
			}
			require.NoError(t, err)

			t.Log("Confirm that the accepted count gives a valid name in the highest replica")
			clique := workloads[0].RequestSpec(0, lastWorkloadReplica).NodeLocal.AgentPodCliqueRef.Name
			hostname := lastPodHostname(clique, int(test.nodes))
			require.Empty(t, validation.IsDNS1123Label(hostname))
			if test.nodes > 1 {
				require.Len(t, hostname, validation.DNS1123LabelMaxLength)
			}
		})
	}
}
