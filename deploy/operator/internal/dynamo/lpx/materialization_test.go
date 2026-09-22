/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"math"
	"strings"
	"testing"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestPlanMaterializationBounds(t *testing.T) {
	t.Parallel()

	t.Log("Project hybrid and LPU-only materialization fixtures")
	snapshot := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	hybrid := newV2CompilerFixture()
	hybrid.compilationMode = manifestcapnp.CompilationMode_lpx
	hybridProjection := projectRenderFixture(t, PipelineLPX, acquireTestSnapshot(t, writeCompilerFixture(t, hybrid)))
	lpuOnlyProjection := projectRenderFixture(t, PipelineSingle, snapshot)

	t.Log("Reserve readable roles at the maximum PCS length and scheduling replica count")
	for _, test := range []struct {
		name       string
		models     int
		projection *ModelProjection
	}{
		{name: "hybrid", models: 1, projection: hybridProjection},
		{name: "LPU-only", models: 1, projection: lpuOnlyProjection},
		{name: "SpecDecode", models: 2, projection: lpuOnlyProjection},
		{name: "maximum draft fanout", models: MaxSpecDecodeNumDrafts + 1, projection: lpuOnlyProjection},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Build independent projections without using component names in identities")
			projections := make([]*ModelProjection, test.models)
			for index := range projections {
				projection := *test.projection
				projection.stage = strings.Repeat("component", 7)
				if test.models > 1 {
					projection.pipeline = PipelineSpecDecode
				}
				projections[index] = &projection
			}
			workload := &Workload{
				modelProjections:     projections,
				scalingGroupReplicas: 2496,
			}
			pcsName := strings.Repeat("a", MaxPodCliqueSetNameLength)
			plan, err := workload.PlanNodeLocalMaterialization(pcsName)
			require.NoError(t, err)
			require.Equal(t, pcsName+"-0-lpx", plan.LPXScalingGroup)

			t.Log("Validate every role's Grove name budget and the last replica's Pod hostnames")
			lastReplica := plan.ForReplica(plan.Replicas - 1)
			hostnames := make([]string, 0, test.models+1)
			if lastReplica.ConductorClique != "" {
				require.Equal(t, "cond", plan.ConductorTemplate)
				hostnames = append(hostnames, materializedPodHostname(lastReplica.ConductorClique, 0))
				require.LessOrEqual(t, len(pcsName)+len(lpxScalingGroupTemplateName)+len(plan.ConductorTemplate),
					commonconsts.MaxCombinedGroveResourceNameLength)
			}
			for _, agent := range lastReplica.Agents {
				require.LessOrEqual(t, len(pcsName)+len(lpxScalingGroupTemplateName)+len(agent.TemplateName), commonconsts.MaxCombinedGroveResourceNameLength)
				hostnames = append(hostnames, materializedPodHostname(agent.CliqueName, agent.Replicas-1))
			}
			if lastReplica.CyborgClique != "" {
				require.Equal(t, "cond", plan.CyborgTemplate)
				hostnames = append(hostnames, materializedPodHostname(lastReplica.CyborgClique, 0))
			}
			for _, hostname := range hostnames {
				require.Empty(t, validation.IsDNS1123Label(hostname))
			}

			t.Log("Renaming authored components leaves every materialized identity unchanged")
			for _, projection := range projections {
				projection.stage = "short"
			}
			renamed, err := workload.PlanNodeLocalMaterialization(pcsName)
			require.NoError(t, err)
			require.Equal(t, plan, renamed)

			t.Log("Reject a PCS name one character beyond Grove's combined name budget")
			_, err = workload.PlanNodeLocalMaterialization(pcsName + "a")
			require.ErrorContains(t, err, "exceeds the LPX maximum of 38 characters")

			t.Log("Bound workload replicas before allocating per-workload request state")
			for _, replicas := range []int32{-1, 0, 1, 2496, 2497, math.MaxInt32} {
				workload.scalingGroupReplicas = replicas
				_, err := workload.PlanNodeLocalMaterialization(pcsName)
				require.Equal(t, replicas < 0 || replicas > 2496, err != nil, "replicas=%d: %v", replicas, err)
			}
		})
	}
}

func TestWorkloadNames(t *testing.T) {
	t.Log("Project a real workload at the maximum scaling-group replica count")
	projection := projectRenderFixture(t, PipelineSingle, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	workload := &Workload{modelProjections: []*ModelProjection{projection}, scalingGroupReplicas: maxWorkloadReplicas}
	for _, pcsName := range []string{"model", strings.Repeat("p", 26)} {
		t.Run(pcsName, func(t *testing.T) {
			t.Log("Keep shared-runtime resource names and the PCS identity")
			plan, err := workload.PlanNodeLocalMaterialization(pcsName)
			require.NoError(t, err)
			before := plan.ForReplica(0)

			t.Log("Use readable component names while bounding and distinguishing shortened names")
			names := make(map[string]bool)
			groups := make(map[string]bool)
			for _, component := range []string{
				"draft", "target", "Draft", "DRAFT",
				strings.Repeat("long-", 12) + "one", strings.Repeat("long-", 12) + "two",
			} {
				scoped, err := plan.WithGroup(component)
				require.NoError(t, err)
				require.Equal(t, pcsName, scoped.PodCliqueSetName)
				require.True(t, strings.HasPrefix(scoped.ResourcePrefix, pcsName+"-"))
				require.False(t, names[scoped.ResourcePrefix], "duplicate prefix for %s", component)
				names[scoped.ResourcePrefix] = true
				require.False(t, groups[scoped.ScalingGroupTemplate], "duplicate group for %s", component)
				groups[scoped.ScalingGroupTemplate] = true
				require.Empty(t, validation.IsDNS1035Label(scoped.ResourcePrefix+"-serve"))
				if component == "draft" || component == "target" {
					require.Equal(t, pcsName+"-"+component, scoped.ResourcePrefix)
					require.Equal(t, component, scoped.ScalingGroupTemplate)
					require.Equal(t, component+"-cond", scoped.ConductorTemplate)
					require.Equal(t, component+"-agt", scoped.Agents[0].TemplateName)
				} else {
					require.Regexp(t, "-(draft|long).*-[a-f0-9]{8}$", scoped.ResourcePrefix)
					require.Regexp(t, "^[a-z].*-[a-f0-9]{4,8}$", scoped.ScalingGroupTemplate)
				}

				t.Log("Keep Grove's combined names and the last replica's Pod hostnames valid")
				last := scoped.ForReplica(scoped.Replicas - 1)
				require.Empty(t, validation.IsDNS1123Label(materializedPodHostname(last.ConductorClique, 0)))
				require.LessOrEqual(t, len(pcsName)+len(scoped.ScalingGroupTemplate)+len(scoped.ConductorTemplate), commonconsts.MaxCombinedGroveResourceNameLength)
				for _, agent := range last.Agents {
					require.LessOrEqual(t, len(pcsName)+len(scoped.ScalingGroupTemplate)+len(agent.TemplateName), commonconsts.MaxCombinedGroveResourceNameLength)
					require.Empty(t, validation.IsDNS1123Label(materializedPodHostname(agent.CliqueName, agent.Replicas-1)))
				}
			}
			require.Equal(t, before, plan)
		})
	}

	t.Log("Use the remaining Grove budget for named workloads")
	for _, test := range []struct {
		name      string
		pcsLength int
		component string
		wantError bool
	}{
		{name: "readable at limit", pcsLength: 28, component: "target"},
		{name: "hashed at limit", pcsLength: 28, component: "TARGET"},
		{name: "sole-workload name leaves too little room", pcsLength: 38, component: "target", wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Scope the workload without renaming its PCS")
			plan, err := workload.PlanNodeLocalMaterialization(strings.Repeat("p", test.pcsLength))
			require.NoError(t, err)
			scoped, err := plan.WithGroup(test.component)
			if test.wantError {
				require.ErrorContains(t, err, "shorten the deployment name")
				require.Equal(t, strings.Repeat("p", test.pcsLength), plan.PodCliqueSetName)
				return
			}
			require.NoError(t, err)
			require.Equal(t, commonconsts.MaxCombinedGroveResourceNameLength,
				len(scoped.PodCliqueSetName)+len(scoped.ScalingGroupTemplate)+len(scoped.ConductorTemplate))
		})
	}
}
