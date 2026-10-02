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

// workloadNames lists every Grove and runtime name assigned to a workload.
func workloadNames(workload *Workload) []string {
	names := []string{workload.scalingGroupTemplate, workload.scalingGroup, workload.resourcePrefix, workload.conductorTemplate}
	for _, model := range workload.models {
		names = append(names, model.agentTemplate)
	}
	return names
}

func TestWorkloadNameBounds(t *testing.T) {
	t.Parallel()

	t.Log("Project hybrid and LPU-only materialization fixtures")
	snapshot := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	hybrid := newV2CompilerFixture()
	hybrid.compilationMode = manifestcapnp.CompilationMode_lpx
	hybridProjection := projectTestModel(t, acquireTestSnapshot(t, writeCompilerFixture(t, hybrid)), PipelineHybrid)
	lpuOnlyProjection := projectTestModel(t, snapshot, PipelineSingle)

	t.Log("Reserve readable roles at the maximum PCS length and scheduling replica count")
	for _, test := range []struct {
		name       string
		models     int
		projection *Model
	}{
		{name: "hybrid", models: 1, projection: hybridProjection},
		{name: "LPU-only", models: 1, projection: lpuOnlyProjection},
		{name: "SpecDecode", models: 2, projection: lpuOnlyProjection},
		{name: "maximum draft fanout", models: MaxSpecDecodeNumDrafts + 1, projection: lpuOnlyProjection},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Build independent models without using component names in identities")
			models := make([]*Model, test.models)
			for index := range models {
				projected := *test.projection.component
				projected.name = strings.Repeat("component", 7)
				if test.models > 1 {
					projected.pipeline = PipelineSpecDecode
				}
				models[index] = &Model{name: test.projection.name, digest: test.projection.digest, component: &projected}
			}
			workload := &Workload{name: models[0].component.name, models: models, scalingGroupReplicas: maxWorkloadReplicas}
			pcsName := strings.Repeat("a", MaxPodCliqueSetNameLength)
			require.NoError(t, workload.nameResources(pcsName, ""))
			require.Equal(t, pcsName+"-0-lpx", workload.ScalingGroup())
			require.Equal(t, "cond", workload.ConductorTemplate())

			t.Log("Validate every role's Grove name budget and the last replica's Pod hostnames")
			lastReplica := int32(maxWorkloadReplicas - 1)
			hostnames := []string{podHostname(workload.cliqueName(workload.conductorTemplate, lastReplica), 0)}
			require.LessOrEqual(t, len(pcsName)+len(lpxScalingGroupTemplateName)+len(workload.conductorTemplate),
				commonconsts.MaxCombinedGroveResourceNameLength)
			for _, model := range models {
				require.LessOrEqual(t, len(pcsName)+len(lpxScalingGroupTemplateName)+len(model.agentTemplate), commonconsts.MaxCombinedGroveResourceNameLength)
				hostnames = append(hostnames, podHostname(workload.cliqueName(model.agentTemplate, lastReplica), model.component.agentReplicas-1))
			}
			for _, hostname := range hostnames {
				require.Empty(t, validation.IsDNS1123Label(hostname))
			}

			t.Log("Renaming authored components leaves every materialized identity unchanged")
			names := workloadNames(workload)
			for _, model := range models {
				model.component.name = "short"
			}
			require.NoError(t, workload.nameResources(pcsName, ""))
			require.Equal(t, names, workloadNames(workload))

			t.Log("Reject a PCS name one character beyond Grove's combined name budget")
			require.ErrorContains(t, workload.nameResources(pcsName+"a", ""), "exceeds the LPX maximum of 38 characters")

			t.Log("Bound workload replicas before allocating per-workload request state")
			for _, replicas := range []int32{-1, 0, 1, 2496, 2497, math.MaxInt32} {
				err := workload.ValidateReplicas(replicas)
				require.Equal(t, replicas < 0 || replicas > 2496, err != nil, "replicas=%d: %v", replicas, err)
			}
		})
	}
}

func TestWorkloadNames(t *testing.T) {
	t.Log("Project a real workload at the maximum scaling-group replica count")
	projection := projectTestModel(t, acquireTestSnapshot(t, writeV2CompilerFixture(t)), PipelineSingle)
	workload := &Workload{name: projection.component.name, models: []*Model{projection}, scalingGroupReplicas: maxWorkloadReplicas}
	for _, pcsName := range []string{"model", strings.Repeat("p", 26)} {
		t.Run(pcsName, func(t *testing.T) {
			t.Log("Keep the unscoped names for a sole workload")
			require.NoError(t, workload.nameResources(pcsName, ""))
			require.Equal(t, pcsName, workload.ResourcePrefix())
			require.Equal(t, lpxScalingGroupTemplateName, workload.scalingGroupTemplate)

			t.Log("Use readable component names while bounding and distinguishing shortened names")
			names := make(map[string]bool)
			groups := make(map[string]bool)
			for _, component := range []string{
				"draft", "target", "Draft", "DRAFT",
				strings.Repeat("long-", 12) + "one", strings.Repeat("long-", 12) + "two",
			} {
				require.NoError(t, workload.nameResources(pcsName, component))
				require.True(t, strings.HasPrefix(workload.resourcePrefix, pcsName+"-"))
				require.False(t, names[workload.resourcePrefix], "duplicate prefix for %s", component)
				names[workload.resourcePrefix] = true
				require.False(t, groups[workload.scalingGroupTemplate], "duplicate group for %s", component)
				groups[workload.scalingGroupTemplate] = true
				require.Empty(t, validation.IsDNS1035Label(workload.resourcePrefix+"-serve"))
				if component == "draft" || component == "target" {
					require.Equal(t, pcsName+"-"+component, workload.resourcePrefix)
					require.Equal(t, component, workload.scalingGroupTemplate)
					require.Equal(t, component+"-cond", workload.conductorTemplate)
					require.Equal(t, component+"-agt", projection.agentTemplate)
				} else {
					require.Regexp(t, "-(draft|long).*-[a-f0-9]{8}$", workload.resourcePrefix)
					require.Regexp(t, "^[a-z].*-[a-f0-9]{4,8}$", workload.scalingGroupTemplate)
				}

				t.Log("Keep Grove's combined names and the last replica's Pod hostnames valid")
				lastReplica := workload.scalingGroupReplicas - 1
				require.Empty(t, validation.IsDNS1123Label(podHostname(workload.cliqueName(workload.conductorTemplate, lastReplica), 0)))
				require.LessOrEqual(t, len(pcsName)+len(workload.scalingGroupTemplate)+len(workload.conductorTemplate), commonconsts.MaxCombinedGroveResourceNameLength)
				require.LessOrEqual(t, len(pcsName)+len(workload.scalingGroupTemplate)+len(projection.agentTemplate), commonconsts.MaxCombinedGroveResourceNameLength)
				require.Empty(t, validation.IsDNS1123Label(podHostname(workload.cliqueName(projection.agentTemplate, lastReplica), projection.component.agentReplicas-1)))
			}
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
			t.Log("Scope the workload within the shared PCS")
			pcsName := strings.Repeat("p", test.pcsLength)
			err := workload.nameResources(pcsName, test.component)
			if test.wantError {
				require.ErrorContains(t, err, "shorten the deployment name")
				return
			}
			require.NoError(t, err)
			require.Equal(t, commonconsts.MaxCombinedGroveResourceNameLength,
				len(pcsName)+len(workload.scalingGroupTemplate)+len(workload.conductorTemplate))
		})
	}
}
