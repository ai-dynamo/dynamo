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

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/validation"
)

// materializedPodHostname returns the hostname Grove gives a Pod in a clique.
func materializedPodHostname(cliqueName string, podIndex int) string {
	return fmt.Sprintf("%s-%d", cliqueName, podIndex)
}

func TestWorkloadMaterializationBounds(t *testing.T) {
	t.Parallel()

	t.Log("Project hybrid and LPU-only workloads")
	hybrid := projectRenderFixture(t, PipelineHybrid, acquireTestSnapshot(t, writeCompilerFixture(t, newV2CompilerFixture())))
	lpuOnly := projectRenderFixture(t, PipelineSingle, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	for _, test := range []struct {
		name           string
		models         int
		projection     *ModelProjection
		cyborgReplicas int32
		wantCyborg     string
		wantServing    string
	}{
		{name: "hybrid", models: 1, projection: hybrid, cyborgReplicas: 1, wantCyborg: "cond", wantServing: "cond"},
		{name: "LPU-only", models: 1, projection: lpuOnly, wantServing: "agt"},
		{name: "SpecDecode", models: 2, projection: lpuOnly, wantServing: "agt1"},
		{name: "maximum draft fanout", models: MaxSpecDecodeNumDrafts + 1, projection: lpuOnly, wantServing: "agt8"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Resolve independent models at the maximum PCS length and replica count")
			projections := make([]*ModelProjection, test.models)
			for index := range projections {
				projection := *test.projection
				projection.stage = strings.Repeat("component", 7)
				if test.models > 1 {
					projection.pipeline = PipelineSpecDecode
				}
				projections[index] = &projection
			}
			pcsName := strings.Repeat("a", MaxPodCliqueSetNameLength)
			workload, err := newWorkload(pcsName, false, projections, maxWorkloadReplicas, test.cyborgReplicas)
			require.NoError(t, err)
			require.Empty(t, workload.Group())
			require.Equal(t, pcsName+"-0-lpx", workload.ScalingGroup())
			require.Equal(t, test.wantCyborg, workload.CyborgTemplate())
			require.Equal(t, test.wantServing, workload.ServingTemplate())

			t.Log("Check every role's name budget and the last replica's request addresses")
			require.LessOrEqual(t, len(pcsName)+len(lpxScalingGroupTemplateName)+len(workload.ServingTemplate()), commonconsts.MaxCombinedGroveResourceNameLength)
			if test.cyborgReplicas > 0 {
				hostname := materializedPodHostname(workload.CyborgCliqueName(maxWorkloadReplicas-1), 0)
				require.Empty(t, validation.IsDNS1123Label(hostname))
			}
			for model, projection := range projections {
				require.LessOrEqual(t, len(pcsName)+len(lpxScalingGroupTemplateName)+len(workload.agentTemplate(model)), commonconsts.MaxCombinedGroveResourceNameLength)
				spec := workload.RequestSpec(model, maxWorkloadReplicas-1)
				require.Empty(t, validation.IsDNS1123Label(materializedPodHostname(spec.NodeLocal.AgentPodCliqueRef.Name, projection.agentReplicas-1)))
			}

			t.Log("Keep resource identities independent of authored component names")
			renamedProjections := make([]*ModelProjection, len(projections))
			for model, projection := range projections {
				copied := *projection
				copied.stage = "short"
				renamedProjections[model] = &copied
			}
			renamed, err := newWorkload(pcsName, false, renamedProjections, maxWorkloadReplicas, test.cyborgReplicas)
			require.NoError(t, err)
			require.Equal(t, workload.ResourcePrefix(), renamed.ResourcePrefix())
			require.Equal(t, workload.ServingTemplate(), renamed.ServingTemplate())
			for model := range projections {
				require.Equal(t, workload.RequestSpec(model, 0), renamed.RequestSpec(model, 0))
			}

			t.Log("Reject an oversized PCS name and an authored capacity outside the supported range")
			_, err = newWorkload(pcsName+"a", false, projections, maxWorkloadReplicas, test.cyborgReplicas)
			require.ErrorContains(t, err, "exceeds the LPX maximum of 38 characters")
			for _, replicas := range []int32{-1, 0, 1, 2496, 2497, math.MaxInt32} {
				_, err := newWorkload(pcsName, false, projections, replicas, test.cyborgReplicas)
				require.Equal(t, replicas < 0 || replicas > 2496, err != nil, "replicas=%d: %v", replicas, err)
			}
		})
	}
}

func TestWorkloadNames(t *testing.T) {
	t.Log("Project a hybrid workload at the maximum scaling-group replica count")
	projection := projectRenderFixture(t, PipelineHybrid, acquireTestSnapshot(t, writeCompilerFixture(t, newV2CompilerFixture())))
	for _, pcsName := range []string{"model", strings.Repeat("p", 26)} {
		t.Run(pcsName, func(t *testing.T) {
			t.Log("Keep component names readable and distinguish shortened names")
			names := make(map[string]bool)
			groups := make(map[string]bool)
			for _, component := range []string{
				"draft", "target", "Draft", "DRAFT",
				strings.Repeat("long-", 12) + "one", strings.Repeat("long-", 12) + "two",
			} {
				model := *projection
				model.stage = component
				workload, err := newWorkload(pcsName, true, []*ModelProjection{&model}, maxWorkloadReplicas, 1)
				require.NoError(t, err)
				require.Equal(t, component, workload.Group())
				require.True(t, strings.HasPrefix(workload.ResourcePrefix(), pcsName+"-"))
				require.False(t, names[workload.ResourcePrefix()], "duplicate prefix for %s", component)
				names[workload.ResourcePrefix()] = true
				require.False(t, groups[workload.scalingGroupTemplate], "duplicate group for %s", component)
				groups[workload.scalingGroupTemplate] = true
				require.Empty(t, validation.IsDNS1035Label(workload.ResourcePrefix()+"-serve"))
				if component == "draft" || component == "target" {
					require.Equal(t, pcsName+"-"+component, workload.ResourcePrefix())
					require.Equal(t, component, workload.scalingGroupTemplate)
					require.Equal(t, component+"-cond", workload.CyborgTemplate())
					require.Equal(t, component+"-agt", workload.agentTemplate(0))
				} else {
					require.Regexp(t, "-(draft|long).*-[a-f0-9]{8}$", workload.ResourcePrefix())
					require.Regexp(t, "^[a-z].*-[a-f0-9]{4,8}$", workload.scalingGroupTemplate)
				}

				t.Log("Keep Grove names and the last replica's Pod hostnames valid")
				clique := workload.CyborgCliqueName(maxWorkloadReplicas - 1)
				require.Empty(t, validation.IsDNS1123Label(materializedPodHostname(clique, 0)))
				require.LessOrEqual(t, len(pcsName)+len(workload.scalingGroupTemplate)+len(workload.CyborgTemplate()), commonconsts.MaxCombinedGroveResourceNameLength)
				require.LessOrEqual(t, len(pcsName)+len(workload.scalingGroupTemplate)+len(workload.agentTemplate(0)), commonconsts.MaxCombinedGroveResourceNameLength)
				request := workload.RequestSpec(0, maxWorkloadReplicas-1)
				require.Empty(t, validation.IsDNS1123Label(materializedPodHostname(request.NodeLocal.AgentPodCliqueRef.Name, model.agentReplicas-1)))
			}
		})
	}

	t.Log("Use the remaining Grove budget for named workloads")
	lpuOnly := projectRenderFixture(t, PipelineSingle, acquireTestSnapshot(t, writeV2CompilerFixture(t)))
	for _, test := range []struct {
		name           string
		projection     *ModelProjection
		cyborgReplicas int32
		pcsLength      int
		component      string
		wantError      bool
	}{
		{name: "readable at limit", projection: projection, cyborgReplicas: 1, pcsLength: 28, component: "target"},
		{name: "hashed at limit", projection: projection, cyborgReplicas: 1, pcsLength: 28, component: "TARGET"},
		{name: "sole-workload name leaves too little room", projection: projection, cyborgReplicas: 1, pcsLength: 38, component: "target", wantError: true},
		{name: "LPU-only readable at limit", projection: lpuOnly, pcsLength: 29, component: "target"},
		{name: "LPU-only hashed at limit", projection: lpuOnly, pcsLength: 29, component: "TARGET"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Name this workload within the unchanged PCS identity")
			model := *test.projection
			model.stage = test.component
			workload, err := newWorkload(strings.Repeat("p", test.pcsLength), true, []*ModelProjection{&model}, maxWorkloadReplicas, test.cyborgReplicas)
			if test.wantError {
				require.ErrorContains(t, err, "shorten the deployment name")
				return
			}
			require.NoError(t, err)
			require.Equal(t, commonconsts.MaxCombinedGroveResourceNameLength, test.pcsLength+len(workload.scalingGroupTemplate)+len(workload.ServingTemplate()))
		})
	}
}

func TestWorkloadBoundsActualGPUHostnames(t *testing.T) {
	t.Parallel()

	t.Log("Project a hybrid workload at the combined name limit")
	projection := projectRenderFixture(t, PipelineHybrid, acquireTestSnapshot(t, writeCompilerFixture(t, newV3CompilerFixture())))
	projection.stage = "target"
	projection.configuredBuild.ioFPGACount = 1
	projection.configuredBuild.ioFanoutFactor = 1
	for _, test := range []struct {
		name      string
		replicas  int32
		wantError bool
	}{
		{name: "one GPU per engine", replicas: 1},
		{name: "hostname at DNS limit", replicas: 100_000_000},
		{name: "hostname over DNS limit", replicas: 100_000_001, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Bound the authored width at the highest replica while the group has one")
			_, err := newWorkload(strings.Repeat("a", 28), true, []*ModelProjection{projection}, 1, test.replicas)
			if test.wantError {
				require.ErrorContains(t, err, "materialized Cyborg Pod hostname")
				require.ErrorContains(t, err, "must be no more than 63 characters")
			} else {
				require.NoError(t, err)
			}

			t.Log("Apply the same bound to an externally managed Cyborg width")
			workload, err := newWorkload(strings.Repeat("a", 28), true, []*ModelProjection{projection}, 1, 1)
			require.NoError(t, err)
			err = workload.ValidateCyborgReplicas(test.replicas)
			if test.wantError {
				require.ErrorContains(t, err, "materialized Cyborg Pod hostname")
				require.ErrorContains(t, err, "must be no more than 63 characters")
				return
			}
			require.NoError(t, err)

			t.Log("Confirm that the accepted width gives a valid name in the highest replica")
			hostname := materializedPodHostname(workload.CyborgCliqueName(maxWorkloadReplicas-1), int(test.replicas)-1)
			require.Empty(t, validation.IsDNS1123Label(hostname))
			if test.replicas > 1 {
				require.Len(t, hostname, validation.DNS1123LabelMaxLength)
			}
		})
	}
}
