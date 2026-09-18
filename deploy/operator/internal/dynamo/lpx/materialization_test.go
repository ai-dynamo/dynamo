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
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestPlanMaterializationBounds(t *testing.T) {
	t.Parallel()

	t.Log("Project hybrid and LPU-only materialization fixtures")
	snapshot := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	hybrid := newV2CompilerFixture()
	hybrid.compilationMode = manifestcapnp.CompilationMode_lpx
	hybridProjection := projectRenderFixture(t, lpxv1alpha1.TargetFamilyXt8888, PipelineLPX, acquireTestSnapshot(t, writeCompilerFixture(t, hybrid)))
	lpuOnlyProjection := projectRenderFixture(t, lpxv1alpha1.TargetFamilyXt8888, PipelineSingle, snapshot)

	t.Log("Reserve readable roles at the maximum PCS length and scheduling replica count")
	for _, test := range []struct {
		name       string
		models     int
		projection *ModelProjection
	}{
		{name: "hybrid", models: 1, projection: hybridProjection},
		{name: "LPU-only", models: 1, projection: lpuOnlyProjection},
		{name: "SpecDecode", models: 2, projection: lpuOnlyProjection},
		{name: "maximum draft fanout", models: maxSpecDecodeNumDrafts + 1, projection: lpuOnlyProjection},
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
			workload := &SelectedWorkload{
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
				require.Equal(t, commonconsts.MaxCombinedGroveResourceNameLength,
					len(pcsName)+len(lpxScalingGroupTemplateName)+len(plan.ConductorTemplate))
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

			t.Log("Bound engine replicas before allocating per-engine request state")
			for _, replicas := range []int32{-1, 0, 1, 2496, 2497, math.MaxInt32} {
				workload.scalingGroupReplicas = replicas
				_, err := workload.PlanNodeLocalMaterialization(pcsName)
				require.Equal(t, replicas < 0 || replicas > 2496, err != nil, "replicas=%d: %v", replicas, err)
			}
		})
	}
}
