/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"strings"
	"testing"

	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/capnp/gbuild_manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/thirdparty/lpxscheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestPlanMaterializationBoundsGeneratedPodHostnames(t *testing.T) {
	t.Parallel()

	t.Log("Project hybrid and LPU-only materialization fixtures")
	snapshot := acquireTestSnapshot(t, writeV2CompilerFixture(t))
	hybrid := newV2CompilerFixture()
	hybrid.compilationMode = manifestcapnp.CompilationMode_lpx
	hybridProjection := projectRenderFixture(t, lpxv1alpha1.TargetFamilyXt8888, PipelineLPX, acquireTestSnapshot(t, writeCompilerFixture(t, hybrid)))
	lpuOnlyProjection := projectRenderFixture(t, lpxv1alpha1.TargetFamilyXt8888, PipelineSingle, snapshot)

	t.Log("Use component-derived names at the minimum collision-resistant group budget")
	tests := []struct {
		name       string
		pcsLength  int
		projection *ModelProjection
	}{
		{
			name: "hybrid", pcsLength: 14,
			projection: hybridProjection,
		},
		{
			name: "LPU-only", pcsLength: 26,
			projection: lpuOnlyProjection,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			test.projection.stage = "longcomponent"
			workload := &SelectedWorkload{modelProjections: []*ModelProjection{test.projection}, scalingGroupReplicas: 1}
			pcsName := strings.Repeat("a", test.pcsLength)
			t.Log("Plan and verify materialization within the DNS hostname budget")
			plan, err := workload.PlanNodeLocalMaterialization(pcsName)
			require.NoError(t, err)
			require.Equal(t, "longcomponent-wkr-m-0", plan.Agents[0].TemplateName)
			require.Len(t, plan.LPXScalingGroupTemplate, 8)
			hostname := materializedPodHostname(plan.Agents[0].CliqueName, plan.Agents[0].Replicas-1)
			if test.projection.pipeline == PipelineLPX {
				require.Equal(t, "longcomponent-engine-gpu", plan.CyborgTemplate)
				hostname = materializedPodHostname(plan.CyborgClique, maximumCyborgPodIndex)
			}
			require.Len(t, hostname, validation.DNS1123LabelMaxLength)
			require.Empty(t, validation.IsDNS1123Label(hostname))

			t.Log("Reject a PCS name that leaves one character too little for the group")
			_, err = workload.PlanNodeLocalMaterialization(pcsName + "a")
			require.ErrorContains(t, err, "derive LPU scaling-group template name")
		})
	}
}
