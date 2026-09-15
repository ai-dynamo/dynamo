/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	manifestcapnp "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/manifest/v2"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
)

func TestMaterializationBoundsCompleteSchedulingAttempt(t *testing.T) {
	t.Parallel()

	t.Log("Account for every populated API field and the array separator")
	row := nvidiacomv1beta1.LPXAttemptRequestStatus{
		Name:          strings.Repeat("a", validation.DNS1123LabelMaxLength),
		AttemptDigest: "sha256:" + strings.Repeat("f", 64),
		UID:           types.UID("01234567-89ab-cdef-0123-456789abcdef"),
	}
	encoded, err := json.Marshal(row)
	require.NoError(t, err)
	require.Equal(t, maximumSchedulingAttemptRequestBytes, len(encoded)+1)

	const maximumRequests = int32(schedulingAttemptRequestBytesBudget / maximumSchedulingAttemptRequestBytes)

	t.Log("Accept the exact row limit and reject larger single- and multi-model allocations")
	for _, test := range []struct {
		name     string
		replicas int32
		models   int
		wantErr  bool
	}{
		{name: "zero", models: 1},
		{name: "nine replicas", replicas: 9, models: 1},
		{name: "twelve replicas", replicas: 12, models: 1},
		{name: "limit", replicas: maximumRequests, models: 1},
		{name: "one over", replicas: maximumRequests + 1, models: 1, wantErr: true},
		{name: "two models at limit", replicas: maximumRequests / 2, models: 2},
		{name: "two models over limit", replicas: maximumRequests/2 + 1, models: 2, wantErr: true},
		{name: "16384 replicas", replicas: 16384, models: 1, wantErr: true},
		{name: "negative", replicas: -1, models: 1, wantErr: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Validate the complete identity storage before materializing replicas")
			plan := &MaterializationPlan{LPXScalingGroup: "group", Replicas: test.replicas, Agents: make([]ExpectedAgent, test.models)}
			for i := range plan.Agents {
				plan.Agents[i] = ExpectedAgent{TemplateName: fmt.Sprintf("agent-%d", i), Replicas: 1}
			}
			if test.wantErr {
				require.ErrorContains(t, plan.ValidateReplicaCount(), "scheduling status size budget")
			} else {
				require.NoError(t, plan.ValidateReplicaCount())
			}
		})
	}
}

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
