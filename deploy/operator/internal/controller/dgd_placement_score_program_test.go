/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/ptr"
)

// TestComponentProgramReportsUnsupportedPlacement covers the placement contract
// on the pathway that has no scheduler score. Every backend must set
// status.placement after its first reconciliation, and a score produced by an
// earlier Grove reconcile must not survive the switch.
//
// The placement projection is established before any workload operation, so the
// contract is asserted on the returned status whether or not the rest of the
// component pathway completes in this fixture.
func TestComponentProgramReportsUnsupportedPlacement(t *testing.T) {
	t.Log("Given a DGD that still carries a placement score from a Grove reconcile")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: commonconsts.ComponentTypeWorker},
	})
	dgd.Status.Placement = &nvidiacomv1beta1.PlacementStatus{
		Score: ptr.To(0.42),
		State: nvidiacomv1beta1.PlacementScoreStateReported,
	}
	dgd.Status.State = nvidiacomv1beta1.DGDStateSuccessful
	reconciler := createTestDGDReconcilerWithStatus(dgd)
	program := reconciler.newComponentProgram()

	t.Log("When the component pathway reconciles the graph")
	result, _ := program.Reconcile(context.Background(), workloadProgramRequest{DGD: dgd})

	t.Log("Then it reports Unsupported and clears the stale score")
	require.NotNil(t, result.Status.Placement, "placement status must always be set")
	assert.Equal(t, nvidiacomv1beta1.PlacementScoreStateUnsupported, result.Status.Placement.State)
	assert.Nil(t, result.Status.Placement.Score, "the Grove score must not survive the pathway switch")
}
