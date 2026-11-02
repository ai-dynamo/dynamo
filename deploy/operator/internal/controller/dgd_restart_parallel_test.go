/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
)

func TestComputeParallelRestartStatus(t *testing.T) {
	tests := []struct {
		name         string
		dgd          *nvidiacomv1beta1.DynamoGraphDeployment
		resolved     []string
		wantStatus   *nvidiacomv1beta1.RestartStatus
		wantResolved []string // nil means the resolver must not be called
	}{
		{
			name: "new request returns all components sorted before resolving",
			dgd:  newParallelRestartDGD("restart-1", nil, namedComponents("worker", "frontend", "planner")...),
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-1",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"frontend", "planner", "worker"},
			},
		},
		{
			name: "new request replacing an older ID resets to all components",
			dgd: newParallelRestartDGD("restart-2", &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-1",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"worker"},
			}, namedComponents("worker", "frontend")...),
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"frontend", "worker"},
			},
		},
		{
			name: "new request with no components completes",
			dgd:  newParallelRestartDGD("restart-1", nil),
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-1",
				Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
			},
		},
		{
			name: "ongoing restart resolves exactly the stored in-progress subset",
			dgd: newParallelRestartDGD("restart-1", &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-1",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"worker", "frontend"},
			}, namedComponents("frontend", "planner", "worker")...),
			resolved:     []string{"worker"},
			wantResolved: []string{"worker", "frontend"},
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-1",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"worker"},
			},
		},
		{
			name: "ongoing restart with empty in-progress resolves all components sorted",
			dgd: newParallelRestartDGD("restart-1", &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-1",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			}, namedComponents("worker", "frontend")...),
			resolved:     []string{"frontend"},
			wantResolved: []string{"frontend", "worker"},
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-1",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"frontend"},
			},
		},
		{
			name: "empty resolver result completes",
			dgd: newParallelRestartDGD("restart-1", &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-1",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"worker"},
			}, namedComponents("worker")...),
			wantResolved: []string{"worker"},
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-1",
				Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Record which components the progress resolver observes")
			ctx := context.Background()

			var calls [][]string
			resolver := func(_ context.Context, _ *nvidiacomv1beta1.DynamoGraphDeployment, inProgress []string) []string {
				calls = append(calls, append([]string(nil), inProgress...))
				return tt.resolved
			}

			t.Log("Check the restart transition and resolver calls")
			got := computeParallelRestartStatus(ctx, logr.Discard(), tt.dgd, resolver)
			require.Equal(t, tt.wantStatus, got)
			if tt.wantResolved == nil {
				require.Empty(t, calls)
			} else {
				require.Equal(t, [][]string{tt.wantResolved}, calls)
			}
		})
	}
}
