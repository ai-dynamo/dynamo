/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"strings"
	"sync"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	parallelRestartNewRequestMsg = "New restart request detected, resetting to all components"
	parallelRestartCompletedMsg  = "Restart completed for all components"
)

// capturedLogs collects the messages written through a funcr logger.
type capturedLogs struct {
	mu   sync.Mutex
	msgs []string
}

func newCapturingLogger() (logr.Logger, *capturedLogs) {
	captured := &capturedLogs{}
	logger := funcr.New(func(prefix, args string) {
		captured.mu.Lock()
		defer captured.mu.Unlock()
		captured.msgs = append(captured.msgs, args)
	}, funcr.Options{Verbosity: 10})
	return logger, captured
}

func (c *capturedLogs) contains(msg string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, line := range c.msgs {
		if strings.Contains(line, msg) {
			return true
		}
	}
	return false
}

// newParallelRestartDGD builds a DGD that requests a parallel restart with the
// given ID and components (in the given order).
func newParallelRestartDGD(
	restartID string,
	status *nvidiacomv1beta1.RestartStatus,
	components ...nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "parallel-restart", Namespace: "default", UID: "parallel-restart-uid"},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: components,
			Restart: &nvidiacomv1beta1.Restart{
				ID:       restartID,
				Strategy: &nvidiacomv1beta1.RestartStrategy{Type: nvidiacomv1beta1.RestartStrategyTypeParallel},
			},
		},
		Status: nvidiacomv1beta1.DynamoGraphDeploymentStatus{Restart: status},
	}
}

func namedComponents(names ...string) []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	components := make([]nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec, 0, len(names))
	for _, name := range names {
		components = append(components, nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: name})
	}
	return components
}

// dsParallelRestartComponents returns two DisaggregatedSet-eligible worker
// components, listed in unsorted order.
func dsParallelRestartComponents() []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
	return []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
		{
			ComponentName: "prefill",
			ComponentType: nvidiacomv1beta1.ComponentTypePrefill,
			Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
		},
		{
			ComponentName: "decode",
			ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
			Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
		},
	}
}

// TestParallelRestartSharedEntryPath drives the component/Grove entry point and
// checks it uses the caller's resolver and logs through the context logger.
func TestParallelRestartSharedEntryPath(t *testing.T) {
	tests := []struct {
		name         string
		dgd          *nvidiacomv1beta1.DynamoGraphDeployment
		resolved     []string
		wantStatus   *nvidiacomv1beta1.RestartStatus
		wantResolved []string
		wantLog      string
	}{
		{
			name: "new request starts all components sorted without resolving",
			dgd:  newParallelRestartDGD("restart-2", nil, namedComponents("worker", "frontend", "planner")...),
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"frontend", "planner", "worker"},
			},
			wantLog: parallelRestartNewRequestMsg,
		},
		{
			name: "ongoing restart completes when resolver reports nothing pending",
			dgd: newParallelRestartDGD("restart-2", &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"worker", "frontend"},
			}, namedComponents("frontend", "worker")...),
			resolved:     nil,
			wantResolved: []string{"worker", "frontend"},
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
			},
			wantLog: parallelRestartCompletedMsg,
		},
		{
			name: "ongoing restart keeps resolver result in progress",
			dgd: newParallelRestartDGD("restart-2", &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"worker", "frontend"},
			}, namedComponents("frontend", "worker")...),
			resolved:     []string{"worker"},
			wantResolved: []string{"worker", "frontend"},
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"worker"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, captured := newCapturingLogger()
			ctx := log.IntoContext(context.Background(), logger)

			var calls [][]string
			resolver := func(_ context.Context, _ *nvidiacomv1beta1.DynamoGraphDeployment, inProgress []string) []string {
				calls = append(calls, append([]string(nil), inProgress...))
				return tt.resolved
			}

			got := newDGDRestartReconciler().computeRestartStatusWithProgressResolver(ctx, tt.dgd, resolver)
			require.Equal(t, tt.wantStatus, got)
			if tt.wantResolved == nil {
				require.Empty(t, calls)
			} else {
				require.Equal(t, [][]string{tt.wantResolved}, calls)
			}
			if tt.wantLog != "" {
				require.True(t, captured.contains(tt.wantLog), "expected log %q, got %v", tt.wantLog, captured.msgs)
			}
		})
	}
}

// TestParallelRestartDisaggregatedSetEntryPath drives the DisaggregatedSet
// entry point with the real DisaggregatedSet progress resolver and checks that
// it does not write the parallel restart progress messages.
func TestParallelRestartDisaggregatedSetEntryPath(t *testing.T) {
	tests := []struct {
		name       string
		dgd        *nvidiacomv1beta1.DynamoGraphDeployment
		wantStatus *nvidiacomv1beta1.RestartStatus
	}{
		{
			name: "new request starts all components sorted",
			dgd:  newParallelRestartDGD("restart-2", nil, dsParallelRestartComponents()...),
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"decode", "prefill"},
			},
		},
		{
			name: "new request with no components completes",
			dgd:  newParallelRestartDGD("restart-2", nil),
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
			},
		},
		{
			name: "ongoing subset stays in progress while DisaggregatedSet is missing",
			dgd: newParallelRestartDGD("restart-2", &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"prefill"},
			}, dsParallelRestartComponents()...),
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
				InProgress: []string{"prefill"},
			},
		},
		{
			name: "ongoing restart with no components completes",
			dgd: newParallelRestartDGD("restart-2", &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseRestarting,
			}),
			wantStatus: &nvidiacomv1beta1.RestartStatus{
				ObservedID: "restart-2",
				Phase:      nvidiacomv1beta1.RestartPhaseCompleted,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scheme := runtime.NewScheme()
			require.NoError(t, nvidiacomv1beta1.AddToScheme(scheme))
			scheme.AddKnownTypeWithName(disaggregatedSetGVK, &unstructured.Unstructured{})
			scheme.AddKnownTypeWithName(disaggregatedSetGVK.GroupVersion().WithKind("DisaggregatedSetList"), &unstructured.UnstructuredList{})
			k8sClient := fake.NewClientBuilder().WithScheme(scheme).Build()
			r := &disaggregatedSetWorkloadsReconciler{
				reader: k8sClient,
				restartProgress: newDisaggregatedSetRestartProgressResolver(
					k8sClient,
					newDisaggregatedSetReadinessResolver(k8sClient),
					newComponentRestartProgressResolver(k8sClient),
				),
			}

			logger, captured := newCapturingLogger()
			ctx := log.IntoContext(context.Background(), logger)

			got := r.computeRestartStatus(ctx, tt.dgd)
			require.Equal(t, tt.wantStatus, got)
			require.False(t, captured.contains(parallelRestartNewRequestMsg), "unexpected log: %v", captured.msgs)
			require.False(t, captured.contains(parallelRestartCompletedMsg), "unexpected log: %v", captured.msgs)
		})
	}
}
