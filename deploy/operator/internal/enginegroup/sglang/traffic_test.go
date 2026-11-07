/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/kubejournal"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestTrafficProjectionFollowsEngineWithoutAdmissionTargets(t *testing.T) {
	t.Log("project the initial world without any controller traffic target")
	engineSize := int32(1)
	observationFailed := false
	control, err := NewClient("http://sglang.test:9090", &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if request.URL.Path != statePath || request.Method != http.MethodGet {
			return nil, fmt.Errorf("unexpected serving mutation %s %s", request.Method, request.URL.Path)
		}
		if observationFailed {
			return nil, fmt.Errorf("engine observation unavailable")
		}
		return jsonResponse(fmt.Sprintf(`{"effective_ep_size":%d,"scale_phase":"serving_expanded","runtime_health":"healthy"}`, engineSize)), nil
	})})
	require.NoError(t, err)
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	capacity := &testCapacityObserver{allocations: []enginegroup.CapacityAllocation{testAllocation(0)}}
	projection := &TrafficProjection{
		Client: control, Capacity: capacity,
		Journal: kubejournal.NewStore(kubeClient, "test", "group", "group-uid", "traffic"),
	}
	observed, err := projection.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	require.Len(t, observed.Admitted, 1)
	assert.Zero(t, observed.AppliedRevision)

	t.Log("accept an empty withdrawal target without sending a serving command")
	target := enginegroup.TrafficTarget{ControlRevision: 1, TransitionID: "grow", TopologyGeneration: 1}
	result, err := projection.Apply(t.Context(), "group-uid", target)
	require.NoError(t, err)
	require.Nil(t, result.Rejection)

	t.Log("observe independently committed growth after reconstructing the adapter, without a new traffic revision")
	engineSize = 2
	capacity.allocations = append(capacity.allocations, testAllocation(1))
	restarted := &TrafficProjection{Client: control, Capacity: capacity, Journal: projection.Journal}
	observed, err = restarted.Observe(t.Context(), "group-uid")
	require.NoError(t, err)
	assert.Len(t, observed.Admitted, 2)
	assert.Equal(t, int64(1), observed.AppliedRevision)

	t.Log("a failed engine read never republishes the previous projection as current evidence")
	observationFailed = true
	observed, err = restarted.Observe(t.Context(), "group-uid")
	require.ErrorContains(t, err, "engine observation unavailable")
	assert.Empty(t, observed.Admitted)
}

func TestTrafficProjectionCannotInventDrainEvidence(t *testing.T) {
	t.Log("reject a drain target before persisting compatibility evidence")
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	kubeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	journal := kubejournal.NewStore(kubeClient, "test", "group", types.UID("group-uid"), "traffic")
	projection := &TrafficProjection{Journal: journal}
	target := enginegroup.TrafficTarget{
		ControlRevision: 1,
		Drain: []enginegroup.TrafficDrainTarget{{
			Membership: enginegroup.ReplicaMembership{
				ReplicaID: "replica-0", Members: []enginegroup.NativeMemberIncarnation{{ID: "dp-0", RuntimeIncarnation: "pod-0"}},
			},
			Mode: enginegroup.TrafficDrainModeGraceful,
		}},
	}
	result, err := projection.Apply(t.Context(), "group", target)
	require.NoError(t, err)
	require.NotNil(t, result.Rejection)
	assert.Equal(t, "DrainUnsupported", result.Rejection.Reason)
	snapshot, err := journal.Load(t.Context(), &trafficJournal{})
	require.NoError(t, err)
	assert.False(t, snapshot.Exists())

	t.Log("refuse to translate a persisted drain intent into completed drain evidence")
	_, err = journal.Save(t.Context(), snapshot, trafficJournal{AppliedRevision: 1, Target: target})
	require.NoError(t, err)
	observation, err := projection.Observe(t.Context(), "group")
	require.ErrorContains(t, err, "cannot prove drain completion")
	assert.Empty(t, observation.Drained)
}
