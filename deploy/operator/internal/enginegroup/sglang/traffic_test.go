/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
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
				ReplicaID: "replica-0", RuntimeIncarnation: "pod-0",
				NativeMembers: []enginegroup.NativeMemberID{"dp-0"},
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
