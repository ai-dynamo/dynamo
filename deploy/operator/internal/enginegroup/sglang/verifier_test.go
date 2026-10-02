/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServingVerifierReturnsCanonicalTopologyProof(t *testing.T) {
	httpClient := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		assert.Equal(t, "/v1/completions", request.URL.Path)
		return jsonResponse(`{"choices":[{"text":"ok"}]}`), nil
	})}
	verifier, err := NewServingVerifier("http://frontend.test:8000/v1/completions", "model", httpClient)
	require.NoError(t, err)
	now := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	verifier.Now = func() time.Time { return now }
	topology := enginegroup.MembershipTopology{Generation: 2, Replicas: []enginegroup.ReplicaMembership{
		{ReplicaID: "replica-0", RuntimeIncarnation: "pod-0", NativeMembers: []enginegroup.NativeMemberID{"dp-0"}},
		{ReplicaID: "replica-1", RuntimeIncarnation: "pod-1", NativeMembers: []enginegroup.NativeMemberID{"dp-1"}},
	}}

	t.Log("prove progress through the real serving path and bind it to the coordinator digest")
	result, err := verifier.Verify(context.Background(), "group", topology)
	require.NoError(t, err)
	require.NotNil(t, result.Proof)
	assert.Equal(t, enginegroup.TopologyRuntimeDigest(topology), result.Proof.RuntimeDigest)
	assert.Equal(t, now, result.Proof.ObservedAt)
}
