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
		{ReplicaID: "replica-0", Members: []enginegroup.NativeMemberIncarnation{{ID: "dp-0", RuntimeIncarnation: "pod-0"}}},
		{ReplicaID: "replica-1", Members: []enginegroup.NativeMemberIncarnation{{ID: "dp-1", RuntimeIncarnation: "pod-1"}}},
	}}

	t.Log("prove progress through the real serving path and bind it to the coordinator digest")
	result, err := verifier.Verify(context.Background(), "group", topology)
	require.NoError(t, err)
	require.NotNil(t, result.Proof)
	assert.Equal(t, enginegroup.TopologyRuntimeDigest(topology), result.Proof.RuntimeDigest)
	assert.Equal(t, now, result.Proof.ObservedAt)
}

func TestServingVerifierRetriesInconclusiveProgress(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		body       string
	}{
		{name: "warming serving endpoint", statusCode: http.StatusServiceUnavailable, body: `{}`},
		{name: "no completion progress", statusCode: http.StatusOK, body: `{"choices":[]}`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("return transient serving evidence rather than a definitive topology rejection")
			verifier, err := NewServingVerifier("http://sglang.test/v1/completions", "model", &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
				response := jsonResponse(test.body)
				response.StatusCode = test.statusCode
				return response, nil
			})})
			require.NoError(t, err)

			t.Log("leave verification pending by returning an ordinary retryable error")
			result, err := verifier.Verify(t.Context(), "group", enginegroup.MembershipTopology{Generation: 2})
			require.Error(t, err)
			assert.Nil(t, result.Proof)
			assert.Nil(t, result.Failure)
		})
	}
}
