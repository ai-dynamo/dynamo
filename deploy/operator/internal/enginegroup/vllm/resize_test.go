/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/controlhttp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResizeFansOutConcurrentlyToEveryOldRank(t *testing.T) {
	for _, target := range []int32{3, 1} {
		t.Run(fmt.Sprintf("DP2 to DP%d", target), func(t *testing.T) {
			t.Log("make every old rank wait until the complete fan-out has arrived")
			arrived := make(chan int32, 2)
			release := make(chan struct{})
			participants := make([]Participant, 0, 2)
			for rank := range int32(2) {
				instance := fmt.Sprintf("api-%d", rank)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var request ScaleRequest
					if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&request)) {
						return
					}
					assert.Equal(t, ScaleRequest{
						OperationID: "resize-1", ExpectedInstanceID: instance,
						NewDataParallelSize: target, DrainTimeout: 120,
						BaseTopologyToken: "base-token",
					}, request)
					arrived <- rank
					select {
					case <-release:
					case <-r.Context().Done():
						return
					}
					_ = json.NewEncoder(w).Encode(SubmissionReceipt{Request: request, FirstRank: rank, LocalSize: 1})
				}))
				t.Cleanup(server.Close)
				control, err := NewClient(server.URL, server.Client())
				require.NoError(t, err)
				participants = append(participants, Participant{Rank: rank, LocalSize: 1, ExpectedInstanceID: instance, BaseTopologyToken: "base-token", Client: control})
			}
			resize := Resize{
				OperationID: "resize-1", BaseDataParallelSize: 2, NewDataParallelSize: target,
				DrainTimeout: 120, Participants: participants,
			}

			t.Log("start the complete resize; suffix shrink must still contact departing rank one")
			type outcome struct {
				results []RankResult
				err     error
			}
			done := make(chan outcome, 1)
			go func() {
				results, err := resize.Apply(t.Context())
				done <- outcome{results: results, err: err}
			}()
			seen := make(map[int32]bool)
			for range participants {
				select {
				case rank := <-arrived:
					seen[rank] = true
				case <-time.After(5 * time.Second):
					close(release)
					t.Fatal("resize did not contact every old rank concurrently")
				}
			}
			close(release)

			t.Log("only return after each rank has independently supplied correlated evidence")
			result := <-done
			require.NoError(t, result.err)
			require.Len(t, result.results, 2)
			assert.Len(t, seen, 2)
			for i, observation := range result.results {
				assert.Equal(t, participants[i].Rank, observation.Rank)
				assert.Equal(t, participants[i].Rank, observation.Receipt.FirstRank)
				assert.False(t, observation.Receipt.Finished, "acceptance does not wait for collective completion")
				assert.NoError(t, observation.Err)
			}
		})
	}
}

func TestResizePreservesMixedFanoutEvidence(t *testing.T) {
	t.Log("one old rank reports conflict while its peer has already accepted the collective")
	var calls atomic.Int32
	conflict := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		http.Error(w, "operation already active", http.StatusConflict)
	}))
	t.Cleanup(conflict.Close)
	accepted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var target ScaleRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&target))
		_ = json.NewEncoder(w).Encode(SubmissionReceipt{Request: target, FirstRank: 1, LocalSize: 1, Started: true})
	}))
	t.Cleanup(accepted.Close)
	control0, err := NewClient(conflict.URL, conflict.Client())
	require.NoError(t, err)
	control1, err := NewClient(accepted.URL, accepted.Client())
	require.NoError(t, err)
	resize := Resize{
		OperationID: "grow-1", BaseDataParallelSize: 2, NewDataParallelSize: 3, DrainTimeout: 120,
		Participants: []Participant{
			{Rank: 0, LocalSize: 1, ExpectedInstanceID: "api-0", BaseTopologyToken: "base-0", Client: control0},
			{Rank: 1, LocalSize: 1, ExpectedInstanceID: "api-1", BaseTopologyToken: "base-1", Client: control1},
		},
	}

	t.Log("preserve successful peer evidence and the typed conflict without claiming rejection")
	results, err := resize.Apply(t.Context())
	var conflictErr *controlhttp.HTTPError
	require.ErrorAs(t, err, &conflictErr)
	assert.Equal(t, http.StatusConflict, conflictErr.StatusCode)
	require.Len(t, results, 2)
	assert.Error(t, results[0].Err)
	assert.NoError(t, results[1].Err)
	assert.True(t, results[1].Receipt.Started)
	assert.False(t, results[1].Receipt.Finished)
	assert.Equal(t, int32(2), calls.Load())
}

func TestResizeValidatesCompleteParticipantsBeforeEffects(t *testing.T) {
	tests := []struct {
		name      string
		base      int32
		target    int32
		ranks     []int32
		widths    []int32
		instances []string
	}{
		{name: "missing old rank", base: 2, target: 3, ranks: []int32{0}, instances: []string{"api-0"}},
		{name: "duplicate old rank", base: 2, target: 3, ranks: []int32{0, 0}, instances: []string{"api-0", "api-1"}},
		{name: "joiner in place of old rank", base: 2, target: 3, ranks: []int32{0, 2}, instances: []string{"api-0", "api-2"}},
		{name: "negative rank", base: 2, target: 3, ranks: []int32{0, -1}, instances: []string{"api-0", "api-1"}},
		{name: "same API process twice", base: 2, target: 3, ranks: []int32{0, 1}, instances: []string{"api-0", "api-0"}},
		{name: "missing process precondition", base: 2, target: 3, ranks: []int32{0, 1}, instances: []string{"api-0", ""}},
		{name: "zero target", base: 2, target: 0, ranks: []int32{0, 1}, instances: []string{"api-0", "api-1"}},
		{name: "no-op is not correlated commit", base: 2, target: 2, ranks: []int32{0, 1}, instances: []string{"api-0", "api-1"}},
		{name: "empty base", target: 3},
		{name: "target bisects an owner", base: 4, target: 3, ranks: []int32{0, 2}, widths: []int32{2, 2}, instances: []string{"api-0", "api-2"}},
		{name: "overlapping owner ranges", base: 4, target: 6, ranks: []int32{0, 2}, widths: []int32{3, 2}, instances: []string{"api-0", "api-2"}},
		{name: "owner extends beyond base", base: 4, target: 6, ranks: []int32{0, 2}, widths: []int32{2, 3}, instances: []string{"api-0", "api-2"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("prepare malformed native fan-out with a client that must never be invoked")
			var calls atomic.Int32
			control, err := NewClient("http://rank.test:8000", &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
				calls.Add(1)
				return nil, errors.New("must not be called")
			})})
			require.NoError(t, err)
			participants := make([]Participant, len(test.ranks))
			for i, rank := range test.ranks {
				participants[i] = Participant{Rank: rank, LocalSize: 1, ExpectedInstanceID: test.instances[i], Client: control}
				if test.widths != nil {
					participants[i].LocalSize = test.widths[i]
				}
			}
			resize := Resize{
				OperationID: "resize-1", BaseDataParallelSize: test.base, NewDataParallelSize: test.target,
				DrainTimeout: 120, Participants: participants,
			}

			t.Log("reject the entire invocation before any rank receives a request")
			results, err := resize.Apply(t.Context())
			require.Error(t, err)
			assert.Nil(t, results)
			assert.Zero(t, calls.Load())
		})
	}
}
