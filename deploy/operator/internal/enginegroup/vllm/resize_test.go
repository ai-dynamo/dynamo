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
			for rank := int32(0); rank < 2; rank++ {
				instance := fmt.Sprintf("api-%d", rank)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var request ScaleRequest
					if !assert.NoError(t, json.NewDecoder(r.Body).Decode(&request)) {
						return
					}
					assert.Equal(t, ScaleRequest{
						OperationID: "resize-1", ExpectedInstanceID: instance,
						NewDataParallelSize: target, DrainTimeout: 120,
					}, request)
					arrived <- rank
					select {
					case <-release:
					case <-r.Context().Done():
						return
					}
					_ = json.NewEncoder(w).Encode(map[string]any{
						"instance_id": instance, "operation_id": "resize-1", "epoch": "epoch-1",
						"phase": "completed", "requested_data_parallel_size": target,
						"effective_data_parallel_size": target, "topology_known": true,
					})
				}))
				t.Cleanup(server.Close)
				control, err := NewClient(server.URL, server.Client())
				require.NoError(t, err)
				participants = append(participants, Participant{Rank: rank, ExpectedInstanceID: instance, Client: control})
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
				assert.Equal(t, PhaseCompleted, observation.Status.Phase)
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
	accepted := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"instance_id": "api-1", "operation_id": "grow-1", "epoch": "epoch-1",
			"phase": "preparing", "requested_data_parallel_size": 3,
			"effective_data_parallel_size": 2, "topology_known": true,
		})
	}))
	t.Cleanup(accepted.Close)
	control0, err := NewClient(conflict.URL, conflict.Client())
	require.NoError(t, err)
	control1, err := NewClient(accepted.URL, accepted.Client())
	require.NoError(t, err)
	resize := Resize{
		OperationID: "grow-1", BaseDataParallelSize: 2, NewDataParallelSize: 3, DrainTimeout: 120,
		Participants: []Participant{
			{Rank: 0, ExpectedInstanceID: "api-0", Client: control0},
			{Rank: 1, ExpectedInstanceID: "api-1", Client: control1},
		},
	}

	t.Log("preserve successful peer evidence and the typed conflict without claiming rejection")
	results, err := resize.Apply(t.Context())
	var conflictErr *HTTPError
	require.ErrorAs(t, err, &conflictErr)
	assert.Equal(t, http.StatusConflict, conflictErr.StatusCode)
	require.Len(t, results, 2)
	assert.Error(t, results[0].Err)
	assert.NoError(t, results[1].Err)
	assert.Equal(t, PhasePreparing, results[1].Status.Phase)
	assert.Equal(t, int32(2), calls.Load())
}

func TestResizeValidatesCompleteParticipantsBeforeEffects(t *testing.T) {
	tests := []struct {
		name      string
		base      int32
		target    int32
		ranks     []int32
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
				participants[i] = Participant{Rank: rank, ExpectedInstanceID: test.instances[i], Client: control}
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
