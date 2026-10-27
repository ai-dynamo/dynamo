/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClientObservesElasticEPState(t *testing.T) {
	t.Log("read the engine-authoritative EP size and asynchronous phase")
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		assert.Equal(t, statePath, r.URL.Path)
		return jsonResponse(`{"is_scaling_elastic_ep":false,"effective_ep_size":2,"scale_phase":"serving_expanded","last_error":null,"runtime_health":"healthy","runtime_error":null}`), nil
	})}
	client, err := NewClient("http://sglang.test:9090", httpClient)
	require.NoError(t, err)

	state, err := client.Observe(context.Background())
	require.NoError(t, err)
	assert.Equal(t, int32(2), state.EffectiveEPSize)
	assert.False(t, state.Scaling)
	assert.Equal(t, "serving_expanded", state.Phase)
	assert.Equal(t, "healthy", state.RuntimeHealth)
	assert.Empty(t, state.RuntimeError)
}

func TestClientUsesSeparateObservationAndScaleBudgets(t *testing.T) {
	tests := []struct {
		name         string
		parentBudget time.Duration
	}{
		{name: "independent request deadlines"},
		{name: "shorter caller deadline", parentBudget: time.Second},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("inspect the deadline on each real control request")
			var observedDeadline, scaleDeadline time.Time
			control, err := NewClient("http://sglang.test:9090", &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
				deadline, ok := request.Context().Deadline()
				require.True(t, ok)
				if request.Method == http.MethodGet {
					observedDeadline = deadline
					return jsonResponse(`{"effective_ep_size":2,"runtime_health":"healthy"}`), nil
				}
				scaleDeadline = deadline
				return jsonResponse(`{"status":"ok"}`), nil
			})})
			require.NoError(t, err)
			ctx := t.Context()
			if test.parentBudget > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, test.parentBudget)
				defer cancel()
			}

			t.Log("observe with a short budget and dispatch with the legacy resize budget")
			started := time.Now()
			_, err = control.Observe(ctx)
			require.NoError(t, err)
			_, err = control.Scale(ctx, ScaleRequest{NewEPSize: 2, OperationID: "grow-1-2", ExpectedInstanceID: "runtime-1", ExpectedJoiningAllocationIDs: []string{"pod-1"}})
			require.NoError(t, err)

			t.Log("preserve a shorter caller deadline and otherwise separate the two budgets")
			if test.parentBudget > 0 {
				deadline, _ := ctx.Deadline()
				assert.Equal(t, deadline, observedDeadline)
				assert.Equal(t, deadline, scaleDeadline)
			} else {
				assert.WithinDuration(t, started.Add(defaultObservationTimeout), observedDeadline, time.Second)
				assert.WithinDuration(t, started.Add(defaultScaleTimeout), scaleDeadline, time.Second)
			}
		})
	}
}

func TestClientRequestsAbsoluteElasticEPTarget(t *testing.T) {
	t.Log("send the absolute target through the Dynamo SGLang control route")
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, scalePath, r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		return jsonResponse(`{"status":"ok","old_ep_size":1,"new_ep_size":2}`), nil
	})}
	client, err := NewClient("http://sglang.test:9090", httpClient)
	require.NoError(t, err)

	response, err := client.Scale(context.Background(), ScaleRequest{NewEPSize: 2, OperationID: "grow-1-2", ExpectedInstanceID: "runtime-1", ExpectedJoiningAllocationIDs: []string{"pod-1"}})
	require.NoError(t, err)
	assert.Equal(t, "ok", response.Status)
	assert.Equal(t, int32(2), response.NewEPSize)
}

func TestClientRejectsInvalidState(t *testing.T) {
	cases := []struct {
		name             string
		body             string
		message          string
		recoveryRequired bool
	}{
		{name: "disabled", body: `{"status":"error","message":"elastic EP is disabled"}`, message: "elastic EP is disabled"},
		{name: "invalid size", body: `{"effective_ep_size":0,"runtime_health":"healthy"}`, message: "invalid effective EP size"},
		{name: "unsupported recovery", body: `{"effective_ep_size":3,"operation_succeeded":true,"runtime_health":"recovery_unsupported","runtime_error":"append rank recovery is unsupported"}`,
			message: "append rank recovery is unsupported", recoveryRequired: true},
		{name: "missing health", body: `{"effective_ep_size":3}`, message: "unknown runtime health"},
		{name: "unknown health", body: `{"effective_ep_size":3,"runtime_health":"unrecognized"}`, message: "unknown runtime health"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("read native state without substituting a completed resize for current runtime health")
			control, err := NewClient("http://sglang.test:9090", &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
				return jsonResponse(tc.body), nil
			})})
			require.NoError(t, err)

			t.Log("fail closed without returning usable membership or traffic evidence")
			state, err := control.Observe(t.Context())
			require.ErrorContains(t, err, tc.message)
			assert.Equal(t, ScaleState{}, state)
			if tc.recoveryRequired {
				assert.ErrorIs(t, err, enginegroup.ErrRecoveryRequired)
			}
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func jsonResponse(body string) *http.Response {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
