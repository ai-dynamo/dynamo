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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClientObservesElasticEPState(t *testing.T) {
	t.Log("read the engine-authoritative EP size and asynchronous phase")
	httpClient := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		assert.Equal(t, statePath, r.URL.Path)
		return jsonResponse(`{"is_scaling_elastic_ep":false,"effective_ep_size":2,"scale_phase":"serving_expanded","last_error":null}`), nil
	})}
	client, err := NewClient("http://sglang.test:9090", httpClient)
	require.NoError(t, err)

	state, err := client.Observe(context.Background())
	require.NoError(t, err)
	assert.Equal(t, int32(2), state.EffectiveEPSize)
	assert.False(t, state.Scaling)
	assert.Equal(t, "serving_expanded", state.Phase)
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

	response, err := client.Scale(context.Background(), 2)
	require.NoError(t, err)
	assert.Equal(t, "ok", response.Status)
	assert.Equal(t, int32(2), response.NewEPSize)
}

func TestClientRejectsInvalidState(t *testing.T) {
	t.Log("fail closed when the worker route does not report an authoritative EP size")
	httpClient := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		return jsonResponse(`{"status":"error","message":"elastic EP is disabled"}`), nil
	})}
	client, err := NewClient("http://sglang.test:9090", httpClient)
	require.NoError(t, err)

	_, err = client.Observe(context.Background())
	assert.ErrorContains(t, err, "elastic EP is disabled")
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
