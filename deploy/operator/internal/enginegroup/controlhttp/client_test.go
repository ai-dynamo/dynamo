/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controlhttp

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRejectAmbiguousEndpoints(t *testing.T) {
	for _, endpoint := range []string{
		"ftp://engine.test", "http:///missing-host", "http://user:password@engine.test",
		"http://engine.test/prefix", "http://engine.test?query=1", "http://engine.test?",
		"http://engine.test#fragment", "http://engine.test/%2f",
	} {
		t.Run(endpoint, func(t *testing.T) {
			t.Log("refuse endpoint components that can change the identity-bound destination")
			_, err := New(endpoint, http.DefaultClient)
			require.Error(t, err)
		})
	}
}

func TestRedirectCannotRetargetProcess(t *testing.T) {
	t.Log("create a redirect to a different engine process")
	var redirected atomic.Int32
	destination := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirected.Add(1)
		w.WriteHeader(http.StatusOK)
	}))
	defer destination.Close()
	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, destination.URL, http.StatusTemporaryRedirect)
	}))
	defer origin.Close()
	original := origin.Client()
	control, err := New(origin.URL, original)
	require.NoError(t, err)

	t.Log("retain HTTP refusal without contacting the other process or changing the caller client")
	_, err = control.Request(t.Context(), http.MethodPost, "/resize", map[string]int{"size": 4})
	var refusal *HTTPError
	require.ErrorAs(t, err, &refusal)
	assert.Equal(t, http.StatusTemporaryRedirect, refusal.StatusCode)
	assert.Zero(t, redirected.Load())
	assert.Nil(t, original.CheckRedirect)
}

func TestResponseLimitsAndContext(t *testing.T) {
	for _, tc := range []struct {
		name      string
		status    int
		body      string
		wantError bool
	}{
		{"bounded JSON response", http.StatusOK, "{}", false},
		{"oversized success", http.StatusOK, strings.Repeat(" ", MaximumResponseBytes+1), true},
		{"bounded refusal", http.StatusConflict, strings.Repeat("x", 8192), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("read the entire response within the shared byte limit")
			control, err := New("http://engine.test", &http.Client{Transport: testTransport{
				status: tc.status, body: tc.body,
			}})
			require.NoError(t, err)
			body, err := control.Request(t.Context(), http.MethodGet, "/status", nil)
			if !tc.wantError {
				require.NoError(t, err)
				assert.Equal(t, tc.body, string(body))
			} else {
				require.Error(t, err)
				assert.Nil(t, body)
				if tc.status == http.StatusConflict {
					var refusal *HTTPError
					require.ErrorAs(t, err, &refusal)
					assert.Len(t, refusal.Body, 4096)
				}
			}

			t.Log("preserve cancellation for the backend to interpret without inventing an operation outcome")
			ctx, cancel := context.WithCancel(t.Context())
			cancel()
			_, err = control.Request(ctx, http.MethodGet, "/status", nil)
			require.ErrorIs(t, err, context.Canceled)
		})
	}
}

type testTransport struct {
	status int
	body   string
}

func (t testTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	if err := request.Context().Err(); err != nil {
		return nil, err
	}
	return &http.Response{StatusCode: t.status, Body: io.NopCloser(strings.NewReader(t.body)), Header: make(http.Header)}, nil
}
