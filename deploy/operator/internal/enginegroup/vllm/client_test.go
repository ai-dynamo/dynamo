/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package vllm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestObserveNativeStatus(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		phase   Phase
		known   bool
		wantErr string
	}{
		{name: "initial world", payload: `{"instance_id":"api-0","operation_id":null,"epoch":null,"phase":"idle","requested_data_parallel_size":null,"effective_data_parallel_size":2,"topology_known":true,"error":null}`, phase: PhaseIdle, known: true},
		{name: "prepare remains observable", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"preparing","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":true,"error":null}`, phase: PhasePreparing, known: true},
		{name: "commit does not claim last known size authoritative", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"committing","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":false,"error":null}`, phase: PhaseCommitting},
		{name: "completed topology", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"completed","requested_data_parallel_size":3,"effective_data_parallel_size":3,"topology_known":true,"error":null,"message":"additive field"}`, phase: PhaseCompleted, known: true},
		{name: "failed prepare with known base", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"failed","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":true,"error":"prepare failed"}`, phase: PhaseFailed, known: true},
		{name: "failed commit with unknown topology", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"failed","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":false,"error":"commit failed"}`, phase: PhaseFailed},
		{name: "uncorrelated upstream rejected", payload: `{"phase":"preparing","epoch":"epoch-1","requested_data_parallel_size":3,"error":null}`, wantErr: "operation-status API is required"},
		{name: "null certainty rejected", payload: `{"instance_id":"api-0","phase":"idle","effective_data_parallel_size":2,"topology_known":null}`, wantErr: "lacks correlated topology evidence"},
		{name: "missing process identity rejected", payload: `{"phase":"idle","effective_data_parallel_size":2,"topology_known":true}`, wantErr: "process instance ID"},
		{name: "zero effective size rejected", payload: `{"instance_id":"api-0","phase":"idle","effective_data_parallel_size":0,"topology_known":true}`, wantErr: "positive effective DP size"},
		{name: "missing target rejected", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"preparing","effective_data_parallel_size":2,"topology_known":true}`, wantErr: "positive requested DP size"},
		{name: "empty epoch rejected", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"","phase":"preparing","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":true}`, wantErr: "operation status requires"},
		{name: "unknown phase rejected", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"mystery","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":true}`, wantErr: "unknown operation phase"},
		{name: "completion with wrong size rejected", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"completed","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":true}`, wantErr: "completed status requires"},
		{name: "completion with unknown topology rejected", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"completed","requested_data_parallel_size":3,"effective_data_parallel_size":3,"topology_known":false}`, wantErr: "completed status requires"},
		{name: "terminal error not attached to pending", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"preparing","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":true,"error":"stale error"}`, wantErr: "active operation cannot carry"},
		{name: "failure without explanation rejected", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"failed","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":false}`, wantErr: "failed status requires"},
		{name: "idle cannot hide operation", payload: `{"instance_id":"api-0","operation_id":"grow-1","phase":"idle","effective_data_parallel_size":2,"topology_known":true}`, wantErr: "idle status must describe"},
		{name: "idle cannot claim unknown", payload: `{"instance_id":"api-0","phase":"idle","effective_data_parallel_size":2,"topology_known":false}`, wantErr: "idle status must describe"},
		{name: "trailing JSON rejected", payload: `{"instance_id":"api-0","phase":"idle","effective_data_parallel_size":2,"topology_known":true} {}`, wantErr: "decode vLLM"},
		{name: "oversized response rejected", payload: strings.Repeat(" ", maximumResponseBytes+1), wantErr: "exceeds"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("serve one native status response through the actual HTTP route")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, http.MethodPost, r.Method)
				assert.Equal(t, statusPath, r.URL.Path)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, test.payload)
			}))
			t.Cleanup(server.Close)
			control, err := NewClient(server.URL, server.Client())
			require.NoError(t, err)

			t.Log("preserve native evidence or reject an incomplete or contradictory observation")
			status, err := control.Observe(t.Context())
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
				assert.Equal(t, Status{}, status)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, test.phase, status.Phase)
			assert.Equal(t, test.known, status.TopologyKnown)
			assert.Equal(t, "api-0", status.InstanceID)
		})
	}
}

func TestNativeAndWorkerRoutesPreserveTheSameProtocol(t *testing.T) {
	tests := []struct {
		name   string
		prefix string
	}{
		{name: "native API"},
		{name: "Dynamo worker", prefix: "/engine/control"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("expose the same native status through the selected control routes")
			type receivedRequest struct {
				path   string
				target ScaleRequest
			}
			requests := make(chan receivedRequest, 2)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, http.MethodPost, r.Method)
				var received ScaleRequest
				if r.URL.Path == test.prefix+scalePath {
					assert.NoError(t, json.NewDecoder(r.Body).Decode(&received))
				} else {
					assert.Equal(t, test.prefix+statusPath, r.URL.Path)
				}
				requests <- receivedRequest{path: r.URL.Path, target: received}
				_, _ = io.WriteString(w, `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"completed","requested_data_parallel_size":3,"effective_data_parallel_size":3,"topology_known":true,"error":null}`)
			}))
			t.Cleanup(server.Close)

			t.Log("choose the route placement once without changing transaction ownership")
			var control *Client
			var err error
			if test.prefix == "" {
				control, err = NewClient(server.URL, server.Client())
			} else {
				control, err = NewWorkerClient(server.URL, server.Client())
			}
			require.NoError(t, err)
			target := ScaleRequest{OperationID: "grow-1", ExpectedInstanceID: "api-0", NewDataParallelSize: 3, DrainTimeout: 120}

			t.Log("retain exact identity and operation correlation for both mutation and observation")
			scaled, err := control.Scale(t.Context(), target)
			require.NoError(t, err)
			observed, err := control.Observe(t.Context())
			require.NoError(t, err)
			scaleRequest, statusRequest := <-requests, <-requests
			assert.Equal(t, target, scaleRequest.target)
			assert.Equal(t, scaled, observed)
			assert.Equal(t, test.prefix+scalePath, scaleRequest.path)
			assert.Equal(t, test.prefix+statusPath, statusRequest.path)
		})
	}
}

func TestScalePreservesCorrelatedRequestAcrossRetry(t *testing.T) {
	t.Log("record the exact payload while the first reply is lost after native acceptance")
	requests := make(chan ScaleRequest, 2)
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, scalePath, r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		var request ScaleRequest
		if err := json.NewDecoder(r.Body).Decode(&request); !assert.NoError(t, err) {
			return
		}
		requests <- request
		if calls.Add(1) == 1 {
			connection, _, err := w.(http.Hijacker).Hijack()
			if assert.NoError(t, err) {
				_ = connection.Close()
			}
			return
		}
		_, _ = io.WriteString(w, `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"completed","requested_data_parallel_size":3,"effective_data_parallel_size":3,"topology_known":true,"error":null}`)
	}))
	t.Cleanup(server.Close)
	control, err := NewClient(server.URL, server.Client())
	require.NoError(t, err)
	request := ScaleRequest{OperationID: "grow-1", ExpectedInstanceID: "api-0", NewDataParallelSize: 3, DrainTimeout: 120}

	t.Log("return an ambiguous transport error without manufacturing a terminal rejection")
	_, err = control.Scale(t.Context(), request)
	require.ErrorContains(t, err, "call vLLM")
	var responseErr *HTTPError
	assert.False(t, errors.As(err, &responseErr))

	t.Log("a restarted client can replay the identical operation and instance precondition")
	restarted, err := NewClient(server.URL, server.Client())
	require.NoError(t, err)
	status, err := restarted.Scale(t.Context(), request)
	require.NoError(t, err)
	assert.Equal(t, PhaseCompleted, status.Phase)
	assert.Equal(t, request, <-requests)
	assert.Equal(t, request, <-requests)
	assert.Equal(t, int32(2), calls.Load())
}

func TestScaleChecksResponseCorrelation(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		wantErr string
	}{
		{name: "pending replay", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"preparing","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":true}`},
		{name: "failed commit remains native evidence not rejection", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"failed","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":false,"error":"collective failed"}`},
		{name: "different operation", payload: `{"instance_id":"api-0","operation_id":"other-grow","epoch":"epoch-1","phase":"completed","requested_data_parallel_size":3,"effective_data_parallel_size":3,"topology_known":true}`, wantErr: "does not match"},
		{name: "different target", payload: `{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"preparing","requested_data_parallel_size":4,"effective_data_parallel_size":2,"topology_known":true}`, wantErr: "does not match"},
		{name: "different process", payload: `{"instance_id":"replacement-api","operation_id":"grow-1","epoch":"epoch-1","phase":"preparing","requested_data_parallel_size":3,"effective_data_parallel_size":2,"topology_known":true}`, wantErr: "does not match"},
		{name: "same count without recorded operation", payload: `{"instance_id":"api-0","phase":"idle","effective_data_parallel_size":3,"topology_known":true}`, wantErr: "does not match"},
		{name: "legacy success payload", payload: `{"message":"Scaled to 3 data parallel engines"}`, wantErr: "operation-status API is required"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("answer one correlated request with native status")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				_, _ = io.WriteString(w, test.payload)
			}))
			t.Cleanup(server.Close)
			control, err := NewClient(server.URL, server.Client())
			require.NoError(t, err)

			t.Log("never infer this operation's success from a matching count alone")
			_, err = control.Scale(t.Context(), ScaleRequest{
				OperationID: "grow-1", ExpectedInstanceID: "api-0", NewDataParallelSize: 3, DrainTimeout: 120,
			})
			if test.wantErr != "" {
				require.ErrorContains(t, err, test.wantErr)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestHTTPFailuresStayInconclusive(t *testing.T) {
	for _, code := range []int{http.StatusBadRequest, http.StatusConflict, http.StatusRequestTimeout, http.StatusInternalServerError, http.StatusServiceUnavailable} {
		t.Run(http.StatusText(code), func(t *testing.T) {
			t.Log("return an unsuccessful response after the request may have reached native code")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "native outcome must be observed", code)
			}))
			t.Cleanup(server.Close)
			control, err := NewClient(server.URL, server.Client())
			require.NoError(t, err)

			t.Log("preserve the HTTP evidence without classifying the entire collective as rejected")
			_, err = control.Scale(t.Context(), ScaleRequest{
				OperationID: "grow-1", ExpectedInstanceID: "api-0", NewDataParallelSize: 3, DrainTimeout: 120,
			})
			var responseErr *HTTPError
			require.ErrorAs(t, err, &responseErr)
			assert.Equal(t, code, responseErr.StatusCode)
			assert.Equal(t, scalePath, responseErr.Path)
		})
	}
}

func TestClientBudgetsAndCancellation(t *testing.T) {
	tests := []struct {
		name   string
		budget time.Duration
	}{
		{name: "separate native budgets"},
		{name: "short caller deadline wins", budget: time.Second},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("capture the bounded contexts supplied to both native routes")
			deadlines := make(map[string]time.Time)
			control, err := NewClient("http://rank-0.test:8000", &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				deadline, exists := r.Context().Deadline()
				assert.True(t, exists)
				deadlines[r.URL.Path] = deadline
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(
					`{"instance_id":"api-0","operation_id":"grow-1","epoch":"epoch-1","phase":"completed","requested_data_parallel_size":3,"effective_data_parallel_size":3,"topology_known":true}`,
				))}, nil
			})})
			require.NoError(t, err)
			ctx := t.Context()
			if test.budget > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, test.budget)
				defer cancel()
			}

			t.Log("observe once and submit once without extending the caller's deadline")
			started := time.Now()
			_, err = control.Observe(ctx)
			require.NoError(t, err)
			_, err = control.Scale(ctx, ScaleRequest{
				OperationID: "grow-1", ExpectedInstanceID: "api-0", NewDataParallelSize: 3, DrainTimeout: 120,
			})
			require.NoError(t, err)
			if test.budget > 0 {
				deadline, _ := ctx.Deadline()
				assert.Equal(t, deadline, deadlines[statusPath])
				assert.Equal(t, deadline, deadlines[scalePath])
			} else {
				assert.WithinDuration(t, started.Add(defaultObservationTimeout), deadlines[statusPath], time.Second)
				assert.WithinDuration(t, started.Add(defaultScaleTimeout), deadlines[scalePath], time.Second)
			}
		})
	}
}

func TestCancelledNativeCallsRemainInconclusive(t *testing.T) {
	for _, scaling := range []bool{false, true} {
		t.Run(fmt.Sprintf("scaling=%t", scaling), func(t *testing.T) {
			t.Log("make native transport wait for the caller's cancellation")
			control, err := NewClient("http://rank-0.test:8000", &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
				<-r.Context().Done()
				return nil, r.Context().Err()
			})})
			require.NoError(t, err)
			ctx, cancel := context.WithCancel(t.Context())
			cancel()

			t.Log("preserve context cancellation without converting it into a native rejection")
			var status Status
			if scaling {
				status, err = control.Scale(ctx, ScaleRequest{
					OperationID: "grow-1", ExpectedInstanceID: "api-0", NewDataParallelSize: 3, DrainTimeout: 120,
				})
			} else {
				status, err = control.Observe(ctx)
			}
			require.ErrorIs(t, err, context.Canceled)
			assert.Equal(t, Status{}, status)
		})
	}
}

func TestControlURLAndRedirectSafety(t *testing.T) {
	t.Log("reject endpoint ambiguity before constructing a control client")
	for _, address := range []string{"", "file:///tmp/control", "http:///missing-host", "http://user:password@rank-0", "http://rank-0/engine", "http://rank-0/a/../engine", "http://rank-0/a%2Fengine", "http://rank-0?", "http://rank-0?query=x", "http://rank-0#fragment"} {
		_, err := NewClient(address, http.DefaultClient)
		assert.Error(t, err, address)
	}

	t.Log("refuse to forward a native mutation to a redirected process")
	var redirected atomic.Int32
	other := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, _ *http.Request) {
		redirected.Add(1)
	}))
	t.Cleanup(other.Close)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(server.Close)
	control, err := NewClient(server.URL, server.Client())
	require.NoError(t, err)
	_, err = control.Scale(t.Context(), ScaleRequest{
		OperationID: "grow-1", ExpectedInstanceID: "api-0", NewDataParallelSize: 3, DrainTimeout: 120,
	})
	var responseErr *HTTPError
	require.ErrorAs(t, err, &responseErr)
	assert.Equal(t, http.StatusTemporaryRedirect, responseErr.StatusCode)
	assert.Zero(t, redirected.Load())
}

func TestInvalidScaleRequestsHaveNoEffects(t *testing.T) {
	tests := []struct {
		name      string
		operation string
		instance  string
		target    int32
		drain     int32
	}{
		{name: "missing operation", instance: "api-0", target: 3, drain: 120},
		{name: "blank operation", operation: "  ", instance: "api-0", target: 3, drain: 120},
		{name: "long operation", operation: strings.Repeat("x", 257), instance: "api-0", target: 3, drain: 120},
		{name: "missing lifetime", operation: "grow-1", target: 3, drain: 120},
		{name: "zero target", operation: "grow-1", instance: "api-0", drain: 120},
		{name: "negative target", operation: "grow-1", instance: "api-0", target: -1, drain: 120},
		{name: "zero drain budget", operation: "grow-1", instance: "api-0", target: 3},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("a malformed target must be rejected without contacting any engine")
			var calls atomic.Int32
			control, err := NewClient("http://rank-0.test:8000", &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
				calls.Add(1)
				return nil, errors.New("must not be called")
			})})
			require.NoError(t, err)
			_, err = control.Scale(t.Context(), ScaleRequest{
				OperationID: test.operation, ExpectedInstanceID: test.instance,
				NewDataParallelSize: test.target, DrainTimeout: test.drain,
			})
			require.Error(t, err)
			assert.Zero(t, calls.Load())
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}
