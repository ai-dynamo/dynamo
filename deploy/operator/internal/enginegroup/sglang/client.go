/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package sglang implements the Engine Group adapter boundary for SGLang Elastic EP.
package sglang

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

const (
	scalePath                 = "/engine/control/scale_elastic_ep"
	statePath                 = "/engine/control/is_scaling_elastic_ep"
	defaultObservationTimeout = 5 * time.Second
	defaultScaleTimeout       = 12 * time.Minute
)

// ScaleState is SGLang's current engine-authoritative Elastic EP state.
type ScaleState struct {
	InstanceID           string   `json:"instance_id"`
	OperationID          string   `json:"operation_id"`
	TargetEPSize         int32    `json:"target_ep_size"`
	OperationSucceeded   *bool    `json:"operation_succeeded"`
	PendingEPSize        *int32   `json:"pending_ep_size"`
	InitialEPSize        int32    `json:"initial_ep_size"`
	AllocationWidth      int32    `json:"allocation_width"`
	JoiningAllocationIDs []string `json:"joining_allocation_ids"`
	Scaling              bool     `json:"is_scaling_elastic_ep"`
	EffectiveEPSize      int32    `json:"effective_ep_size"`
	Phase                string   `json:"scale_phase"`
	LastError            string   `json:"last_error"`
	RuntimeHealth        string   `json:"runtime_health"`
	RuntimeError         string   `json:"runtime_error"`
	Status               string   `json:"status"`
	Message              string   `json:"message"`
}

// ScaleRequest binds one absolute resize to the native runtime and joining Pod identities.
type ScaleRequest struct {
	NewEPSize                    int32    `json:"new_ep_size"`
	OperationID                  string   `json:"operation_id"`
	ExpectedInstanceID           string   `json:"expected_instance_id"`
	ExpectedJoiningAllocationIDs []string `json:"expected_joining_allocation_ids"`
}

// ScaleResponse is the synchronous response to one SGLang scale request. A
// successful response does not replace observing ScaleState to convergence.
type ScaleResponse struct {
	Status        string `json:"status"`
	Message       string `json:"message"`
	OldEPSize     int32  `json:"old_ep_size"`
	NewEPSize     int32  `json:"new_ep_size"`
	PendingEPSize int32  `json:"pending_ep_size"`
}

// Client calls the Dynamo SGLang worker's backend-neutral system routes.
// BaseURL identifies one primary worker and must not contain a route path.
type Client struct {
	BaseURL            *url.URL
	HTTPClient         *http.Client
	ObservationTimeout time.Duration
	ScaleTimeout       time.Duration
}

// NewClient validates and constructs a control client.
func NewClient(rawBaseURL string, httpClient *http.Client) (*Client, error) {
	parsed, err := url.Parse(rawBaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse SGLang control URL: %w", err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("SGLang control URL must use http or https")
	}
	if parsed.Host == "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, fmt.Errorf("SGLang control URL must identify a host without query or fragment")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/")
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &Client{
		BaseURL: parsed, HTTPClient: httpClient,
		ObservationTimeout: defaultObservationTimeout, ScaleTimeout: defaultScaleTimeout,
	}, nil
}

// Observe returns SGLang's current Elastic EP state.
func (c *Client) Observe(ctx context.Context) (ScaleState, error) {
	// Native status reads have a shorter budget than synchronous resize submission.
	ctx, cancel := context.WithTimeout(ctx, c.ObservationTimeout)
	defer cancel()

	var state ScaleState
	if err := c.doJSON(ctx, http.MethodGet, statePath, nil, &state); err != nil {
		return ScaleState{}, err
	}
	if state.Status == "error" {
		return ScaleState{}, fmt.Errorf("SGLang rejected state observation: %s", state.Message)
	}
	if state.EffectiveEPSize <= 0 {
		return ScaleState{}, fmt.Errorf("SGLang returned invalid effective EP size %d", state.EffectiveEPSize)
	}

	// A retained successful resize does not establish the runtime's current health.
	switch state.RuntimeHealth {
	case "healthy":
	case "recovery_unsupported":
		return ScaleState{}, fmt.Errorf("%w: SGLang runtime recovery is unsupported: %s", enginegroup.ErrRecoveryRequired, state.RuntimeError)
	default:
		return ScaleState{}, fmt.Errorf("SGLang returned unknown runtime health %q: %s", state.RuntimeHealth, state.RuntimeError)
	}
	return state, nil
}

// Scale requests one absolute EP size. Callers must observe the engine after
// every return, including transport errors, because acceptance can be ambiguous.
func (c *Client) Scale(ctx context.Context, target ScaleRequest) (ScaleResponse, error) {
	if target.NewEPSize <= 0 || target.OperationID == "" || target.ExpectedInstanceID == "" || len(target.ExpectedJoiningAllocationIDs) != 1 || target.ExpectedJoiningAllocationIDs[0] == "" {
		return ScaleResponse{}, fmt.Errorf("SGLang scale requires a positive target, operation ID, runtime ID, and one joining allocation ID")
	}
	// Bound synchronous native dispatch without weakening the caller's deadline.
	ctx, cancel := context.WithTimeout(ctx, c.ScaleTimeout)
	defer cancel()

	var response ScaleResponse
	if err := c.doJSON(ctx, http.MethodPost, scalePath, target, &response); err != nil {
		return ScaleResponse{}, err
	}
	return response, nil
}

func (c *Client) doJSON(ctx context.Context, method, path string, body any, out any) error {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("encode SGLang request: %w", err)
		}
		reader = bytes.NewReader(encoded)
	}
	endpoint := *c.BaseURL
	endpoint.Path = strings.TrimRight(endpoint.Path, "/") + path
	request, err := http.NewRequestWithContext(ctx, method, endpoint.String(), reader)
	if err != nil {
		return fmt.Errorf("build SGLang request: %w", err)
	}
	if body != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := c.HTTPClient.Do(request)
	if err != nil {
		return fmt.Errorf("call SGLang %s: %w", path, err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		payload, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return fmt.Errorf("SGLang %s returned HTTP %d: %s", path, response.StatusCode, strings.TrimSpace(string(payload)))
	}
	if err := json.NewDecoder(response.Body).Decode(out); err != nil {
		return fmt.Errorf("decode SGLang %s response: %w", path, err)
	}
	return nil
}
