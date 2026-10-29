/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package sglang implements the Engine Group adapter boundary for SGLang Elastic EP.
package sglang

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup/controlhttp"
)

const (
	scalePath                 = "/engine/control/scale_elastic_ep"
	statePath                 = "/engine/control/is_scaling_elastic_ep"
	defaultObservationTimeout = 5 * time.Second
	// Native submission waits at most 30s for scheduler acknowledgment, not collective completion.
	defaultScaleTimeout = 35 * time.Second
)

// ScaleState is SGLang's current engine-authoritative Elastic EP state.
type ScaleState struct {
	InstanceID           string   `json:"instance_id"`
	TopologyToken        string   `json:"topology_token"`
	OperationID          string   `json:"operation_id"`
	TargetEPSize         int32    `json:"target_ep_size"`
	OperationSucceeded   *bool    `json:"operation_succeeded"`
	PendingEPSize        *int32   `json:"pending_ep_size"`
	InitialEPSize        int32    `json:"initial_ep_size"`
	AllocationWidth      int32    `json:"allocation_width"`
	MaxCommittedEPSize   int32    `json:"max_committed_ep_size"`
	JoiningAllocationIDs []string `json:"joining_allocation_ids"`
	RetireeLifecycle     string   `json:"retiree_lifecycle"`
	SafeToTerminateRanks []int32  `json:"safe_to_terminate_ranks"`
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
	ExpectedTopologyToken        string   `json:"expected_topology_token"`
	ExpectedJoiningAllocationIDs []string `json:"expected_joining_allocation_ids,omitempty"`
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
// Its transport identifies one primary worker and must not contain a route path.
type Client struct {
	transport          *controlhttp.Client
	ObservationTimeout time.Duration
	ScaleTimeout       time.Duration
}

// NewClient validates and constructs a control client.
func NewClient(rawBaseURL string, httpClient *http.Client) (*Client, error) {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	transport, err := controlhttp.New(rawBaseURL, httpClient)
	if err != nil {
		return nil, fmt.Errorf("SGLang control URL: %w", err)
	}
	return &Client{
		transport:          transport,
		ObservationTimeout: defaultObservationTimeout, ScaleTimeout: defaultScaleTimeout,
	}, nil
}

// Observe returns SGLang's current Elastic EP state.
func (c *Client) Observe(ctx context.Context) (ScaleState, error) {
	// Observation does not wait for a collective to finish.
	ctx, cancel := context.WithTimeout(ctx, c.ObservationTimeout)
	defer cancel()

	data, err := c.transport.Request(ctx, http.MethodGet, statePath, nil)
	if err != nil {
		return ScaleState{}, fmt.Errorf("SGLang %w", err)
	}
	var state ScaleState
	if err := json.Unmarshal(data, &state); err != nil {
		return ScaleState{}, fmt.Errorf("decode SGLang %s response: %w", statePath, err)
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
	if target.NewEPSize <= 0 || target.OperationID == "" || target.ExpectedInstanceID == "" || target.ExpectedTopologyToken == "" {
		return ScaleResponse{}, fmt.Errorf("SGLang scale requires a positive target, operation ID, runtime ID and base topology token")
	}
	if len(target.ExpectedJoiningAllocationIDs) > 1 || (len(target.ExpectedJoiningAllocationIDs) == 1 && target.ExpectedJoiningAllocationIDs[0] == "") {
		return ScaleResponse{}, fmt.Errorf("SGLang growth requires one non-empty joining allocation ID; retirement has none")
	}
	// A timeout leaves the exact request ambiguous and replayable; it does not cancel scheduler-owned work.
	ctx, cancel := context.WithTimeout(ctx, c.ScaleTimeout)
	defer cancel()

	data, err := c.transport.Request(ctx, http.MethodPost, scalePath, target)
	if err != nil {
		return ScaleResponse{}, fmt.Errorf("SGLang %w", err)
	}
	var response ScaleResponse
	if err := json.Unmarshal(data, &response); err != nil {
		return ScaleResponse{}, fmt.Errorf("decode SGLang %s response: %w", scalePath, err)
	}
	return response, nil
}
