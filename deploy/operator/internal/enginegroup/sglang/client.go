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
)

const (
	scalePath = "/engine/control/scale_elastic_ep"
	statePath = "/engine/control/is_scaling_elastic_ep"
)

// ScaleState is SGLang's current engine-authoritative Elastic EP state.
type ScaleState struct {
	Scaling         bool   `json:"is_scaling_elastic_ep"`
	EffectiveEPSize int32  `json:"effective_ep_size"`
	Phase           string `json:"scale_phase"`
	LastError       string `json:"last_error"`
	Status          string `json:"status"`
	Message         string `json:"message"`
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
	BaseURL    *url.URL
	HTTPClient *http.Client
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
	return &Client{BaseURL: parsed, HTTPClient: httpClient}, nil
}

// Observe returns SGLang's current Elastic EP state.
func (c *Client) Observe(ctx context.Context) (ScaleState, error) {
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
	return state, nil
}

// Scale requests one absolute EP size. Callers must observe the engine after
// every return, including transport errors, because acceptance can be ambiguous.
func (c *Client) Scale(ctx context.Context, target int32) (ScaleResponse, error) {
	if target <= 0 {
		return ScaleResponse{}, fmt.Errorf("SGLang EP target must be positive, got %d", target)
	}
	request := struct {
		NewEPSize int32 `json:"new_ep_size"`
	}{NewEPSize: target}
	var response ScaleResponse
	if err := c.doJSON(ctx, http.MethodPost, scalePath, request, &response); err != nil {
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
