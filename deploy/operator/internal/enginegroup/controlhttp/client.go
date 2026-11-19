/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package controlhttp supplies transport safeguards for identity-bound engine control.
// Backend clients retain ownership of routes, schemas, deadlines and operation semantics.
package controlhttp

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

// MaximumResponseBytes bounds the evidence returned by one engine control request.
const MaximumResponseBytes = 1 << 20

// HTTPError records an HTTP refusal, never a definitive membership rejection.
type HTTPError struct {
	Path       string
	StatusCode int
	Body       string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("%s returned HTTP %d: %s", e.Path, e.StatusCode, e.Body)
}

// Client freezes one process endpoint and never follows redirects to another process.
type Client struct {
	baseURL    url.URL
	httpClient *http.Client
}

// New validates an unambiguous endpoint and copies httpClient without changing it.
// httpClient must not be nil. The endpoint must not include a route prefix.
func New(rawBaseURL string, httpClient *http.Client) (*Client, error) {
	parsed, err := url.Parse(rawBaseURL)
	if err != nil {
		return nil, fmt.Errorf("parse control URL: %w", err)
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" ||
		parsed.User != nil || parsed.Opaque != "" || parsed.RawQuery != "" || parsed.ForceQuery ||
		parsed.Fragment != "" || parsed.RawPath != "" || (parsed.Path != "" && parsed.Path != "/") {
		return nil, fmt.Errorf("control URL must identify an http(s) process endpoint without credentials, path, query, or fragment")
	}

	// Preserve caller transport settings while fencing redirects to different runtime identities.
	transport := *httpClient
	transport.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return &Client{baseURL: *parsed, httpClient: &transport}, nil
}

// Request sends one bounded JSON exchange using the caller's context and method.
// A nil payload means no request body. Callers must fully decode the returned JSON.
func (c *Client) Request(ctx context.Context, method, path string, payload any) ([]byte, error) {
	var body io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("encode %s request: %w", path, err)
		}
		body = bytes.NewReader(encoded)
	}
	endpoint := c.baseURL
	endpoint.Path = path
	request, err := http.NewRequestWithContext(ctx, method, endpoint.String(), body)
	if err != nil {
		return nil, fmt.Errorf("build %s request: %w", path, err)
	}
	if payload != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := c.httpClient.Do(request)
	if err != nil {
		return nil, fmt.Errorf("call %s: %w", path, err)
	}
	defer func() { _ = response.Body.Close() }()

	// Limit successful and error bodies alike; HTTP refusal is transport evidence only.
	data, err := io.ReadAll(io.LimitReader(response.Body, MaximumResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read %s response: %w", path, err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, &HTTPError{Path: path, StatusCode: response.StatusCode,
			Body: strings.TrimSpace(string(data[:min(len(data), 4096)]))}
	}
	if len(data) > MaximumResponseBytes {
		return nil, fmt.Errorf("%s response exceeds %d bytes", path, MaximumResponseBytes)
	}
	return data, nil
}
