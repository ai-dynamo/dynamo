/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
)

// defaultRequestTimeout is the fallback HTTP request timeout used when the caller
// does not supply a positive timeout or a custom HTTP client.
const defaultRequestTimeout = 5 * time.Second

type MetadataClient struct {
	baseURL    *url.URL
	apiKey     string
	httpClient *http.Client
}

func NewMetadataClient(baseURL string, timeout time.Duration) (*MetadataClient, error) {
	if timeout <= 0 {
		timeout = defaultRequestTimeout
	}
	return NewMetadataClientWithHTTPClient(baseURL, "", &http.Client{Timeout: timeout})
}

func NewMetadataClientWithHTTPClient(baseURL string, apiKey string, httpClient *http.Client) (*MetadataClient, error) {
	normalized, err := normalizeSGLangEndpointURL(baseURL)
	if err != nil {
		return nil, err
	}
	if httpClient == nil {
		httpClient = &http.Client{Timeout: defaultRequestTimeout}
	}
	return &MetadataClient{
		baseURL:    normalized,
		apiKey:     apiKey,
		httpClient: httpClient,
	}, nil
}

func (c *MetadataClient) Fetch(ctx context.Context) (MetadataSnapshot, error) {
	var models ModelsResponse
	if err := c.getJSON(ctx, "/v1/models", &models); err != nil {
		return MetadataSnapshot{}, err
	}
	var modelInfo ModelInfo
	if err := c.getJSON(ctx, "/model_info", &modelInfo); err != nil {
		return MetadataSnapshot{}, err
	}
	var serverInfo ServerInfo
	if err := c.getJSON(ctx, "/server_info", &serverInfo); err != nil {
		return MetadataSnapshot{}, err
	}
	return MetadataSnapshot{
		Models:     models,
		ModelInfo:  modelInfo,
		ServerInfo: serverInfo,
	}, nil
}

func (c *MetadataClient) getJSON(ctx context.Context, path string, target any) error {
	operation := http.MethodGet + " " + path
	requestURL, err := c.requestURL(path)
	if err != nil {
		return fmt.Errorf("build SGLang metadata %s URL: %w", operation, err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, requestURL, nil)
	if err != nil {
		return fmt.Errorf("build SGLang metadata %s request: %w", operation, err)
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("SGLang metadata %s failed: %w", operation, err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			log.FromContext(ctx).V(1).Info("Failed to close SGLang metadata response body", "operation", operation, "error", closeErr)
		}
	}()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read SGLang metadata %s response: %w", operation, err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("SGLang metadata %s failed with HTTP %d: %s", operation, resp.StatusCode, string(body))
	}

	if err := json.Unmarshal(body, target); err != nil {
		return fmt.Errorf("decode SGLang metadata %s response JSON: %w", operation, err)
	}
	return nil
}

func normalizeSGLangEndpointURL(raw string) (*url.URL, error) {
	normalized := strings.TrimRight(strings.TrimSpace(raw), "/")
	parsed, err := url.Parse(normalized)
	if err != nil {
		return nil, fmt.Errorf("parse SGLang endpoint %q: %w", raw, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("SGLang endpoint %q must use http or https", raw)
	}
	if parsed.Host == "" {
		return nil, fmt.Errorf("SGLang endpoint %q must include a host", raw)
	}
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed, nil
}

func (c *MetadataClient) requestURL(path string) (string, error) {
	return url.JoinPath(c.baseURL.String(), strings.TrimLeft(path, "/"))
}
