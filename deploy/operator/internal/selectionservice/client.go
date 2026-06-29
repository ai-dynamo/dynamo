/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package selectionservice

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

	"sigs.k8s.io/controller-runtime/pkg/log"
)

const workersPath = "/workers"

// defaultRequestTimeout is the fallback HTTP request timeout used when the caller
// does not supply a positive timeout or a custom HTTP client.
const defaultRequestTimeout = 5 * time.Second

// Client reconciles workers through the runtime-free selection service HTTP API.
type Client struct {
	baseURL    *url.URL
	httpClient *http.Client
}

func NewClient(baseURL string, timeout time.Duration) (*Client, error) {
	if timeout <= 0 {
		timeout = defaultRequestTimeout
	}
	return NewClientWithHTTPClient(baseURL, &http.Client{Timeout: timeout})
}

func NewClientWithHTTPClient(baseURL string, httpClient *http.Client) (*Client, error) {
	normalized, err := normalizeSelectionServiceURL(baseURL)
	if err != nil {
		return nil, err
	}
	if httpClient == nil {
		httpClient = &http.Client{Timeout: defaultRequestTimeout}
	}
	return &Client{baseURL: normalized, httpClient: httpClient}, nil
}

func (c *Client) UpsertWorker(ctx context.Context, worker WorkerRequest) (WorkerRecord, error) {
	worker.ModelName = NormalizeModelName(worker.ModelName)
	worker.TenantID = NormalizeTenantID(worker.TenantID)
	payload, err := json.Marshal(worker)
	if err != nil {
		return WorkerRecord{}, fmt.Errorf("marshal selection worker request: %w", err)
	}
	respBody, err := c.doRequest(ctx, http.MethodPost, workersPath, nil, payload, false)
	if err != nil {
		return WorkerRecord{}, err
	}
	return decodeWorkerRecord(respBody, "POST /workers")
}

func (c *Client) ListWorkers(ctx context.Context, modelName string, tenantID string) ([]WorkerRecord, error) {
	query := url.Values{}
	if modelName != "" {
		query.Set("model_name", modelName)
	}
	if tenantID != "" {
		query.Set("tenant_id", tenantID)
	}

	respBody, err := c.doRequest(ctx, http.MethodGet, workersPath, query, nil, false)
	if err != nil {
		return nil, err
	}
	var workers []WorkerRecord
	if err := json.Unmarshal(respBody, &workers); err != nil {
		return nil, fmt.Errorf("decode selection service GET /workers response JSON: %w", err)
	}
	return workers, nil
}

// DeleteWorker calls DELETE /workers/{worker_id}. The selection service
// deactivates the worker and returns the record; it does not purge catalog state.
func (c *Client) DeleteWorker(ctx context.Context, workerID uint64, missingOK bool) (WorkerRecord, error) {
	path := fmt.Sprintf("/workers/%d", workerID)
	respBody, err := c.doRequest(ctx, http.MethodDelete, path, nil, nil, missingOK)
	if err != nil || respBody == nil {
		return WorkerRecord{}, err
	}
	return decodeWorkerRecord(respBody, http.MethodDelete+" "+path)
}

func decodeWorkerRecord(body []byte, operation string) (WorkerRecord, error) {
	var record WorkerRecord
	if err := json.Unmarshal(body, &record); err != nil {
		return WorkerRecord{}, fmt.Errorf("decode selection service %s response JSON: %w", operation, err)
	}
	return record, nil
}

func (c *Client) doRequest(ctx context.Context, method string, path string, query url.Values, payload []byte, missingOK bool) ([]byte, error) {
	operation := method + " " + path
	var body io.Reader
	if payload != nil {
		body = bytes.NewReader(payload)
	}
	requestURL, err := c.requestURL(path, query)
	if err != nil {
		return nil, fmt.Errorf("build selection service %s URL: %w", operation, err)
	}
	req, err := http.NewRequestWithContext(ctx, method, requestURL, body)
	if err != nil {
		return nil, fmt.Errorf("build selection service %s request: %w", operation, err)
	}
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("selection service %s failed: %w", operation, err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			log.FromContext(ctx).V(1).Info("Failed to close selection service response body", "operation", operation, "error", closeErr)
		}
	}()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read selection service %s response: %w", operation, err)
	}
	if missingOK && resp.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("selection service %s failed with HTTP %d: %s", operation, resp.StatusCode, string(respBody))
	}
	return respBody, nil
}

func normalizeSelectionServiceURL(raw string) (*url.URL, error) {
	normalized := strings.TrimRight(strings.TrimSpace(raw), "/")
	parsed, err := url.Parse(normalized)
	if err != nil {
		return nil, fmt.Errorf("parse selection service URL %q: %w", raw, err)
	}
	if parsed.Scheme != "http" && parsed.Scheme != "https" {
		return nil, fmt.Errorf("selection service URL %q must use http or https", raw)
	}
	if parsed.Host == "" {
		return nil, fmt.Errorf("selection service URL %q must include a host", raw)
	}
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed, nil
}

func (c *Client) requestURL(path string, query url.Values) (string, error) {
	joined, err := url.JoinPath(c.baseURL.String(), strings.TrimLeft(path, "/"))
	if err != nil {
		return "", err
	}
	if len(query) == 0 {
		return joined, nil
	}
	parsed, err := url.Parse(joined)
	if err != nil {
		return "", err
	}
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}
