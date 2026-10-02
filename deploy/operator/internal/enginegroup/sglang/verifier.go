/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package sglang

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginegroup"
)

// ServingVerifier sends one minimal completion through the real serving path.
type ServingVerifier struct {
	URL        *url.URL
	Model      string
	HTTPClient *http.Client
	Now        func() time.Time
}

// NewServingVerifier validates a serving endpoint and model.
func NewServingVerifier(rawURL, model string, httpClient *http.Client) (*ServingVerifier, error) {
	endpoint, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("parse SGLang verification URL: %w", err)
	}
	if (endpoint.Scheme != "http" && endpoint.Scheme != "https") || endpoint.Host == "" {
		return nil, fmt.Errorf("SGLang verification URL must be an absolute HTTP URL")
	}
	if model == "" {
		return nil, fmt.Errorf("SGLang verification model is required")
	}
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &ServingVerifier{URL: endpoint, Model: model, HTTPClient: httpClient, Now: time.Now}, nil
}

// Verify proves that one request completes after the exact topology commits.
func (v *ServingVerifier) Verify(
	ctx context.Context,
	_ enginegroup.GroupID,
	topology enginegroup.MembershipTopology,
) (enginegroup.VerificationResult, error) {
	requestBody := struct {
		Model     string `json:"model"`
		Prompt    string `json:"prompt"`
		MaxTokens int    `json:"max_tokens"`
	}{Model: v.Model, Prompt: "Reply with one token.", MaxTokens: 1}
	payload, err := json.Marshal(requestBody)
	if err != nil {
		return enginegroup.VerificationResult{}, fmt.Errorf("encode serving verification request: %w", err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, v.URL.String(), bytes.NewReader(payload))
	if err != nil {
		return enginegroup.VerificationResult{}, fmt.Errorf("build serving verification request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := v.HTTPClient.Do(request)
	if err != nil {
		return enginegroup.VerificationResult{}, fmt.Errorf("call SGLang serving path: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return enginegroup.VerificationResult{Failure: &enginegroup.Failure{
			Classification: enginegroup.FailureClassificationRetryable,
			Reason:         "ServingRequestFailed",
			Message:        fmt.Sprintf("serving path returned HTTP %d", response.StatusCode),
		}}, nil
	}
	var result struct {
		Choices []json.RawMessage `json:"choices"`
	}
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		return enginegroup.VerificationResult{}, fmt.Errorf("decode SGLang serving response: %w", err)
	}
	if len(result.Choices) == 0 {
		return enginegroup.VerificationResult{Failure: &enginegroup.Failure{
			Classification: enginegroup.FailureClassificationRetryable,
			Reason:         "NoServingProgress",
			Message:        "serving response contained no completion choices",
		}}, nil
	}
	return enginegroup.VerificationResult{Proof: &enginegroup.ServingProof{
		TopologyGeneration: topology.Generation,
		RuntimeDigest:      enginegroup.TopologyRuntimeDigest(topology),
		ObservedAt:         v.Now().UTC(),
	}}, nil
}
