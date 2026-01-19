/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package modelendpoint

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"sigs.k8s.io/controller-runtime/pkg/log"
)

// loadLoRA loads a LoRA model on a single endpoint
func (c *Client) loadLoRA(ctx context.Context, address, modelName, sourceURI string) error {
	logs := log.FromContext(ctx)

	// Build request body with source object
	loadReq := map[string]interface{}{
		"lora_name": modelName,
		"source": map[string]interface{}{
			"uri": sourceURI,
		},
	}

	loadBody, err := json.Marshal(loadReq)
	if err != nil {
		return fmt.Errorf("failed to marshal load LoRA request: %w", err)
	}

	// Build URL robustly using url.JoinPath to handle trailing slashes
	// Pass path segments without leading slash to preserve any existing path in address (e.g., /v1)
	apiURL, err := url.JoinPath(address, "v1", "loras")
	if err != nil {
		return fmt.Errorf("failed to construct load LoRA URL: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(loadBody))
	if err != nil {
		return fmt.Errorf("failed to create load LoRA request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call load LoRA endpoint: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			logs.V(1).Info("Failed to close response body", "error", closeErr)
		}
	}()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		logs.V(1).Info("Load LoRA failed", "address", address, "status", resp.StatusCode, "body", string(body))
		return fmt.Errorf("load LoRA failed with status %d: %s", resp.StatusCode, string(body))
	}

	logs.Info("Successfully loaded LoRA", "address", address, "modelName", modelName, "sourceURI", sourceURI)
	return nil
}

// unloadLoRA unloads a LoRA model from a single endpoint
func (c *Client) unloadLoRA(ctx context.Context, address, modelName string) error {
	logs := log.FromContext(ctx)

	// Build URL robustly using url.JoinPath to handle trailing slashes and encode modelName
	// Pass path segments without leading slash to preserve any existing path in address (e.g., /v1)
	apiURL, err := url.JoinPath(address, "v1", "loras", modelName)
	if err != nil {
		logs.V(1).Info("Failed to construct unload LoRA URL", "error", err)
		return fmt.Errorf("failed to construct unload LoRA URL: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "DELETE", apiURL, nil)
	if err != nil {
		logs.V(1).Info("Failed to create unload LoRA request", "error", err)
		return fmt.Errorf("failed to create unload LoRA request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		logs.V(1).Info("Failed to call unload LoRA endpoint", "address", address, "error", err)
		return fmt.Errorf("failed to call unload LoRA endpoint: %w", err)
	}
	defer func() {
		if closeErr := resp.Body.Close(); closeErr != nil {
			logs.V(1).Info("Failed to close response body", "error", closeErr)
		}
	}()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		logs.V(1).Info("Unload LoRA endpoint returned error status",
			"address", address,
			"status", resp.StatusCode,
			"body", string(body))
		return fmt.Errorf("unload LoRA failed with status %d: %s", resp.StatusCode, string(body))
	}

	logs.V(1).Info("Successfully unloaded LoRA", "address", address, "modelName", modelName)
	return nil
}
