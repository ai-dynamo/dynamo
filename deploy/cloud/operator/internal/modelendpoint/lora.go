/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

	"sigs.k8s.io/controller-runtime/pkg/log"
)

// LoadLoRA loads a LoRA model on the specified endpoint
func (p *Prober) LoadLoRA(ctx context.Context, address, modelName string) error {
	logs := log.FromContext(ctx)

	loadReq := map[string]interface{}{
		"lora_name": modelName,
		"lora_path": modelName, // May need to adjust based on actual API
	}

	loadBody, err := json.Marshal(loadReq)
	if err != nil {
		return fmt.Errorf("failed to marshal load LoRA request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", address+"/v1/load_lora", bytes.NewBuffer(loadBody))
	if err != nil {
		return fmt.Errorf("failed to create load LoRA request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call load LoRA endpoint: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		logs.V(1).Info("Load LoRA failed", "address", address, "status", resp.StatusCode, "body", string(body))
		return fmt.Errorf("load LoRA failed with status %d: %s", resp.StatusCode, string(body))
	}

	logs.Info("Successfully loaded LoRA", "address", address, "modelName", modelName)
	return nil
}

// VerifyLoRALoaded checks if a LoRA model is present in the GET /loras response
func (p *Prober) VerifyLoRALoaded(ctx context.Context, address, modelName string) bool {
	logs := log.FromContext(ctx)

	req, err := http.NewRequestWithContext(ctx, "GET", address+"/v1/loras", nil)
	if err != nil {
		logs.V(1).Info("Failed to create GET loras request", "error", err)
		return false
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		logs.V(1).Info("Failed to call GET loras endpoint", "address", address, "error", err)
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		logs.V(1).Info("GET loras failed", "address", address, "status", resp.StatusCode)
		return false
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		logs.V(1).Info("Failed to read loras response", "error", err)
		return false
	}

	// Parse the response - expecting a JSON array or object with lora names
	var lorasResp struct {
		Loras []string `json:"loras"`
	}

	if err := json.Unmarshal(body, &lorasResp); err != nil {
		// Try parsing as a simple array
		var lorasList []string
		if err := json.Unmarshal(body, &lorasList); err != nil {
			logs.V(1).Info("Failed to parse loras response", "error", err, "body", string(body))
			return false
		}
		lorasResp.Loras = lorasList
	}

	// Check if our model is in the list
	for _, lora := range lorasResp.Loras {
		if lora == modelName {
			logs.V(1).Info("LoRA model verified as loaded", "address", address, "modelName", modelName)
			return true
		}
	}

	logs.V(1).Info("LoRA model not found in list", "address", address, "modelName", modelName, "availableLoras", lorasResp.Loras)
	return false
}

// probeLoRAEndpoint checks if LoRA is already loaded, and loads it if not
func (p *Prober) probeLoRAEndpoint(ctx context.Context, address, modelName string) bool {
	logs := log.FromContext(ctx)

	// Step 1: Check if LoRA is already loaded
	if p.VerifyLoRALoaded(ctx, address, modelName) {
		logs.V(1).Info("LoRA already loaded", "address", address, "modelName", modelName)
		return true
	}

	// Step 2: Load the LoRA since it's not present
	if err := p.LoadLoRA(ctx, address, modelName); err != nil {
		logs.V(1).Info("Failed to load LoRA", "address", address, "error", err)
		return false
	}

	// Step 3: Verify the LoRA was loaded successfully
	if p.VerifyLoRALoaded(ctx, address, modelName) {
		return true
	}

	logs.V(1).Info("LoRA load appeared successful but verification failed", "address", address)
	return false
}

// unloadLoRA unloads a LoRA model from a single endpoint
func (p *Prober) unloadLoRA(ctx context.Context, address, modelName string) error {
	logs := log.FromContext(ctx)

	req, err := http.NewRequestWithContext(ctx, "DELETE", address+"/v1/loras/"+modelName, nil)
	if err != nil {
		logs.V(1).Info("Failed to create unload LoRA request", "error", err)
		return fmt.Errorf("failed to create unload LoRA request: %w", err)
	}

	resp, err := p.httpClient.Do(req)
	if err != nil {
		logs.V(1).Info("Failed to call unload LoRA endpoint", "address", address, "error", err)
		return fmt.Errorf("failed to call unload LoRA endpoint: %w", err)
	}
	defer resp.Body.Close()

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
