/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
)

// The engine control plane serves its routes over HTTP on the system port under /engine/.
// The reconciler reaches them over the network at the Phase-3 headless leader Service -- NOT
// by kubectl exec like the manual scale script, which only did so because port-forward proved
// unreliable through the API proxy.
const (
	epCapacityRoute = "/engine/control/ep_capacity"
	epScaleRoute    = "/engine/control/scale_elastic_ep"
)

// elasticEPLeaderBaseURL is the http://host:port the leader's engine control plane answers on.
// host is the Phase-3 headless Service <component>-ray, addressed cluster-internally.
func elasticEPLeaderBaseURL(componentName, namespace string) string {
	svc := dynamo.ElasticEPLeaderServiceName(componentName)
	return fmt.Sprintf("http://%s.%s.svc:%d", svc, namespace, consts.DynamoSystemPort)
}

// readEPCapacity GETs (POSTs, per the route contract) the current elastic-EP capacity.
func readEPCapacity(ctx context.Context, hc *http.Client, baseURL string) (dynamo.EPCapacity, error) {
	var cap dynamo.EPCapacity
	if err := postEngineRoute(ctx, hc, baseURL+epCapacityRoute, map[string]any{}, &cap); err != nil {
		return dynamo.EPCapacity{}, err
	}
	if cap.Status != "ok" {
		return dynamo.EPCapacity{}, fmt.Errorf("ep_capacity returned status %q", cap.Status)
	}
	return cap, nil
}

// scaleElasticEPResult is the scale_elastic_ep reply; both fields are checked because the
// endpoint takes a target, not a delta -- a reply that succeeded at the wrong size is a failure.
type scaleElasticEPResult struct {
	Status              string `json:"status"`
	NewDataParallelSize int    `json:"new_data_parallel_size"`
}

// scaleElasticEP asks the engine to converge on target ranks. It accepts the result only when
// status is "ok" AND the echoed new_data_parallel_size matches target. It returns errEngineBusy
// when the engine is mid-scale so the caller can requeue rather than treat it as a failure.
func scaleElasticEP(ctx context.Context, hc *http.Client, baseURL string, target int) error {
	var res scaleElasticEPResult
	err := postEngineRoute(ctx, hc, baseURL+epScaleRoute,
		map[string]any{"new_data_parallel_size": target}, &res)
	if err != nil {
		return err
	}
	if res.Status == "busy" {
		return errEngineBusy
	}
	if res.Status != "ok" {
		return fmt.Errorf("scale_elastic_ep returned status %q", res.Status)
	}
	if res.NewDataParallelSize != target {
		return fmt.Errorf("scale_elastic_ep acknowledged size %d, wanted %d", res.NewDataParallelSize, target)
	}
	return nil
}

// errEngineBusy signals a scale is already in progress; treat as requeue, not failure, or a
// slow scale reads as a broken deployment.
var errEngineBusy = fmt.Errorf("engine busy: scale already in progress")

func postEngineRoute(ctx context.Context, hc *http.Client, url string, body any, out any) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := hc.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: HTTP %d: %s", url, resp.StatusCode, string(raw))
	}
	if out != nil {
		if err := json.Unmarshal(raw, out); err != nil {
			return fmt.Errorf("%s: decode response: %w", url, err)
		}
	}
	return nil
}
