/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pool_selector

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const poolLabelKey = "nvidia.com/dynamo-pool"

// ep builds an endpoint carrying just the pool label the filter keys off.
func ep(t *testing.T, pool string) schedtypes.Endpoint {
	t.Helper()
	labels := map[string]string{}
	if pool != "" {
		labels[poolLabelKey] = pool
	}
	return schedtypes.NewEndpoint(
		&fwkdl.EndpointMetadata{Labels: labels},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
}

// poolOf returns the pool label of an endpoint (for asserting which survived).
func poolOf(e schedtypes.Endpoint) string { return e.GetMetadata().Labels[poolLabelKey] }

// completionReq builds a /v1/completions request whose prompt is nChars long,
// optionally carrying nvext.router.ttft_target.
func completionReq(nChars int, ttft *float64) *schedtypes.InferenceRequest {
	body := &fwkrh.InferenceRequestBody{
		Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: strings.Repeat("a", nChars)}},
	}
	if ttft != nil {
		body.Payload = fwkrh.PayloadMap{
			"nvext": map[string]any{"router": map[string]any{"ttft_target": *ttft}},
		}
	}
	return &schedtypes.InferenceRequest{Body: body}
}

// islGrid: 2 ISL bands over [0,4000), 1 SLA band, mapping short→tp1, long→tp2.
func islGrid(t *testing.T) *PoolSelector {
	t.Helper()
	p, err := PoolSelectorFactory("ps", json.RawMessage(goodParams), nil)
	if err != nil {
		t.Fatalf("factory: %v", err)
	}
	return p.(*PoolSelector)
}

func TestFilter_RoutesByISL(t *testing.T) {
	p := islGrid(t)
	eps := []schedtypes.Endpoint{ep(t, "tp1"), ep(t, "tp2")}
	ctx := context.Background()

	// 400 chars → ISL 100 → band 0 → tp1.
	short := p.Filter(ctx, nil, completionReq(400, nil), eps)
	if len(short) != 1 || poolOf(short[0]) != "tp1" {
		t.Fatalf("short prompt: got %d endpoints (%v), want 1 in tp1", len(short), poolsOf(short))
	}
	// 12000 chars → ISL 3000 → band 1 → tp2.
	long := p.Filter(ctx, nil, completionReq(12000, nil), eps)
	if len(long) != 1 || poolOf(long[0]) != "tp2" {
		t.Fatalf("long prompt: got %d endpoints (%v), want 1 in tp2", len(long), poolsOf(long))
	}
}

func TestFilter_SelectedPoolEmpty_PassesThrough(t *testing.T) {
	p := islGrid(t)
	// Only tp2 endpoints present, but a short prompt selects tp1 (no members).
	eps := []schedtypes.Endpoint{ep(t, "tp2"), ep(t, "tp2")}
	got := p.Filter(context.Background(), nil, completionReq(400, nil), eps)
	if len(got) != len(eps) {
		t.Fatalf("empty selected pool: got %d endpoints, want pass-through of all %d", len(got), len(eps))
	}
}

func TestFilter_UnlabeledEndpointExcluded(t *testing.T) {
	p := islGrid(t)
	// Short prompt → tp1. Unlabeled endpoint must not match tp1.
	eps := []schedtypes.Endpoint{ep(t, "tp1"), ep(t, "")}
	got := p.Filter(context.Background(), nil, completionReq(400, nil), eps)
	if len(got) != 1 || poolOf(got[0]) != "tp1" {
		t.Fatalf("unlabeled endpoint: got %v, want only tp1", poolsOf(got))
	}
}

// ttftGrid: 1 ISL band, 2 SLA bands over [0,200) with lower edges 0 and 100.
// tight target (<100ms) → tp1, looser → tp2.
func ttftGrid(t *testing.T) *PoolSelector {
	t.Helper()
	params := `{
	  "poolLabel": "nvidia.com/dynamo-pool",
	  "poolLabels": ["tp1", "tp2"],
	  "sizeMin": 0, "sizeMax": 4000, "sizeResolution": 1,
	  "latencyMinMs": 0, "latencyMaxMs": 200, "latencyResolution": 2,
	  "mapping": [[0, 1]]
	}`
	p, err := PoolSelectorFactory("ps", json.RawMessage(params), nil)
	if err != nil {
		t.Fatalf("factory: %v", err)
	}
	return p.(*PoolSelector)
}

func TestFilter_NvextTtftSelectsPool(t *testing.T) {
	p := ttftGrid(t)
	eps := []schedtypes.Endpoint{ep(t, "tp1"), ep(t, "tp2")}
	f := func(v float64) float64 { return v }

	ttft50 := f(50) // band 0 → tp1
	if got := p.Filter(context.Background(), nil, completionReq(40, &ttft50), eps); len(got) != 1 || poolOf(got[0]) != "tp1" {
		t.Fatalf("ttft=50: got %v, want tp1", poolsOf(got))
	}
	ttft150 := f(150) // band 1 → tp2
	if got := p.Filter(context.Background(), nil, completionReq(40, &ttft150), eps); len(got) != 1 || poolOf(got[0]) != "tp2" {
		t.Fatalf("ttft=150: got %v, want tp2", poolsOf(got))
	}
}

func TestFilter_ExplicitZeroTtftIsStrict(t *testing.T) {
	p := ttftGrid(t)
	eps := []schedtypes.Endpoint{ep(t, "tp1"), ep(t, "tp2")}
	zero := 0.0 // explicit 0 = strictest target → band 0 → tp1 (not mid-range)
	if got := p.Filter(context.Background(), nil, completionReq(40, &zero), eps); len(got) != 1 || poolOf(got[0]) != "tp1" {
		t.Fatalf("ttft=0: got %v, want tp1 (strict)", poolsOf(got))
	}
}

func poolsOf(eps []schedtypes.Endpoint) []string {
	out := make([]string, len(eps))
	for i, e := range eps {
		out[i] = poolOf(e)
	}
	return out
}
