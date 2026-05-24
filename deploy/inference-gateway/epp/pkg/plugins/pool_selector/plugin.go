/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pool_selector

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"
	plugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	// PoolSelectorType is the plugin type used in the EndpointPickerConfig.
	PoolSelectorType = "pool-selector"

	// defaultCharsPerToken approximates ISL from prompt length when exact
	// tokenization is unavailable at filter time. See the package doc / the
	// "ISL estimate" decision flagged for review: the FFI router tokenizes
	// inside the route call (in the scorer, which runs after filters), and we
	// avoid a second tokenization pass here. Coarse ISL bands tolerate the
	// approximation; swap for exact tokenization if a standalone FFI is added.
	defaultCharsPerToken = 4
)

// compile-time assertion: PoolSelector is a scheduling Filter.
var _ schedtypes.Filter = &PoolSelector{}

// PoolSelectorConfig is the plugin's eppConfig `parameters`. It embeds the grid
// (flat sizeMin/.../mapping fields) plus the label wiring that maps a grid pool
// index to the endpoint label value identifying that pool.
type PoolSelectorConfig struct {
	GridSelectionStrategy
	// PoolLabel is the endpoint label key whose value names an endpoint's pool
	// (e.g. "nvidia.com/dynamo-pool").
	PoolLabel string `json:"poolLabel"`
	// PoolLabels maps grid pool index -> PoolLabel value (index N = PoolLabels[N]).
	PoolLabels []string `json:"poolLabels"`
}

// PoolSelectorFactory builds a PoolSelector from eppConfig parameters.
func PoolSelectorFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	cfg := PoolSelectorConfig{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin parameters: %w", PoolSelectorType, err)
		}
	}
	if cfg.PoolLabel == "" {
		return nil, fmt.Errorf("%s plugin parameter 'poolLabel' must not be empty", PoolSelectorType)
	}
	if len(cfg.PoolLabels) == 0 {
		return nil, fmt.Errorf("%s plugin parameter 'poolLabels' must contain at least one pool", PoolSelectorType)
	}
	if err := cfg.GridSelectionStrategy.Validate(len(cfg.PoolLabels)); err != nil {
		return nil, fmt.Errorf("%s plugin grid invalid: %w", PoolSelectorType, err)
	}
	return NewPoolSelector(cfg).WithName(name), nil
}

// NewPoolSelector constructs a PoolSelector from validated config.
func NewPoolSelector(cfg PoolSelectorConfig) *PoolSelector {
	return &PoolSelector{
		typedName:  plugins.TypedName{Type: PoolSelectorType, Name: PoolSelectorType},
		label:      cfg.PoolLabel,
		poolLabels: cfg.PoolLabels,
		grid:       cfg.GridSelectionStrategy,
	}
}

// PoolSelector is a Filter that picks a target pool from (ISL, TTFT-target) via
// the grid, then keeps only endpoints in that pool — so the downstream
// dyn-decode-scorer (FFI KV/load router) selects an endpoint *within* the chosen
// pool. This folds the standalone Global Router's cross-pool selection into the
// gateway path; both consume the same profiler-derived grid.
type PoolSelector struct {
	typedName  plugins.TypedName
	label      string
	poolLabels []string
	grid       GridSelectionStrategy
}

func (p *PoolSelector) TypedName() plugins.TypedName { return p.typedName }

func (p *PoolSelector) WithName(name string) *PoolSelector {
	p.typedName.Name = name
	return p
}

// Filter restricts endpoints to the pool selected for this request's ISL/SLA.
// If the selected pool currently has no endpoints, it passes all endpoints
// through rather than black-holing the request (logged).
func (p *PoolSelector) Filter(ctx context.Context, _ *schedtypes.CycleState, req *schedtypes.InferenceRequest, endpoints []schedtypes.Endpoint) []schedtypes.Endpoint {
	logger := log.FromContext(ctx)

	isl := estimateISL(req)
	ttftMs := readNvextFloat(req, "router", "ttft_target") // <=0 → grid uses mid-range
	priority := readNvextPriority(req)                     // <0 → no override

	poolIdx := p.grid.SelectPool(float64(isl), ttftMs, priority)
	if poolIdx < 0 || poolIdx >= len(p.poolLabels) {
		logger.Info("pool-selector: pool index out of range, passing through", "poolIdx", poolIdx)
		return endpoints
	}
	target := p.poolLabels[poolIdx]

	filtered := make([]schedtypes.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		if ep == nil || ep.GetMetadata() == nil {
			continue
		}
		if ep.GetMetadata().Labels[p.label] == target {
			filtered = append(filtered, ep)
		}
	}

	if len(filtered) == 0 {
		logger.Info("pool-selector: selected pool has no ready endpoints; passing through all",
			"isl", isl, "ttftMs", ttftMs, "pool", target)
		return endpoints
	}
	logger.Info("pool-selector: routed",
		"isl", isl, "ttftMs", ttftMs, "priority", priority, "pool", target, "candidates", len(filtered))
	return filtered
}

// estimateISL approximates the input sequence length (tokens) from the prompt's
// character count. See defaultCharsPerToken for the rationale + review flag.
func estimateISL(req *schedtypes.InferenceRequest) int {
	if req == nil || req.Body == nil {
		return 0
	}
	chars := 0
	switch {
	case req.Body.ChatCompletions != nil:
		for _, m := range req.Body.ChatCompletions.Messages {
			chars += len(m.Content.PlainText())
		}
	case req.Body.Completions != nil:
		chars += len(req.Body.Completions.Prompt.PlainText())
	}
	return chars / defaultCharsPerToken
}

// nvextOf returns the caller-supplied nvext object, mirroring the dynamo_kv_scorer
// extractNvext helper (the same hint channel the FFI router reads).
func nvextOf(req *schedtypes.InferenceRequest) map[string]any {
	if req == nil || req.Body == nil {
		return nil
	}
	pm, ok := req.Body.Payload.(fwkrh.PayloadMap)
	if !ok {
		return nil
	}
	nv, _ := pm["nvext"].(map[string]any)
	return nv
}

// readNvextFloat reads nvext[group][field] as a float (e.g. nvext.router.ttft_target).
// Returns -1 when absent, matching SelectPool's "unspecified" sentinel.
func readNvextFloat(req *schedtypes.InferenceRequest, group, field string) float64 {
	nv := nvextOf(req)
	if nv == nil {
		return -1
	}
	g, _ := nv[group].(map[string]any)
	if g == nil {
		return -1
	}
	if f, ok := toFloat(g[field]); ok {
		return f
	}
	return -1
}

// readNvextPriority reads nvext.agent_hints.priority. Returns -1 when absent.
func readNvextPriority(req *schedtypes.InferenceRequest) int {
	nv := nvextOf(req)
	if nv == nil {
		return -1
	}
	ah, _ := nv["agent_hints"].(map[string]any)
	if ah == nil {
		return -1
	}
	if f, ok := toFloat(ah["priority"]); ok {
		return int(f)
	}
	return -1
}

// toFloat coerces JSON-decoded numerics (float64 / json.Number / int) to float64.
func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int:
		return float64(x), true
	case int64:
		return float64(x), true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	}
	return 0, false
}
