/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package pool_selector

import (
	"encoding/json"
	"testing"
)

// A well-formed eppConfig `parameters` block: embedded grid (flat fields) +
// label wiring. 2 ISL bands × 1 SLA band over 2 pools (tp1/tp2).
const goodParams = `{
  "poolLabel": "nvidia.com/dynamo-pool",
  "poolLabels": ["tp1", "tp2"],
  "sizeMin": 0, "sizeMax": 4000, "sizeResolution": 2,
  "latencyMinMs": 100, "latencyMaxMs": 5000, "latencyResolution": 1,
  "mapping": [[0],[1]]
}`

func TestFactory_AcceptsValidConfig(t *testing.T) {
	p, err := PoolSelectorFactory("ps", json.RawMessage(goodParams), nil)
	if err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if p.TypedName().Type != PoolSelectorType || p.TypedName().Name != "ps" {
		t.Fatalf("unexpected typed name: %+v", p.TypedName())
	}
}

func TestFactory_Rejects(t *testing.T) {
	cases := map[string]string{
		"missing poolLabel": `{"poolLabels":["tp1"],"sizeMin":0,"sizeMax":1,"sizeResolution":1,"latencyMinMs":1,"latencyMaxMs":2,"latencyResolution":1,"mapping":[[0]]}`,
		"grid dim mismatch": `{"poolLabel":"x","poolLabels":["tp1"],"sizeMin":0,"sizeMax":1,"sizeResolution":2,"latencyMinMs":1,"latencyMaxMs":2,"latencyResolution":1,"mapping":[[0]]}`,
		"malformed json":    `{"poolLabel":`,
	}
	for name, params := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := PoolSelectorFactory("ps", json.RawMessage(params), nil); err == nil {
				t.Fatalf("expected rejection for %q", name)
			}
		})
	}
}

func TestToFloat(t *testing.T) {
	cases := []struct {
		in   any
		want float64
		ok   bool
	}{
		{float64(3.5), 3.5, true},
		{int(4), 4, true},
		{int64(5), 5, true},
		{json.Number("2.5"), 2.5, true},
		{"nope", 0, false},
		{nil, 0, false},
	}
	for _, c := range cases {
		got, ok := toFloat(c.in)
		if ok != c.ok || (ok && got != c.want) {
			t.Fatalf("toFloat(%v)=(%v,%v), want (%v,%v)", c.in, got, ok, c.want, c.ok)
		}
	}
}
