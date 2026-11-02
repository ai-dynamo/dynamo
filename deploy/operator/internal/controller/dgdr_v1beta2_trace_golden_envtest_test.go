//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"bytes"
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"testing"

	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// updateTraceRequestLoweringGolden rewrites the golden file instead of comparing with it:
//
//	go test ./internal/controller -run '^TestDGDRV1beta2TraceRequestLoweringGolden$' -update
var updateTraceRequestLoweringGolden = flag.Bool("update", false,
	"rewrite internal/aisimulate/testdata/trace_request_lowering.golden.json")

// traceRequestLoweringGoldenPath is read by the live-retune planner's tests, which check that the
// planner builds request_spec and accepts run_input as the effective input of its Run.
const traceRequestLoweringGoldenPath = "../aisimulate/testdata/trace_request_lowering.golden.json"

// liveRetuneShardRequestSpec is the spec of one grid shard the live-retune planner submits: its
// build_request for an aggregated vLLM deployment with 3 replicas of tensor parallelism 2, the window
// ending at 1793000000000 ms, and a 2x2x2 router grid. It omits workload.trace.format, which the
// CRD schema defaults. Change it together with the planner and rerun with -update.
const liveRetuneShardRequestSpec = `{
  "model": {"name": "Qwen/Qwen3-32B"},
  "backends": ["vllm"],
  "hardware": {"gpu": {"sku": "h200_sxm", "budget": 6}},
  "workload": {
    "trace": {
      "source": {
        "persistentVolumeClaim": {
          "claimName": "request-traces",
          "subPath": "windows/llm/1793000000000.jsonl.gz"
        }
      }
    }
  },
  "objective": {
    "mode": "Optimize",
    "metric": "GoodputPerGPU",
    "sla": {"ttftMs": 500.0, "itlMs": 30.0}
  },
  "recommendation": {"maxCandidates": 8},
  "aisimulate": {
    "inputOverrides": {
      "engine": {
        "mode": "aggregated",
        "backend": "vllm",
        "context_length": 32768,
        "workers": {
          "aggregated": {
            "parallelism": {
              "preset": [
                {"tensor": 2, "pipeline": 1, "attention_data": 1, "moe_tensor": 1, "moe_expert": 1, "replicas": 3}
              ]
            }
          },
          "prefill": null,
          "decode": null
        }
      },
      "router": {
        "policy": "kv_router",
        "prefill_load_model": {"type": "none"},
        "overlap_score_credit": {"choices": [0.5, 1.0]},
        "prefill_load_scale": {"choices": [1.0, 2.0]},
        "temperature": {"choices": [0.0, 0.5]}
      },
      "optimizer": {
        "algorithm": "random",
        "max_trials": 16,
        "parallelism": 8,
        "candidate_timeout_seconds": 300.0
      },
      "optimization": {"strict_sla": false},
      "dgdr_run": {"generator": "direct"}
    }
  }
}`

// The live-retune planner submits a trace Request and checks the input its Run executes. Both
// sides share one golden file, so a change to the trace API, admission or lowering that the planner
// does not follow fails here or in the planner's tests.
func TestDGDRV1beta2TraceRequestLoweringGolden(t *testing.T) {
	env := sharedEnv.ForTest(t)
	c := env.Client()
	ctx := t.Context()

	t.Log("Decode the planner's shard Request; every field must belong to the v1beta2 API")
	var spec nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	decoder := json.NewDecoder(bytes.NewReader([]byte(liveRetuneShardRequestSpec)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&spec); err != nil {
		t.Fatalf("decode the planner's Request spec: %v", err)
	}

	t.Log("Create the Request through admission and the CRD schema, which complete it")
	request := &nvidiacomv1beta2.DynamoGraphDeploymentRequest{Name: "llm-rt-golden", Namespace: env.Namespace(), Spec: spec}
	if err := c.Create(ctx, request); err != nil {
		t.Fatalf("create the Request: %v", err)
	}
	if err := c.Get(ctx, client.ObjectKeyFromObject(request), request); err != nil {
		t.Fatalf("get the Request: %v", err)
	}

	t.Log("Freeze the admitted Request into its Run as the Request controller does, and store the Run")
	run, err := (&DynamoGraphDeploymentRequestV1beta2Reconciler{Client: c}).buildRun(request)
	if err != nil {
		t.Fatalf("build the Run: %v", err)
	}
	if err := c.Create(ctx, run); err != nil {
		t.Fatalf("create the Run: %v", err)
	}
	if err := c.Get(ctx, client.ObjectKeyFromObject(run), run); err != nil {
		t.Fatalf("get the Run: %v", err)
	}

	t.Log("Lower the stored Run as the Run controller does, in a fixed namespace so the golden file is stable")
	run.Namespace = "test-ns"
	input, err := dgdrRunInput(run, "example.com/dynamo-planner:1.2.3", "1.2.3")
	if err != nil {
		t.Fatalf("lower the Run: %v", err)
	}
	got, err := json.MarshalIndent(struct {
		RequestSpec json.RawMessage `json:"request_spec"`
		RunInput    map[string]any  `json:"run_input"`
	}{RequestSpec: json.RawMessage(liveRetuneShardRequestSpec), RunInput: input}, "", "  ")
	if err != nil {
		t.Fatalf("encode the golden file: %v", err)
	}
	got = append(got, '\n')

	t.Log("Compare with the golden file, or rewrite it with -update")
	if *updateTraceRequestLoweringGolden {
		if err := os.MkdirAll(filepath.Dir(traceRequestLoweringGoldenPath), 0o755); err != nil {
			t.Fatalf("create the golden directory: %v", err)
		}
		if err := os.WriteFile(traceRequestLoweringGoldenPath, got, 0o644); err != nil {
			t.Fatalf("write the golden file: %v", err)
		}
	}
	want, err := os.ReadFile(traceRequestLoweringGoldenPath)
	if err != nil {
		t.Fatalf("read the golden file: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("%s is stale; rerun with -update and update the planner if needed\n got: %s\nwant: %s",
			traceRequestLoweringGoldenPath, got, want)
	}
}
