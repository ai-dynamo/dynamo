/*
Copyright 2025 NVIDIA Corporation.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Dynamo EPP - Custom Endpoint Picker Plugin for NVIDIA Dynamo
//
// This EPP integrates with the Gateway API Inference Extension to provide
// KV-aware routing for Dynamo inference backends.
//
// # Header-Based Routing
//
// The Dynamo KV scorer sets routing headers that the Lua filter at the
// gateway uses to inject nvext into the request body:
//
//   - x-worker-instance-id: Selected worker ID (decode worker in disagg mode)
//   - x-prefiller-host-port: Prefill worker ID (disaggregated mode only)
//   - x-dynamo-routing-mode: "aggregated" or "disaggregated"
//
// The Lua filter reads these headers and injects:
//   - Aggregated: {"nvext": {"backend_instance_id": <worker_id>}}
//   - Disaggregated: {"nvext": {"prefill_worker_id": <prefill>, "decode_worker_id": <decode>}}
package main

import (
	"os"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/gateway-api-inference-extension/cmd/epp/runner"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"

	// Dynamo plugins
	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

func main() {
	// Register Dynamo custom plugins:
	// - kv-aware-scorer: Implements Scorer, PreRequest, and ResponseComplete interfaces
	//   - Score: Calls Dynamo router to select workers based on KV cache, sets routing headers
	//   - PreRequest: Registers request with router bookkeeping after scheduling is finalized
	//   - ResponseComplete: Cleans up router bookkeeping when response completes
	plugins.Register("kv-aware-scorer", dynscorer.KVAwareScorerFactory)

	// Run using standard GAIE runner (it registers built-in plugins automatically)
	if err := runner.NewRunner().Run(ctrl.SetupSignalHandler()); err != nil {
		os.Exit(1)
	}
}
