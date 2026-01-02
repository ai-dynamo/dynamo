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
// # Header-Only Routing
//
// The Dynamo plugins communicate routing decisions via HTTP headers:
//
//   - x-worker-instance-id: Selected worker ID
//   - x-prefiller-host-port: Prefill worker (disagg mode)
//   - x-dynamo-token-data: JSON token IDs for KV cache
//   - x-dynamo-routing-mode: "aggregated" or "disaggregated"
//   - x-dynamo-backend-instance-id: Worker ID (aggregated)
//   - x-dynamo-prefill-worker-id: Prefill worker (disagg)
//   - x-dynamo-decode-worker-id: Decode worker (disagg)
//
// Backend workers must read these headers for routing decisions.
package main

import (
	"os"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/gateway-api-inference-extension/cmd/epp/runner"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"

	// Dynamo plugins
	dyncleanup "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_cleanup"
	dynprereq "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_inject_workerid"
	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

func main() {
	// Register Dynamo custom plugins:
	// - kv-aware-scorer: Calls Dynamo router to select workers based on KV cache
	// - dynamo-inject-workerid: Normalizes routing headers for backend consumption
	// - dynamo-cleanup: Cleans up router state after request completion
	plugins.Register("kv-aware-scorer", dynscorer.KVAwareScorerFactory)
	plugins.Register("dynamo-inject-workerid", dynprereq.InjectWorkerIDPreRequestFactory)
	plugins.Register("dynamo-cleanup", dyncleanup.DynamoCleanupPluginFactory)

	// Run using standard GAIE runner (it registers built-in plugins automatically)
	if err := runner.NewRunner().Run(ctrl.SetupSignalHandler()); err != nil {
		os.Exit(1)
	}
}
