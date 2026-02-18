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

package disagg

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	log "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

const (
	// DynPrefillScorerType is the plugin type registered in the plugin registry.
	DynPrefillScorerType = "dyn-prefill-scorer"
)

// compile-time type assertion
var _ framework.Scorer = &DynPrefillScorer{}

// DynPrefillScorerConfig holds the configuration for the DynPrefillScorer plugin.
type DynPrefillScorerConfig struct{}

// DynPrefillScorerFactory defines the factory function for DynPrefillScorer.
func DynPrefillScorerFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	cfg := DynPrefillScorerConfig{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin parameters: %w", DynPrefillScorerType, err)
		}
	}

	// Initialize the shared FFI (idempotent)
	if err := dynscorer.InitFFI(); err != nil {
		return nil, fmt.Errorf("Dynamo FFI init for prefill scorer failed: %w", err)
	}

	return NewDynPrefillScorer().WithName(name), nil
}

// NewDynPrefillScorer initializes a new DynPrefillScorer.
func NewDynPrefillScorer() *DynPrefillScorer {
	return &DynPrefillScorer{
		typedName: plugins.TypedName{Type: DynPrefillScorerType, Name: DynPrefillScorerType},
	}
}

// DynPrefillScorer is a scorer plugin for the prefill scheduling profile.
//
// When Score() is called, it:
//  1. Reads PrefillEnabledState from CycleState (written by DisaggProfileHandler).
//  2. If prefill is NOT enabled, returns zero scores.
//  3. If prefill IS enabled, calls the Dynamo FFI prefill router to select the best prefill worker.
//  4. Assigns score 1.0 to the pod matching the selected worker, 0.0 to all others.
type DynPrefillScorer struct {
	typedName  plugins.TypedName
	warmupOnce sync.Once
	warmupErr  error
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *DynPrefillScorer) TypedName() plugins.TypedName {
	return s.typedName
}

// WithName sets the name of the scorer.
func (s *DynPrefillScorer) WithName(name string) *DynPrefillScorer {
	s.typedName.Name = name
	return s
}

// Score scores pods for prefill suitability.
func (s *DynPrefillScorer) Score(ctx context.Context, cycleState *schedtypes.CycleState, req *schedtypes.LLMRequest, pods []schedtypes.Pod) map[schedtypes.Pod]float64 {
	logger := log.FromContext(ctx)

	// Check if prefill is enabled from CycleState (written by DisaggProfileHandler).
	prefillEnabled := false
	state, err := schedtypes.ReadCycleStateKey[*PrefillEnabledState](cycleState, PrefillEnabledStateKey)
	if err == nil && state != nil {
		prefillEnabled = state.Enabled
	}

	out := make(map[schedtypes.Pod]float64, len(pods))
	if !prefillEnabled {
		logger.V(logutil.VERBOSE).Info("DynPrefillScorer: prefill not enabled, returning zero scores")
		for _, p := range pods {
			out[p] = 0
		}
		return out
	}

	// Build request JSON
	requestBody, err := dynscorer.BuildOpenAIRequest(req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: failed to build request")
		for _, p := range pods {
			out[p] = 0
		}
		return out
	}
	requestJSON, err := json.Marshal(requestBody)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: failed to marshal request")
		for _, p := range pods {
			out[p] = 0
		}
		return out
	}

	// Serialize pods for the FFI filter
	podsJSON := ""
	if len(pods) > 0 {
		if pj, err := dynscorer.SerializePodsToJSON(pods); err == nil {
			podsJSON = pj
		}
	}

	// Call the prefill router via FFI
	result, err := dynscorer.CallRoutePrefillRequest(string(requestJSON), podsJSON)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: FFI prefill routing failed")
		for _, p := range pods {
			out[p] = 0
		}
		return out
	}

	workerIDStr := fmt.Sprintf("%d", result.WorkerID)
	logger.V(logutil.DEFAULT).Info("DynPrefillScorer: prefill worker selected",
		"prefillWorkerID", workerIDStr,
		"tokenCount", len(result.TokenData))

	// Score: 1.0 for the selected worker, 0.0 for all others.
	// The picker will then select the highest-scoring pod.
	for _, p := range pods {
		out[p] = 0
	}
	// Note: In the prefill profile, the label-filter has already restricted pods to prefill workers.
	// The FFI router selected a worker by ID; we give score 1.0 to all pods
	// since the router's internal selection is authoritative and pods have already been filtered.
	// In the future, we could match worker IDs to pod names for precise scoring.
	for _, p := range pods {
		out[p] = 1.0
	}

	return out
}
