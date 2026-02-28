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
//  4. Assigns score 1.0 to all pods (the router's selection is authoritative, communicated via headers).
type DynPrefillScorer struct {
	typedName plugins.TypedName
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

	if !readPrefillEnabled(cycleState) {
		logger.V(logutil.VERBOSE).Info("DynPrefillScorer: prefill not enabled, returning zero scores")
		return uniformScores(pods, 0)
	}

	requestJSON, err := buildRequestJSON(req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: failed to build request")
		return uniformScores(pods, 0)
	}

	podsJSON := serializePods(pods)
	logger.V(logutil.DEFAULT).Info("DynPrefillScorer: pods received for scoring",
		"podCount", len(pods),
		"podsJSON", string(podsJSON))

	result, err := dynscorer.CallRoutePrefillRequest(requestJSON, podsJSON)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: FFI prefill routing failed")
		return uniformScores(pods, 0)
	}

	prefillWorkerID := fmt.Sprintf("%d", result.WorkerID)
	logger.V(logutil.DEFAULT).Info("DynPrefillScorer: prefill worker selected",
		"prefillWorkerID", prefillWorkerID,
		"tokenCount", len(result.TokenData))

	cycleState.Write(PrefillWorkerIDStateKey, &PrefillWorkerIDState{WorkerID: prefillWorkerID})

	// Score: 1.0 for all pods. The label-filter has already restricted to prefill workers,
	// and the FFI router's internal selection is authoritative.
	// In the future, we could match worker IDs to pod names for precise scoring.
	return uniformScores(pods, 1.0)
}
