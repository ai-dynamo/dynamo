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
	"strconv"

	log "sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	plugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	dynscorer "github.com/nvidia/dynamo/deploy/inference-gateway/pkg/plugins/dynamo_kv_scorer"
)

const (
	// DynPrefillScorerType is the plugin type registered in the plugin registry.
	DynPrefillScorerType = "dyn-prefill-scorer"
)

// compile-time type assertion
var _ schedtypes.Scorer = &DynPrefillScorer{}

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

// Category returns the scorer category.
func (s *DynPrefillScorer) Category() schedtypes.ScorerCategory {
	return schedtypes.Affinity
}

// Score scores endpoints for prefill suitability.
func (s *DynPrefillScorer) Score(ctx context.Context, cycleState *schedtypes.CycleState, req *schedtypes.InferenceRequest, endpoints []schedtypes.Endpoint) map[schedtypes.Endpoint]float64 {
	logger := log.FromContext(ctx)

	if !readPrefillEnabled(cycleState) {
		logger.V(logutil.VERBOSE).Info("DynPrefillScorer: prefill not enabled, returning zero scores")
		return uniformScores(endpoints, 0)
	}

	requestJSON, err := buildRequestJSON(req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: failed to build request")
		return uniformScores(endpoints, 0)
	}

	endpointsJSON := serializeEndpoints(endpoints)
	logger.V(logutil.DEFAULT).Info("DynPrefillScorer: endpoints received for scoring",
		"endpointCount", len(endpoints),
		"endpointsJSON", string(endpointsJSON))

	result, err := dynscorer.CallRoutePrefillRequest(requestJSON, endpointsJSON)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "DynPrefillScorer: FFI prefill routing failed")
		cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
		return uniformScores(endpoints, 0)
	}

	prefillWorkerID := strconv.FormatUint(result.WorkerID, 10)
	logger.V(logutil.DEFAULT).Info("DynPrefillScorer: prefill worker selected",
		"prefillWorkerID", prefillWorkerID,
		"prefillDpRank", result.DpRank,
		"tokenCount", len(result.TokenData))

	if req.Headers == nil {
		req.Headers = map[string]string{}
	}
	req.Headers[PrefillWorkerIDHeader] = prefillWorkerID
	if result.DpRank != dynscorer.UnsetDpRank {
		req.Headers[PrefillDpRankHeader] = strconv.FormatUint(uint64(result.DpRank), 10)
	} else {
		delete(req.Headers, PrefillDpRankHeader)
	}

	return uniformScores(endpoints, 1.0)
}
