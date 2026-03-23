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
	"errors"
	"fmt"

	log "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	DisaggProfileHandlerType = "disagg-profile-handler"
)

// compile-time type assertion
var _ framework.ProfileHandler = &DisaggProfileHandler{}

// DisaggProfileHandlerConfig holds the configuration for the DisaggProfileHandler.
type DisaggProfileHandlerConfig struct{}

// DisaggProfileHandlerFactory defines the factory function for DisaggProfileHandler.
func DisaggProfileHandlerFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	cfg := DisaggProfileHandlerConfig{}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin parameters: %w", DisaggProfileHandlerType, err)
		}
	}
	return NewDisaggProfileHandler().WithName(name), nil
}

// NewDisaggProfileHandler initializes a new DisaggProfileHandler.
func NewDisaggProfileHandler() *DisaggProfileHandler {
	return &DisaggProfileHandler{
		typedName: plugins.TypedName{Type: DisaggProfileHandlerType, Name: DisaggProfileHandlerType},
	}
}

// DisaggProfileHandler is a ProfileHandler that orchestrates prefill/decode disaggregated serving.
//
// # Disaggregated mode detection
//
// In Dynamo's native architecture, disaggregated mode is determined by whether prefill workers
// actually exist at runtime (the is_disaggregated flag in the Rust KV router). However, the
// GAIE EPP framework determines profile availability at configuration time, not at runtime.
// To bridge this gap, DisaggProfileHandler uses the EPP profile mechanism as a proxy:
// it checks whether a "prefill" scheduling profile is registered in the config. If prefill
// workers are configured but none are actually running, the prefill profile's label-filter
// will find zero pods, causing the profile to fail — and the handler gracefully degrades
// to aggregated mode (see below).
//
// # Scheduling flow
//
// On each scheduling cycle it:
//  1. Checks whether a "prefill" profile is registered in the config.
//  2. Writes PrefillEnabledState into CycleState so scorer plugins can read it.
//  3. If a prefill profile exists: runs the "prefill" profile first, then the "decode" profile.
//     The "decode" profile is the primary (the pod the request is ultimately sent to).
//  4. If no prefill profile exists: runs only the "decode" profile (pure aggregated mode).
//
// # Graceful degradation
//
// When a prefill profile is configured but no prefill workers are available at runtime,
// the handler degrades gracefully to aggregated mode on a per-request basis:
//
//  1. Pick (iteration 1): prefill profile exists → writes PrefillEnabled=true → runs prefill profile.
//  2. Prefill profile runs: label-filter finds 0 prefill pods → profile fails → result is nil.
//  3. Pick (iteration 2): sees prefill result is nil → overwrites PrefillEnabled=false → runs decode profile.
//  4. Decode scorer runs: reads PrefillEnabled=false → passes isDisaggregated=false to the Rust
//     decode router → full KV cache overlap scoring is used (overlap_score_weight=1.0).
//
// This means the same YAML config works transparently for both aggregated and disaggregated
// deployments. If prefill workers come up later, subsequent requests automatically use
// disaggregated routing. If they go down, requests fall back to aggregated mode.
type DisaggProfileHandler struct {
	typedName plugins.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (h *DisaggProfileHandler) TypedName() plugins.TypedName {
	return h.typedName
}

// WithName sets the name of the profile handler.
func (h *DisaggProfileHandler) WithName(name string) *DisaggProfileHandler {
	h.typedName.Name = name
	return h
}

// Pick selects which profiles to run in the current iteration.
//
// Iteration 1 (no results yet):
//   - Writes PrefillEnabledState into CycleState.
//   - If a "prefill" profile exists → returns it alone (run prefill first).
//   - Otherwise → returns the "decode" profile.
//
// Iteration 2 (prefill result exists, decode not yet):
//   - Returns the "decode" profile.
//
// Iteration 3+ (all results collected):
//   - Returns empty map to stop the loop.
func (h *DisaggProfileHandler) Pick(ctx context.Context, cycleState *schedtypes.CycleState, _ *schedtypes.LLMRequest,
	profiles map[string]*framework.SchedulerProfile, profileResults map[string]*schedtypes.ProfileRunResult) map[string]*framework.SchedulerProfile {

	logger := log.FromContext(ctx).V(logutil.VERBOSE)

	// First call: determine if prefill is enabled and write state.
	if len(profileResults) == 0 {
		_, prefillExists := profiles[PrefillProfileName]
		state := &PrefillEnabledState{Enabled: prefillExists}
		cycleState.Write(PrefillEnabledStateKey, state)
		logger.Info("DisaggProfileHandler: prefill enabled state determined", "prefillEnabled", prefillExists)

		if prefillExists {
			// Run prefill profile first.
			return map[string]*framework.SchedulerProfile{
				PrefillProfileName: profiles[PrefillProfileName],
			}
		}
		// No prefill profile — run decode only.
		if decodeProfile, ok := profiles[DecodeProfileName]; ok {
			return map[string]*framework.SchedulerProfile{
				DecodeProfileName: decodeProfile,
			}
		}
		// Fallback: return all profiles.
		return profiles
	}

	// Second call: prefill has run, now run decode.
	if prefillResult, prefillDone := profileResults[PrefillProfileName]; prefillDone {
		if _, decodeDone := profileResults[DecodeProfileName]; !decodeDone {
			// If the prefill profile failed (nil result = no prefill pods available),
			// update PrefillEnabledState to false so the decode scorer uses normal
			// KV cache overlap scoring instead of disaggregated mode (overlap_score_weight=0).
			if prefillResult == nil {
				logger.Info("DisaggProfileHandler: prefill profile failed (no workers?), falling back to aggregated decode")
				cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
			}

			if decodeProfile, ok := profiles[DecodeProfileName]; ok {
				return map[string]*framework.SchedulerProfile{
					DecodeProfileName: decodeProfile,
				}
			}
		}
	}

	// All profiles have been executed.
	return map[string]*framework.SchedulerProfile{}
}

// ProcessResults aggregates the profile run results and designates the primary profile.
// The "decode" profile is always the primary (the pod that handles the request).
func (h *DisaggProfileHandler) ProcessResults(_ context.Context, _ *schedtypes.CycleState, _ *schedtypes.LLMRequest,
	profileResults map[string]*schedtypes.ProfileRunResult) (*schedtypes.SchedulingResult, error) {

	if len(profileResults) == 0 {
		return nil, errors.New("disagg profile handler received no profile results")
	}

	// Determine primary profile name.
	primaryProfile := DecodeProfileName
	if _, ok := profileResults[DecodeProfileName]; !ok {
		// If there's no decode result, pick whichever profile ran.
		for name := range profileResults {
			primaryProfile = name
			break
		}
	}

	if profileResults[primaryProfile] == nil {
		return nil, fmt.Errorf("primary profile '%s' failed to produce a result", primaryProfile)
	}

	return &schedtypes.SchedulingResult{
		ProfileResults:     profileResults,
		PrimaryProfileName: primaryProfile,
	}, nil
}
