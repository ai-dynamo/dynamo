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
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	plugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	DisaggProfileHandlerType = "disagg-profile-handler"
)

// compile-time type assertion
var _ schedtypes.ProfileHandler = &DisaggProfileHandler{}

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
	warnDeprecatedEnforceDisagg(log.Log.WithName(DisaggProfileHandlerType))
	return NewDisaggProfileHandler().WithName(name), nil
}

// NewDisaggProfileHandler initializes a new DisaggProfileHandler.
func NewDisaggProfileHandler() *DisaggProfileHandler {
	return &DisaggProfileHandler{
		typedName: plugins.TypedName{Type: DisaggProfileHandlerType, Name: DisaggProfileHandlerType},
	}
}

// DisaggProfileHandler is a ProfileHandler that orchestrates prefill/decode disaggregated serving.
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
func (h *DisaggProfileHandler) Pick(ctx context.Context, cycleState *schedtypes.CycleState, _ *schedtypes.InferenceRequest,
	profiles map[string]schedtypes.SchedulerProfile, profileResults map[string]*schedtypes.ProfileRunResult) map[string]schedtypes.SchedulerProfile {

	logger := log.FromContext(ctx).V(logutil.VERBOSE)

	if len(profileResults) == 0 {
		_, prefillExists := profiles[PrefillProfileName]
		state := &PrefillEnabledState{Enabled: prefillExists}
		cycleState.Write(PrefillEnabledStateKey, state)
		logger.Info("DisaggProfileHandler: prefill enabled state determined", "prefillEnabled", prefillExists)

		if prefillExists {
			return map[string]schedtypes.SchedulerProfile{
				PrefillProfileName: profiles[PrefillProfileName],
			}
		}
		if decodeProfile, ok := profiles[DecodeProfileName]; ok {
			return map[string]schedtypes.SchedulerProfile{
				DecodeProfileName: decodeProfile,
			}
		}
		return profiles
	}

	if prefillResult, prefillDone := profileResults[PrefillProfileName]; prefillDone {
		if _, decodeDone := profileResults[DecodeProfileName]; !decodeDone {
			if prefillResult == nil {
				logger.Info("DisaggProfileHandler: prefill profile failed (no workers?), falling back to aggregated decode")
				cycleState.Write(PrefillEnabledStateKey, &PrefillEnabledState{Enabled: false})
			}

			if decodeProfile, ok := profiles[DecodeProfileName]; ok {
				return map[string]schedtypes.SchedulerProfile{
					DecodeProfileName: decodeProfile,
				}
			}
		}
	}

	return map[string]schedtypes.SchedulerProfile{}
}

// ProcessResults aggregates the profile run results and designates the primary profile.
func (h *DisaggProfileHandler) ProcessResults(_ context.Context, _ *schedtypes.CycleState, _ *schedtypes.InferenceRequest,
	profileResults map[string]*schedtypes.ProfileRunResult) (*schedtypes.SchedulingResult, error) {

	if len(profileResults) == 0 {
		return nil, errors.New("disagg profile handler received no profile results")
	}

	primaryProfile := DecodeProfileName
	if _, ok := profileResults[DecodeProfileName]; !ok {
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
