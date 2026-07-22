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

package label_filter

import (
	"context"
	"encoding/json"
	"fmt"

	plugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	schedtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	LabelFilterType = "label-filter"
)

// compile-time type assertion
var _ schedtypes.Filter = &LabelFilter{}

// LabelFilterConfig holds the configuration for the LabelFilter plugin.
type LabelFilterConfig struct {
	Label         string   `json:"label"`
	ValidValues   []string `json:"validValues"`
	AllowsNoLabel bool     `json:"allowsNoLabel"`
}

func LabelFilterFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	cfg := LabelFilterConfig{}
	if rawParameters == nil {
		return nil, fmt.Errorf("%s plugin requires parameters with 'label' and 'validValues' fields", LabelFilterType)
	}
	if err := json.Unmarshal(rawParameters, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse %s plugin parameters: %w", LabelFilterType, err)
	}
	if cfg.Label == "" {
		return nil, fmt.Errorf("%s plugin parameter 'label' must not be empty", LabelFilterType)
	}
	if len(cfg.ValidValues) == 0 {
		return nil, fmt.Errorf("%s plugin parameter 'validValues' must contain at least one value", LabelFilterType)
	}
	return NewLabelFilter(cfg.Label, cfg.ValidValues, cfg.AllowsNoLabel).WithName(name), nil
}

func NewLabelFilter(label string, validValues []string, allowsNoLabel bool) *LabelFilter {
	valuesSet := make(map[string]struct{}, len(validValues))
	for _, v := range validValues {
		valuesSet[v] = struct{}{}
	}
	return &LabelFilter{
		typedName:     plugins.TypedName{Type: LabelFilterType, Name: LabelFilterType},
		label:         label,
		validValues:   valuesSet,
		allowsNoLabel: allowsNoLabel,
	}
}

type LabelFilter struct {
	typedName     plugins.TypedName
	label         string
	validValues   map[string]struct{}
	allowsNoLabel bool
}

func (f *LabelFilter) TypedName() plugins.TypedName {
	return f.typedName
}

func (f *LabelFilter) WithName(name string) *LabelFilter {
	f.typedName.Name = name
	return f
}

// Filter returns only the endpoints whose label matches one of the configured valid values.
func (f *LabelFilter) Filter(_ context.Context, _ *schedtypes.CycleState, _ *schedtypes.InferenceRequest, endpoints []schedtypes.Endpoint) []schedtypes.Endpoint {
	filtered := make([]schedtypes.Endpoint, 0, len(endpoints))
	for _, ep := range endpoints {
		if ep == nil || ep.GetMetadata() == nil {
			continue
		}
		labelValue, hasLabel := ep.GetMetadata().Labels[f.label]
		if !hasLabel {
			if f.allowsNoLabel {
				filtered = append(filtered, ep)
			}
			continue
		}
		if _, ok := f.validValues[labelValue]; ok {
			filtered = append(filtered, ep)
		}
	}
	return filtered
}
