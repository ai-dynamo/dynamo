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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	// LabelFilterType is the filter type that is used in the plugin registry.
	LabelFilterType = "label-filter"
)

// compile-time type assertion
var _ framework.Filter = &LabelFilter{}

// LabelFilterConfig holds the configuration for the LabelFilter plugin.
type LabelFilterConfig struct {
	// Key is the label key to match on each pod's labels.
	Key string `json:"key"`
	// Value is the expected label value. Only pods whose label matches this value are kept.
	Value string `json:"value"`
}

// LabelFilterFactory defines the factory function for LabelFilter.
// It parses the JSON parameters to obtain the label key and value.
func LabelFilterFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	cfg := LabelFilterConfig{}
	if rawParameters == nil {
		return nil, fmt.Errorf("%s plugin requires parameters with 'key' and 'value' fields", LabelFilterType)
	}
	if err := json.Unmarshal(rawParameters, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse %s plugin parameters: %w", LabelFilterType, err)
	}
	if cfg.Key == "" {
		return nil, fmt.Errorf("%s plugin parameter 'key' must not be empty", LabelFilterType)
	}
	return NewLabelFilter(cfg.Key, cfg.Value).WithName(name), nil
}

// NewLabelFilter initializes a new LabelFilter with the given label key and value.
func NewLabelFilter(key, value string) *LabelFilter {
	return &LabelFilter{
		typedName: plugins.TypedName{Type: LabelFilterType, Name: LabelFilterType},
		key:       key,
		value:     value,
	}
}

// LabelFilter filters pods based on a pod label key/value pair.
// Only pods whose pod label[key] equals the configured value are kept.
type LabelFilter struct {
	typedName plugins.TypedName
	key       string
	value     string
}

// TypedName returns the type and name tuple of this plugin instance.
func (f *LabelFilter) TypedName() plugins.TypedName {
	return f.typedName
}

// WithName sets the name of the filter.
func (f *LabelFilter) WithName(name string) *LabelFilter {
	f.typedName.Name = name
	return f
}

// Filter returns only the pods whose label matches the configured key/value.
func (f *LabelFilter) Filter(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, pods []types.Pod) []types.Pod {
	filtered := make([]types.Pod, 0, len(pods))
	for _, pod := range pods {
		if pod == nil || pod.GetPod() == nil {
			continue
		}
		if pod.GetPod().Labels[f.key] == f.value {
			filtered = append(filtered, pod)
		}
	}
	return filtered
}
