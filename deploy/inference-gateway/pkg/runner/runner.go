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

// Package runner provides a custom EPP runner that wraps the GAIE
// director with Dynamo body mutation support.
//
// This package re-exports the GAIE runner with registration of built-in plugins.
// The body mutation happens via the BodyMutatingDirector wrapper which is injected
// by replacing the Director in the ExtProcServerRunner before starting.
package runner

import (
	"sigs.k8s.io/gateway-api-inference-extension/cmd/epp/runner"
	dlmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	testresponsereceived "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol/plugins/test/responsereceived"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/slo_aware_router"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
	testfilter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/test/filter"
)

// RegisterInTreePlugins registers all the built-in GAIE plugins.
// This should be called before creating the runner.
func RegisterInTreePlugins() {
	plugins.Register(prefix.PrefixCachePluginType, prefix.PrefixCachePluginFactory)
	plugins.Register(picker.MaxScorePickerType, picker.MaxScorePickerFactory)
	plugins.Register(picker.RandomPickerType, picker.RandomPickerFactory)
	plugins.Register(picker.WeightedRandomPickerType, picker.WeightedRandomPickerFactory)
	plugins.Register(profile.SingleProfileHandlerType, profile.SingleProfileHandlerFactory)
	plugins.Register(scorer.KvCacheUtilizationScorerType, scorer.KvCacheUtilizationScorerFactory)
	plugins.Register(scorer.QueueScorerType, scorer.QueueScorerFactory)
	plugins.Register(scorer.RunningRequestsSizeScorerType, scorer.RunningRequestsSizeScorerFactory)
	plugins.Register(scorer.LoraAffinityScorerType, scorer.LoraAffinityScorerFactory)
	plugins.Register(slo_aware_router.SLOAwareRouterPluginType, slo_aware_router.SLOAwareRouterFactory)
	plugins.Register(testfilter.HeaderBasedTestingFilterType, testfilter.HeaderBasedTestingFilterFactory)
	plugins.Register(testresponsereceived.DestinationEndpointServedVerifierType, testresponsereceived.DestinationEndpointServedVerifierFactory)
	plugins.Register(dlmetrics.MetricsDataSourceType, dlmetrics.MetricsDataSourceFactory)
	plugins.Register(dlmetrics.MetricsExtractorType, dlmetrics.ModelServerExtractorFactory)
}

// NewRunner creates a new GAIE EPP runner.
// Call RegisterInTreePlugins() before this to register built-in plugins.
func NewRunner() *runner.Runner {
	return runner.NewRunner()
}
