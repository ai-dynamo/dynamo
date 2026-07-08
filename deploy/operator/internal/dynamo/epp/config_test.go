/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/eppconfig"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	configv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

func TestGenerateConfigMapPreservesGAIEV15Config(t *testing.T) {
	weight := 0.5
	maxRequests := resource.MustParse("100")
	parameters := json.RawMessage("{}")
	config := &eppconfig.EndpointPickerConfig{
		FeatureGates: configv1alpha1.FeatureGates{"dataLayer", "flowControl"},
		Plugins: []configv1alpha1.PluginSpec{
			{Name: "source", Type: "metrics-source", Parameters: parameters},
			{Name: "extractor", Type: "metrics-extractor", Parameters: parameters},
			{Name: "scorer", Type: "prefix-cache-scorer", Parameters: parameters},
			{Name: "detector", Type: "utilization-detector", Parameters: parameters},
			{Name: "parser", Type: "openai-parser", Parameters: parameters},
		},
		SchedulingProfiles: []configv1alpha1.SchedulingProfile{
			{Name: "default", Plugins: []configv1alpha1.SchedulingPlugin{{PluginRef: "scorer", Weight: &weight}}},
		},
		SaturationDetector: &eppconfig.SaturationDetectorConfig{PluginRef: "detector"},
		DataLayer: &configv1alpha1.DataLayerConfig{
			Sources: []configv1alpha1.DataLayerSource{
				{PluginRef: "source", Extractors: []configv1alpha1.DataLayerExtractor{{PluginRef: "extractor"}}},
			},
		},
		FlowControl: &configv1alpha1.FlowControlConfig{MaxRequests: &maxRequests},
		Parser:      &configv1alpha1.ParserConfig{PluginRef: "parser"},
	}
	want := &configv1alpha1.EndpointPickerConfig{
		TypeMeta:           metav1.TypeMeta{APIVersion: configv1alpha1.GroupVersion.String(), Kind: "EndpointPickerConfig"},
		FeatureGates:       config.FeatureGates,
		Plugins:            config.Plugins,
		SchedulingProfiles: config.SchedulingProfiles,
		SaturationDetector: &configv1alpha1.SaturationDetectorConfig{PluginRef: "detector"},
		DataLayer:          config.DataLayer,
		FlowControl:        config.FlowControl,
		Parser:             config.Parser,
	}
	original := config.DeepCopy()

	decoded := generateAndDecodeConfig(t, config)
	if diff := cmp.Diff(want, decoded); diff != "" {
		t.Errorf("generated config mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(original, config); diff != "" {
		t.Errorf("GenerateConfigMap() mutated its input (-want +got):\n%s", diff)
	}
}

func TestGenerateConfigMapNormalizesDeprecatedSaturationDetector(t *testing.T) {
	config := &eppconfig.EndpointPickerConfig{
		Plugins:            []configv1alpha1.PluginSpec{{Name: normalizedSaturationDetectorPlugin, Type: "another-plugin"}},
		SchedulingProfiles: []configv1alpha1.SchedulingProfile{},
		SaturationDetector: &eppconfig.SaturationDetectorConfig{
			QueueDepthThreshold:       7,
			KVCacheUtilThreshold:      0.9,
			MetricsStalenessThreshold: metav1.Duration{Duration: 3 * time.Second},
		},
	}
	original := config.DeepCopy()

	decoded := generateAndDecodeConfig(t, config)
	if len(decoded.Plugins) != 2 {
		t.Fatalf("generated plugins = %d, want 2", len(decoded.Plugins))
	}
	detectorPlugin := decoded.Plugins[1]
	if detectorPlugin.Name != normalizedSaturationDetectorPlugin+"-2" {
		t.Errorf("generated detector plugin name = %q, want %q", detectorPlugin.Name, normalizedSaturationDetectorPlugin+"-2")
	}
	if detectorPlugin.Type != utilizationDetectorPluginType {
		t.Errorf("generated detector plugin type = %q, want %q", detectorPlugin.Type, utilizationDetectorPluginType)
	}
	if decoded.SaturationDetector == nil || decoded.SaturationDetector.PluginRef != detectorPlugin.Name {
		t.Errorf("generated saturation detector = %#v, want pluginRef %q", decoded.SaturationDetector, detectorPlugin.Name)
	}

	parameters := utilizationDetectorParameters{}
	if err := json.Unmarshal(detectorPlugin.Parameters, &parameters); err != nil {
		t.Fatalf("generated detector parameters are invalid: %v", err)
	}
	if parameters.QueueDepthThreshold == nil || *parameters.QueueDepthThreshold != 7 {
		t.Errorf("queueDepthThreshold = %v, want 7", parameters.QueueDepthThreshold)
	}
	if parameters.KVCacheUtilThreshold == nil || *parameters.KVCacheUtilThreshold != 0.9 {
		t.Errorf("kvCacheUtilThreshold = %v, want 0.9", parameters.KVCacheUtilThreshold)
	}
	if parameters.MetricsStalenessThreshold == nil || parameters.MetricsStalenessThreshold.Duration != 3*time.Second {
		t.Errorf("metricsStalenessThreshold = %v, want 3s", parameters.MetricsStalenessThreshold)
	}
	if diff := cmp.Diff(original, config); diff != "" {
		t.Errorf("GenerateConfigMap() mutated its input (-want +got):\n%s", diff)
	}
}

func TestGenerateConfigMapPreservesDeprecatedSaturationDetectorDefaults(t *testing.T) {
	config := &eppconfig.EndpointPickerConfig{
		SchedulingProfiles: []configv1alpha1.SchedulingProfile{},
		SaturationDetector: &eppconfig.SaturationDetectorConfig{
			QueueDepthThreshold:       -1,
			KVCacheUtilThreshold:      1,
			MetricsStalenessThreshold: metav1.Duration{Duration: -time.Second},
		},
	}

	decoded := generateAndDecodeConfig(t, config)
	if len(decoded.Plugins) != 1 {
		t.Fatalf("generated plugins = %d, want 1", len(decoded.Plugins))
	}
	parameters := map[string]any{}
	if err := json.Unmarshal(decoded.Plugins[0].Parameters, &parameters); err != nil {
		t.Fatalf("generated detector parameters are invalid: %v", err)
	}
	if len(parameters) != 0 {
		t.Errorf("generated detector parameters = %v, want defaults", parameters)
	}
}

func TestMarshalEndpointPickerConfigRejectsMixedSaturationDetectorForms(t *testing.T) {
	_, err := marshalEndpointPickerConfig(&eppconfig.EndpointPickerConfig{
		SaturationDetector: &eppconfig.SaturationDetectorConfig{
			PluginRef:           "detector",
			QueueDepthThreshold: 7,
		},
	})
	if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
		t.Fatalf("marshalEndpointPickerConfig() error = %v, want mutual-exclusion error", err)
	}
}

func generateAndDecodeConfig(t *testing.T, config *eppconfig.EndpointPickerConfig) *configv1alpha1.EndpointPickerConfig {
	t.Helper()

	configMap, err := GenerateConfigMap(
		context.Background(),
		&v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "ns"}},
		"epp",
		&v1beta1.EPPConfig{Config: config},
	)
	if err != nil {
		t.Fatalf("GenerateConfigMap() error = %v", err)
	}

	scheme := runtime.NewScheme()
	if err := configv1alpha1.Install(scheme); err != nil {
		t.Fatalf("Install() error = %v", err)
	}
	decoded := &configv1alpha1.EndpointPickerConfig{}
	codecs := serializer.NewCodecFactory(scheme, serializer.EnableStrict)
	if err := runtime.DecodeInto(codecs.UniversalDecoder(), []byte(configMap.Data[ConfigKey]), decoded); err != nil {
		t.Fatalf("generated EPP config is not valid GAIE v1.5 configuration: %v", err)
	}
	return decoded
}
