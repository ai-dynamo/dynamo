/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"context"
	"encoding/json"
	"testing"

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
	want := &configv1alpha1.EndpointPickerConfig{
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
		SaturationDetector: &configv1alpha1.SaturationDetectorConfig{PluginRef: "detector"},
		DataLayer: &configv1alpha1.DataLayerConfig{
			Sources: []configv1alpha1.DataLayerSource{
				{PluginRef: "source", Extractors: []configv1alpha1.DataLayerExtractor{{PluginRef: "extractor"}}},
			},
		},
		FlowControl: &configv1alpha1.FlowControlConfig{MaxRequests: &maxRequests},
		Parser:      &configv1alpha1.ParserConfig{PluginRef: "parser"},
	}

	configMap, err := GenerateConfigMap(
		context.Background(),
		&v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "ns"}},
		"epp",
		&v1beta1.EPPConfig{Config: want},
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
	want.TypeMeta = metav1.TypeMeta{APIVersion: configv1alpha1.GroupVersion.String(), Kind: "EndpointPickerConfig"}
	if diff := cmp.Diff(want, decoded); diff != "" {
		t.Errorf("generated config mismatch (-want +got):\n%s", diff)
	}
}
