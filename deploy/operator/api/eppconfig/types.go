/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package eppconfig

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
)

// EndpointPickerConfig is the Schema for the endpointpickerconfigs API.
type EndpointPickerConfig struct {
	metav1.TypeMeta `json:",inline"`

	// +optional
	// FeatureGates is a set of flags that enable various experimental features with the EPP.
	// If omitted none of these experimental features will be enabled.
	FeatureGates apixv1alpha1.FeatureGates `json:"featureGates,omitempty"`

	// +required
	// +kubebuilder:validation:Required
	// Plugins is the list of plugins that will be instantiated.
	Plugins []apixv1alpha1.PluginSpec `json:"plugins"`

	// +required
	// +kubebuilder:validation:Required
	// SchedulingProfiles is the list of named SchedulingProfiles
	// that will be created.
	SchedulingProfiles []apixv1alpha1.SchedulingProfile `json:"schedulingProfiles"`

	// +optional
	// SaturationDetector specifies which saturation detector plugin to use for both Admission and
	// Flow Control. The deprecated threshold fields are normalized to a utilization-detector plugin.
	// If omitted, "utilization-detector" is used by default.
	SaturationDetector *SaturationDetectorConfig `json:"saturationDetector,omitempty"`

	// +optional
	// DataLayer configures the DataLayer. It is required if the new DataLayer is enabled.
	DataLayer *apixv1alpha1.DataLayerConfig `json:"dataLayer,omitempty"`

	// +optional
	// FlowControl configures the Flow Control layer.
	// This configuration is only respected if the "flowControl" FeatureGate is enabled.
	FlowControl *apixv1alpha1.FlowControlConfig `json:"flowControl,omitempty"`

	// +optional
	// Parser specifies the parsing logic used by the EPP to process protocol messages.
	// If unspecified, default parsing behavior will be applied.
	Parser *apixv1alpha1.ParserConfig `json:"parser,omitempty"`
}

// TODO(sttts): Remove the deprecated threshold fields from the Dynamo v1 API
// and align this type with the then-current GAIE schema.

// SaturationDetectorConfig supports the current GAIE plugin reference and the
// deprecated GAIE v1.2 threshold configuration. The two forms are mutually
// exclusive.
// +kubebuilder:validation:XValidation:rule="!has(self.pluginRef) || !(has(self.queueDepthThreshold) || has(self.kvCacheUtilThreshold) || has(self.metricsStalenessThreshold))",message="pluginRef is mutually exclusive with the deprecated saturation detector threshold fields"
type SaturationDetectorConfig struct {
	// +optional
	// PluginRef specifies the name of the plugin instance to use for saturation detection.
	// The reference is to the name of an entry of the Plugins defined in the configuration's Plugins section.
	// If unspecified, "utilization-detector" is used by default.
	PluginRef string `json:"pluginRef,omitempty"`

	// QueueDepthThreshold defines the backend waiting queue size above which a
	// pod is considered to have insufficient capacity for new requests.
	// Deprecated: configure a utilization-detector plugin and use pluginRef.
	// +optional
	QueueDepthThreshold int `json:"queueDepthThreshold,omitempty"`

	// KVCacheUtilThreshold defines the KV cache utilization (0.0 to 1.0) above
	// which a pod is considered to have insufficient capacity.
	// Deprecated: configure a utilization-detector plugin and use pluginRef.
	// +optional
	KVCacheUtilThreshold float64 `json:"kvCacheUtilThreshold,omitempty"`

	// MetricsStalenessThreshold defines how old a pod's metrics can be.
	// If a pod's metrics are older than this, it might be excluded from
	// "good capacity" considerations or treated as having no capacity for
	// safety.
	// Deprecated: configure a utilization-detector plugin and use pluginRef.
	// +optional
	MetricsStalenessThreshold metav1.Duration `json:"metricsStalenessThreshold,omitempty"`
}
