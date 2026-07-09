/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/eppconfig"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apixv1alpha1 "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
	"sigs.k8s.io/yaml"
)

const (
	// ConfigMapSuffix is appended to DGD name to create EPP ConfigMap name
	ConfigMapSuffix = "epp-config"
	// ConfigKey is the key in the ConfigMap containing the EPP configuration
	ConfigKey = "epp-config-dynamo.yaml"

	utilizationDetectorPluginType      = "utilization-detector"
	normalizedSaturationDetectorPlugin = "dynamo-saturation-detector"
)

type utilizationDetectorParameters struct {
	QueueDepthThreshold       *int             `json:"queueDepthThreshold,omitempty"`
	KVCacheUtilThreshold      *float64         `json:"kvCacheUtilThreshold,omitempty"`
	MetricsStalenessThreshold *metav1.Duration `json:"metricsStalenessThreshold,omitempty"`
}

// GenerateConfigMap generates a ConfigMap for EPP configuration
// Returns nil if ConfigMapRef is used (user provides their own ConfigMap)
// Returns error if neither ConfigMapRef nor Config is provided
func GenerateConfigMap(
	ctx context.Context,
	dgd *v1beta1.DynamoGraphDeployment,
	componentName string,
	eppConfig *v1beta1.EPPConfig,
) (*corev1.ConfigMap, error) {
	// If user provides ConfigMapRef, they manage the ConfigMap themselves
	if eppConfig != nil && eppConfig.ConfigMapRef != nil {
		return nil, nil
	}

	// User MUST provide either ConfigMapRef or Config (no default)
	if eppConfig == nil || eppConfig.Config == nil {
		return nil, fmt.Errorf("EPP configuration is required: either eppConfig.configMapRef or eppConfig.config must be specified")
	}

	// User provided inline config as Go struct - marshal to YAML
	configYAML, err := marshalEndpointPickerConfig(eppConfig.Config)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal EPP config: %w", err)
	}

	configMapName := GetConfigMapName(dgd.Name)

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      configMapName,
			Namespace: dgd.Namespace,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
				consts.KubeLabelDynamoComponent:           componentName,
				consts.KubeLabelDynamoComponentType:       consts.ComponentTypeEPP,
			},
		},
		Data: map[string]string{
			ConfigKey: configYAML,
		},
	}

	return configMap, nil
}

// GetConfigMapName returns the ConfigMap name for a given DGD
func GetConfigMapName(dgdName string) string {
	return fmt.Sprintf("%s-%s", dgdName, ConfigMapSuffix)
}

// marshalEndpointPickerConfig marshals EndpointPickerConfig to YAML with proper API metadata
func marshalEndpointPickerConfig(config *eppconfig.EndpointPickerConfig) (string, error) {
	normalized, err := normalizeEndpointPickerConfig(config)
	if err != nil {
		return "", err
	}

	yamlBytes, err := yaml.Marshal(normalized)
	if err != nil {
		return "", fmt.Errorf("failed to marshal EndpointPickerConfig to YAML: %w", err)
	}

	return string(yamlBytes), nil
}

// normalizeEndpointPickerConfig returns a canonical GAIE v1.5 configuration
// without mutating the Dynamo API object.
func normalizeEndpointPickerConfig(config *eppconfig.EndpointPickerConfig) (*apixv1alpha1.EndpointPickerConfig, error) {
	normalized := (&apixv1alpha1.EndpointPickerConfig{
		FeatureGates:       config.FeatureGates,
		Plugins:            config.Plugins,
		SchedulingProfiles: config.SchedulingProfiles,
		DataLayer:          config.DataLayer,
		FlowControl:        config.FlowControl,
		Parser:             config.Parser,
	}).DeepCopy()
	normalized.TypeMeta = metav1.TypeMeta{
		APIVersion: apixv1alpha1.GroupVersion.String(),
		Kind:       "EndpointPickerConfig",
	}

	detector := config.SaturationDetector
	if detector == nil {
		return normalized, nil
	}
	usesDeprecatedFields := hasDeprecatedSaturationDetectorConfig(detector)
	if detector.PluginRef != "" && usesDeprecatedFields {
		return nil, fmt.Errorf("saturationDetector.pluginRef is mutually exclusive with the deprecated threshold fields")
	}

	// The current form passes through unchanged, including an empty reference
	// that GAIE defaults to the built-in utilization detector.
	if !usesDeprecatedFields {
		normalized.SaturationDetector = &apixv1alpha1.SaturationDetectorConfig{PluginRef: detector.PluginRef}
		return normalized, nil
	}

	parameters, err := normalizeDeprecatedSaturationDetectorConfig(detector)
	if err != nil {
		return nil, err
	}
	pluginName := availableSaturationDetectorPluginName(normalized.Plugins)
	normalized.Plugins = append(normalized.Plugins, apixv1alpha1.PluginSpec{
		Name:       pluginName,
		Type:       utilizationDetectorPluginType,
		Parameters: parameters,
	})
	normalized.SaturationDetector = &apixv1alpha1.SaturationDetectorConfig{PluginRef: pluginName}

	return normalized, nil
}

func hasDeprecatedSaturationDetectorConfig(config *eppconfig.SaturationDetectorConfig) bool {
	return config.QueueDepthThreshold != 0 ||
		config.KVCacheUtilThreshold != 0 ||
		config.MetricsStalenessThreshold.Duration != 0
}

func normalizeDeprecatedSaturationDetectorConfig(config *eppconfig.SaturationDetectorConfig) (json.RawMessage, error) {
	parameters := utilizationDetectorParameters{}

	// GAIE v1.2 replaced invalid values with defaults. Omitting them preserves
	// that behavior because the v1.5 utilization detector applies the same defaults.
	if config.QueueDepthThreshold > 0 {
		parameters.QueueDepthThreshold = &config.QueueDepthThreshold
	}
	if config.KVCacheUtilThreshold > 0 && config.KVCacheUtilThreshold < 1 {
		parameters.KVCacheUtilThreshold = &config.KVCacheUtilThreshold
	}
	if config.MetricsStalenessThreshold.Duration > 0 {
		parameters.MetricsStalenessThreshold = &config.MetricsStalenessThreshold
	}

	raw, err := json.Marshal(parameters)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal deprecated saturation detector configuration: %w", err)
	}
	return raw, nil
}

func availableSaturationDetectorPluginName(plugins []apixv1alpha1.PluginSpec) string {
	usedNames := make(map[string]struct{}, len(plugins))
	for _, plugin := range plugins {
		name := plugin.Name
		if name == "" {
			name = plugin.Type
		}
		usedNames[name] = struct{}{}
	}
	if _, exists := usedNames[normalizedSaturationDetectorPlugin]; !exists {
		return normalizedSaturationDetectorPlugin
	}

	for suffix := 2; ; suffix++ {
		name := fmt.Sprintf("%s-%d", normalizedSaturationDetectorPlugin, suffix)
		if _, exists := usedNames[name]; !exists {
			return name
		}
	}
}

// GetConfigMapVolumeMount returns the volume and volumeMount for EPP config
func GetConfigMapVolumeMount(dgdName string, eppConfig *v1beta1.EPPConfig) (corev1.Volume, corev1.VolumeMount) {
	configMapName := dgdName + "-" + ConfigMapSuffix
	configKey := ConfigKey

	// If user provides their own ConfigMap, use that
	if eppConfig != nil && eppConfig.ConfigMapRef != nil {
		configMapName = eppConfig.ConfigMapRef.Name
		// Allow user to specify custom key, default to ConfigKey if not specified
		if eppConfig.ConfigMapRef.Key != "" {
			configKey = eppConfig.ConfigMapRef.Key
		}
	}

	volume := corev1.Volume{
		Name: "epp-config",
		VolumeSource: corev1.VolumeSource{
			ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{
					Name: configMapName,
				},
				Items: []corev1.KeyToPath{
					{
						Key:  configKey,
						Path: ConfigKey, // Always mount to the same path regardless of source key
					},
				},
			},
		},
	}

	volumeMount := corev1.VolumeMount{
		Name:      "epp-config",
		MountPath: "/etc/epp",
		ReadOnly:  true,
	}

	return volume, volumeMount
}

// GetConfigFilePath returns the path where EPP config is mounted in the container
// Note: The config is always mounted at this path regardless of the source ConfigMap key
// because GetConfigMapVolumeMount() maps any custom key to ConfigKey in the Path field
func GetConfigFilePath() string {
	return fmt.Sprintf("/etc/epp/%s", ConfigKey)
}
