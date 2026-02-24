/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"context"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
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
)

// GenerateConfigMap generates a ConfigMap for EPP configuration
// Returns nil if ConfigMapRef is used (user provides their own ConfigMap)
// Returns error if neither ConfigMapRef nor Config is provided
func GenerateConfigMap(
	ctx context.Context,
	dgd *v1alpha1.DynamoGraphDeployment,
	componentName string,
	eppConfig *v1alpha1.EPPConfig,
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
func marshalEndpointPickerConfig(config *apixv1alpha1.EndpointPickerConfig) (string, error) {
	// Set the TypeMeta fields using upstream constants
	config.TypeMeta = metav1.TypeMeta{
		APIVersion: apixv1alpha1.SchemeGroupVersion.String(),
		Kind:       "EndpointPickerConfig",
	}

	yamlBytes, err := yaml.Marshal(config)
	if err != nil {
		return "", fmt.Errorf("failed to marshal EndpointPickerConfig to YAML: %w", err)
	}

	return string(yamlBytes), nil
}

// GetConfigMapVolumeMount returns the volume and volumeMount for EPP config
func GetConfigMapVolumeMount(dgdName string, eppConfig *v1alpha1.EPPConfig) (corev1.Volume, corev1.VolumeMount) {
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
