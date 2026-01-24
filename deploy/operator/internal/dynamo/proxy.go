/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dynamo

import (
	"fmt"
	"maps"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
)

// ProxyConfig holds configuration for generating HAProxy resources
type ProxyConfig struct {
	// DGDName is the name of the parent DynamoGraphDeployment
	DGDName string
	// Namespace is the Kubernetes namespace
	Namespace string
	// Replicas is the number of proxy pod replicas
	Replicas int32
	// Image is the HAProxy container image (e.g., "haproxy:2.9-alpine")
	Image string
	// Resources are the resource requirements for the proxy container
	Resources corev1.ResourceRequirements
	// Tolerations for the proxy pods
	Tolerations []corev1.Toleration
	// Affinity rules for the proxy pods
	Affinity *corev1.Affinity
	// OldBackend is the configuration for the old (current) frontend backend
	OldBackend *BackendConfig
	// NewBackend is the configuration for the new (target) frontend backend
	NewBackend *BackendConfig
	// Labels are additional labels to apply to all resources
	Labels map[string]string
}

// BackendConfig holds configuration for an HAProxy backend
type BackendConfig struct {
	// ServiceName is the Kubernetes Service name for this backend
	ServiceName string
	// ServicePort is the port to connect to
	ServicePort int32
	// Weight is the traffic weight (0-100)
	Weight int32
}

// GetProxyName returns the name for proxy resources
func GetProxyName(dgdName string) string {
	return fmt.Sprintf("%s-traffic-proxy", dgdName)
}

// GetProxyLabels returns the labels for proxy resources
func GetProxyLabels(dgdName string, extraLabels map[string]string) map[string]string {
	labels := map[string]string{
		commonconsts.KubeLabelTrafficProxy:              commonconsts.KubeLabelValueTrue,
		commonconsts.KubeLabelTrafficProxyComponent:     commonconsts.TrafficProxyComponentProxy,
		commonconsts.KubeLabelDynamoGraphDeploymentName: dgdName,
	}
	maps.Copy(labels, extraLabels)
	return labels
}

// GenerateProxyDeployment creates the HAProxy Deployment
func GenerateProxyDeployment(config *ProxyConfig) *appsv1.Deployment {
	proxyName := GetProxyName(config.DGDName)
	labels := GetProxyLabels(config.DGDName, config.Labels)

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      proxyName,
			Namespace: config.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To(config.Replicas),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					commonconsts.KubeLabelTrafficProxy:          commonconsts.KubeLabelValueTrue,
					commonconsts.KubeLabelDynamoGraphDeploymentName: config.DGDName,
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:      commonconsts.HAProxyContainerName,
							Image:     config.Image,
							Resources: config.Resources,
							Ports: []corev1.ContainerPort{
								{
									Name:          commonconsts.HAProxyHTTPPortName,
									ContainerPort: commonconsts.HAProxyHTTPPort,
									Protocol:      corev1.ProtocolTCP,
								},
								{
									Name:          commonconsts.HAProxyStatsPortName,
									ContainerPort: commonconsts.HAProxyStatsPort,
									Protocol:      corev1.ProtocolTCP,
								},
								{
									Name:          commonconsts.HAProxyRuntimePortName,
									ContainerPort: commonconsts.HAProxyRuntimePort,
									Protocol:      corev1.ProtocolTCP,
								},
								{
									Name:          commonconsts.HAProxyMetricsPortName,
									ContainerPort: commonconsts.HAProxyMetricsPort,
									Protocol:      corev1.ProtocolTCP,
								},
							},
							VolumeMounts: []corev1.VolumeMount{
								{
									Name:      "haproxy-config",
									MountPath: commonconsts.HAProxyConfigMountPath,
									ReadOnly:  true,
								},
								{
									Name:      "haproxy-socket",
									MountPath: commonconsts.HAProxySocketMountPath,
								},
							},
							// Probes use /stats - if HAProxy can serve stats, the process is healthy.
							// Backend health is checked by HAProxy's httpchk; unhealthy backends return 503.
							LivenessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/stats",
										Port: intstr.FromInt32(commonconsts.HAProxyStatsPort),
									},
								},
								InitialDelaySeconds: 5,
								PeriodSeconds:       10,
								TimeoutSeconds:      5,
								FailureThreshold:    3,
							},
							ReadinessProbe: &corev1.Probe{
								ProbeHandler: corev1.ProbeHandler{
									HTTPGet: &corev1.HTTPGetAction{
										Path: "/stats",
										Port: intstr.FromInt32(commonconsts.HAProxyStatsPort),
									},
								},
								InitialDelaySeconds: 2,
								PeriodSeconds:       5,
								TimeoutSeconds:      3,
								FailureThreshold:    3,
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "haproxy-config",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{
										Name: proxyName,
									},
								},
							},
						},
						{
							Name: "haproxy-socket",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{},
							},
						},
					},
					Tolerations: config.Tolerations,
					Affinity:    config.Affinity,
				},
			},
		},
	}

	return deployment
}

// GenerateProxyService creates the Service fronting the HAProxy
func GenerateProxyService(config *ProxyConfig) *corev1.Service {
	proxyName := GetProxyName(config.DGDName)
	labels := GetProxyLabels(config.DGDName, config.Labels)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      proxyName,
			Namespace: config.Namespace,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				commonconsts.KubeLabelTrafficProxy:          commonconsts.KubeLabelValueTrue,
				commonconsts.KubeLabelDynamoGraphDeploymentName: config.DGDName,
			},
			Ports: []corev1.ServicePort{
				{
					Name:       commonconsts.HAProxyHTTPPortName,
					Port:       commonconsts.HAProxyHTTPPort,
					TargetPort: intstr.FromInt32(commonconsts.HAProxyHTTPPort),
					Protocol:   corev1.ProtocolTCP,
				},
				{
					Name:       commonconsts.HAProxyStatsPortName,
					Port:       commonconsts.HAProxyStatsPort,
					TargetPort: intstr.FromInt32(commonconsts.HAProxyStatsPort),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}

	return service
}

// GenerateProxyRuntimeService creates a separate Service for the HAProxy runtime API
// This is used by the controller to update weights without going through the main service
func GenerateProxyRuntimeService(config *ProxyConfig) *corev1.Service {
	proxyName := GetProxyName(config.DGDName)
	runtimeServiceName := fmt.Sprintf("%s-runtime", proxyName)
	labels := GetProxyLabels(config.DGDName, config.Labels)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      runtimeServiceName,
			Namespace: config.Namespace,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{
				commonconsts.KubeLabelTrafficProxy:          commonconsts.KubeLabelValueTrue,
				commonconsts.KubeLabelDynamoGraphDeploymentName: config.DGDName,
			},
			Ports: []corev1.ServicePort{
				{
					Name:       commonconsts.HAProxyRuntimePortName,
					Port:       commonconsts.HAProxyRuntimePort,
					TargetPort: intstr.FromInt32(commonconsts.HAProxyRuntimePort),
					Protocol:   corev1.ProtocolTCP,
				},
			},
		},
	}

	return service
}

// GenerateProxyConfigMap creates the HAProxy configuration ConfigMap
func GenerateProxyConfigMap(config *ProxyConfig) *corev1.ConfigMap {
	proxyName := GetProxyName(config.DGDName)
	labels := GetProxyLabels(config.DGDName, config.Labels)

	haproxyConfig := GenerateHAProxyConfig(config.OldBackend, config.NewBackend)

	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      proxyName,
			Namespace: config.Namespace,
			Labels:    labels,
		},
		Data: map[string]string{
			commonconsts.HAProxyConfigMapKey: haproxyConfig,
		},
	}

	return configMap
}

// GenerateHAProxyConfig generates the haproxy.cfg content
func GenerateHAProxyConfig(oldBackend, newBackend *BackendConfig) string {
	// Default weights if not specified
	oldWeight := int32(100)
	newWeight := int32(0)

	oldServer := ""
	newServer := ""

	if oldBackend != nil {
		oldWeight = oldBackend.Weight
		oldServer = fmt.Sprintf("    server old_frontend %s:%d weight %d check\n",
			oldBackend.ServiceName, oldBackend.ServicePort, oldWeight)
	}

	if newBackend != nil {
		newWeight = newBackend.Weight
		newServer = fmt.Sprintf("    server new_frontend %s:%d weight %d check\n",
			newBackend.ServiceName, newBackend.ServicePort, newWeight)
	}

	// If only old backend exists (normal operation), set weight to 100
	if oldBackend != nil && newBackend == nil {
		oldServer = fmt.Sprintf("    server old_frontend %s:%d weight 100 check\n",
			oldBackend.ServiceName, oldBackend.ServicePort)
	}

	config := fmt.Sprintf(`# HAProxy configuration for Dynamo rolling updates
# Auto-generated by Dynamo operator - do not edit manually

global
    stats socket %s mode 660 level admin expose-fd listeners
    stats timeout 30s
    log stdout format raw local0 info

defaults
    mode http
    log global
    option httplog
    option dontlognull
    timeout connect 5s
    timeout client 300s
    timeout server 300s
    timeout http-keep-alive 300s
    retries 3
    option redispatch

frontend http_front
    bind *:%d
    default_backend frontends

backend frontends
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
%s%s
listen stats
    bind *:%d
    stats enable
    stats uri /stats
    stats refresh 10s
    stats show-legends
    stats show-node

# Runtime API for dynamic updates and metrics
# Connect via: echo "show servers state" | nc <host> %d
listen runtime_api
    bind *:%d
    mode tcp
    server runtime_sock unix@%s send-proxy-v2

# Metrics endpoint
listen metrics
    bind *:%d
    http-request use-service prometheus-exporter if { path /metrics }
`,
		commonconsts.HAProxySocketPath,
		commonconsts.HAProxyHTTPPort,
		oldServer,
		newServer,
		commonconsts.HAProxyStatsPort,
		commonconsts.HAProxyRuntimePort,
		commonconsts.HAProxyRuntimePort,
		commonconsts.HAProxySocketPath,
		commonconsts.HAProxyMetricsPort,
	)

	return config
}

// GenerateSingleBackendHAProxyConfig generates haproxy.cfg for normal operation (single backend)
func GenerateSingleBackendHAProxyConfig(backendServiceName string, backendPort int32) string {
	backend := &BackendConfig{
		ServiceName: backendServiceName,
		ServicePort: backendPort,
		Weight:      100,
	}
	return GenerateHAProxyConfig(backend, nil)
}
