/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controller_common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/runtimeversion"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

// ComponentDefaults interface defines how defaults should be provided
type ComponentDefaults interface {
	// GetBaseContainer returns the base container configuration for this component type
	// The numberOfNodes parameter indicates the total number of nodes in the deployment
	GetBaseContainer(context ComponentContext) (corev1.Container, error)

	// GetBasePodSpec returns the base pod spec configuration for this component type
	// The numberOfNodes parameter indicates the total number of nodes in the deployment
	GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error)
}

// ComponentDefaultsFactory creates appropriate defaults based on component type and number of nodes
func ComponentDefaultsFactory(componentType string) ComponentDefaults {
	switch componentType {
	case commonconsts.ComponentTypeFrontend:
		return NewFrontendDefaults()
	case commonconsts.ComponentTypeWorker, commonconsts.ComponentTypePrefill, commonconsts.ComponentTypeDecode:
		return NewWorkerDefaults()
	case commonconsts.ComponentTypePlanner:
		return NewPlannerDefaults()
	case commonconsts.ComponentTypeEPP:
		return NewEPPDefaults()
	default:
		return &BaseComponentDefaults{}
	}
}

// BaseComponentDefaults provides common defaults shared by all components
type BaseComponentDefaults struct{}

// DiscoveryContext holds resolved discovery settings for a component.
type DiscoveryContext struct {
	Backend configv1alpha1.DiscoveryBackend
	Mode    configv1alpha1.KubeDiscoveryMode
}

// NewDiscoveryContext resolves discovery settings from operator config and component annotations.
func NewDiscoveryContext(defaultBackend configv1alpha1.DiscoveryBackend, annotations map[string]string) DiscoveryContext {
	return DiscoveryContext{
		Backend: controller_common.GetDiscoveryBackend(defaultBackend, annotations),
		Mode:    controller_common.GetKubeDiscoveryMode(annotations),
	}
}

type ComponentContext struct {
	numberOfNodes                  int32
	RuntimeContainerName           string // Resolved name of the container hosting this component's Dynamo runtime.
	DynamoNamespace                string
	ComponentType                  string
	ParentGraphDeploymentName      string
	ParentGraphDeploymentNamespace string
	Discovery                      DiscoveryContext
	Infrastructure                 configv1alpha1.InfrastructureConfiguration
	EPPConfig                      *v1beta1.EPPConfig
	WorkerHashSuffix               string
	RuntimeVersion                 *runtimeversion.Version
}

func (b *BaseComponentDefaults) GetBaseContainer(context ComponentContext) (corev1.Container, error) {
	return b.getCommonContainer(context), nil
}

func (b *BaseComponentDefaults) GetBasePodSpec(context ComponentContext) (corev1.PodSpec, error) {
	return b.getCommonPodSpec(), nil
}

func (b *BaseComponentDefaults) getCommonPodSpec() corev1.PodSpec {
	return corev1.PodSpec{
		TerminationGracePeriodSeconds: ptr.To(int64(60)),
		RestartPolicy:                 corev1.RestartPolicyAlways,
	}
}

func (b *BaseComponentDefaults) getCommonContainer(context ComponentContext) corev1.Container {
	// Use the resolved runtime identity for the container and discovery metadata.
	container := corev1.Container{
		Name: context.RuntimeContainerName,
		Command: []string{
			"/bin/sh",
			"-c",
		},
	}

	// Every Dynamo component receives infrastructure and transport defaults.
	AddStandardEnvVars(&container, context.Infrastructure)
	AddTransportTLSEnvVars(&container, context.Infrastructure)

	// Runtime identity is independent of the infrastructure configuration.
	container.Env = append(container.Env, []corev1.EnvVar{
		{
			Name:  commonconsts.DynamoNamespaceEnvVar,
			Value: context.DynamoNamespace,
		},
		{
			Name:  commonconsts.DynamoComponentEnvVar,
			Value: context.ComponentType,
		},
		{
			Name:  "DYN_PARENT_DGD_K8S_NAME",
			Value: context.ParentGraphDeploymentName,
		},
		{
			Name:  "DYN_PARENT_DGD_K8S_NAMESPACE",
			Value: context.ParentGraphDeploymentNamespace,
		},
		{
			Name: "POD_NAME",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.name",
				},
			},
		},
		{
			Name: "POD_NAMESPACE",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.namespace",
				},
			},
		},
		{
			Name: "POD_UID",
			ValueFrom: &corev1.EnvVarSource{
				FieldRef: &corev1.ObjectFieldSelector{
					FieldPath: "metadata.uid",
				},
			},
		},
	}...)

	// Set discovery backend env var to "kubernetes" unless explicitly set to "etcd"
	if context.Discovery.Backend != "etcd" {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  commonconsts.DynamoDiscoveryBackendEnvVar,
			Value: "kubernetes",
		})
	}

	if context.Discovery.Mode == configv1alpha1.KubeDiscoveryModeContainer {
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "CONTAINER_NAME",
			Value: context.RuntimeContainerName,
		})
		container.Env = append(container.Env, corev1.EnvVar{
			Name:  "DYN_KUBE_DISCOVERY_MODE",
			Value: string(configv1alpha1.KubeDiscoveryModeContainer),
		})
	}

	return container
}

// AddStandardEnvVars adds the standard environment variables that are common to
// Dynamo component containers and the DGDR profiler Job.
// container must not be nil; existing environment values take precedence.
func AddStandardEnvVars(container *corev1.Container, infrastructure configv1alpha1.InfrastructureConfiguration) {
	standardEnvVars := []corev1.EnvVar{}
	if infrastructure.NATSAddress != "" {
		standardEnvVars = append(standardEnvVars, corev1.EnvVar{
			Name:  "NATS_SERVER",
			Value: infrastructure.NATSAddress,
		})
	}

	if infrastructure.ETCDAddress != "" {
		standardEnvVars = append(standardEnvVars, corev1.EnvVar{
			Name:  "ETCD_ENDPOINTS",
			Value: infrastructure.ETCDAddress,
		})
	}

	if infrastructure.ModelExpressURL != "" {
		standardEnvVars = append(standardEnvVars, corev1.EnvVar{
			Name:  "MODEL_EXPRESS_URL",
			Value: infrastructure.ModelExpressURL,
		})
	}
	if infrastructure.PrometheusEndpoint != "" {
		standardEnvVars = append(standardEnvVars, corev1.EnvVar{
			Name:  "PROMETHEUS_ENDPOINT",
			Value: infrastructure.PrometheusEndpoint,
		})
	}
	// merge the env vars to allow users to override the standard env vars
	container.Env = MergeEnvs(standardEnvVars, container.Env)
}

// AddTransportTLSEnvVars injects DYN_TCP_TLS_* and NATS_TLS_* certificate path
// environment variables from InfrastructureConfiguration. Unlike
// AddStandardEnvVars, this is scoped to DGD workload pods only — not the
// DGDR profiler Job — because the profiler does not run the TCP/NATS
// transport and does not inherit DGD podTemplate certificate mounts.
// container must not be nil; existing environment values take precedence.
func AddTransportTLSEnvVars(container *corev1.Container, infrastructure configv1alpha1.InfrastructureConfiguration) {
	tlsEnvVars := []corev1.EnvVar{}
	// Inject TLS certificate paths for inter-component encryption (DYN_TCP_TLS_* / NATS_TLS_*).
	if infrastructure.NATSTLSCAPath != "" {
		tlsEnvVars = append(tlsEnvVars, corev1.EnvVar{
			Name:  "NATS_TLS_CA_CERT_PATH",
			Value: infrastructure.NATSTLSCAPath,
		})
	}
	if infrastructure.NATSTLSClientCertPath != "" {
		tlsEnvVars = append(tlsEnvVars, corev1.EnvVar{
			Name:  "NATS_TLS_CLIENT_CERT_PATH",
			Value: infrastructure.NATSTLSClientCertPath,
		})
	}
	if infrastructure.NATSTLSClientKeyPath != "" {
		tlsEnvVars = append(tlsEnvVars, corev1.EnvVar{
			Name:  "NATS_TLS_CLIENT_KEY_PATH",
			Value: infrastructure.NATSTLSClientKeyPath,
		})
	}
	if infrastructure.TCPTLSCertPath != "" {
		tlsEnvVars = append(tlsEnvVars, corev1.EnvVar{
			Name:  "DYN_TCP_TLS_CERT_PATH",
			Value: infrastructure.TCPTLSCertPath,
		})
	}
	if infrastructure.TCPTLSKeyPath != "" {
		tlsEnvVars = append(tlsEnvVars, corev1.EnvVar{
			Name:  "DYN_TCP_TLS_KEY_PATH",
			Value: infrastructure.TCPTLSKeyPath,
		})
	}
	if infrastructure.TCPTLSCAPath != "" {
		tlsEnvVars = append(tlsEnvVars, corev1.EnvVar{
			Name:  "DYN_TCP_TLS_CA_CERT_PATH",
			Value: infrastructure.TCPTLSCAPath,
		})
	}
	if infrastructure.TCPTLSClientCertPath != "" {
		tlsEnvVars = append(tlsEnvVars, corev1.EnvVar{
			Name:  "DYN_TCP_TLS_CLIENT_CERT_PATH",
			Value: infrastructure.TCPTLSClientCertPath,
		})
	}
	if infrastructure.TCPTLSClientKeyPath != "" {
		tlsEnvVars = append(tlsEnvVars, corev1.EnvVar{
			Name:  "DYN_TCP_TLS_CLIENT_KEY_PATH",
			Value: infrastructure.TCPTLSClientKeyPath,
		})
	}
	if infrastructure.TCPTLSClientCAPath != "" {
		tlsEnvVars = append(tlsEnvVars, corev1.EnvVar{
			Name:  "DYN_TCP_TLS_CLIENT_CA_CERT_PATH",
			Value: infrastructure.TCPTLSClientCAPath,
		})
	}
	if infrastructure.TCPTLSServerName != "" {
		tlsEnvVars = append(tlsEnvVars, corev1.EnvVar{
			Name:  "DYN_TCP_TLS_SERVER_NAME",
			Value: infrastructure.TCPTLSServerName,
		})
	}
	container.Env = MergeEnvs(tlsEnvVars, container.Env)
}
