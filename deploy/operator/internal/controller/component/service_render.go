// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package component

import (
	"maps"
	"strings"

	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// ServiceParams contains all the fields needed to generate a Kubernetes
// Service for a Dynamo component, independent of whether the caller is the DGD
// (Grove) or DCD controller.
type ServiceParams struct {
	ServiceName     string
	Namespace       string
	ComponentType   string
	DynamoNamespace string
	ComponentName   string // original user-provided name, used in selector
	Labels          map[string]string
	Annotations     map[string]string
	IsK8sDiscovery  bool
}

// RenderService renders a component Service from resolved metadata and discovery settings.
// The input maps are not modified.
func RenderService(params ServiceParams) (*corev1.Service, error) {
	if !ServiceEnabled(params.ComponentType, params.IsK8sDiscovery) {
		return nil, nil
	}

	var servicePort corev1.ServicePort
	switch params.ComponentType {
	case commonconsts.ComponentTypeFrontend:
		servicePort = corev1.ServicePort{
			Name:       commonconsts.DynamoServicePortName,
			Port:       commonconsts.DynamoServicePort,
			TargetPort: intstr.FromString(commonconsts.DynamoContainerPortName),
			Protocol:   corev1.ProtocolTCP,
		}
	case commonconsts.ComponentTypeEPP:
		servicePort = corev1.ServicePort{
			Name:        commonconsts.EPPGRPCPortName,
			Port:        commonconsts.EPPGRPCPort,
			TargetPort:  intstr.FromInt(commonconsts.EPPGRPCPort),
			Protocol:    corev1.ProtocolTCP,
			AppProtocol: new("http2"),
		}
	default:
		servicePort = corev1.ServicePort{
			Name:       commonconsts.DynamoSystemPortName,
			Port:       commonconsts.DynamoSystemPort,
			TargetPort: intstr.FromString(commonconsts.DynamoSystemPortName),
			Protocol:   corev1.ProtocolTCP,
		}
	}

	labels := make(map[string]string)
	maps.Copy(labels, params.Labels)
	if params.IsK8sDiscovery {
		labels[commonconsts.KubeLabelDynamoDiscoveryBackend] = commonconsts.DiscoveryBackendKubernetes
		labels[commonconsts.KubeLabelDynamoDiscoveryEnabled] = commonconsts.KubeLabelValueTrue
	}

	selector := map[string]string{
		commonconsts.KubeLabelDynamoComponentType: params.ComponentType,
		commonconsts.KubeLabelDynamoNamespace:     params.DynamoNamespace,
		commonconsts.KubeLabelDynamoComponent:     params.ComponentName,
	}

	annotations := make(map[string]string)
	maps.Copy(annotations, params.Annotations)

	service := &corev1.Service{
		// Service names must be DNS-1035 labels (no dots). Replace dots with
		// hyphens so model names like "Qwen3-0.6B" don't cause rejections.
		Name:        strings.ToLower(strings.ReplaceAll(params.ServiceName, ".", "-")),
		Namespace:   params.Namespace,
		Labels:      labels,
		Annotations: annotations,
		Spec: corev1.ServiceSpec{
			Selector: selector,
			Ports:    []corev1.ServicePort{servicePort},
		},
	}
	return service, nil
}

// ServiceEnabled reports whether this component exposes an ordinary discovery Service.
func ServiceEnabled(componentType string, kubernetesDiscovery bool) bool {
	return kubernetesDiscovery || componentType == commonconsts.ComponentTypeFrontend
}
