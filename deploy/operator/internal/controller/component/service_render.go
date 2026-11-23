// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package component

import (
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/shared"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

// RenderService renders a component Service from resolved metadata and discovery settings.
// The input maps are not modified.
func RenderService(params shared.ServiceParams) (*corev1.Service, error) {
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

	service := shared.RenderService(params)
	service.Spec.Ports = []corev1.ServicePort{servicePort}
	return service, nil
}

// ServiceEnabled reports whether this component exposes an ordinary discovery Service.
func ServiceEnabled(componentType string, kubernetesDiscovery bool) bool {
	return kubernetesDiscovery || componentType == commonconsts.ComponentTypeFrontend
}
