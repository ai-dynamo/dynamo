/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
)

// GenerateService generates a Kubernetes Service for EPP component
// This service is referenced by the InferencePool and exposes the EPP gRPC endpoints
func GenerateService(
	dgd *v1alpha1.DynamoGraphDeployment,
	serviceName string,
) *corev1.Service {
	eppServiceName := GetServiceName(dgd.Name)

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      eppServiceName,
			Namespace: dgd.Namespace,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
				consts.KubeLabelDynamoComponent:           serviceName,
				consts.KubeLabelDynamoComponentType:       consts.ComponentTypeEPP,
				"app":                                     fmt.Sprintf("%s-epp", dgd.Name), // Match pod selector
			},
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeClusterIP,
			Selector: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
				consts.KubeLabelDynamoComponentType:       consts.ComponentTypeEPP,
			},
			Ports: []corev1.ServicePort{
				{
					Name:        "grpc",
					Protocol:    corev1.ProtocolTCP,
					Port:        9002,
					TargetPort:  intstr.FromInt(9002),
					AppProtocol: ptr.To("http2"),
				},
				{
					Name:       "grpc-health",
					Protocol:   corev1.ProtocolTCP,
					Port:       9003,
					TargetPort: intstr.FromInt(9003),
				},
				{
					Name:       consts.DynamoMetricsPortName,
					Protocol:   corev1.ProtocolTCP,
					Port:       9090,
					TargetPort: intstr.FromInt(9090),
				},
			},
		},
	}

	return service
}

// GetServiceName returns the service name for EPP
func GetServiceName(dgdName string) string {
	return fmt.Sprintf("%s-epp", dgdName)
}
