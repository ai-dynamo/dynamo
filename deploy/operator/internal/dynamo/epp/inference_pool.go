/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	gaiev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

const (
	// InferencePoolGroup is the API group for InferencePool (stable API)
	// Using the stable v1 API group instead of the experimental x-k8s.io group
	InferencePoolGroup = gaiev1.GroupName
)

// InferencePoolGVK is the GroupVersionKind for InferencePool (stable API)
var InferencePoolGVK = schema.GroupVersionKind{
	Group:   gaiev1.GroupName,
	Version: gaiev1.GroupVersion.Version,
	Kind:    "InferencePool",
}

// GenerateInferencePool generates an InferencePool resource for EPP
// This solves the chicken-and-egg problem: EPP needs the pool name, pool needs EPP service
// Using the stable inference.networking.k8s.io/v1 API (per PR #5592)
func GenerateInferencePool(
	dgd *v1alpha1.DynamoGraphDeployment,
	serviceName string,
	eppConfig *v1alpha1.EPPConfig,
) (*gaiev1.InferencePool, error) {
	poolName := GetPoolName(dgd.Name, eppConfig)
	poolNamespace := GetPoolNamespace(dgd.Namespace, eppConfig)
	dynamoNamespace := dgd.GetDynamoNamespaceForService(dgd.Spec.Services[serviceName])
	eppServiceName := GetServiceName(dgd.Name)

	// Build InferencePool using typed API
	pool := &gaiev1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{
			Name:      poolName,
			Namespace: poolNamespace,
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
				consts.KubeLabelDynamoComponent:           serviceName,
				consts.KubeLabelDynamoComponentType:       consts.ComponentTypeEPP,
			},
		},
		Spec: gaiev1.InferencePoolSpec{
			TargetPorts: []gaiev1.Port{
				{Number: 8000}, // Frontend port
			},
			Selector: gaiev1.LabelSelector{
				MatchLabels: map[gaiev1.LabelKey]gaiev1.LabelValue{
					consts.KubeLabelDynamoComponent: "Frontend",
					consts.KubeLabelDynamoNamespace: gaiev1.LabelValue(dynamoNamespace),
				},
			},
			EndpointPickerRef: gaiev1.EndpointPickerRef{
				Kind: "Service",
				Name: gaiev1.ObjectName(eppServiceName),
				Port: &gaiev1.Port{
					Number: 9002,
				},
			},
		},
	}

	return pool, nil
}

// GetPoolName returns the InferencePool name
// Can be overridden via EPPConfig.PoolName, otherwise defaults to {dgdName}-pool
func GetPoolName(dgdName string, eppConfig *v1alpha1.EPPConfig) string {
	if eppConfig != nil && eppConfig.PoolName != nil && *eppConfig.PoolName != "" {
		return *eppConfig.PoolName
	}
	return fmt.Sprintf("%s-pool", dgdName)
}

// GetPoolNamespace returns the InferencePool namespace
// Can be overridden via EPPConfig.PoolNamespace, otherwise defaults to dgdNamespace
func GetPoolNamespace(dgdNamespace string, eppConfig *v1alpha1.EPPConfig) string {
	if eppConfig != nil && eppConfig.PoolNamespace != nil && *eppConfig.PoolNamespace != "" {
		return *eppConfig.PoolNamespace
	}
	return dgdNamespace
}
