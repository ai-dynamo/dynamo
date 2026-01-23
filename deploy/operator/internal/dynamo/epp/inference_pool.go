/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package epp

import (
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const (
	// InferencePoolGroup is the API group for InferencePool (stable API)
	InferencePoolGroup = "inference.networking.k8s.io"
)

// InferencePoolGVK is the GroupVersionKind for InferencePool (stable API)
var InferencePoolGVK = schema.GroupVersionKind{
	Group:   InferencePoolGroup,
	Version: "v1",
	Kind:    "InferencePool",
}

// GenerateInferencePool generates an InferencePool resource for EPP
// This solves the chicken-and-egg problem: EPP needs the pool name, pool needs EPP service
// Using the stable inference.networking.k8s.io/v1 API (per PR #5592)
func GenerateInferencePool(
	dgd *v1alpha1.DynamoGraphDeployment,
	serviceName string,
	eppConfig *v1alpha1.EPPConfig,
) (*unstructured.Unstructured, error) {
	poolName := GetPoolName(dgd.Name, eppConfig)
	poolNamespace := GetPoolNamespace(dgd.Namespace, eppConfig)
	dynamoNamespace := dgd.GetDynamoNamespaceForService(dgd.Spec.Services[serviceName])
	eppServiceName := GetServiceName(dgd.Name)

	// Build InferencePool using stable v1 API
	pool := &unstructured.Unstructured{}
	pool.SetGroupVersionKind(InferencePoolGVK)
	pool.SetName(poolName)
	pool.SetNamespace(poolNamespace)
	pool.SetLabels(map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
		consts.KubeLabelDynamoComponent:           serviceName,
		consts.KubeLabelDynamoComponentType:       consts.ComponentTypeEPP,
	})

	// Set spec fields
	spec := map[string]interface{}{
		"targetPorts": []map[string]interface{}{
			{
				"number": int64(8000), // Frontend port
			},
		},
		"selector": map[string]interface{}{
			"matchLabels": map[string]string{
				consts.KubeLabelDynamoComponent: "Frontend",
				consts.KubeLabelDynamoNamespace: dynamoNamespace,
			},
		},
		"endpointPickerRef": map[string]interface{}{
			"group": "",
			"kind":  "Service",
			"name":  eppServiceName,
			"port": map[string]interface{}{
				"number": int64(9002),
			},
		},
	}

	if err := unstructured.SetNestedMap(pool.Object, spec, "spec"); err != nil {
		return nil, fmt.Errorf("failed to set InferencePool spec: %w", err)
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
