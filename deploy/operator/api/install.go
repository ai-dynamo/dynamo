/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package api contains the Dynamo Kubernetes APIs.
package api

import (
	"k8s.io/apimachinery/pkg/runtime"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// Install registers all Dynamo API versions in scheme.
func Install(scheme *runtime.Scheme) error {
	if err := v1alpha1.AddToScheme(scheme); err != nil {
		return err
	}
	return v1beta1.AddToScheme(scheme)
}
