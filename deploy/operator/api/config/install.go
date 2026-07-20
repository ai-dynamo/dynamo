/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package config contains the Dynamo operator configuration APIs.
package config

import (
	"k8s.io/apimachinery/pkg/runtime"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
)

// Install registers all operator configuration API versions in scheme.
func Install(scheme *runtime.Scheme) error {
	return v1alpha1.AddToScheme(scheme)
}
