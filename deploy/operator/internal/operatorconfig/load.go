/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package operatorconfig

import (
	"fmt"
	"os"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	configvalidation "github.com/ai-dynamo/dynamo/deploy/operator/api/config/validation"
	k8sruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

// Load reads, defaults, and validates an operator configuration file.
func Load(path string) (*configv1alpha1.OperatorConfiguration, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", path, err)
	}

	scheme := k8sruntime.NewScheme()
	if err := configv1alpha1.AddToScheme(scheme); err != nil {
		return nil, fmt.Errorf("failed to initialize operator config scheme: %w", err)
	}

	codecFactory := serializer.NewCodecFactory(scheme)
	config := &configv1alpha1.OperatorConfiguration{}
	if err := k8sruntime.DecodeInto(codecFactory.UniversalDecoder(), data, config); err != nil {
		return nil, fmt.Errorf("failed to decode config file %s: %w", path, err)
	}

	if errs := configvalidation.ValidateOperatorConfiguration(config); len(errs) > 0 {
		return nil, fmt.Errorf("config validation failed: %s", errs.ToAggregate().Error())
	}

	return config, nil
}
