/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package gms

import (
	"fmt"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

// OverlayClients applies service GMS client config to checkpoint GMS metadata.
func OverlayClients(checkpointGMS **nvidiacomv1alpha1.GPUMemoryServiceSpec, checkpointName string, checkpointExists bool, serviceGMS *nvidiacomv1beta1.GPUMemoryServiceSpec) error {
	if checkpointGMS == nil || serviceGMS == nil {
		return nil
	}
	converted := ToAlphaSpec(serviceGMS)
	if converted == nil || !converted.Enabled {
		return nil
	}
	if checkpointExists && (*checkpointGMS == nil || !(*checkpointGMS).Enabled) {
		return fmt.Errorf("gpuMemoryService restore requires resolved checkpoint %q to enable gpuMemoryService", checkpointName)
	}
	if *checkpointGMS == nil || !(*checkpointGMS).Enabled {
		return nil
	}
	*checkpointGMS = converted.DeepCopy()
	return nil
}

// ToAlphaSpec converts the v1beta1 GMS config into the v1alpha1 compatibility shape.
func ToAlphaSpec(src *nvidiacomv1beta1.GPUMemoryServiceSpec) *nvidiacomv1alpha1.GPUMemoryServiceSpec {
	if src == nil {
		return nil
	}
	dst := &nvidiacomv1alpha1.GPUMemoryServiceSpec{}
	nvidiacomv1alpha1.ConvertToGPUMemoryServiceSpec(src, dst)
	return dst
}
