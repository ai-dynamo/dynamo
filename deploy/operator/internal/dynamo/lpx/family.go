/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	corev1 "k8s.io/api/core/v1"
)

// family is one LPU target family's scheduler and runtime contract. The manifest
// decoder selects xtFamily or hxFamily; every other family-specific decision reads
// this table.
type family struct {
	// target is the scheduler's wire value, also bound into workload digests.
	target lpxv1alpha1.TargetFamily
	// projectionVersion versions the digest fields written by project.
	projectionVersion string
	// project derives the component shared by every model and writes its
	// model-independent digest fields.
	project func(source *Build, pipeline Pipeline, fields digestTranscript) (*component, error)
	// lpuOnlyMode and hybridMode are the scheduler workload modes of each runtime shape.
	lpuOnlyMode, hybridMode lpxv1alpha1.WorkloadMode
	// lpuResource is the extended resource each Agent requests per device.
	lpuResource corev1.ResourceName
	// configOverrides lets an authored "config" volume replace the generated ConfigMap.
	configOverrides bool
	// cyborgServerConfig renders the lpu_servers ConfigMap read by hybrid Cyborg workers.
	cyborgServerConfig bool
	// singleModelColumns keeps the model-identity partition columns for Single pipelines.
	singleModelColumns bool
}

var (
	xtFamily = &family{
		target:             lpxv1alpha1.TargetFamilyXt8888,
		projectionVersion:  "v2-xt-node-local/v1",
		project:            projectV2Component,
		lpuOnlyMode:        lpxv1alpha1.WorkloadModeV2LPUOnly,
		hybridMode:         lpxv1alpha1.WorkloadModeV2StrictHybrid,
		lpuResource:        "lpu.nvidia.com/lpu",
		configOverrides:    true,
		cyborgServerConfig: true,
	}
	hxFamily = &family{
		target:             lpxv1alpha1.TargetFamilyHx16x8x2x3,
		projectionVersion:  "v3-hx-capnp/v1",
		project:            projectV3Component,
		lpuOnlyMode:        lpxv1alpha1.WorkloadModeV3HxLPUOnly,
		hybridMode:         lpxv1alpha1.WorkloadModeV3HxStrictHybrid,
		lpuResource:        "nvidia.com/lpu",
		singleModelColumns: true,
	}
)

// workloadMode returns the scheduler workload mode for pipeline.
func (f *family) workloadMode(pipeline Pipeline) lpxv1alpha1.WorkloadMode {
	if pipeline == PipelineHybrid {
		return f.hybridMode
	}
	return f.lpuOnlyMode
}
