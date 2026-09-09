// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package consts

const (
	KubeAnnotationLPXSchedulerBackend = "scheduling.lpu.nvidia.com/scheduler-backend"
	KubeAnnotationLPXExecutionBackend = "scheduling.lpu.nvidia.com/execution-backend"
	LPXSchedulerBackend               = "lpx-scheduler"
	ComponentTypeLPX                  = "lpx"
	ModelStorageVolumeName            = "model-storage"

	// AnnotationExtraResourcesHash stores the generated LPX configuration hash.
	AnnotationExtraResourcesHash = "nvidia.com/extra-resources-hash"
)
