/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

// Pipeline is the runtime pipeline shape derived while projecting a model build.
type Pipeline string

const (
	PipelineSingle     Pipeline = "single"
	PipelineLPX        Pipeline = "lpx"
	PipelineSpecDecode Pipeline = "specDecode"
)
