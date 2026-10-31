/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package defaulting

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	admissionv1 "k8s.io/api/admission/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
)

// defaultMetricV1beta2 is the scalar target an Optimize objective receives when it names none.
const defaultMetricV1beta2 = nvidiacomv1beta2.ObjectiveMetric("ThroughputPerGPU")

// Defaults of the objects admission creates. Admission is their only owner: the API server
// defaults a mutated object in the webhook's version (v1beta1, whose v1beta2 carrier is
// opaque), so the v1beta2 schema carries no default markers for them.
const (
	defaultStaticInputTokensV1beta2  int32 = 1024
	defaultStaticOutputTokensV1beta2 int32 = 128
	defaultStaticConcurrencyV1beta2  int32 = 10
	defaultSLATTFTMsV1beta2                = 2000.0
	defaultSLAITLMsV1beta2                 = 30.0
	defaultMaxCandidatesV1beta2      int32 = 1
)

// This file defaults the v1beta2 shape of a DynamoGraphDeploymentRequest. It is separate
// from the canonical defaulting because the v1beta1 hub carries a v1beta2 Request as an
// opaque spec.v1beta2 envelope: conversion cannot apply field-level defaults to it, so the
// handler decodes the envelope, runs this traversal and encodes the result. Names carry the
// API version because the canonical names belong to v1beta1.
//
// Admission owns every v1beta2 default, and each default has this one owner: backends,
// objective.mode, the scalar metric, objective.sla, recommendation.maxCandidates,
// workload.static and discovered cluster hardware. workload.static: {} opts out of the static
// workload; a nonempty static object receives any missing inputTokens, outputTokens and load
// defaults. objective.sla: {} opts out of the latency defaults. hardware: {} opts out of
// cluster discovery and typed hardware defaults; a nonempty object is completed.
// spec.aisimulate.inputOverrides never changes a default. It overrides the defaulted typed
// values later, when the input is lowered.

// defaultDynamoGraphDeploymentRequestV1beta2 defaults the v1beta2 spec that dgdr carries in
// spec.v1beta2. The environment is read once, before any mutation, and only for a CREATE or
// an UPDATE that changes spec.hardware (see resolveHardware). oldSpec is the stored spec on
// UPDATE and nil otherwise. The envelope is rewritten
// only when defaulting changed the spec, so a defaulted object stays byte-identical across
// repeated admissions.
func defaultDynamoGraphDeploymentRequestV1beta2(
	ctx context.Context,
	dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	operation admissionv1.Operation,
	oldSpec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	options DGDRV1beta2Options,
) error {
	var spec nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal(dgdr.Spec.V1Beta2.Raw, &spec); err != nil {
		return fmt.Errorf("decode spec.v1beta2: %w", err)
	}
	before, err := json.Marshal(&spec)
	if err != nil {
		return fmt.Errorf("encode spec.v1beta2: %w", err)
	}

	defaults, err := options.resolveDefaults(ctx, operation, oldSpec, &spec)
	if err != nil {
		return admissionVerdictV1beta2(dgdr, err)
	}
	defaultDynamoGraphDeploymentRequestSpecV1beta2(&spec, defaults)

	after, err := json.Marshal(&spec)
	if err != nil {
		return fmt.Errorf("encode spec.v1beta2: %w", err)
	}
	if !bytes.Equal(before, after) {
		dgdr.Spec.V1Beta2 = &apiextensionsv1.JSON{Raw: after}
	}
	return nil
}

// decodeOldSpecV1beta2 returns the stored v1beta2 spec an UPDATE replaces. The API server
// converts the old object to the same v1beta1 carrier view as the new one, so the spec sits
// in the same spec.v1beta2 envelope. It returns nil when there is nothing to compare with.
func decodeOldSpecV1beta2(
	operation admissionv1.Operation,
	oldObject []byte,
) (*nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec, error) {
	if operation != admissionv1.Update || len(oldObject) == 0 {
		return nil, nil
	}
	var old nvidiacomv1beta1.DynamoGraphDeploymentRequest
	if err := json.Unmarshal(oldObject, &old); err != nil {
		return nil, fmt.Errorf("decode old object: %w", err)
	}
	if old.Spec.V1Beta2 == nil {
		return nil, nil
	}
	var spec nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec
	if err := json.Unmarshal(old.Spec.V1Beta2.Raw, &spec); err != nil {
		return nil, fmt.Errorf("decode old spec.v1beta2: %w", err)
	}
	return &spec, nil
}

// dynamoGraphDeploymentRequestV1beta2DefaultOptions carries the request-scoped values the
// traversal needs. An empty field means the environment had nothing to offer.
type dynamoGraphDeploymentRequestV1beta2DefaultOptions struct {
	// hardware is the GPU identity, budget and node width discovered from the cluster.
	hardware hardwareDefaultsV1beta2
}

// hardwareDefaultsV1beta2 is the hardware the cluster can offer for the values a Request omits.
type hardwareDefaultsV1beta2 struct {
	sku            string
	budget         int32
	numGPUsPerNode int32
}

// defaultDynamoGraphDeploymentRequestSpecV1beta2 defaults spec in place. It applies on
// CREATE and UPDATE and is idempotent; opts carries hardware only when the environment was
// asked (see resolveHardware). spec must not be nil. Explicit values, including empty
// objects, are kept. The functions below mirror the Go API tree and visit children in
// declaration order.
func defaultDynamoGraphDeploymentRequestSpecV1beta2(
	spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	opts dynamoGraphDeploymentRequestV1beta2DefaultOptions,
) {
	defaultDynamoGraphDeploymentRequestNativeSpecV1beta2(&spec.DynamoGraphDeploymentRequestNativeSpec, opts)
}

func defaultDynamoGraphDeploymentRequestNativeSpecV1beta2(
	native *nvidiacomv1beta2.DynamoGraphDeploymentRequestNativeSpec,
	opts dynamoGraphDeploymentRequestV1beta2DefaultOptions,
) {
	defaultNativeSearchInputV1beta2(&native.NativeSearchInput, opts)
}

func defaultNativeSearchInputV1beta2(
	input *nvidiacomv1beta2.NativeSearchInput,
	opts dynamoGraphDeploymentRequestV1beta2DefaultOptions,
) {
	if input.Backends == nil {
		input.Backends = []nvidiacomv1beta2.Backend{"vllm", "sglang", "trtllm"}
	}

	// Omitted hardware receives discovered values. An explicit empty object reaches this
	// traversal with empty hardware options and remains the opt-out object.
	if input.Hardware == nil && opts.hardware != (hardwareDefaultsV1beta2{}) {
		input.Hardware = &nvidiacomv1beta2.HardwareSpec{}
	}
	if input.Hardware != nil {
		defaultHardwareSpecV1beta2(input.Hardware, opts.hardware)
	}

	if input.Workload == nil {
		input.Workload = &nvidiacomv1beta2.WorkloadSpec{}
	}
	defaultWorkloadSpecV1beta2(input.Workload)

	if input.Objective == nil {
		input.Objective = &nvidiacomv1beta2.ObjectiveSpec{}
	}
	defaultObjectiveSpecV1beta2(input.Objective)

	if input.Recommendation == nil {
		input.Recommendation = &nvidiacomv1beta2.RecommendationSpec{}
	}
	defaultRecommendationSpecV1beta2(input.Recommendation)
}

func defaultHardwareSpecV1beta2(hardware *nvidiacomv1beta2.HardwareSpec, defaults hardwareDefaultsV1beta2) {
	if hardware.GPU == nil && defaults != (hardwareDefaultsV1beta2{}) {
		hardware.GPU = &nvidiacomv1beta2.GPUHardwareSpec{}
	}
	if hardware.GPU != nil {
		defaultGPUHardwareSpecV1beta2(hardware.GPU, defaults)
	}
	if hardware.NumGPUsPerNode == nil && defaults.numGPUsPerNode > 0 {
		hardware.NumGPUsPerNode = new(defaults.numGPUsPerNode)
	}
}

// defaultGPUHardwareSpecV1beta2 fills each omitted value on its own, so a caller can pin the
// SKU and let admission size the budget.
func defaultGPUHardwareSpecV1beta2(gpu *nvidiacomv1beta2.GPUHardwareSpec, defaults hardwareDefaultsV1beta2) {
	if gpu.SKU == nil && defaults.sku != "" {
		gpu.SKU = new(defaults.sku)
	}
	if gpu.Budget == nil && defaults.budget > 0 {
		gpu.Budget = new(defaults.budget)
	}
}

func defaultWorkloadSpecV1beta2(workload *nvidiacomv1beta2.WorkloadSpec) {
	if workload.Static == nil {
		workload.Static = &nvidiacomv1beta2.StaticWorkloadSpec{}
	} else if workload.Static.InputTokens == nil && workload.Static.OutputTokens == nil && workload.Static.Load == nil {
		// An explicit empty object is the one static-workload opt-out.
		return
	}
	defaultStaticWorkloadSpecV1beta2(workload.Static)
}

// defaultStaticWorkloadSpecV1beta2 completes a static workload with every omitted default.
func defaultStaticWorkloadSpecV1beta2(static *nvidiacomv1beta2.StaticWorkloadSpec) {
	if static.InputTokens == nil {
		static.InputTokens = new(defaultStaticInputTokensV1beta2)
	}
	if static.OutputTokens == nil {
		static.OutputTokens = new(defaultStaticOutputTokensV1beta2)
	}
	if static.Load == nil {
		static.Load = &nvidiacomv1beta2.StaticLoadSpec{
			Concurrency: new(defaultStaticConcurrencyV1beta2),
		}
	}
}

// defaultObjectiveSpecV1beta2 defaults the mode, the Optimize metric and an absent SLA.
// Existing SLA objects are preserved, including the empty opt-out. It reads the typed objective
// alone: spec.aisimulate.inputOverrides overrides the completed typed values when the input is
// lowered, so it never suppresses these defaults.
func defaultObjectiveSpecV1beta2(objective *nvidiacomv1beta2.ObjectiveSpec) {
	if objective.Mode == "" {
		objective.Mode = nvidiacomv1beta2.ObjectiveModeOptimize
	}
	if objective.Mode == nvidiacomv1beta2.ObjectiveModeOptimize && objective.Metric == nil {
		objective.Metric = new(defaultMetricV1beta2)
	}
	if objective.SLA == nil {
		objective.SLA = &nvidiacomv1beta2.ObjectiveSLASpec{}
		defaultObjectiveSLASpecV1beta2(objective.SLA)
	}
}

// defaultObjectiveSLASpecV1beta2 fills the whole-object SLA default admission creates.
func defaultObjectiveSLASpecV1beta2(sla *nvidiacomv1beta2.ObjectiveSLASpec) {
	sla.TTFTMs = new(defaultSLATTFTMsV1beta2)
	sla.ITLMs = new(defaultSLAITLMsV1beta2)
}

func defaultRecommendationSpecV1beta2(recommendation *nvidiacomv1beta2.RecommendationSpec) {
	if recommendation.MaxCandidates == 0 {
		recommendation.MaxCandidates = defaultMaxCandidatesV1beta2
	}
}
