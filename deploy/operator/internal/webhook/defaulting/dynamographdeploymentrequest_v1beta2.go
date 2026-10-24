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
	"k8s.io/utils/ptr"
)

// defaultMetricV1beta2 is the scalar target an Optimize objective receives when it names none.
const defaultMetricV1beta2 = nvidiacomv1beta2.ObjectiveMetric("Throughput")

// Defaults of the objects admission creates. Admission is their only owner: the API server
// defaults a mutated object in the webhook's version (v1beta1, whose v1beta2 carrier is
// opaque), so the v1beta2 schema carries no default markers for them.
const (
	defaultStaticISLV1beta2         int32 = 1024
	defaultStaticOSLV1beta2         int32 = 128
	defaultStaticConcurrencyV1beta2 int32 = 10
	defaultMaxCandidatesV1beta2     int32 = 1
)

// This file defaults the v1beta2 shape of a DynamoGraphDeploymentRequest. It is separate
// from the canonical defaulting because the v1beta1 hub carries a v1beta2 Request as an
// opaque spec.v1beta2 envelope: conversion cannot apply field-level defaults to it, so the
// handler decodes the envelope, runs this traversal and encodes the result. Names carry the
// API version because the canonical names belong to v1beta1.
//
// Admission owns every v1beta2 default, and each default has this one owner: backends,
// objective.mode, the scalar metric, recommendation.maxCandidates, workload.static, the
// cluster hardware and the AISimulate image. workload.static: {} opts out of the static
// workload. objective.sla has no default: an omitted SLA stays absent.
// spec.aisimulate.parameters never changes a default. It overrides the defaulted typed
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

	defaultDynamoGraphDeploymentRequestSpecV1beta2(&spec, options.resolveDefaults(ctx, operation, oldSpec, &spec))

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
	// hardware is the GPU identity and budget discovered from the cluster.
	hardware hardwareDefaultsV1beta2
	// aisimulateImage is the operator's resolved AISimulate image.
	aisimulateImage string
}

// hardwareDefaultsV1beta2 is the hardware the cluster can offer for the values a Request omits.
type hardwareDefaultsV1beta2 struct {
	sku    string
	budget int32
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

	// Hardware has no static default: it exists only when the cluster supplied some.
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

	// The operator image is persisted with the Request, so spec.aisimulate is created to
	// carry it.
	if input.AISimulate == nil && opts.aisimulateImage != "" {
		input.AISimulate = &nvidiacomv1beta2.AISimulateSpec{}
	}
	if input.AISimulate != nil {
		defaultAISimulateSpecV1beta2(input.AISimulate, opts.aisimulateImage)
	}

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
}

// defaultGPUHardwareSpecV1beta2 fills each omitted value on its own, so a caller can pin the
// SKU and let admission size the budget.
func defaultGPUHardwareSpecV1beta2(gpu *nvidiacomv1beta2.GPUHardwareSpec, defaults hardwareDefaultsV1beta2) {
	if gpu.SKU == nil && defaults.sku != "" {
		gpu.SKU = ptr.To(defaults.sku)
	}
	if gpu.Budget == nil && defaults.budget > 0 {
		gpu.Budget = ptr.To(defaults.budget)
	}
}

func defaultWorkloadSpecV1beta2(workload *nvidiacomv1beta2.WorkloadSpec) {
	// An explicit static: {} is the opt-out, so the static defaults apply only to the object
	// created here.
	if workload.Static == nil {
		workload.Static = &nvidiacomv1beta2.StaticWorkloadSpec{}
		defaultStaticWorkloadSpecV1beta2(workload.Static)
	}
}

func defaultStaticWorkloadSpecV1beta2(static *nvidiacomv1beta2.StaticWorkloadSpec) {
	static.ISL = ptr.To(defaultStaticISLV1beta2)
	static.OSL = ptr.To(defaultStaticOSLV1beta2)
	static.Concurrency = ptr.To(defaultStaticConcurrencyV1beta2)
}

// defaultObjectiveSpecV1beta2 defaults the mode and gives an Optimize objective without a
// metric the default scalar target. It reads the typed objective alone: a goal.target in
// spec.aisimulate.parameters overrides the completed typed value when the input is lowered,
// so it never suppresses this default.
func defaultObjectiveSpecV1beta2(objective *nvidiacomv1beta2.ObjectiveSpec) {
	if objective.Mode == "" {
		objective.Mode = nvidiacomv1beta2.ObjectiveModeOptimize
	}
	if objective.Mode == nvidiacomv1beta2.ObjectiveModeOptimize && objective.Metric == nil {
		objective.Metric = ptr.To(defaultMetricV1beta2)
	}
}

func defaultAISimulateSpecV1beta2(aisimulate *nvidiacomv1beta2.AISimulateSpec, image string) {
	if aisimulate.Image == "" && image != "" {
		aisimulate.Image = image
	}
}

func defaultRecommendationSpecV1beta2(recommendation *nvidiacomv1beta2.RecommendationSpec) {
	if recommendation.MaxCandidates == 0 {
		recommendation.MaxCandidates = defaultMaxCandidatesV1beta2
	}
}
