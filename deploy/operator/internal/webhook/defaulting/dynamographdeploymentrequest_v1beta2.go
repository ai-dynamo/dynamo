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
const defaultMetricV1beta2 = nvidiacomv1beta2.ObjectiveMetric("Throughput")

// Nested defaults of the parents admission creates. They must equal the +kubebuilder:default
// markers in api/v1beta2: the API server defaults a mutated object in the webhook's version
// (v1beta1, whose v1beta2 carrier is opaque), so the v1beta2 schema never fills them in.
const (
	defaultStaticISLV1beta2         int32 = 1024
	defaultStaticOSLV1beta2         int32 = 128
	defaultStaticConcurrencyV1beta2 int32 = 10
	defaultMaxCandidatesV1beta2     int32 = 1
)

func defaultPtrV1beta2[T any](v T) *T { return &v }

// This file defaults the v1beta2 shape of a DynamoGraphDeploymentRequest. It is separate
// from the canonical defaulting because the v1beta1 hub carries a v1beta2 Request as an
// opaque spec.v1beta2 envelope: conversion cannot apply field-level defaults to it, so the
// handler decodes the envelope, runs this traversal and encodes the result. Names carry the
// API version because the canonical names belong to v1beta1.
//
// The CRD schema owns the unconditional defaults: backends, objective.mode,
// recommendation.maxCandidates and workload.static as a whole (workload.static: {} opts
// out). objective.sla has no default: an omitted SLA stays absent. Admission owns what depends on
// other fields or on the environment: the scalar metric, the cluster hardware and the
// AISimulate image. spec.aisimulate.parameters never changes a default. It overrides the
// defaulted typed values later, when the input is lowered.

// defaultDynamoGraphDeploymentRequestV1beta2 defaults the v1beta2 spec that dgdr carries in
// spec.v1beta2. The environment is read once, before any mutation, and only on CREATE:
// discovered hardware is persisted with the Request, so a later write never replaces what
// the Request was created with. The envelope is rewritten
// only when defaulting changed the spec, so a defaulted object stays byte-identical across
// repeated admissions.
func defaultDynamoGraphDeploymentRequestV1beta2(
	ctx context.Context,
	dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest,
	operation admissionv1.Operation,
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

	defaultDynamoGraphDeploymentRequestSpecV1beta2(&spec, options.resolveDefaults(ctx, operation, &spec))

	after, err := json.Marshal(&spec)
	if err != nil {
		return fmt.Errorf("encode spec.v1beta2: %w", err)
	}
	if !bytes.Equal(before, after) {
		dgdr.Spec.V1Beta2 = &apiextensionsv1.JSON{Raw: after}
	}
	return nil
}

// dynamoGraphDeploymentRequestV1beta2DefaultOptions carries the request-scoped values the
// traversal needs. An empty field means the environment had nothing to offer.
type dynamoGraphDeploymentRequestV1beta2DefaultOptions struct {
	// hardware is the GPU identity and budget discovered from the cluster.
	hardware hardwareDefaultsV1beta2
	// aisimulateImage is the operator's configured AISimulate image.
	aisimulateImage string
}

// hardwareDefaultsV1beta2 is the hardware the cluster can offer for the values a Request omits.
type hardwareDefaultsV1beta2 struct {
	sku    string
	budget int32
}

func (h hardwareDefaultsV1beta2) empty() bool { return h.sku == "" && h.budget == 0 }

// defaultDynamoGraphDeploymentRequestSpecV1beta2 defaults spec in place. It applies on
// CREATE and UPDATE and is idempotent; opts carries no hardware on UPDATE. spec must not be nil.
func defaultDynamoGraphDeploymentRequestSpecV1beta2(
	spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
	opts dynamoGraphDeploymentRequestV1beta2DefaultOptions,
) {
	// Structural defaults live here, not in the CRD schema: a schema default would also be
	// applied to an object carrying spec.v1beta1 and put native fields next to the carrier,
	// which conversion rejects. This function only runs for native requests. Explicit values,
	// including empty objects, are kept. A parent the client sent already received its nested
	// schema defaults at decode; a parent created here gets them explicitly.
	if spec.Backends == nil {
		spec.Backends = []nvidiacomv1beta2.Backend{"vllm", "sglang", "trtllm"}
	}
	if spec.Workload == nil {
		spec.Workload = &nvidiacomv1beta2.WorkloadSpec{Static: &nvidiacomv1beta2.StaticWorkloadSpec{
			ISL:         defaultPtrV1beta2(defaultStaticISLV1beta2),
			OSL:         defaultPtrV1beta2(defaultStaticOSLV1beta2),
			Concurrency: defaultPtrV1beta2(defaultStaticConcurrencyV1beta2),
		}}
	}
	if spec.Objective == nil {
		spec.Objective = &nvidiacomv1beta2.ObjectiveSpec{Mode: nvidiacomv1beta2.ObjectiveModeOptimize}
	}
	if spec.Recommendation == nil {
		spec.Recommendation = &nvidiacomv1beta2.RecommendationSpec{MaxCandidates: defaultMaxCandidatesV1beta2}
	}

	if spec.Hardware == nil && !opts.hardware.empty() {
		spec.Hardware = &nvidiacomv1beta2.HardwareSpec{}
	}
	if spec.Hardware != nil {
		defaultHardwareSpecV1beta2(spec.Hardware, opts.hardware)
	}

	if spec.Objective != nil {
		defaultObjectiveSpecV1beta2(spec.Objective)
	}

	// The operator image is persisted with the Request, so spec.aisimulate is created to
	// carry it.
	if spec.AISimulate == nil && opts.aisimulateImage != "" {
		spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{}
	}
	if spec.AISimulate != nil {
		defaultAISimulateSpecV1beta2(spec.AISimulate, opts.aisimulateImage)
	}
}

func defaultHardwareSpecV1beta2(hardware *nvidiacomv1beta2.HardwareSpec, defaults hardwareDefaultsV1beta2) {
	if hardware.GPU == nil && !defaults.empty() {
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
		sku := defaults.sku
		gpu.SKU = &sku
	}
	if gpu.Budget == nil && defaults.budget > 0 {
		budget := defaults.budget
		gpu.Budget = &budget
	}
}

// defaultObjectiveSpecV1beta2 gives an Optimize objective without a metric the default
// scalar target. It reads the typed objective alone: a goal.target in
// spec.aisimulate.parameters overrides the completed typed value when the input is lowered,
// so it never suppresses this default.
func defaultObjectiveSpecV1beta2(objective *nvidiacomv1beta2.ObjectiveSpec) {
	if (objective.Mode == "" || objective.Mode == nvidiacomv1beta2.ObjectiveModeOptimize) && objective.Metric == nil {
		metric := defaultMetricV1beta2
		objective.Metric = &metric
	}
}

func defaultAISimulateSpecV1beta2(aisimulate *nvidiacomv1beta2.AISimulateSpec, image string) {
	if aisimulate.Image == "" && image != "" {
		aisimulate.Image = image
	}
}
