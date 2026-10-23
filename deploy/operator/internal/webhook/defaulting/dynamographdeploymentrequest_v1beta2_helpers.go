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
	"context"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	admissionv1 "k8s.io/api/admission/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// defaultMaxHardwareBudget caps a discovered GPU budget. It matches the limit DGDR
	// v1beta1 applies to auto-discovered hardware, which keeps a search from using every GPU
	// in a large cluster. Set a budget explicitly to exceed it.
	defaultMaxHardwareBudget int32 = 32

	// defaultHardwareDiscoveryTimeout bounds GPU discovery inside the admission call. Keep it
	// below the webhook's timeoutSeconds, or the API server gives up first and the request
	// fails under the webhook's failurePolicy instead of reaching validation.
	defaultHardwareDiscoveryTimeout = 5 * time.Second
)

// HardwareDiscoverer finds the cluster's GPUs. filterSKU, when set, restricts the result to
// GPUs of that SKU.
type HardwareDiscoverer interface {
	Discover(ctx context.Context, filterSKU nvidiacomv1beta1.GPUSKUType) (*gpu.GPUInfo, error)
}

// DGDRV1beta2Options configures the defaults of v1beta2 Requests.
type DGDRV1beta2Options struct {
	// AISimulateImage overrides the version-matched Dynamo planner image persisted
	// into spec.aisimulate.image when the caller omits it.
	AISimulateImage string
	// Hardware discovers the cluster's GPUs. Nil disables hardware defaulting, and
	// validation then requires the hardware to be spelled out.
	Hardware HardwareDiscoverer
	// HardwareDiscoveryTimeout defaults to 5 seconds.
	HardwareDiscoveryTimeout time.Duration
	// MaxHardwareBudget caps a discovered GPU budget; zero means 32.
	MaxHardwareBudget int32
}

// resolveDefaults reads the environment for the values a v1beta2 spec omits. It does not
// mutate spec, and it never fails: a value the environment cannot supply stays unset and
// validating admission reports the exact field.
func (o DGDRV1beta2Options) resolveDefaults(
	ctx context.Context,
	operation admissionv1.Operation,
	oldSpec, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
) dynamoGraphDeploymentRequestV1beta2DefaultOptions {
	return dynamoGraphDeploymentRequestV1beta2DefaultOptions{
		hardware:        o.resolveHardware(ctx, operation, oldSpec, spec),
		aisimulateImage: o.AISimulateImage,
	}
}

// resolveHardware returns the SKU and budget the typed spec omits. It asks the cluster only
// when something is missing, restricted to the typed SKU when there is one.
//
// Discovery runs on CREATE and on an UPDATE that changes spec.hardware. A Request stays
// mutable and every Run snapshots it, so a caller may change or remove typed hardware before
// the next Run. An UPDATE that leaves spec.hardware unchanged, metadata-only ones included,
// never asks the cluster: a Request created while discovery had nothing to offer must not
// pick hardware up once discovery recovers, which would change its spec and generation behind
// the writer's back. oldSpec is the stored spec on UPDATE and nil otherwise. The decision reads
// the typed hardware alone, never spec.aisimulate.parameters.
func (o DGDRV1beta2Options) resolveHardware(
	ctx context.Context,
	operation admissionv1.Operation,
	oldSpec, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
) hardwareDefaultsV1beta2 {
	changed := operation == admissionv1.Update && oldSpec != nil &&
		!apiequality.Semantic.DeepEqual(oldSpec.Hardware, spec.Hardware)
	if o.Hardware == nil || (operation != admissionv1.Create && !changed) {
		return hardwareDefaultsV1beta2{}
	}
	var sku string
	var hasBudget bool
	if spec.Hardware != nil && spec.Hardware.GPU != nil {
		if spec.Hardware.GPU.SKU != nil {
			sku = *spec.Hardware.GPU.SKU
		}
		hasBudget = spec.Hardware.GPU.Budget != nil
	}
	if sku != "" && hasBudget {
		return hardwareDefaultsV1beta2{}
	}

	timeout := o.HardwareDiscoveryTimeout
	if timeout <= 0 {
		timeout = defaultHardwareDiscoveryTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	found, err := o.Hardware.Discover(ctx, nvidiacomv1beta1.GPUSKUType(sku))
	if err != nil {
		log.FromContext(ctx).Info("hardware discovery failed; leaving hardware unresolved", "error", err.Error())
		return hardwareDefaultsV1beta2{}
	}

	// The cluster facts are mapped into the v1beta2 API here: the AISimulate catalog name of
	// the GPU model (empty when the catalog does not know it) and the total GPU count.
	var gpus int
	if found != nil {
		gpus = found.GPUsPerNode * found.NodesWithGPUs
	}
	var defaults hardwareDefaultsV1beta2
	if sku == "" {
		defaults.sku = string(gpu.KnownSKU(found))
	}
	if !hasBudget && gpus > 0 {
		maxBudget := o.MaxHardwareBudget
		if maxBudget <= 0 {
			maxBudget = defaultMaxHardwareBudget
		}
		defaults.budget = int32(min(gpus, int(maxBudget)))
	}
	return defaults
}
