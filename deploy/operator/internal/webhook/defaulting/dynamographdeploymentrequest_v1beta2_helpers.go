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
	"errors"
	"fmt"
	"strings"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/validation/field"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	admissionv1 "k8s.io/api/admission/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
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
	// AISimulateVersionOverride is the AISimulate version bundled in AISimulateImage, for an image
	// whose Dynamo release the release contract does not map (a nightly or custom planner image).
	// Empty maps the image's release. WithV1beta2 completes it, so admission persists the image
	// together with its AISimulate version, and only next to an image it supplies itself, never
	// next to an image the caller chose.
	AISimulateVersionOverride string
	// Hardware discovers the cluster's GPUs. Nil disables hardware defaulting, and
	// validation then requires the hardware to be spelled out.
	Hardware HardwareDiscoverer
	// HardwareDiscoveryTimeout defaults to 5 seconds.
	HardwareDiscoveryTimeout time.Duration
	// MaxHardwareBudget caps a discovered GPU budget; zero means 32.
	MaxHardwareBudget int32
}

// resolveDefaults reads the environment for the values a v1beta2 spec omits. It does not
// mutate spec. It fails when the typed hardware cannot be completed: the error is the
// admission verdict, so a Request is never stored with hardware that depends on AISimulate's
// implicit defaults or on a cluster that could not be read.
func (o DGDRV1beta2Options) resolveDefaults(
	ctx context.Context,
	operation admissionv1.Operation,
	oldSpec, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
) (dynamoGraphDeploymentRequestV1beta2DefaultOptions, error) {
	hardware, err := o.resolveHardware(ctx, operation, oldSpec, spec)
	if err != nil {
		return dynamoGraphDeploymentRequestV1beta2DefaultOptions{}, err
	}
	return dynamoGraphDeploymentRequestV1beta2DefaultOptions{
		hardware:                  hardware,
		aisimulateImage:           o.AISimulateImage,
		aisimulateVersionOverride: o.AISimulateVersionOverride,
	}, nil
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
//
// After a successful return the typed hardware is complete. An omitted SKU becomes the
// cluster's AISimulate SKU, and an omitted budget becomes min(matching GPUs, cap). An explicit
// budget is authoritative: it is never capped, whatever the cluster holds. What the cluster
// cannot supply is rejected on the field that fixes it: a failed discovery, a GPU model the
// AISimulate catalog does not know, and a SKU without any matching GPU.
func (o DGDRV1beta2Options) resolveHardware(
	ctx context.Context,
	operation admissionv1.Operation,
	oldSpec, spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec,
) (hardwareDefaultsV1beta2, error) {
	changed := operation == admissionv1.Update && oldSpec != nil &&
		!apiequality.Semantic.DeepEqual(oldSpec.Hardware, spec.Hardware)
	if o.Hardware == nil || (operation != admissionv1.Create && !changed) {
		return hardwareDefaultsV1beta2{}, nil
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
		return hardwareDefaultsV1beta2{}, nil
	}

	timeout := o.HardwareDiscoveryTimeout
	if timeout <= 0 {
		timeout = defaultHardwareDiscoveryTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	gpuPath := field.NewPath("spec", "hardware", "gpu")
	found, err := o.Hardware.Discover(ctx, nvidiacomv1beta1.GPUSKUType(sku))
	if err != nil {
		return hardwareDefaultsV1beta2{}, rejectHardwareV1beta2(field.Required(gpuPath, fmt.Sprintf(
			"the cluster's GPUs could not be discovered (%v); set spec.hardware.gpu.sku and spec.hardware.gpu.budget", err)))
	}

	// The cluster facts are mapped into the v1beta2 API here: the AISimulate catalog name of
	// the GPU model (empty when the catalog does not know it) and the total GPU count.
	var defaults hardwareDefaultsV1beta2
	if sku == "" {
		if found != nil && len(found.OtherModels) > 0 {
			return hardwareDefaultsV1beta2{}, rejectHardwareV1beta2(field.Required(gpuPath.Child("sku"), fmt.Sprintf(
				"the cluster has several GPU types (%s); set spec.hardware.gpu.sku",
				strings.Join(append([]string{found.Model}, found.OtherModels...), ", "))))
		}
		sku = string(gpu.KnownSKU(found))
		if sku == "" {
			detail := "the cluster's GPUs map to no AISimulate SKU; set spec.hardware.gpu.sku"
			if found != nil && found.Model != "" {
				detail = fmt.Sprintf("the cluster's GPU model %q maps to no AISimulate SKU; set spec.hardware.gpu.sku", found.Model)
			}
			return hardwareDefaultsV1beta2{}, rejectHardwareV1beta2(field.Required(gpuPath.Child("sku"), detail))
		}
		defaults.sku = sku
	}
	if !hasBudget {
		var gpus int
		if found != nil {
			// TODO(#17192): Use the discovered aggregate GPU count so mixed node sizes of one SKU are counted correctly.
			gpus = found.GPUsPerNode * found.NodesWithGPUs
		}
		if gpus <= 0 {
			return hardwareDefaultsV1beta2{}, rejectHardwareV1beta2(field.Required(gpuPath.Child("budget"), fmt.Sprintf(
				"no GPUs of SKU %q were found in the cluster; set spec.hardware.gpu.budget", sku)))
		}
		maxBudget := o.MaxHardwareBudget
		if maxBudget <= 0 {
			maxBudget = defaultMaxHardwareBudget
		}
		defaults.budget = int32(min(gpus, int(maxBudget)))
	}
	return defaults, nil
}

// hardwareRejectionV1beta2 is hardware the cluster cannot complete, with the field that fixes it.
type hardwareRejectionV1beta2 struct{ err *field.Error }

func (e *hardwareRejectionV1beta2) Error() string { return e.err.Error() }

func rejectHardwareV1beta2(err *field.Error) error { return &hardwareRejectionV1beta2{err: err} }

// admissionVerdictV1beta2 turns a rejection into the invalid-object verdict validation uses, so a
// client sees the field and its cause. Any other error is returned as it is.
func admissionVerdictV1beta2(dgdr *nvidiacomv1beta1.DynamoGraphDeploymentRequest, err error) error {
	var rejection *hardwareRejectionV1beta2
	if errors.As(err, &rejection) {
		return apierrors.NewInvalid(nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK.GroupKind(), dgdr.Name,
			field.ErrorList{rejection.err})
	}
	return err
}
