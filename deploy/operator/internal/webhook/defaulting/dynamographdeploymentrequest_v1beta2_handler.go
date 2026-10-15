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
	"fmt"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	nvidiacomv1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrhardware"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrinput"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/gpu"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
	admissionv1 "k8s.io/api/admission/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	dgdrV1Beta2DefaultingWebhookName = "dynamographdeploymentrequest-v1beta2-defaulting-webhook"
	dgdrV1Beta2DefaultingWebhookPath = "/mutate-nvidia-com-v1beta2-dynamographdeploymentrequest"

	// defaultHardwareDiscoveryTimeout bounds GPU discovery inside the admission call. Keep it
	// below the webhook's timeoutSeconds, or the API server gives up first and the request
	// fails under the webhook's failurePolicy instead of reaching validation.
	defaultHardwareDiscoveryTimeout = 5 * time.Second
)

// DGDRV1Beta2DefaulterOptions configures the v1beta2 DGDR defaulter.
type DGDRV1Beta2DefaulterOptions struct {
	// Hardware discovers the cluster's GPUs. Nil disables hardware defaulting, and the
	// validating webhook then requires the hardware to be spelled out.
	Hardware dgdrhardware.Discoverer
	// HardwareDiscoveryTimeout defaults to 5 seconds.
	HardwareDiscoveryTimeout time.Duration
	// MaxBudget caps a discovered GPU budget; zero means dgdrhardware.DefaultMaxBudget.
	MaxBudget int32
	// AISimulateImage is put into spec.aisimulate.image when unset; empty leaves it unset.
	AISimulateImage string
}

// DGDRV1Beta2Defaulter fills the Request values the CRD schema cannot: the scalar objective
// metric, the AISimulate runner image, and the hardware discovered from the cluster. It
// acts on CREATE only. An UPDATE is admitted unchanged, so a value a caller removed is not
// silently written back, and the validating webhook decides whether the result is complete.
type DGDRV1Beta2Defaulter struct {
	options DGDRV1Beta2DefaulterOptions
}

// NewDGDRV1Beta2Defaulter creates the v1beta2 DGDR defaulter.
func NewDGDRV1Beta2Defaulter(options DGDRV1Beta2DefaulterOptions) *DGDRV1Beta2Defaulter {
	return &DGDRV1Beta2Defaulter{options: options}
}

// Default implements admission.CustomDefaulter.
func (d *DGDRV1Beta2Defaulter) Default(ctx context.Context, obj runtime.Object) error {
	logger := log.FromContext(ctx).WithName(dgdrV1Beta2DefaultingWebhookName)

	if err := internalwebhook.ValidateAdmissionGVK(ctx, nvidiacomv1beta2.DynamoGraphDeploymentRequestGVK); err != nil {
		return err
	}

	dgdr, ok := obj.(*nvidiacomv1beta2.DynamoGraphDeploymentRequest)
	if !ok {
		return fmt.Errorf("expected DynamoGraphDeploymentRequest but got %T", obj)
	}

	req, err := admission.RequestFromContext(ctx)
	if err != nil {
		logger.Error(err, "failed to get admission request from context, skipping defaulting")
		return nil
	}
	if req.Operation != admissionv1.Create {
		return nil
	}

	spec := &dgdr.Spec
	d.defaultAISimulateImage(spec)
	if metric, ok := dgdrinput.MetricDefault(spec); ok {
		spec.Objective.Metric = &metric
	}
	d.defaultHardware(ctx, dgdr)
	return nil
}

func (d *DGDRV1Beta2Defaulter) defaultAISimulateImage(spec *nvidiacomv1beta2.DynamoGraphDeploymentRequestSpec) {
	if d.options.AISimulateImage == "" {
		return
	}
	if spec.AISimulate == nil {
		spec.AISimulate = &nvidiacomv1beta2.AISimulateSpec{}
	}
	if spec.AISimulate.Image == "" {
		spec.AISimulate.Image = d.options.AISimulateImage
	}
}

// defaultHardware fills the missing hardware from the cluster. A discovery failure is not
// an admission failure: the hardware stays unset and the validating webhook rejects the
// Request with a message that names the missing fields.
func (d *DGDRV1Beta2Defaulter) defaultHardware(ctx context.Context, dgdr *nvidiacomv1beta2.DynamoGraphDeploymentRequest) {
	if d.options.Hardware == nil {
		return
	}
	logger := log.FromContext(ctx).WithName(dgdrV1Beta2DefaultingWebhookName)

	timeout := d.options.HardwareDiscoveryTimeout
	if timeout <= 0 {
		timeout = defaultHardwareDiscoveryTimeout
	}
	discoveryCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	defaults, err := dgdrhardware.Resolve(discoveryCtx, d.options.Hardware, &dgdr.Spec,
		dgdrhardware.Options{MaxBudget: d.options.MaxBudget})
	if err != nil {
		logger.Info("hardware discovery failed; leaving hardware unresolved",
			"name", dgdr.Name, "namespace", dgdr.Namespace, "error", err.Error())
		return
	}
	dgdrhardware.Apply(&dgdr.Spec, defaults)
	if defaults != (dgdrhardware.Defaults{}) {
		logger.Info("defaulted spec.hardware.gpu from the cluster",
			"name", dgdr.Name, "namespace", dgdr.Namespace, "sku", defaults.SKU, "budget", defaults.Budget)
	}
}

// RegisterWithManager registers the v1beta2 DGDR defaulting webhook with the manager.
func (d *DGDRV1Beta2Defaulter) RegisterWithManager(mgr manager.Manager) error {
	defaulter := internalwebhook.NewLeaseAwareDefaulter(d, internalwebhook.GetExcludedNamespaces())
	webhook := admission.
		WithCustomDefaulter(mgr.GetScheme(), &nvidiacomv1beta2.DynamoGraphDeploymentRequest{}, defaulter).
		WithRecoverPanic(true)
	mgr.GetWebhookServer().Register(dgdrV1Beta2DefaultingWebhookPath, webhook)
	return nil
}

// gpuHardwareDiscoverer backs dgdrhardware.Discoverer with the operator's GPU discovery.
type gpuHardwareDiscoverer struct {
	discoverer *gpu.Discoverer
}

// NewGPUHardwareDiscoverer adapts the operator's GPU discovery for hardware defaulting.
func NewGPUHardwareDiscoverer(discoverer *gpu.Discoverer) dgdrhardware.Discoverer {
	return gpuHardwareDiscoverer{discoverer: discoverer}
}

func (g gpuHardwareDiscoverer) Discover(ctx context.Context, filterSKU string) (dgdrhardware.Discovered, error) {
	info, err := g.discoverer.Discover(ctx, nvidiacomv1beta1.GPUSKUType(filterSKU))
	if err != nil {
		return dgdrhardware.Discovered{}, err
	}
	return dgdrhardware.Discovered{
		SKU:  string(gpu.KnownSKU(info)),
		GPUs: info.GPUsPerNode * info.NodesWithGPUs,
	}, nil
}
