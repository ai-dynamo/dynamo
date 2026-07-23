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

package validation

import (
	"context"
	"fmt"
	"strings"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/epp"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

const (
	unsetValue = "<unset>"

	vllmDistributedExecutorBackendMP  = "mp"
	vllmDistributedExecutorBackendRay = "ray"
)

// runtimeVersionValidationSource identifies the API representation whose field
// paths must be used for runtime-version validation errors.
type runtimeVersionValidationSource uint8

const (
	runtimeVersionSourceV1Beta1 runtimeVersionValidationSource = iota
	runtimeVersionSourceV1Alpha1
)

// runtimeVersionValidationSourceForRequest uses RequestKind because it preserves
// the GVK the client submitted when the API server converts the object for
// an equivalent-version webhook. For unconverted requests, RequestKind is nil;
// the handler endpoint GVK is then the source representation.
func runtimeVersionValidationSourceForRequest(ctx context.Context, fallbackGVK schema.GroupVersionKind) runtimeVersionValidationSource {
	request, err := admission.RequestFromContext(ctx)
	if err == nil && request.RequestKind != nil {
		return runtimeVersionValidationSourceForGVK(schema.GroupVersionKind{
			Group:   request.RequestKind.Group,
			Version: request.RequestKind.Version,
			Kind:    request.RequestKind.Kind,
		})
	}
	return runtimeVersionValidationSourceForGVK(fallbackGVK)
}

func runtimeVersionValidationSourceForGVK(gvk schema.GroupVersionKind) runtimeVersionValidationSource {
	if gvk.GroupVersion() == nvidiacomv1alpha1.GroupVersion {
		return runtimeVersionSourceV1Alpha1
	}
	return runtimeVersionSourceV1Beta1
}

func (v *sharedValidation) validatesRuntimeVersionFor(source runtimeVersionValidationSource) bool {
	return v.runtimeVersionSource == source
}

func hasContainerNamed(containers []corev1.Container, name string) bool {
	for i := range containers {
		if containers[i].Name == name {
			return true
		}
	}
	return false
}

func invalidVLLMDistributedExecutorBackendAnnotation(annotations map[string]string) (string, bool) {
	value, exists := annotations[consts.KubeAnnotationVLLMDistributedExecutorBackend]
	if !exists {
		return "", false
	}

	switch strings.ToLower(value) {
	case vllmDistributedExecutorBackendMP, vllmDistributedExecutorBackendRay:
		return "", false
	default:
		return value, true
	}
}

// inferencePoolAvailabilityError checks the InferencePool API.
// ctx and mgr must not be nil.
func inferencePoolAvailabilityError(ctx context.Context, mgr ctrl.Manager) error {
	if features.DetectInferencePoolAvailability(ctx, mgr) {
		return nil
	}
	return fmt.Errorf(
		"InferencePool API group (%s) is not available in the cluster; install the Gateway API Inference Extension before deploying EPP components",
		epp.InferencePoolGroup,
	)
}

func gpuMemoryServiceFor(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) *nvidiacomv1beta1.GPUMemoryServiceSpec {
	return gpuMemoryServiceForExperimental(component.Experimental)
}

func gpuMemoryServiceForExperimental(experimental *nvidiacomv1beta1.ExperimentalSpec) *nvidiacomv1beta1.GPUMemoryServiceSpec {
	if experimental == nil {
		return nil
	}
	return experimental.GPUMemoryService
}

func failoverFor(
	component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec,
) *nvidiacomv1beta1.FailoverSpec {
	return failoverForExperimental(component.Experimental)
}

func failoverForExperimental(experimental *nvidiacomv1beta1.ExperimentalSpec) *nvidiacomv1beta1.FailoverSpec {
	if experimental == nil {
		return nil
	}
	return experimental.Failover
}

func effectiveGMSMode(mode nvidiacomv1beta1.GPUMemoryServiceMode) nvidiacomv1beta1.GPUMemoryServiceMode {
	if mode == "" {
		return nvidiacomv1beta1.GMSModeIntraPod
	}
	return mode
}

func isInterPodGMS(gms *nvidiacomv1beta1.GPUMemoryServiceSpec) bool {
	return gms != nil && effectiveGMSMode(gms.Mode) == nvidiacomv1beta1.GMSModeInterPod
}

func isInterPodFailover(failover *nvidiacomv1beta1.FailoverSpec) bool {
	return failover != nil && effectiveGMSMode(failover.Mode) == nvidiacomv1beta1.GMSModeInterPod
}

func effectiveNumShadows(failover *nvidiacomv1beta1.FailoverSpec) int32 {
	if failover.NumShadows < 1 {
		return 1
	}
	return failover.NumShadows
}
