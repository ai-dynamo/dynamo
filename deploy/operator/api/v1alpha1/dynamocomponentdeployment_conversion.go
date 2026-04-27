/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Conversion between v1alpha1 and v1beta1 DynamoComponentDeployment.
// See dynamographdeployment_conversion.go for the design rationale.

package v1alpha1

import (
	"encoding/json"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/conversion"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

const (
	annDCDHubSpec     = "nvidia.com/dcd-hub-spec"
	annDCDSpokeSpec   = "nvidia.com/dcd-spoke-spec"
	annDCDSpokeStatus = "nvidia.com/dcd-spoke-status"
	annDCDHubOrigin   = "nvidia.com/dcd-hub-origin"
)

// ConvertTo converts this DynamoComponentDeployment (v1alpha1) into the hub
// version (v1beta1).
func (src *DynamoComponentDeployment) ConvertTo(dstRaw conversion.Hub) error {
	dst, ok := dstRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", dstRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	restoredHubSpec := false

	if raw, ok := dst.ObjectMeta.Annotations[annDCDHubSpec]; ok && raw != "" {
		if spec, ok := restoreDCDHubSpec(raw); ok {
			dst.Spec = spec
			restoredHubSpec = true
			delAnnFromObj(&dst.ObjectMeta, annDCDHubSpec)
		}
	}
	hubOrigin := restoredHubSpec || dst.ObjectMeta.Annotations[annDCDHubOrigin] == annotationTrue
	delAnnFromObj(&dst.ObjectMeta, annDCDHubOrigin)

	var semantic v1beta1.DynamoComponentDeploymentSpec
	semantic.BackendFramework = src.Spec.BackendFramework
	carrier := newDCDCarrier(&dst.ObjectMeta)
	if err := convertSharedSpecTo(&src.Spec.DynamoComponentDeploymentSharedSpec,
		&semantic.DynamoComponentDeploymentSharedSpec, carrier); err != nil {
		return err
	}

	// v1beta1 requires DCD.spec.componentName (it is the +listMapKey on
	// DGD.spec.components and is enforced as Required by the schema). When a
	// v1alpha1 caller omits ServiceName -- the common case for standalone
	// DCDs -- fall back to ObjectMeta.Name so the converted object is
	// schema-valid. The deferred defaulting webhook (see DEP #8069) will
	// own the same defaulting at admission time once v1beta1 is the storage
	// version.
	if semantic.ComponentName == "" && !hubOrigin {
		semantic.ComponentName = dst.ObjectMeta.Name
	}
	if restoredHubSpec {
		overlayDCDHubSpec(&dst.Spec, &semantic)
	} else {
		dst.Spec = semantic
	}

	if !hubOrigin {
		if data, err := marshalDCDSpokeSpec(&src.Spec); err == nil {
			if dst.ObjectMeta.Annotations == nil {
				dst.ObjectMeta.Annotations = map[string]string{}
			}
			dst.ObjectMeta.Annotations[annDCDSpokeSpec] = string(data)
		}
		if data, err := json.Marshal(src.Status); err == nil {
			if dst.ObjectMeta.Annotations == nil {
				dst.ObjectMeta.Annotations = map[string]string{}
			}
			dst.ObjectMeta.Annotations[annDCDSpokeStatus] = string(data)
		}
	} else {
		scrubDCDAnnotations(&dst.ObjectMeta)
	}
	convertDCDStatusTo(&src.Status, &dst.Status)
	return nil
}

func overlayDCDHubSpec(base *v1beta1.DynamoComponentDeploymentSpec, semantic *v1beta1.DynamoComponentDeploymentSpec) {
	hubPodTemplate := base.PodTemplate
	hubFrontendSidecar := base.FrontendSidecar
	hubExperimental := base.Experimental

	*base = *semantic.DeepCopy()
	if hubPodTemplate != nil {
		base.PodTemplate = hubPodTemplate
	}
	if base.FrontendSidecar == nil {
		base.FrontendSidecar = hubFrontendSidecar
	}
	if base.Experimental == nil {
		base.Experimental = hubExperimental
	}
}

// ConvertFrom converts from the hub (v1beta1) DynamoComponentDeployment into
// this v1alpha1 instance.
func (dst *DynamoComponentDeployment) ConvertFrom(srcRaw conversion.Hub) error {
	src, ok := srcRaw.(*v1beta1.DynamoComponentDeployment)
	if !ok {
		return fmt.Errorf("expected *v1beta1.DynamoComponentDeployment but got %T", srcRaw)
	}

	dst.ObjectMeta = *src.ObjectMeta.DeepCopy()
	dst.Spec.BackendFramework = src.Spec.BackendFramework

	if raw, ok := dst.ObjectMeta.Annotations[annDCDSpokeSpec]; ok && raw != "" {
		if spec, ok := restoreDCDSpokeSpec(raw); ok {
			dst.Spec = spec
			if rawStatus, ok := dst.ObjectMeta.Annotations[annDCDSpokeStatus]; ok && rawStatus != "" {
				_ = json.Unmarshal([]byte(rawStatus), &dst.Status)
			} else {
				convertDCDStatusFrom(&src.Status, &dst.Status)
			}
			scrubDCDAnnotations(&dst.ObjectMeta)
			delAnnFromObj(&dst.ObjectMeta, annDCDHubOrigin)
			return nil
		}
	}

	generatedPodTemplate := src.ObjectMeta.Annotations[annDCDPrefix+suffixPodTemplateOrig] == "generated"
	carrier := newDCDCarrier(&dst.ObjectMeta)
	if err := convertSharedSpecFrom(&src.Spec.DynamoComponentDeploymentSharedSpec,
		&dst.Spec.DynamoComponentDeploymentSharedSpec, carrier); err != nil {
		return err
	}

	convertDCDStatusFrom(&src.Status, &dst.Status)
	scrubDCDAnnotations(&dst.ObjectMeta)
	if dcdNeedsHubSpecPreservation(&src.Spec, generatedPodTemplate) {
		data, err := marshalDCDHubSpec(&src.Spec)
		if err != nil {
			return nil
		}
		if dst.ObjectMeta.Annotations == nil {
			dst.ObjectMeta.Annotations = map[string]string{}
		}
		dst.ObjectMeta.Annotations[annDCDHubSpec] = string(data)
	} else {
		if dst.ObjectMeta.Annotations == nil {
			dst.ObjectMeta.Annotations = map[string]string{}
		}
		dst.ObjectMeta.Annotations[annDCDHubOrigin] = annotationTrue
	}
	return nil
}

func marshalDCDHubSpec(src *v1beta1.DynamoComponentDeploymentSpec) ([]byte, error) {
	return marshalPreservedSpec(*src.DeepCopy(), func(spec *v1beta1.DynamoComponentDeploymentSpec, records *[]preservedRawJSON) {
		if spec.EPPConfig != nil {
			preserveEPPPluginParameters(spec.EPPConfig.Config, "eppConfig/config", records)
		}
	})
}

func restoreDCDHubSpec(raw string) (v1beta1.DynamoComponentDeploymentSpec, bool) {
	return restorePreservedSpec(raw, func(spec *v1beta1.DynamoComponentDeploymentSpec, records []preservedRawJSON) {
		if spec.EPPConfig != nil {
			restoreEPPPluginParameters(spec.EPPConfig.Config, "eppConfig/config", records)
		}
	})
}

func marshalDCDSpokeSpec(src *DynamoComponentDeploymentSpec) ([]byte, error) {
	return marshalPreservedSpec(*src.DeepCopy(), func(spec *DynamoComponentDeploymentSpec, records *[]preservedRawJSON) {
		if spec.EPPConfig != nil {
			preserveEPPPluginParameters(spec.EPPConfig.Config, "eppConfig/config", records)
		}
	})
}

func restoreDCDSpokeSpec(raw string) (DynamoComponentDeploymentSpec, bool) {
	return restorePreservedSpec(raw, func(spec *DynamoComponentDeploymentSpec, records []preservedRawJSON) {
		if spec.EPPConfig != nil {
			restoreEPPPluginParameters(spec.EPPConfig.Config, "eppConfig/config", records)
		}
	})
}

func dcdNeedsHubSpecPreservation(src *v1beta1.DynamoComponentDeploymentSpec, generatedPodTemplate bool) bool {
	if generatedPodTemplate {
		return false
	}
	return src.FrontendSidecar != nil ||
		src.PodTemplate != nil ||
		(src.Experimental != nil &&
			src.Experimental.GPUMemoryService == nil &&
			src.Experimental.Failover == nil &&
			src.Experimental.Checkpoint == nil)
}

func convertDCDStatusTo(src *DynamoComponentDeploymentStatus, dst *v1beta1.DynamoComponentDeploymentStatus) {
	dst.ObservedGeneration = src.ObservedGeneration
	if len(src.Conditions) > 0 {
		dst.Conditions = make([]metav1.Condition, 0, len(src.Conditions))
		for _, c := range src.Conditions {
			dst.Conditions = append(dst.Conditions, *c.DeepCopy())
		}
	}
	if src.Service != nil {
		dst.Component = convertReplicaStatusTo(src.Service)
	}
	// PodSelector is dropped in v1beta1 (the field was never populated by the
	// controller). No annotation is needed: the round-trip invariant is on
	// v1beta1 inputs, which do not carry PodSelector.
}

func convertDCDStatusFrom(src *v1beta1.DynamoComponentDeploymentStatus, dst *DynamoComponentDeploymentStatus) {
	dst.ObservedGeneration = src.ObservedGeneration
	if len(src.Conditions) > 0 {
		dst.Conditions = make([]metav1.Condition, 0, len(src.Conditions))
		for _, c := range src.Conditions {
			dst.Conditions = append(dst.Conditions, *c.DeepCopy())
		}
	}
	if src.Component != nil {
		dst.Service = convertReplicaStatusFrom(src.Component)
	}
}

// scrubDCDAnnotations removes any lingering "nvidia.com/dcd-*" keys that
// convertSharedSpecFrom did not consume.
func scrubDCDAnnotations(obj *metav1.ObjectMeta) {
	scrubAnnotationsByPrefix(obj, annDCDPrefix)
}
