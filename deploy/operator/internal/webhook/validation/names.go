// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package validation

import (
	"fmt"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8svalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// validateService checks the rendered name at the user-chosen source field.
// newService and fldPath must be non-nil; oldService is nil when no Service previously existed.
func (v *sharedValidation) validateService(newService, oldService *corev1.Service, fldPath *field.Path) field.ErrorList {
	if oldService != nil && newService.Name == oldService.Name {
		return nil
	}
	if messages := k8svalidation.IsDNS1035Label(newService.Name); len(messages) > 0 {
		return field.ErrorList{field.Invalid(fldPath, newService.Name, "generated Service name is invalid: "+strings.Join(messages, "; ")+"; choose a shorter name starting with a letter")}
	}
	return nil
}

// validateComponentPodMetadata checks the selector that the lowering actually emits.
// newMetadata and fldPath must be non-nil; oldMetadata is nil when no matching Pod template existed.
func (v *sharedValidation) validateComponentPodMetadata(newMetadata, oldMetadata *metav1.ObjectMeta, fldPath *field.Path) field.ErrorList {
	value, exists := newMetadata.Labels[consts.KubeLabelDynamoSelector]
	if !exists {
		return nil
	}
	if oldMetadata != nil {
		if oldValue, hadSelector := oldMetadata.Labels[consts.KubeLabelDynamoSelector]; hadSelector && value == oldValue {
			return nil
		}
	}
	if messages := k8svalidation.IsValidLabelValue(value); len(messages) > 0 {
		return field.ErrorList{field.Invalid(fldPath, value, fmt.Sprintf("generated %s label value is invalid: %s; choose a shorter deployment or component name", consts.KubeLabelDynamoSelector, strings.Join(messages, "; ")))}
	}
	return nil
}

// validateIngressBackend checks a rendered Ingress Service backend at its source field.
// newBackend, its Service and fldPath must be non-nil; oldBackend is nil when no Ingress existed.
// A non-nil oldBackend must also have a Service.
func (v *sharedValidation) validateIngressBackend(newBackend, oldBackend *networkingv1.IngressBackend, fldPath *field.Path) field.ErrorList {
	name := newBackend.Service.Name
	if oldBackend != nil && name == oldBackend.Service.Name {
		return nil
	}
	if messages := k8svalidation.IsDNS1035Label(name); len(messages) > 0 {
		return field.ErrorList{field.Invalid(fldPath, name, "generated Ingress backend Service name is invalid: "+strings.Join(messages, "; ")+"; choose a shorter name starting with a letter")}
	}
	return nil
}
