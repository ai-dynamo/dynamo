// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package power

import (
	"context"
	"fmt"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// fieldManager owns the PowerCapPropagationReady entry of the DGD's status conditions.
const fieldManager = "dynamo-power-cap"

// applyCondition server-side applies the controller's only DGD status field. A nil condition applies
// an empty list, which removes the entry this field manager owns. Every pass applies the condition;
// an unchanged cached status keeps its lastTransitionTime. dgd must be non-nil.
func applyCondition(ctx context.Context, c client.Client, dgd *v1beta1.DynamoGraphDeployment, condition *metav1.Condition) error {
	conditions := []any{}
	if condition != nil {
		current := meta.FindStatusCondition(dgd.Status.Conditions, ConditionType)
		desired := *condition
		desired.LastTransitionTime = metav1.Now()
		if current != nil && current.Status == condition.Status {
			desired.LastTransitionTime = current.LastTransitionTime
		}
		entry, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&desired)
		if err != nil {
			return fmt.Errorf("convert %s condition: %w", ConditionType, err)
		}
		conditions = append(conditions, entry)
	}

	// Name only the DGD identity and the owned condition so other status fields stay with their owners.
	apply := &unstructured.Unstructured{}
	apply.SetGroupVersionKind(v1beta1.DynamoGraphDeploymentGVK)
	apply.SetNamespace(dgd.Namespace)
	apply.SetName(dgd.Name)
	apply.SetUID(dgd.UID)
	if err := unstructured.SetNestedSlice(apply.Object, conditions, "status", "conditions"); err != nil {
		return fmt.Errorf("build %s condition apply: %w", ConditionType, err)
	}
	if err := c.Status().Apply(ctx, client.ApplyConfigurationFromUnstructured(apply), client.FieldOwner(fieldManager), client.ForceOwnership); err != nil {
		return fmt.Errorf("apply %s condition: %w", ConditionType, err)
	}
	return nil
}
