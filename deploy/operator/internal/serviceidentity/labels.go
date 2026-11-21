// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package serviceidentity defines the component identity exposed by Services,
// independently of the workload resources selected by a lowering.
package serviceidentity

import (
	"context"
	"errors"
	"fmt"
	"maps"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	corev1apply "k8s.io/client-go/applyconfigurations/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const fieldOwner = "dynamo-operator-service-identity"

// ErrCacheObservationPending marks an identity write rejected by its stale observation.
var ErrCacheObservationPending = errors.New("waiting for the Service cache observation")

// Labels copies authored labels and stamps the canonical component identity.
// An empty graphDeploymentName supports standalone DCDs without inventing a DGD.
func Labels(authored map[string]string, graphDeploymentName, componentName, componentType, dynamoNamespace string) map[string]string {
	labels := maps.Clone(authored)
	if labels == nil {
		labels = make(map[string]string)
	}

	// Service identity uses the same component values as the selected Pods.
	labels[consts.KubeLabelDynamoComponent] = componentName
	if componentType != "" {
		labels[consts.KubeLabelDynamoComponentType] = componentType
	}
	labels[consts.KubeLabelDynamoNamespace] = dynamoNamespace
	if graphDeploymentName != "" {
		labels[consts.KubeLabelDynamoGraphDeploymentName] = graphDeploymentName
	}
	return labels
}

// Apply repairs only identity labels after the caller has checked Service ownership.
// desired and observed must be non-nil. It preserves the spec and external metadata;
// UID and resourceVersion bind the write to the Service the caller observed.
func Apply(ctx context.Context, writer client.Writer, desired, observed *corev1.Service) (bool, error) {
	labels := make(map[string]string)
	changed := false
	for _, key := range [...]string{
		consts.KubeLabelDynamoGraphDeploymentName,
		consts.KubeLabelDynamoComponent,
		consts.KubeLabelDynamoComponentType,
		consts.KubeLabelDynamoNamespace,
	} {
		if value, found := desired.Labels[key]; found {
			labels[key] = value
			current, present := observed.Labels[key]
			changed = changed || !present || current != value
		}
	}
	if !changed {
		return false, nil
	}

	// Apply no spec or owner fields, and reject replacement or concurrent handoff.
	configuration := corev1apply.Service(observed.Name, observed.Namespace).
		WithUID(observed.UID).
		WithResourceVersion(observed.ResourceVersion).
		WithLabels(labels)
	if err := writer.Apply(ctx, configuration, client.FieldOwner(fieldOwner), client.ForceOwnership); err != nil {
		if apierrors.IsConflict(err) {
			return false, fmt.Errorf("%w: %w", ErrCacheObservationPending, err)
		}
		return false, fmt.Errorf("apply Service identity %s/%s: %w", observed.Namespace, observed.Name, err)
	}
	return true, nil
}
