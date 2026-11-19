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

package controller_common

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"sort"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// ManagedServiceMetadataAnnotation records the label and annotation keys owned by the operator.
const ManagedServiceMetadataAnnotation = "nvidia.com/dynamo-managed-service-metadata"

type managedServiceMetadata struct {
	Labels      []string `json:"labels,omitempty"`
	Annotations []string `json:"annotations,omitempty"`
}

// ReconcileServiceMetadata converges recorded Service metadata while preserving other keys.
// existing is nil for a create request; desired must be non-nil.
func ReconcileServiceMetadata(
	existing *corev1.Service,
	desired *corev1.Service,
) (map[string]string, map[string]string, error) {
	labels := map[string]string{}
	annotations := map[string]string{}
	previous := managedServiceMetadata{}
	if existing != nil {
		maps.Copy(labels, existing.Labels)
		maps.Copy(annotations, existing.Annotations)
		// An unreadable inventory prunes nothing; it is rewritten from desired below.
		if raw := existing.Annotations[ManagedServiceMetadataAnnotation]; raw != "" {
			if err := json.Unmarshal([]byte(raw), &previous); err != nil {
				previous = managedServiceMetadata{}
			}
		}
	}
	for _, key := range previous.Labels {
		delete(labels, key)
	}
	for _, key := range previous.Annotations {
		delete(annotations, key)
	}

	desiredAnnotations := maps.Clone(desired.Annotations)
	delete(desiredAnnotations, ManagedServiceMetadataAnnotation)
	maps.Copy(labels, desired.Labels)
	maps.Copy(annotations, desiredAnnotations)

	next := managedServiceMetadata{
		Labels:      sortedMetadataKeys(desired.Labels),
		Annotations: sortedMetadataKeys(desiredAnnotations),
	}
	raw, err := json.Marshal(next)
	if err != nil {
		return nil, nil, fmt.Errorf("encode managed Service metadata inventory: %w", err)
	}
	annotations[ManagedServiceMetadataAnnotation] = string(raw)
	labels, annotations = serviceMetadataWithinAnnotationLimit(existing, labels, annotations, 0)
	return labels, annotations, nil
}

// syncResourceBookkeepingAnnotations are owned by SyncResource. Authored values never replace them.
var syncResourceBookkeepingAnnotations = []string{
	NvidiaAnnotationHashKey,
	NvidiaAnnotationGenerationKey,
}

// StampManagedServiceMetadata gives a Service rendered for SyncResource its managed-key inventory,
// so the inventory is written in the create request.
func StampManagedServiceMetadata(service *corev1.Service) error {
	labels, annotations, err := managedServiceMetadataFor(nil, service)
	if err != nil {
		return err
	}
	service.Labels = labels
	service.Annotations = annotations
	return nil
}

// SyncManagedServiceMetadata converges a synced Service's labels and annotations, pruning only
// keys it recorded. Copy desired before SyncResource, which stamps bookkeeping annotations into it.
func SyncManagedServiceMetadata(
	ctx context.Context,
	writer client.Writer,
	synced *corev1.Service,
	desired *corev1.Service,
) (bool, error) {
	labels, annotations, err := managedServiceMetadataFor(synced, desired)
	if err != nil {
		return false, err
	}
	if equality.Semantic.DeepEqual(synced.Labels, labels) &&
		equality.Semantic.DeepEqual(synced.Annotations, annotations) {
		return false, nil
	}
	synced.Labels = labels
	synced.Annotations = annotations
	if err := writer.Update(ctx, synced); err != nil {
		return false, err
	}
	return true, nil
}

// managedServiceMetadataFor is ReconcileServiceMetadata for Services synced by
// SyncResource, which owns the bookkeeping annotations: desired copies of them are ignored and
// existing values are kept. existing is nil for a Service that does not exist yet.
func managedServiceMetadataFor(existing, desired *corev1.Service) (map[string]string, map[string]string, error) {
	desired = desired.DeepCopy()
	for _, key := range syncResourceBookkeepingAnnotations {
		delete(desired.Annotations, key)
	}
	labels, annotations, err := ReconcileServiceMetadata(existing, desired)
	if err != nil {
		return nil, nil, err
	}

	// Restore the stamped values, which an older inventory may have pruned.
	for _, key := range syncResourceBookkeepingAnnotations {
		delete(annotations, key)
		if existing == nil {
			continue
		}
		if value, ok := existing.Annotations[key]; ok {
			annotations[key] = value
		}
	}

	// A create request will also receive a SHA-256 hash and an int64 generation from SyncResource.
	reservedBytes := 0
	if existing == nil {
		reservedBytes = len(NvidiaAnnotationHashKey) + 64 +
			len(NvidiaAnnotationGenerationKey) + 20
	}
	labels, annotations = serviceMetadataWithinAnnotationLimit(existing, labels, annotations, reservedBytes)
	return labels, annotations, nil
}

// serviceMetadataWithinAnnotationLimit defers existing metadata edits when the inventory
// would exceed Kubernetes' annotation limit. Creates omit the inventory until space is available.
// existing is nil when preparing a create request.
func serviceMetadataWithinAnnotationLimit(
	existing *corev1.Service,
	labels, annotations map[string]string,
	reservedBytes int,
) (map[string]string, map[string]string) {
	// Kubernetes counts the byte lengths of every annotation key and value.
	size := reservedBytes
	for key, value := range annotations {
		size += len(key) + len(value)
	}
	if size <= 256*1024 {
		return labels, annotations
	}

	// Keep the prior inventory and metadata intact so a later retry can still prune recorded keys.
	if existing != nil {
		return maps.Clone(existing.Labels), maps.Clone(existing.Annotations)
	}
	delete(annotations, ManagedServiceMetadataAnnotation)
	return labels, annotations
}

func sortedMetadataKeys(metadata map[string]string) []string {
	keys := make([]string, 0, len(metadata))
	for key := range metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
