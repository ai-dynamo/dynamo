/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package workermetadata reads the discovery records that Dynamo processes
// publish in DynamoWorkerMetadata resources and defines the representation of
// those resources retained by the shared controller-runtime cache.
//
// Every Dynamo pod that uses Kubernetes discovery owns one resource per
// process. Its spec.data holds the process's discovery registrations. The
// operator reads two of them: the model cards a worker registers, and the
// admission records a frontend publishes for the workers it serves.
package workermetadata

import (
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/cache"
)

const (
	// FrontendAdmissionTopic is the discovery event-source topic of frontend
	// admission records. lib/llm/src/discovery/frontend_admission.rs owns the
	// record format.
	FrontendAdmissionTopic = "frontend-model-admission"
	// FrontendAdmissionProtocol is the admission record format this operator reads.
	FrontendAdmissionProtocol = "v1"
)

var (
	// GVK identifies the DynamoWorkerMetadata kind.
	GVK = schema.GroupVersionKind{Group: "nvidia.com", Version: "v1alpha1", Kind: "DynamoWorkerMetadata"}
	// ListGVK identifies the DynamoWorkerMetadata list kind.
	ListGVK = schema.GroupVersionKind{Group: "nvidia.com", Version: "v1alpha1", Kind: "DynamoWorkerMetadataList"}
)

// New returns an empty DynamoWorkerMetadata object for typed client calls.
func New() *unstructured.Unstructured {
	obj := &unstructured.Unstructured{}
	obj.SetGroupVersionKind(GVK)
	return obj
}

// NewList returns an empty DynamoWorkerMetadata list for typed client calls.
func NewList() *unstructured.UnstructuredList {
	list := &unstructured.UnstructuredList{}
	list.SetGroupVersionKind(ListGVK)
	return list
}

// Configure projects DynamoWorkerMetadata objects with Project before they
// enter the shared informer cache. options must be non-nil, and Configure must
// run before the manager is created.
//
// The projection is a default transform rather than a per-type one because a
// per-type cache setting requires the API to be served when the manager
// starts, and the operator must start without this optional resource. Other
// objects pass through to any default transform already configured.
func Configure(options *cache.Options) {
	next := options.DefaultTransform
	options.DefaultTransform = func(obj any) (any, error) {
		if metadata, ok := obj.(*unstructured.Unstructured); ok && metadata.GroupVersionKind() == GVK {
			return Project(metadata), nil
		}
		if next != nil {
			return next(obj)
		}
		return obj, nil
	}
}

// Project reduces obj in place to the fields the operator reads and returns it.
//
// Worker resources carry full model deployment cards, which are large and
// unused here. The projection keeps object metadata without managed fields,
// each model card's identity, and frontend admission records. Objects read
// from the cached client are therefore partial and must not be written back.
func Project(obj *unstructured.Unstructured) *unstructured.Unstructured {
	obj.SetManagedFields(nil)

	// Keep the identity of each model card and drop the card itself.
	modelCards := make(map[string]any)
	for path, raw := range nestedMap(obj.Object, "spec", "data", "model_cards") {
		card, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		modelCards[path] = map[string]any{
			"type":         card["type"],
			"namespace":    card["namespace"],
			"model_suffix": card["model_suffix"],
		}
	}

	// Keep only the event sources that are frontend admission records.
	eventSources := make(map[string]any)
	for path, raw := range nestedMap(obj.Object, "spec", "data", "event_sources") {
		source, ok := raw.(map[string]any)
		if ok && source["topic"] == FrontendAdmissionTopic {
			eventSources[path] = source
		}
	}

	obj.Object["spec"] = map[string]any{"data": map[string]any{
		"model_cards":   modelCards,
		"event_sources": eventSources,
	}}
	return obj
}

// OwningPod returns the name of the pod that owns obj, which is the process
// that published it.
func OwningPod(obj *unstructured.Unstructured) (string, bool) {
	for _, owner := range obj.GetOwnerReferences() {
		if owner.APIVersion == "v1" && owner.Kind == "Pod" && owner.Name != "" {
			return owner.Name, true
		}
	}
	return "", false
}

// AddBaseModelCards adds to cards the discovery keys of the model cards that
// obj registers, excluding LoRA adapter cards. Adapters are loaded on demand
// and never gate a rollout. A key embeds the card's runtime namespace, so keys
// from different worker generations never collide.
func AddBaseModelCards(obj *unstructured.Unstructured, cards map[string]struct{}) {
	for path, raw := range nestedMap(obj.Object, "spec", "data", "model_cards") {
		card, ok := raw.(map[string]any)
		if !ok || card["type"] != "Model" {
			continue
		}
		if suffix, present := card["model_suffix"]; present && suffix != nil {
			continue
		}
		cards[path] = struct{}{}
	}
}

// FrontendAdmission is what one frontend process publishes about the workers
// it serves.
type FrontendAdmission struct {
	// Capable reports that the frontend publishes admission records.
	Capable bool
	// Members holds the model-card keys of the workers the frontend serves.
	Members map[string]struct{}
}

// AddFrontendAdmission merges the admission records that obj publishes into
// admission. Records of unknown protocol versions are ignored.
func AddFrontendAdmission(obj *unstructured.Unstructured, admission *FrontendAdmission) {
	for _, raw := range nestedMap(obj.Object, "spec", "data", "event_sources") {
		source, ok := raw.(map[string]any)
		if !ok || source["topic"] != FrontendAdmissionTopic {
			continue
		}
		metadata, ok := source["metadata"].(map[string]any)
		if !ok || metadata["protocol"] != FrontendAdmissionProtocol {
			continue
		}
		if capability, ok := metadata["capability"].(bool); ok && capability {
			admission.Capable = true
		}
		members, _ := metadata["members"].([]any)
		for _, member := range members {
			if key, ok := member.(string); ok {
				admission.Members[key] = struct{}{}
			}
		}
	}
}

// nestedMap returns the map at fields without copying it, or nil when the
// field is absent or not a map. Callers must not mutate the result.
func nestedMap(obj map[string]any, fields ...string) map[string]any {
	value, found, err := unstructured.NestedFieldNoCopy(obj, fields...)
	if !found || err != nil {
		return nil
	}
	nested, _ := value.(map[string]any)
	return nested
}
