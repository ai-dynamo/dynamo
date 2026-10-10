/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package workermetadata

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/cache"
)

func TestProjectKeepsWhatTheOperatorReads(t *testing.T) {
	t.Log("Build a resource carrying a full model card, an adapter, an endpoint, and two event sources")
	metadata := New()
	metadata.SetName("worker-0")
	metadata.SetNamespace("inference")
	metadata.SetManagedFields([]metav1.ManagedFieldsEntry{{Manager: "dynamo"}})
	metadata.Object["spec"] = map[string]any{"data": map[string]any{
		"endpoints": map[string]any{
			"graph-a1/backend/generate/1": map[string]any{"type": "Endpoint"},
		},
		"model_cards": map[string]any{
			"graph-a1/backend/generate/1": map[string]any{
				"type": "Model", "namespace": "graph-a1", "component": "backend",
				"card_json": map[string]any{"display_name": "model", "chat_template": "large"},
			},
			"graph-a1/backend/generate/1/adapter": map[string]any{
				"type": "Model", "namespace": "graph-a1", "model_suffix": "adapter",
				"card_json": map[string]any{"display_name": "adapter"},
			},
		},
		"event_sources": map[string]any{
			"admission": map[string]any{
				"type": "EventSource", "topic": FrontendAdmissionTopic, "publisher_id": int64(7),
				"scope":    map[string]any{"kind": "namespace", "name": "graph"},
				"metadata": map[string]any{"protocol": FrontendAdmissionProtocol, "capability": true},
			},
			"kv": map[string]any{"type": "EventSource", "topic": "kv-events"},
		},
	}}

	t.Log("Project the resource as the cache would")
	projected := Project(metadata)

	t.Log("Only card identities and admission records remain")
	assert.Nil(t, projected.GetManagedFields())
	assert.Equal(t, map[string]any{"data": map[string]any{
		"model_cards": map[string]any{
			"graph-a1/backend/generate/1": map[string]any{
				"type": "Model", "namespace": "graph-a1", "model_suffix": nil,
			},
			"graph-a1/backend/generate/1/adapter": map[string]any{
				"type": "Model", "namespace": "graph-a1", "model_suffix": "adapter",
			},
		},
		"event_sources": map[string]any{
			"admission": map[string]any{
				"type": "EventSource", "topic": FrontendAdmissionTopic, "publisher_id": int64(7),
				"scope":    map[string]any{"kind": "namespace", "name": "graph"},
				"metadata": map[string]any{"protocol": FrontendAdmissionProtocol, "capability": true},
			},
		},
	}}, projected.Object["spec"])

	t.Log("Projecting again changes nothing")
	again := Project(projected.DeepCopy())
	assert.Equal(t, projected.Object, again.Object)

	t.Log("The projection still yields the base model cards")
	cards := map[string]struct{}{}
	AddBaseModelCards(again, cards)
	assert.Equal(t, map[string]struct{}{"graph-a1/backend/generate/1": {}}, cards)
}

func TestAddFrontendAdmissionReadsOnlyKnownRecords(t *testing.T) {
	t.Log("Build a frontend resource with current, foreign-topic, and future-protocol records")
	metadata := New()
	metadata.Object["spec"] = map[string]any{"data": map[string]any{
		"event_sources": map[string]any{
			"capability": map[string]any{
				"topic":    FrontendAdmissionTopic,
				"scope":    map[string]any{"kind": "namespace", "name": "graph"},
				"metadata": map[string]any{"protocol": FrontendAdmissionProtocol, "capability": true},
			},
			"members": map[string]any{
				"topic": FrontendAdmissionTopic,
				"scope": map[string]any{"kind": "namespace", "name": "graph-a1"},
				"metadata": map[string]any{
					"protocol": FrontendAdmissionProtocol,
					"members":  []any{"graph-a1/prefill/generate/1", "graph-a1/backend/generate/2"},
				},
			},
			"future": map[string]any{
				"topic": FrontendAdmissionTopic,
				"scope": map[string]any{"kind": "namespace", "name": "graph-b2"},
				"metadata": map[string]any{
					"protocol": "v2", "members": []any{"graph-b2/backend/generate/3"},
				},
			},
			"kv": map[string]any{
				"topic":    "kv-events",
				"metadata": map[string]any{"protocol": FrontendAdmissionProtocol, "members": []any{"other"}},
			},
		},
	}}

	t.Log("Merge the records")
	admission := FrontendAdmission{Members: map[string]struct{}{}}
	AddFrontendAdmission(metadata, &admission)

	t.Log("The capability and the current members are read; the rest is ignored")
	assert.True(t, admission.Capable)
	assert.Equal(t, map[string]struct{}{
		"graph-a1/prefill/generate/1": {},
		"graph-a1/backend/generate/2": {},
	}, admission.Members)
}

func TestOwningPod(t *testing.T) {
	tests := []struct {
		name    string
		owners  []metav1.OwnerReference
		wantPod string
		wantOK  bool
	}{
		{name: "pod owner", owners: []metav1.OwnerReference{{APIVersion: "v1", Kind: "Pod", Name: "worker-0"}}, wantPod: "worker-0", wantOK: true},
		{name: "non-pod owner", owners: []metav1.OwnerReference{{APIVersion: "apps/v1", Kind: "ReplicaSet", Name: "rs"}}},
		{name: "no owner"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metadata := New()
			metadata.SetOwnerReferences(tt.owners)
			pod, ok := OwningPod(metadata)
			assert.Equal(t, tt.wantPod, pod)
			assert.Equal(t, tt.wantOK, ok)
		})
	}
}

func TestConfigureProjectsOnlyWorkerMetadata(t *testing.T) {
	t.Log("Configure a cache that already has a default transform")
	options := cache.Options{DefaultTransform: func(obj any) (any, error) {
		return "next", nil
	}}
	Configure(&options)

	t.Log("DynamoWorkerMetadata objects are projected")
	transformed, err := options.DefaultTransform(New())
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"data": map[string]any{
		"model_cards":   map[string]any{},
		"event_sources": map[string]any{},
	}}, transformed.(*unstructured.Unstructured).Object["spec"])

	t.Log("Other objects reach the transform that was configured before")
	other := &unstructured.Unstructured{}
	other.SetGroupVersionKind(schema.GroupVersionKind{Group: "nvidia.com", Version: "v1beta1", Kind: "DynamoGraphDeployment"})
	transformed, err = options.DefaultTransform(other)
	require.NoError(t, err)
	assert.Equal(t, "next", transformed)

	t.Log("Without an earlier transform, other objects pass through unchanged")
	plain := cache.Options{}
	Configure(&plain)
	transformed, err = plain.DefaultTransform(other)
	require.NoError(t, err)
	assert.Same(t, other, transformed)
}
