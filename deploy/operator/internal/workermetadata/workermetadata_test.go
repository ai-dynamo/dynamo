/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package workermetadata

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
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
				"metadata": map[string]any{"protocol": FrontendAdmissionProtocol, "synced": true, "members": map[string]any{"graph-a1/backend/generate": []any{"1"}}},
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
				"metadata": map[string]any{"protocol": FrontendAdmissionProtocol, "synced": true, "members": map[string]any{"graph-a1/backend/generate": []any{"1"}}},
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

func TestAddFrontendAdmission(t *testing.T) {
	record := func(protocol string, synced bool, members map[string]any) map[string]any {
		return map[string]any{
			"topic":    FrontendAdmissionTopic,
			"scope":    map[string]any{"kind": "namespace", "name": "graph"},
			"metadata": map[string]any{"protocol": protocol, "synced": synced, "members": members},
		}
	}
	tests := []struct {
		name    string
		sources map[string]any
		want    FrontendAdmission
	}{
		{
			name: "a synced record lists the workers the frontend serves",
			sources: map[string]any{"a": record(FrontendAdmissionProtocol, true, map[string]any{
				"graph-a1/prefill/generate": []any{"1"},
				"graph-a1/backend/generate": []any{"2"},
			})},
			want: FrontendAdmission{Capable: true, Synced: true, Members: map[string]struct{}{
				"graph-a1/prefill/generate/1": {}, "graph-a1/backend/generate/2": {},
			}},
		},
		{
			name:    "a frontend without live discovery is capable but not synced",
			sources: map[string]any{"a": record(FrontendAdmissionProtocol, false, map[string]any{"graph-a1/backend/generate": []any{"2"}})},
			want:    FrontendAdmission{Capable: true, Members: map[string]struct{}{"graph-a1/backend/generate/2": {}}},
		},
		{
			name: "while one record replaces another, members merge and both must be synced",
			sources: map[string]any{
				"a": record(FrontendAdmissionProtocol, true, map[string]any{"graph-a1/backend/generate": []any{"2"}}),
				"b": record(FrontendAdmissionProtocol, false, map[string]any{"graph-b2/backend/generate": []any{"3"}}),
			},
			want: FrontendAdmission{Capable: true, Members: map[string]struct{}{
				"graph-a1/backend/generate/2": {}, "graph-b2/backend/generate/3": {},
			}},
		},
		{
			name: "records of other topics and protocols are ignored",
			sources: map[string]any{
				"future": record("v2", true, map[string]any{"graph-b2/backend/generate": []any{"3"}}),
				"kv":     map[string]any{"topic": "kv-events", "metadata": map[string]any{"protocol": FrontendAdmissionProtocol, "synced": true}},
			},
			want: FrontendAdmission{Members: map[string]struct{}{}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build a frontend resource with the records")
			metadata := New()
			metadata.Object["spec"] = map[string]any{"data": map[string]any{"event_sources": tt.sources}}

			t.Log("Merge the records")
			admission := FrontendAdmission{Members: map[string]struct{}{}}
			AddFrontendAdmission(metadata, &admission)
			assert.Equal(t, tt.want, admission)
		})
	}
}

// projectedTestRecords returns the projected resource of each process in the
// golden discovery records that the Rust tests in
// lib/llm/src/discovery/frontend_admission.rs produce.
func projectedTestRecords(t *testing.T) map[string]*unstructured.Unstructured {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "discovery_records.json"))
	require.NoError(t, err)
	var dataByProcess map[string]map[string]any
	require.NoError(t, json.Unmarshal(raw, &dataByProcess))
	resources := make(map[string]*unstructured.Unstructured, len(dataByProcess))
	for process, data := range dataByProcess {
		metadata := New()
		metadata.Object["spec"] = map[string]any{"data": data}
		resources[process] = Project(metadata)
	}
	return resources
}

func TestReadsTheRecordsDynamoPublishes(t *testing.T) {
	t.Log("Load the frontend and worker records produced by Dynamo's discovery code")
	resources := projectedTestRecords(t)

	t.Log("The worker's base card counts and its adapter card does not")
	cards := map[string]struct{}{}
	AddBaseModelCards(resources["worker"], cards)
	assert.Equal(t, map[string]struct{}{"graph-1/backend/generate/2a": {}}, cards)

	t.Log("The frontend's synced record lists that base card")
	admission := FrontendAdmission{Members: map[string]struct{}{}}
	AddFrontendAdmission(resources["frontend"], &admission)
	assert.Equal(t, FrontendAdmission{Capable: true, Synced: true, Members: cards}, admission)
}

func TestOwningPod(t *testing.T) {
	tests := []struct {
		name    string
		owners  []metav1.OwnerReference
		wantPod string
		wantUID types.UID
		wantOK  bool
	}{
		{name: "pod owner", owners: []metav1.OwnerReference{{APIVersion: "v1", Kind: "Pod", Name: "worker-0", UID: "uid-0"}}, wantPod: "worker-0", wantUID: "uid-0", wantOK: true},
		{name: "non-pod owner", owners: []metav1.OwnerReference{{APIVersion: "apps/v1", Kind: "ReplicaSet", Name: "rs"}}},
		{name: "no owner"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metadata := New()
			metadata.SetOwnerReferences(tt.owners)
			pod, uid, ok := OwningPod(metadata)
			assert.Equal(t, tt.wantPod, pod)
			assert.Equal(t, tt.wantUID, uid)
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
