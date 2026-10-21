//go:build clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clusterenv

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	extensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestClusterHistoryStorage(t *testing.T) {
	t.Log("Install a candidate schema that cannot express a historical field")
	config, err := loadRESTConfig()
	require.NoError(t, err)
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, extensions.AddToScheme(scheme))
	require.NoError(t, admissionv1.AddToScheme(scheme))
	c, err := client.New(config, client.Options{Scheme: scheme})
	require.NoError(t, err)
	crd := &extensions.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "records.upgrade.example.com"},
		Spec: extensions.CustomResourceDefinitionSpec{Group: "upgrade.example.com", Scope: extensions.NamespaceScoped,
			Names: extensions.CustomResourceDefinitionNames{Plural: "records", Singular: "record", Kind: "Record", ListKind: "RecordList"},
			Versions: []extensions.CustomResourceDefinitionVersion{{Name: "v1", Served: true, Storage: true,
				Schema: &extensions.CustomResourceValidation{OpenAPIV3Schema: &extensions.JSONSchemaProps{Type: "object", Properties: map[string]extensions.JSONSchemaProps{
					"spec":   {Type: "object", Properties: map[string]extensions.JSONSchemaProps{"current": {Type: "string"}}},
					"status": {Type: "object", Properties: map[string]extensions.JSONSchemaProps{"observed": {Type: "string"}}},
				}}}, Subresources: &extensions.CustomResourceSubresources{Status: &extensions.CustomResourceSubresourceStatus{}},
			}},
		},
	}
	require.NoError(t, c.Create(t.Context(), crd))
	t.Cleanup(func() { require.NoError(t, c.Delete(context.Background(), crd)) })
	require.Eventually(t, func() bool {
		if c.Get(t.Context(), client.ObjectKeyFromObject(crd), crd) != nil {
			return false
		}
		for _, condition := range crd.Status.Conditions {
			if condition.Type == extensions.Established && condition.Status == extensions.ConditionTrue {
				return true
			}
		}
		return false
	}, 20*time.Second, 100*time.Millisecond)
	original := crd.Spec.DeepCopy()
	for _, broken := range []bool{false, true} {
		name := "stored"
		if broken {
			name = "unresolved-owner"
		}
		t.Run(name, func(t *testing.T) {
			ns := "upgrade-storage-" + name
			record := unstructured.Unstructured{Object: map[string]any{
				"apiVersion": "upgrade.example.com/v1", "kind": "Record",
				"metadata": map[string]any{"name": "old", "namespace": ns, "uid": "old-parent"},
				"spec":     map[string]any{"current": "value", "removedHistoricalField": "preserve-until-takeover"},
				"status":   map[string]any{"observed": "historical"},
			}}
			child := unstructured.Unstructured{Object: map[string]any{
				"apiVersion": "v1", "kind": "ConfigMap", "metadata": map[string]any{"name": "child", "namespace": ns, "uid": "old-child"}, "data": map[string]any{"key": "value"},
			}}
			owner := metav1.OwnerReference{APIVersion: "upgrade.example.com/v1", Kind: "Record", Name: "old", UID: "old-parent"}
			if broken {
				owner.UID = "missing"
			}
			child.SetOwnerReferences([]metav1.OwnerReference{owner})
			path := filepath.Join(t.TempDir(), "input.yaml")
			require.NoError(t, golden.WriteManifests(path, []unstructured.Unstructured{record, child}))
			snapshot, err := golden.ReadManifests(path)
			require.NoError(t, err)
			cleanup, prepareErr := PrepareHistory(t.Context(), c, map[string][]unstructured.Unstructured{ns: snapshot})
			require.NotNil(t, cleanup)
			t.Cleanup(func() {
				ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
				defer cancel()
				require.NoError(t, cleanup(ctx))
				require.NoError(t, c.Get(ctx, client.ObjectKeyFromObject(crd), crd))
				require.Equal(t, original, &crd.Spec)
			})
			// PrepareHistory verifies exact historical spec/status before schema restoration.
			if broken {
				require.ErrorContains(t, prepareErr, "unresolved owner")
			} else {
				require.NoError(t, prepareErr)
				t.Log("Wait for the restored candidate schema to prune historical fields on reads")
				require.Eventually(t, func() bool {
					current := &unstructured.Unstructured{}
					current.SetGroupVersionKind(record.GroupVersionKind())
					if err := c.Get(t.Context(), client.ObjectKeyFromObject(&record), current); err != nil {
						return false
					}
					_, found, err := unstructured.NestedString(current.Object, "spec", "removedHistoricalField")
					if err != nil || found {
						return false
					}
					record = *current
					return true
				}, 20*time.Second, 100*time.Millisecond, "candidate schema should prune the historical field on reads")
				require.Equal(t, map[string]any{"observed": "historical"}, record.Object["status"])
				require.NotEqual(t, "old-parent", string(record.GetUID()))
				require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(&child), &child))
				require.Equal(t, record.GetUID(), child.GetOwnerReferences()[0].UID)
			}
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(crd), crd))
		})
	}
}
