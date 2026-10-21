//go:build clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package clusterenv

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"github.com/stretchr/testify/require"
	admissionv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	extensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
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
				require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(&record), &record))
				_, found, err := unstructured.NestedString(record.Object, "spec", "removedHistoricalField")
				require.NoError(t, err)
				require.False(t, found, "candidate schema should now prune the historical field on reads")
				require.Equal(t, map[string]any{"observed": "historical"}, record.Object["status"])
				require.NotEqual(t, "old-parent", string(record.GetUID()))
				require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(&child), &child))
				require.Equal(t, record.GetUID(), child.GetOwnerReferences()[0].UID)
			}
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(crd), crd))
		})
	}
}

// historyQuotaClient observes the real snapshot CREATE boundary without changing it.
type historyQuotaClient struct {
	client.Client
	deploymentCreates int
}

func (c *historyQuotaClient) Create(ctx context.Context, object client.Object, opts ...client.CreateOption) error {
	// Observe effective admission limits before forwarding the workload to the API server.
	if object.GetObjectKind().GroupVersionKind().Kind == "Deployment" {
		quota := &corev1.ResourceQuota{}
		if err := c.Client.Get(ctx, client.ObjectKey{Namespace: object.GetNamespace(), Name: workloadBlockQuotaName}, quota); err != nil {
			return fmt.Errorf("get quota before historical Deployment CREATE: %w", err)
		}
		if !quotaInitialized(quota, workloadBlockQuota(object.GetNamespace()).Spec.Hard) {
			return fmt.Errorf("quota is not initialized before historical Deployment CREATE")
		}
		c.deploymentCreates++
	}
	return c.Client.Create(ctx, object, opts...)
}

func TestClusterComponentHistoryBlocksWorkloads(t *testing.T) {
	t.Log("Connect before candidate admission or managers start")
	config, err := loadRESTConfig()
	require.NoError(t, err)
	scheme := workloadClusterTestScheme()
	require.NoError(t, extensions.AddToScheme(scheme))
	require.NoError(t, admissionv1.AddToScheme(scheme))
	c, err := client.New(config, client.Options{Scheme: scheme})
	require.NoError(t, err)

	t.Log("Build a populated Component snapshot with positive Deployment replicas")
	ns := "upgrade-component-quota"
	component := unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "nvidia.com/v1beta1", "kind": "DynamoComponentDeployment",
		"metadata": map[string]any{"name": "worker", "namespace": ns, "uid": "old-component"},
		"spec":     map[string]any{"name": "worker", "type": "worker", "backendFramework": "sglang", "replicas": int64(2)},
		"status":   map[string]any{"observedGeneration": int64(1)},
	}}
	labels := map[string]string{"app": "history-worker"}
	deployment := &appsv1.Deployment{
		TypeMeta: metav1.TypeMeta{APIVersion: "apps/v1", Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: ns, UID: "old-workload",
			Annotations:     map[string]string{"nvidia.com/last-applied-generation": "1"},
			OwnerReferences: []metav1.OwnerReference{{APIVersion: component.GetAPIVersion(), Kind: component.GetKind(), Name: component.GetName(), UID: component.GetUID(), Controller: ptr.To(true)}},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To[int32](2), Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: corev1.PodTemplateSpec{ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "invalid.example/never-used", Env: []corev1.EnvVar{{Name: "HISTORICAL", Value: "preserved"}}}}}},
		},
	}
	workload, err := runtime.DefaultUnstructuredConverter.ToUnstructured(deployment)
	require.NoError(t, err)
	snapshot := []unstructured.Unstructured{component, {Object: workload}}
	original := deployment.DeepCopy()

	t.Log("Import through PrepareHistory and check quota initialization at Deployment CREATE")
	observed := &historyQuotaClient{Client: c}
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	cleanup, prepareErr := PrepareHistory(ctx, observed, map[string][]unstructured.Unstructured{ns: snapshot})
	require.NotNil(t, cleanup)
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		require.NoError(t, cleanup(ctx))
	})
	require.NoError(t, prepareErr)
	require.Equal(t, 1, observed.deploymentCreates)

	t.Log("Keep captured workload intent and reconstruct the Component owner")
	stored := &appsv1.Deployment{}
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(deployment), stored))
	require.True(t, equality.Semantic.DeepDerivative(original.Spec, stored.Spec), "API defaulting may add fields but must preserve captured spec values")
	require.Equal(t, int32(2), *stored.Spec.Replicas)
	require.Equal(t, "1", stored.Annotations["nvidia.com/last-applied-generation"])
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(&component), &component))
	require.Equal(t, component.GetUID(), stored.OwnerReferences[0].UID)

	t.Log("Reject ReplicaSets and Pods before any candidate manager has started")
	replicaSet := &appsv1.ReplicaSet{ObjectMeta: metav1.ObjectMeta{Name: "blocked", Namespace: ns},
		Spec: appsv1.ReplicaSetSpec{Replicas: ptr.To[int32](1), Selector: original.Spec.Selector, Template: original.Spec.Template}}
	assertQuotaDenied(t, c.Create(t.Context(), replicaSet), "replicasets")
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "blocked", Namespace: ns}, Spec: original.Spec.Template.Spec}
	assertQuotaDenied(t, c.Create(t.Context(), pod), "pods")
}
