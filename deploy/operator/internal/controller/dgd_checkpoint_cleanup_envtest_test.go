//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
)

func TestAutomaticCheckpointCleanupUsesObservedUID(t *testing.T) {
	t.Log("Start a real API server to enforce deletion preconditions")
	env := operatorenv.New(operatorenv.Options{
		SetupWebhooks: setupProductionWebhooks,
	}).RunT(t)

	t.Log("Register Snapshot resource kinds for metadata lifecycle testing")
	// Metadata-focused schemas isolate generic deletion preconditions from Snapshot's spec validation.
	var crds []*apiextensionsv1.CustomResourceDefinition
	for _, resource := range []struct{ kind, plural string }{
		{"SnapshotJob", "snapshotjobs"},
		{"PodSnapshot", "podsnapshots"},
	} {
		crds = append(crds, &apiextensionsv1.CustomResourceDefinition{
			Name: resource.plural + "." + snapshotv1alpha1.GroupVersion.Group,
			Spec: apiextensionsv1.CustomResourceDefinitionSpec{
				Group: snapshotv1alpha1.GroupVersion.Group, Scope: apiextensionsv1.NamespaceScoped,
				Names: apiextensionsv1.CustomResourceDefinitionNames{Kind: resource.kind, ListKind: resource.kind + "List", Plural: resource.plural},
				Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
					Name: snapshotv1alpha1.GroupVersion.Version, Served: true, Storage: true,
					Schema: &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Type: "object", XPreserveUnknownFields: new(true)}},
				}},
			},
		})
	}
	_, err := envtest.InstallCRDs(env.RESTConfig(), envtest.CRDInstallOptions{CRDs: crds})
	require.NoError(t, err)
	apiClient, err := client.NewWithWatch(env.RESTConfig(), client.Options{Scheme: env.Client().Scheme()})
	require.NoError(t, err)

	for _, kind := range []string{"job", "snapshot"} {
		for _, change := range []string{"replacement", "metadata-update", "already-deleted"} {
			t.Run(kind+"/"+change, func(t *testing.T) {
				t.Log("Create an automatic checkpoint resource owned by the graph")
				ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
				defer cancel()
				dgd := &v1beta1.DynamoGraphDeployment{
					Name:      "cleanup-" + kind + "-" + change,
					Namespace: env.Namespace(),
					UID:       types.UID("dgd-uid"),
				}
				metadata := metav1.ObjectMeta{
					Name:      dgd.Name,
					Namespace: dgd.Namespace,
					Labels: map[string]string{
						commonconsts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
					},
					Annotations: map[string]string{
						commonconsts.CheckpointAutoAnnotation:           commonconsts.KubeLabelValueTrue,
						commonconsts.CheckpointOwnerUIDAnnotation:       string(dgd.UID),
						commonconsts.CheckpointDeletionPolicyAnnotation: string(v1alpha1.CheckpointDeletionPolicyDelete),
					},
				}
				var original client.Object
				if kind == "job" {
					original = &snapshotv1alpha1.SnapshotJob{
						ObjectMeta: metadata,
						Spec: snapshotv1alpha1.SnapshotJobSpec{
							PodTemplate: corev1.PodTemplateSpec{Spec: corev1.PodSpec{
								Containers:    []corev1.Container{{Name: "main", Image: "example.invalid/capture:v1"}},
								RestartPolicy: corev1.RestartPolicyNever,
							}},
							PodSnapshotTemplate: snapshotv1alpha1.PodSnapshotTemplate{TargetContainers: []string{"main"}},
						},
					}
				} else {
					original = &snapshotv1alpha1.PodSnapshot{
						ObjectMeta: metadata,
						Spec: snapshotv1alpha1.PodSnapshotSpec{Source: snapshotv1alpha1.PodSnapshotSource{
							PodRef: snapshotv1alpha1.PodReference{Name: "capture", Containers: []string{"main"}},
						}},
					}
				}
				require.NoError(t, apiClient.Create(ctx, original))
				key := client.ObjectKeyFromObject(original)

				t.Log("Change the live resource after cleanup observes it and before the API receives its Delete")
				changed := false
				var replacementUID types.UID
				kubeClient := interceptor.NewClient(apiClient, interceptor.Funcs{
					Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
						if !changed && client.ObjectKeyFromObject(obj) == key {
							changed = true
							current := obj.DeepCopyObject().(client.Object)
							if change == "metadata-update" {
								current.GetAnnotations()["test.dynamo.nvidia.com/observed"] = "true"
								require.NoError(t, c.Update(ctx, current))
								require.NotEqual(t, obj.GetResourceVersion(), current.GetResourceVersion())
							} else {
								require.NoError(t, c.Delete(ctx, current))
								if change == "replacement" {
									current.SetUID("")
									current.SetResourceVersion("")
									current.SetCreationTimestamp(metav1.Time{})
									current.SetManagedFields(nil)
									current.GetAnnotations()[commonconsts.CheckpointOwnerUIDAnnotation] = "other-dgd-uid"
									current.GetAnnotations()[commonconsts.CheckpointDeletionPolicyAnnotation] = string(v1alpha1.CheckpointDeletionPolicyRetain)
									require.NoError(t, c.Create(ctx, current))
									replacementUID = current.GetUID()
									require.NotEqual(t, original.GetUID(), replacementUID)
								}
							}
						}
						return c.Delete(ctx, obj, opts...)
					},
				})
				reconciler := newTestDGDCheckpointsReconciler(&DynamoGraphDeploymentReconciler{Client: kubeClient})

				t.Log("Run automatic cleanup through its complete job-before-artifact path")
				err := reconciler.deleteAutoCheckpointsForDGD(ctx, dgd)
				require.True(t, changed)
				observed := original.DeepCopyObject().(client.Object)
				if change == "replacement" {
					assert.True(t, apierrors.IsConflict(err), "expected an API UID conflict, got %v", err)
					require.NoError(t, apiClient.Get(ctx, key, observed))
					assert.Equal(t, replacementUID, observed.GetUID())
					assert.Equal(t, "other-dgd-uid", observed.GetAnnotations()[commonconsts.CheckpointOwnerUIDAnnotation])
					assert.Equal(t, string(v1alpha1.CheckpointDeletionPolicyRetain), observed.GetAnnotations()[commonconsts.CheckpointDeletionPolicyAnnotation])
				} else {
					if kind == "job" {
						require.ErrorIs(t, err, errAutomaticSnapshotCleanupPending)
					} else {
						require.NoError(t, err)
					}
					assert.True(t, apierrors.IsNotFound(apiClient.Get(ctx, key, observed)))
				}

				t.Log("Reconcile the refreshed state to finish cleanup without deleting a replacement")
				require.NoError(t, reconciler.deleteAutoCheckpointsForDGD(ctx, dgd))
				if change == "replacement" {
					require.NoError(t, apiClient.Get(ctx, key, observed))
					assert.Equal(t, replacementUID, observed.GetUID())
				}
			})
		}
	}
}
