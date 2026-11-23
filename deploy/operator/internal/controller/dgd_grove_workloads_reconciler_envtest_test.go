//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"testing"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGroveProviderOverridesUseObservedPCSReconciliation(t *testing.T) {
	t.Log("Build one Grove reconciler with an injectable ordinary update failure")
	env := sharedEnv.ForTest(t)
	ctx := t.Context()
	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{
		Name: "graph", Namespace: env.Namespace(), UID: types.UID("graph-uid"),
	}
	desired := &grovev1alpha1.PodCliqueSet{
		APIVersion: provideroverride.GroveAPIVersion, Kind: provideroverride.TargetPodCliqueSet,
		Name: "graph", Namespace: env.Namespace(),
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{},
		},
	}
	rejectUpdate := false
	updateCalls := 0
	baseClient, err := client.NewWithWatch(env.RESTConfig(), client.Options{Scheme: newDynamoGraphDeploymentControllerTestScheme(t)})
	require.NoError(t, err)
	kubeClient := interceptor.NewClient(baseClient, interceptor.Funcs{
		Apply: func(ctx context.Context, writer client.WithWatch, object runtime.ApplyConfiguration, opts ...client.ApplyOption) error {
			wire, ok := object.(runtime.Unstructured)
			require.True(t, ok, "opaque PCS applies must serialize only at the write boundary")
			current := newUnstructuredGrovePodCliqueSet()
			require.NoError(t, writer.Get(ctx, client.ObjectKeyFromObject(object.(client.Object)), current))
			before, _, err := unstructured.NestedFieldNoCopy(current.Object, "spec", "template", "topologyConstraint")
			require.NoError(t, err)
			after, _, err := unstructured.NestedFieldNoCopy(wire.UnstructuredContent(), "spec", "template", "topologyConstraint")
			require.NoError(t, err)
			if !apiequality.Semantic.DeepEqual(before, after) {
				updateCalls++
				if rejectUpdate {
					return errors.New("provider rejected update")
				}
			}
			return writer.Apply(ctx, object, opts...)
		},
	})
	reconciler := &groveWorkloadsReconciler{
		syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(10)),
	}

	t.Log("Create the PCS after composing an opaque root topology override")
	dgd.Spec.ProviderOverride = rootTopologyOverride(`{"topologyName":"gpu-topology","futureProviderField":{"enabled":true}}`)
	_, _, err = reconciler.reconcilePodCliqueSet(ctx, dgd, &grovePodCliqueSetRender{desired: desired})
	require.NoError(t, err)
	assertLiveRootTopologyValue(t, kubeClient, env.Namespace(), "topologyName", "gpu-topology")
	assertLiveRootTopologyValue(t, kubeClient, env.Namespace(), "futureProviderField", map[string]any{"enabled": true})

	t.Log("Reuse the typed render observation when the desired PCS is unchanged")
	observed := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(desired), observed))
	_, _, err = reconciler.reconcilePodCliqueSet(ctx, dgd, &grovePodCliqueSetRender{existing: observed, desired: desired})
	require.NoError(t, err)
	assert.Zero(t, updateCalls)

	t.Log("Reject an ordinary PCS update and verify the live resource remains unchanged")
	dgd.Spec.ProviderOverride = rootTopologyOverride(`{"topologyName":"changed"}`)
	rejectUpdate = true
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(desired), observed))
	_, _, err = reconciler.reconcilePodCliqueSet(ctx, dgd, &grovePodCliqueSetRender{existing: observed, desired: desired})
	require.ErrorContains(t, err, "provider rejected update")
	assert.Equal(t, 1, updateCalls)
	assertLiveRootTopologyValue(t, kubeClient, env.Namespace(), "topologyName", "gpu-topology")
	assertLiveRootTopologyValue(t, kubeClient, env.Namespace(), "futureProviderField", map[string]any{"enabled": true})

	t.Log("Remove the override and let the same PCS update path prune its subtree")
	rejectUpdate = false
	dgd.Spec.ProviderOverride = nil
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(desired), observed))
	_, _, err = reconciler.reconcilePodCliqueSet(ctx, dgd, &grovePodCliqueSetRender{existing: observed, desired: desired})
	require.NoError(t, err)
	assert.Equal(t, 2, updateCalls)
	live := newUnstructuredGrovePodCliqueSet()
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(desired), live))
	_, found, nestedErr := unstructured.NestedFieldNoCopy(live.Object, "spec", "template", "topologyConstraint")
	require.NoError(t, nestedErr)
	assert.False(t, found)
}

func assertLiveRootTopologyValue(
	t *testing.T,
	kubeClient client.Client,
	namespace, field string,
	want any,
) {
	t.Helper()
	live := newUnstructuredGrovePodCliqueSet()
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKey{Name: "graph", Namespace: namespace}, live))
	got, found, err := unstructured.NestedFieldNoCopy(
		live.Object,
		"spec",
		"template",
		"topologyConstraint",
		field,
	)
	require.NoError(t, err)
	require.True(t, found)
	assert.Equal(t, want, got)
}
