//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/service"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

func TestSyncResourceServicePreservesOwnedIntentAndExternalMetadata(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx, kubeClient := t.Context(), env.Client()
	syncer := newDGDResourceSyncer(kubeClient, nil)

	t.Log("Create a rendered Service without an optional type label and retain its allocated network identity")
	parent := &corev1.ConfigMap{Name: "service-parent", Namespace: env.Namespace()}
	require.NoError(t, kubeClient.Create(ctx, parent))
	params := service.Params{
		ServiceName: "graph-frontend", Namespace: env.Namespace(), DGDName: "graph",
		ComponentName: "frontend", DynamoNamespace: "default-graph",
		Labels:      map[string]string{"example.com/operator-label": "before"},
		Annotations: map[string]string{"example.com/optional": "before"},
	}
	desired := service.Render(params)
	desired.Spec.Ports = []corev1.ServicePort{{Name: "http", Protocol: corev1.ProtocolTCP, Port: 8000, TargetPort: intstr.FromInt32(8000)}}
	input := desired.DeepCopy()
	modified, observed, err := commoncontroller.SyncResource(ctx, &syncer, parent, func(context.Context) (*corev1.Service, bool, error) {
		return desired, false, nil
	})
	require.NoError(t, err)
	require.True(t, modified)
	require.Equal(t, input, desired, "synchronization must leave rendered intent unchanged")
	require.NotEmpty(t, observed.UID)
	require.NotEmpty(t, observed.Spec.ClusterIP)
	require.NotEmpty(t, observed.Spec.ClusterIPs)
	require.NotEmpty(t, observed.Spec.IPFamilies)
	require.NotNil(t, observed.Spec.IPFamilyPolicy)
	require.True(t, metav1.IsControlledBy(observed, parent))
	require.NotContains(t, observed.Labels, consts.KubeLabelDynamoComponentType)
	original := observed.DeepCopy()
	var applied bool
	for _, entry := range observed.ManagedFields {
		applied = applied || entry.Operation == metav1.ManagedFieldsOperationApply
	}
	require.True(t, applied, "creation must also establish SSA ownership")

	t.Log("An independent manager adds metadata without taking over the operator's fields")
	external := observed.DeepCopy()
	external.Labels["example.com/external-label"] = "preserved"
	external.Annotations["example.com/external-note"] = "preserved"
	require.NoError(t, kubeClient.Patch(ctx, external, client.MergeFrom(observed), client.FieldOwner("external-service-manager")))

	t.Log("Introduce a type label through Apply, update metadata and change the target port")
	params.ComponentType = "frontend"
	params.Labels["example.com/operator-label"] = "after"
	params.Annotations["example.com/optional"] = "after"
	desired = service.Render(params)
	desired.Spec.Ports = []corev1.ServicePort{{Name: "http", Protocol: corev1.ProtocolTCP, Port: 8000, TargetPort: intstr.FromInt32(9000)}}
	modified, observed, err = commoncontroller.SyncResource(ctx, &syncer, parent, func(context.Context) (*corev1.Service, bool, error) {
		return desired, false, nil
	})
	require.NoError(t, err)
	require.True(t, modified)
	require.Equal(t, "after", observed.Labels["example.com/operator-label"])
	require.Equal(t, "after", observed.Annotations["example.com/optional"])
	require.Equal(t, intstr.FromInt32(9000), observed.Spec.Ports[0].TargetPort)
	require.Equal(t, "frontend", observed.Labels[consts.KubeLabelDynamoComponentType])
	require.Equal(t, "frontend", observed.Spec.Selector[consts.KubeLabelDynamoComponentType])

	t.Log("Omitting exclusively Apply-owned fields removes them while preserving external metadata")
	params.ComponentType = ""
	params.Annotations = nil
	next := service.Render(params)
	next.Spec.Ports = desired.Spec.Ports
	require.NotContains(t, next.Labels, consts.KubeLabelDynamoComponentType)
	modified, observed, err = commoncontroller.SyncResource(ctx, &syncer, parent, func(context.Context) (*corev1.Service, bool, error) {
		return next, false, nil
	})
	require.NoError(t, err)
	require.True(t, modified)
	require.NotContains(t, observed.Labels, consts.KubeLabelDynamoComponentType)
	require.NotContains(t, observed.Annotations, "example.com/optional")
	require.Equal(t, "preserved", observed.Labels["example.com/external-label"])
	require.Equal(t, "preserved", observed.Annotations["example.com/external-note"])
	require.Equal(t, original.UID, observed.UID)
	require.Equal(t, original.Spec.ClusterIP, observed.Spec.ClusterIP)
	require.Equal(t, original.Spec.ClusterIPs, observed.Spec.ClusterIPs)
	require.Equal(t, original.Spec.IPFamilies, observed.Spec.IPFamilies)
	require.Equal(t, original.Spec.IPFamilyPolicy, observed.Spec.IPFamilyPolicy)
	require.Equal(t, original.OwnerReferences, observed.OwnerReferences)

	t.Log("Reapplying unchanged intent does not change the server resource version")
	version := observed.ResourceVersion
	modified, observed, err = commoncontroller.SyncResource(ctx, &syncer, parent, func(context.Context) (*corev1.Service, bool, error) {
		return next, false, nil
	})
	require.NoError(t, err)
	require.False(t, modified)
	require.Equal(t, version, observed.ResourceVersion)
	stored := &corev1.Service{}
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(next), stored))
	require.Equal(t, version, stored.ResourceVersion)
}

// staleServiceObservationClient keeps only Service reads stale; every write uses the API server.
type staleServiceObservationClient struct {
	client.Client
	observed *corev1.Service
}

func (c *staleServiceObservationClient) Get(ctx context.Context, key client.ObjectKey, object client.Object, opts ...client.GetOption) error {
	if service, ok := object.(*corev1.Service); ok && key == client.ObjectKeyFromObject(c.observed) {
		c.observed.DeepCopyInto(service)
		return nil
	}
	return c.Client.Get(ctx, key, object, opts...)
}

func TestSyncResourceServiceUsesStaleObservationWithoutWritingToAnotherUID(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx, kubeClient := t.Context(), env.Client()
	syncer := newDGDResourceSyncer(kubeClient, nil)

	t.Log("Retain a cached Service observation before another manager changes its metadata")
	parent := &corev1.ConfigMap{Name: "service-parent", Namespace: env.Namespace()}
	require.NoError(t, kubeClient.Create(ctx, parent))
	desired := service.Render(service.Params{
		ServiceName: "graph-frontend", Namespace: env.Namespace(), DGDName: "graph",
		ComponentName: "frontend", ComponentType: "frontend", DynamoNamespace: "default-graph",
	})
	desired.Spec.Ports = []corev1.ServicePort{{Name: "http", Port: 8000, TargetPort: intstr.FromInt32(8000)}}
	_, observed, err := commoncontroller.SyncResource(ctx, &syncer, parent, func(context.Context) (*corev1.Service, bool, error) {
		return desired, false, nil
	})
	require.NoError(t, err)
	cached := &staleServiceObservationClient{Client: kubeClient, observed: observed.DeepCopy()}
	cachedSyncer := newDGDResourceSyncer(cached, nil)
	external := observed.DeepCopy()
	external.Annotations = map[string]string{"example.com/external-note": "newer-than-cache"}
	require.NoError(t, kubeClient.Patch(ctx, external, client.MergeFrom(observed), client.FieldOwner("external-service-manager")))
	require.NotEqual(t, cached.observed.ResourceVersion, external.ResourceVersion)

	t.Log("Reject a stale version before applying metadata, then retry the fresh observation")
	desired.Labels["example.com/operator-label"] = "updated"
	_, _, err = commoncontroller.SyncResource(ctx, &cachedSyncer, parent, func(context.Context) (*corev1.Service, bool, error) {
		return desired, false, nil
	})
	require.True(t, apierrors.IsConflict(err), "%v", err)
	live := &corev1.Service{}
	require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(desired), live))
	require.Equal(t, external.ObjectMeta, live.ObjectMeta, "a stale snapshot must not overwrite the concurrent edit")
	require.Equal(t, external.Spec, live.Spec)
	require.Equal(t, external.Status, live.Status)
	_, observed, err = commoncontroller.SyncResource(ctx, &syncer, parent, func(context.Context) (*corev1.Service, bool, error) {
		return desired, false, nil
	})
	require.NoError(t, err)
	require.Equal(t, "updated", observed.Labels["example.com/operator-label"])
	require.Equal(t, "newer-than-cache", observed.Annotations["example.com/external-note"])
	require.Equal(t, cached.observed.UID, observed.UID)

	t.Log("A cached observation of a deleted Service must not recreate it")
	require.NoError(t, kubeClient.Delete(ctx, observed))
	key := client.ObjectKeyFromObject(desired)
	require.True(t, apierrors.IsNotFound(kubeClient.Get(ctx, key, &corev1.Service{})))
	_, _, err = commoncontroller.SyncResource(ctx, &cachedSyncer, parent, func(context.Context) (*corev1.Service, bool, error) {
		return desired, false, nil
	})
	require.Error(t, err)
	require.True(t, apierrors.IsNotFound(kubeClient.Get(ctx, key, &corev1.Service{})))

	t.Log("A replacement Service remains untouched when the cache still contains the old UID")
	replacement := desired.DeepCopy()
	replacement.Labels = map[string]string{"example.com/replacement": "untouched"}
	replacement.Spec.Ports[0].TargetPort = intstr.FromInt32(7000)
	require.NoError(t, kubeClient.Create(ctx, replacement))
	require.NotEqual(t, cached.observed.UID, replacement.UID)
	before := replacement.DeepCopy()
	_, _, err = commoncontroller.SyncResource(ctx, &cachedSyncer, parent, func(context.Context) (*corev1.Service, bool, error) {
		return desired, false, nil
	})
	require.Error(t, err)
	require.NoError(t, kubeClient.Get(ctx, key, replacement))
	require.Equal(t, before, replacement)
}

func TestSyncResourceServiceRefusesForeignOrUnownedServices(t *testing.T) {
	for _, controlled := range []bool{false, true} {
		name := "unowned"
		if controlled {
			name = "foreign controller"
		}
		t.Run(name, func(t *testing.T) {
			env := sharedEnv.ForTest(t)
			ctx, kubeClient := t.Context(), env.Client()
			syncer := newDGDResourceSyncer(kubeClient, nil)

			t.Log("Create a same-name Service outside the requesting parent's control")
			parent := &corev1.ConfigMap{Name: "requesting-parent", Namespace: env.Namespace()}
			other := &corev1.ConfigMap{Name: "other-parent", Namespace: env.Namespace()}
			require.NoError(t, kubeClient.Create(ctx, parent))
			require.NoError(t, kubeClient.Create(ctx, other))
			desired := service.Render(service.Params{
				ServiceName: "graph-frontend", Namespace: env.Namespace(), DGDName: "graph",
				ComponentName: "frontend", ComponentType: "frontend", DynamoNamespace: "default-graph",
			})
			desired.Spec.Ports = []corev1.ServicePort{{Name: "http", Port: 8000, TargetPort: intstr.FromInt32(8000)}}
			existing := desired.DeepCopy()
			if controlled {
				require.NoError(t, controllerutil.SetControllerReference(other, existing, kubeClient.Scheme()))
			}
			require.NoError(t, kubeClient.Create(ctx, existing))
			before := existing.DeepCopy()

			t.Log("Refuse synchronization without changing any field on the existing Service")
			_, _, err := commoncontroller.SyncResource(ctx, &syncer, parent, func(context.Context) (*corev1.Service, bool, error) {
				return desired, false, nil
			})
			var conflict *commoncontroller.OwnershipConflictError
			require.ErrorAs(t, err, &conflict)
			require.NoError(t, kubeClient.Get(ctx, client.ObjectKeyFromObject(existing), existing))
			require.Equal(t, before, existing)
		})
	}
}

func TestSyncResourceSharedOwnershipPreservesEachParentsMetadata(t *testing.T) {
	env := sharedEnv.ForTest(t)
	ctx, kubeClient := t.Context(), env.Client()
	syncer := newDGDResourceSyncer(kubeClient, nil)

	t.Log("Create two distinct parents that contribute metadata to one shared ConfigMap")
	first := &corev1.ConfigMap{Name: "first-parent", Namespace: env.Namespace()}
	second := &corev1.ConfigMap{Name: "second-parent", Namespace: env.Namespace()}
	require.NoError(t, kubeClient.Create(ctx, first))
	require.NoError(t, kubeClient.Create(ctx, second))
	require.NotEqual(t, first.UID, second.UID)
	firstDesired := &corev1.ConfigMap{
		Name: "shared-resource", Namespace: env.Namespace(),
		Labels:      map[string]string{"example.com/first-label": "first"},
		Annotations: map[string]string{"example.com/first-note": "first"},
		Data:        map[string]string{"shared": "value"},
	}
	secondDesired := &corev1.ConfigMap{
		Name: "shared-resource", Namespace: env.Namespace(),
		Labels:      map[string]string{"example.com/second-label": "second"},
		Annotations: map[string]string{"example.com/second-note": "second"},
		Data:        map[string]string{"shared": "value"},
	}
	_, observed, err := commoncontroller.SyncResource(ctx, &syncer, first, func(context.Context) (*corev1.ConfigMap, bool, error) {
		return firstDesired, false, nil
	}, commoncontroller.WithSharedOwnership())
	require.NoError(t, err)
	require.True(t, metav1.IsControlledBy(observed, first))
	controllerReferences := observed.DeepCopy().OwnerReferences

	t.Log("Alternating parents preserve both contributions and the original controller reference")
	for _, actor := range []struct {
		parent  *corev1.ConfigMap
		desired *corev1.ConfigMap
	}{
		{parent: second, desired: secondDesired},
		{parent: first, desired: firstDesired},
		{parent: second, desired: secondDesired},
		{parent: first, desired: firstDesired},
	} {
		_, observed, err = commoncontroller.SyncResource(ctx, &syncer, actor.parent, func(context.Context) (*corev1.ConfigMap, bool, error) {
			return actor.desired, false, nil
		}, commoncontroller.WithSharedOwnership())
		require.NoError(t, err)
		require.Equal(t, "first", observed.Labels["example.com/first-label"])
		require.Equal(t, "first", observed.Annotations["example.com/first-note"])
		require.Equal(t, "second", observed.Labels["example.com/second-label"])
		require.Equal(t, "second", observed.Annotations["example.com/second-note"])
		require.Equal(t, controllerReferences, observed.OwnerReferences)
	}

	t.Log("The second parent can withdraw only its own metadata contribution")
	secondDesired.Labels = nil
	secondDesired.Annotations = nil
	_, observed, err = commoncontroller.SyncResource(ctx, &syncer, second, func(context.Context) (*corev1.ConfigMap, bool, error) {
		return secondDesired, false, nil
	}, commoncontroller.WithSharedOwnership())
	require.NoError(t, err)
	require.NotContains(t, observed.Labels, "example.com/second-label")
	require.NotContains(t, observed.Annotations, "example.com/second-note")
	require.Equal(t, "first", observed.Labels["example.com/first-label"])
	require.Equal(t, "first", observed.Annotations["example.com/first-note"])
	require.Equal(t, firstDesired.Data, observed.Data)
	require.Equal(t, controllerReferences, observed.OwnerReferences)

	t.Log("Reapplying the first parent's unchanged contribution leaves the server version unchanged")
	version := observed.ResourceVersion
	modified, observed, err := commoncontroller.SyncResource(ctx, &syncer, first, func(context.Context) (*corev1.ConfigMap, bool, error) {
		return firstDesired, false, nil
	}, commoncontroller.WithSharedOwnership())
	require.NoError(t, err)
	require.False(t, modified)
	require.Equal(t, version, observed.ResourceVersion)
}
