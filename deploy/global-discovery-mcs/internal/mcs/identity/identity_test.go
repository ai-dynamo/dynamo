// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package identity_test

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/config"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/identity"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/mcstest"
)

func node(name, region, zone, providerID string) *corev1.Node {
	n := &corev1.Node{Name: name, Labels: map[string]string{}, Spec: corev1.NodeSpec{ProviderID: providerID}}
	if region != "" {
		n.Labels[identity.LabelRegion] = region
	}
	if zone != "" {
		n.Labels[identity.LabelZone] = zone
	}
	return n
}

func property(name, value string) *unstructured.Unstructured {
	u := identity.NewClusterProperty()
	u.SetName(name)
	_ = unstructured.SetNestedField(u.Object, value, "spec", "value")
	return u
}

func TestResolveFromFlagsAndNodes(t *testing.T) {
	c := mcstest.FakeClient(t,
		node("n1", "us-east-1", "us-east-1a", "aws:///us-east-1a/i-1"),
		node("n2", "us-east-1", "us-east-1b", ""),
		node("n3", "us-west-2", "us-east-1a", ""),
		property(api.ClusterPropertyClusterSet, "ignored"),
	)
	id, err := identity.Resolve(mcstest.Ctx(t), c, identity.FromConfig(&config.Config{
		IdentitySource: config.IdentitySourceAuto, ClusterSetID: "prod", ClusterID: "a",
		ClusterLabels: map[string]string{"env": "prod"},
	}))
	require.NoError(t, err)
	assert.Equal(t, identity.Identity{
		ClusterSet: "prod", Cluster: "a", Region: "us-east-1", Zones: []string{"us-east-1a", "us-east-1b"},
		Provider: "aws", Labels: map[string]string{"env": "prod"},
	}, id)
}

func TestResolveFromClusterProperties(t *testing.T) {
	c := mcstest.FakeClient(t,
		property(api.ClusterPropertyClusterSet, "prod"),
		property(api.ClusterPropertyClusterID, " b "),
	)
	id, err := identity.Resolve(mcstest.Ctx(t), c, identity.Options{Region: "eu-west-1"})
	require.NoError(t, err)
	assert.Equal(t, "prod", id.ClusterSet)
	assert.Equal(t, "b", id.Cluster)
	assert.Equal(t, "eu-west-1", id.Region)
	assert.Empty(t, id.Zones)

	// The flag wins in auto mode.
	id, err = identity.Resolve(mcstest.Ctx(t), c, identity.Options{ClusterID: "c"})
	require.NoError(t, err)
	assert.Equal(t, "c", id.Cluster)
	assert.Equal(t, "prod", id.ClusterSet)

	// clusterproperty mode accepts agreeing flags and rejects others.
	_, err = identity.Resolve(mcstest.Ctx(t), c, identity.Options{Source: config.IdentitySourceClusterProperty, ClusterID: "b"})
	require.NoError(t, err)
	_, err = identity.Resolve(mcstest.Ctx(t), c, identity.Options{Source: config.IdentitySourceClusterProperty, ClusterID: "c"})
	require.ErrorContains(t, err, "disagrees")
}

func TestResolveErrors(t *testing.T) {
	ctx := mcstest.Ctx(t)
	_, err := identity.Resolve(ctx, mcstest.FakeClient(t), identity.Options{Source: config.IdentitySourceClusterProperty})
	require.ErrorContains(t, err, "missing or empty")

	_, err = identity.Resolve(ctx, mcstest.FakeClient(t), identity.Options{ClusterSetID: "prod"})
	require.ErrorContains(t, err, "no cluster id")

	_, err = identity.Resolve(ctx, mcstest.FakeClient(t, property(api.ClusterPropertyClusterID, "Bad_Id")),
		identity.Options{ClusterSetID: "prod"})
	require.ErrorContains(t, err, "not a DNS label")

	// Flags mode never reads ClusterProperty objects.
	failing := mcstest.FakeClientWith(t, interceptor.Funcs{
		Get: func(context.Context, ctrlclient.WithWatch, ctrlclient.ObjectKey, ctrlclient.Object, ...ctrlclient.GetOption) error {
			return errors.New("boom")
		},
	})
	id, err := identity.Resolve(ctx, failing, identity.Options{Source: config.IdentitySourceFlags, ClusterSetID: "prod", ClusterID: "a"})
	require.NoError(t, err)
	assert.Equal(t, "a", id.Cluster)
	_, err = identity.Resolve(ctx, failing, identity.Options{ClusterSetID: "prod"})
	require.ErrorContains(t, err, "boom")
	// Auto mode with both flags reads no ClusterProperty either.
	_, err = identity.Resolve(ctx, failing, identity.Options{ClusterSetID: "prod", ClusterID: "a"})
	require.NoError(t, err)

	// A missing CRD counts as no ClusterProperty.
	noCRD := mcstest.FakeClientWith(t, interceptor.Funcs{
		Get: func(context.Context, ctrlclient.WithWatch, ctrlclient.ObjectKey, ctrlclient.Object, ...ctrlclient.GetOption) error {
			return &meta.NoKindMatchError{GroupKind: schema.GroupKind{Group: "about.k8s.io", Kind: "ClusterProperty"}}
		},
	})
	_, err = identity.Resolve(ctx, noCRD, identity.Options{ClusterSetID: "prod"})
	require.ErrorContains(t, err, "no cluster id")

	listFails := mcstest.FakeClientWith(t, interceptor.Funcs{
		List: func(context.Context, ctrlclient.WithWatch, ctrlclient.ObjectList, ...ctrlclient.ListOption) error {
			return errors.New("forbidden")
		},
	})
	_, err = identity.Resolve(ctx, listFails, identity.Options{Source: config.IdentitySourceFlags, ClusterSetID: "prod", ClusterID: "a"})
	require.ErrorContains(t, err, "list nodes")

	bad := property(api.ClusterPropertyClusterSet, "x")
	bad.Object["spec"] = map[string]any{"value": int64(3)}
	_, err = identity.Resolve(ctx, mcstest.FakeClient(t, bad), identity.Options{ClusterID: "a"})
	require.Error(t, err)
}

func TestWriteClusterProperties(t *testing.T) {
	ctx := mcstest.Ctx(t)
	c := mcstest.FakeClient(t, property(api.ClusterPropertyClusterSet, "prod"))
	id := identity.Identity{ClusterSet: "prod", Cluster: "a"}
	require.NoError(t, identity.WriteClusterProperties(ctx, c, id))
	v, err := identity.GetClusterProperty(ctx, c, api.ClusterPropertyClusterID)
	require.NoError(t, err)
	assert.Equal(t, "a", v)
	got := identity.NewClusterProperty()
	require.NoError(t, c.Get(ctx, ctrlclient.ObjectKey{Name: api.ClusterPropertyClusterID}, got))
	assert.Equal(t, api.ManagedByController, got.GetLabels()[api.LabelManagedBy])
	// Idempotent.
	require.NoError(t, identity.WriteClusterProperties(ctx, c, id))
	// Never overwrites another identity.
	err = identity.WriteClusterProperties(ctx, c, identity.Identity{ClusterSet: "prod", Cluster: "b"})
	require.ErrorContains(t, err, "refusing to overwrite")

	// A concurrent creator with the same value is fine, with another is not.
	for _, tt := range []struct {
		value string
		ok    bool
	}{{"a", true}, {"z", false}} {
		created := false
		racy := mcstest.FakeClientWith(t, interceptor.Funcs{
			Create: func(ctx context.Context, c ctrlclient.WithWatch, obj ctrlclient.Object, opts ...ctrlclient.CreateOption) error {
				if !created {
					created = true
					require.NoError(t, c.Create(ctx, property(obj.GetName(), tt.value)))
				}
				return c.Create(ctx, obj, opts...)
			},
		}, property(api.ClusterPropertyClusterSet, "prod"))
		err := identity.WriteClusterProperties(ctx, racy, id)
		if tt.ok {
			require.NoError(t, err)
		} else {
			require.Error(t, err)
		}
	}

	failing := mcstest.FakeClientWith(t, interceptor.Funcs{
		Create: func(context.Context, ctrlclient.WithWatch, ctrlclient.Object, ...ctrlclient.CreateOption) error {
			return errors.New("forbidden")
		},
	})
	require.ErrorContains(t, identity.WriteClusterProperties(ctx, failing, id), "forbidden")
	getFails := mcstest.FakeClientWith(t, interceptor.Funcs{
		Get: func(context.Context, ctrlclient.WithWatch, ctrlclient.ObjectKey, ctrlclient.Object, ...ctrlclient.GetOption) error {
			return errors.New("unavailable")
		},
	})
	require.ErrorContains(t, identity.WriteClusterProperties(ctx, getFails, id), "unavailable")
}
