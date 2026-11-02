// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package export

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/mcstest"
)

func autoReconcile(t *testing.T, a *AutoExporter, name string) {
	t.Helper()
	_, err := a.Reconcile(mcstest.Ctx(t), ctrl.Request{Namespace: ns, Name: name})
	require.NoError(t, err)
}

func getExport(t *testing.T, c ctrlclient.Client, name string) (*mcsv1beta1.ServiceExport, bool) {
	t.Helper()
	var se mcsv1beta1.ServiceExport
	err := c.Get(mcstest.Ctx(t), types.NamespacedName{Namespace: ns, Name: name}, &se)
	if apierrors.IsNotFound(err) {
		return nil, false
	}
	require.NoError(t, err)
	return &se, true
}

func TestAutoExportFollowsTheLabel(t *testing.T) {
	svc := mcstest.Service(ns, "web", map[string]string{api.LabelExport: api.ExportEnabled})
	c := mcstest.FakeClient(t, svc)
	a := NewAutoExporter(AutoExporterOptions{Client: c, Identity: mcstest.Identity()})

	autoReconcile(t, a, "web")
	se, ok := getExport(t, c, "web")
	require.True(t, ok)
	assert.True(t, Managed(se))
	assert.Equal(t, mcstest.ClusterSet, se.Labels[api.LabelClusterSet])
	require.Len(t, se.OwnerReferences, 1)
	assert.Equal(t, svc.UID, se.OwnerReferences[0].UID)
	assert.Equal(t, "Service", se.OwnerReferences[0].Kind)

	autoReconcile(t, a, "web") // idempotent
	svc.Labels = nil
	require.NoError(t, c.Update(mcstest.Ctx(t), svc))
	autoReconcile(t, a, "web")
	_, ok = getExport(t, c, "web")
	assert.False(t, ok, "deleted when the label goes")

	// A Service that disappears takes its export along.
	svc.Labels = map[string]string{api.LabelExport: api.ExportEnabled}
	require.NoError(t, c.Update(mcstest.Ctx(t), svc))
	autoReconcile(t, a, "web")
	require.NoError(t, c.Delete(mcstest.Ctx(t), svc))
	autoReconcile(t, a, "web")
	_, ok = getExport(t, c, "web")
	assert.False(t, ok)
}

func TestAutoExportNeverTouchesForeignExports(t *testing.T) {
	explicit := mcstest.ServiceExport(ns, "web")
	c := mcstest.FakeClient(t, mcstest.Service(ns, "web", nil), explicit)
	a := NewAutoExporter(AutoExporterOptions{Client: c, Identity: mcstest.Identity()})
	autoReconcile(t, a, "web")
	_, ok := getExport(t, c, "web")
	assert.True(t, ok, "an unlabelled Service keeps a ServiceExport it did not get from the controller")

	svc := mcstest.Service(ns, "api", map[string]string{api.LabelExport: api.ExportEnabled})
	require.NoError(t, c.Create(mcstest.Ctx(t), svc))
	require.NoError(t, c.Create(mcstest.Ctx(t), mcstest.ServiceExport(ns, "api")))
	autoReconcile(t, a, "api")
	se, _ := getExport(t, c, "api")
	assert.False(t, Managed(se))
	assert.Empty(t, se.OwnerReferences)
}

func TestAutoExportOwnsItsExportAgain(t *testing.T) {
	t.Log("A managed ServiceExport left behind by a deleted Service of the same name")
	svc := mcstest.Service(ns, "web", map[string]string{api.LabelExport: api.ExportEnabled})
	stale := mcstest.ServiceExport(ns, "web")
	stale.Labels = managedLabels(mcstest.ClusterSet)
	stale.OwnerReferences = []metav1.OwnerReference{{APIVersion: "v1", Kind: "Service", Name: "web", UID: "old-uid"}}
	c := mcstest.FakeClient(t, svc, stale)
	a := NewAutoExporter(AutoExporterOptions{Client: c, Identity: mcstest.Identity()})

	t.Log("The re-created Service becomes its only Service owner")
	autoReconcile(t, a, "web")
	se, ok := getExport(t, c, "web")
	require.True(t, ok)
	require.Len(t, se.OwnerReferences, 1)
	assert.Equal(t, svc.UID, se.OwnerReferences[0].UID)

	t.Log("A namespace the controller does not watch is left alone")
	a.opts.WatchesNamespace = func(string) bool { return false }
	require.NoError(t, c.Delete(mcstest.Ctx(t), svc))
	autoReconcile(t, a, "web")
	_, ok = getExport(t, c, "web")
	assert.True(t, ok)
}
