// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package importer

import (
	"context"
	"errors"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/mcstest"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
)

type ownCall struct {
	key string
	st  *mcsv1.ClusterStatus
}

type env struct {
	k8s ctrlclient.WithWatch
	imp *Importer
	reg *prometheus.Registry

	mu  sync.Mutex
	own []ownCall
}

func newEnv(t *testing.T, opts Options, objs ...ctrlclient.Object) *env {
	t.Helper()
	return newEnvWith(t, opts, interceptor.Funcs{}, objs...)
}

func newEnvWith(t *testing.T, opts Options, funcs interceptor.Funcs, objs ...ctrlclient.Object) *env {
	t.Helper()
	e := &env{k8s: mcstest.FakeClientWith(t, funcs, objs...), reg: prometheus.NewRegistry()}
	m := metrics.New()
	require.NoError(t, m.Register(e.reg))
	opts.Client, opts.Metrics, opts.Logger = e.k8s, m, testr.New(t)
	opts.Identity = mcstest.Identity()
	opts.OwnStatus = func(namespace, name string, st *mcsv1.ClusterStatus) {
		e.mu.Lock()
		defer e.mu.Unlock()
		e.own = append(e.own, ownCall{namespace + "/" + name, st})
	}
	e.imp = New(opts)
	return e
}

// testNamespace returns the namespace testNS.
func testNamespace() *corev1.Namespace {
	return &corev1.Namespace{Name: testNS}
}

func (e *env) ownCalls() []ownCall {
	e.mu.Lock()
	defer e.mu.Unlock()
	return append([]ownCall(nil), e.own...)
}

// drain returns the enqueued keys.
func (e *env) drain() []string {
	var keys []string
	for {
		select {
		case ev := <-e.imp.events:
			keys = append(keys, ev.Object.Namespace+"/"+ev.Object.Name)
		default:
			sort.Strings(keys)
			return keys
		}
	}
}

func (e *env) reconcile(t *testing.T, ns, name string) error {
	t.Helper()
	_, err := e.imp.Reconcile(mcstest.Ctx(t), ctrl.Request{Namespace: ns, Name: name})
	return err
}

// sync delivers a snapshot with imps and reconciles everything enqueued.
func (e *env) sync(t *testing.T, imps ...*mcsv1.ServiceImport) {
	t.Helper()
	e.imp.SnapshotStart()
	for _, imp := range imps {
		e.imp.Upsert(imp)
	}
	e.imp.Synced("rev")
	e.reconcileAll(t)
}

func (e *env) reconcileAll(t *testing.T) {
	t.Helper()
	for _, key := range e.drain() {
		ns, name, _ := cut(key)
		require.NoError(t, e.reconcile(t, ns, name), key)
	}
}

func cut(key string) (string, string, bool) {
	for i := range key {
		if key[i] == '/' {
			return key[:i], key[i+1:], true
		}
	}
	return "", key, false
}

func (e *env) serviceImport(t *testing.T, name string) (*mcsv1beta1.ServiceImport, bool) {
	t.Helper()
	var si mcsv1beta1.ServiceImport
	err := e.k8s.Get(mcstest.Ctx(t), types.NamespacedName{Namespace: testNS, Name: name}, &si)
	if apierrors.IsNotFound(err) {
		return nil, false
	}
	require.NoError(t, err)
	return &si, true
}

func (e *env) derived(t *testing.T, name string) (*corev1.Service, bool) {
	t.Helper()
	var svc corev1.Service
	err := e.k8s.Get(mcstest.Ctx(t), types.NamespacedName{Namespace: testNS, Name: api.DerivedServiceName(testNS, name)}, &svc)
	if apierrors.IsNotFound(err) {
		return nil, false
	}
	require.NoError(t, err)
	return &svc, true
}

func (e *env) slices(t *testing.T, name string) []discoveryv1.EndpointSlice {
	t.Helper()
	var list discoveryv1.EndpointSliceList
	require.NoError(t, e.k8s.List(mcstest.Ctx(t), &list, ctrlclient.MatchingLabels{api.LabelMCSServiceName: name}))
	sort.Slice(list.Items, func(i, j int) bool { return list.Items[i].Name < list.Items[j].Name })
	return list.Items
}

func TestApplyCreatesImportDerivedServiceAndSlices(t *testing.T) {
	e := newEnv(t, Options{EndpointsPerSlice: 1}, testNamespace())
	imp := testImport()
	e.sync(t, imp)

	si, ok := e.serviceImport(t, "frontend")
	require.True(t, ok)
	assert.Equal(t, api.ManagedByController, si.Labels[api.LabelManagedBy])
	assert.Equal(t, mcstest.ClusterSet, si.Labels[api.LabelClusterSet])
	assert.Equal(t, "gold", si.Labels[api.Prefix+"tier"])
	assert.Equal(t, api.DerivedServiceName(testNS, "frontend"), si.Annotations[api.AnnotationMCSDerivedService])
	assert.Equal(t, mcsv1beta1.ClusterSetIP, si.Spec.Type)
	assert.Equal(t, []mcsv1beta1.ClusterStatus{{Cluster: "a"}, {Cluster: "b"}}, si.Status.Clusters)
	require.NotEmpty(t, si.Status.Conditions)
	assert.Equal(t, metav1.ConditionTrue, si.Status.Conditions[0].Status)

	svc, ok := e.derived(t, "frontend")
	require.True(t, ok)
	assert.Empty(t, svc.Spec.Selector)
	assert.Equal(t, int32(80), svc.Spec.Ports[0].Port)
	require.Len(t, svc.OwnerReferences, 1)
	assert.Equal(t, "ServiceImport", svc.OwnerReferences[0].Kind)
	assert.Equal(t, si.UID, svc.OwnerReferences[0].UID)

	slices := e.slices(t, "frontend")
	require.Len(t, slices, 3, "one endpoint per slice, two clusters")
	derived := api.DerivedServiceName(testNS, "frontend")
	assert.Equal(t, derived+"-a-0", slices[0].Name)
	assert.Equal(t, derived, slices[0].Labels[discoveryv1.LabelServiceName])
	assert.Equal(t, "a", slices[0].Labels[api.LabelMCSSourceCluster])
	assert.Equal(t, api.EndpointSliceManagedBy, slices[0].Labels[api.LabelEndpointSliceManagedBy])
	assert.Equal(t, si.UID, slices[0].OwnerReferences[0].UID)
	assert.Equal(t, "b", slices[2].Labels[api.LabelMCSSourceCluster])
	assert.Equal(t, 1.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_imports", nil))
	assert.Equal(t, 1.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_import_events_total", map[string]string{"type": "upsert"}))

	// Our cluster's status went to the export side.
	calls := e.ownCalls()
	require.Len(t, calls, 1)
	assert.Equal(t, testNS+"/frontend", calls[0].key)
	assert.Equal(t, "a", calls[0].st.GetCluster())

	// Re-applying the same import leaves the status alone.
	conds := si.Status.Conditions
	e.imp.Upsert(imp)
	e.reconcileAll(t)
	si, _ = e.serviceImport(t, "frontend")
	assert.Equal(t, conds, si.Status.Conditions)

	// Cluster b leaves: its slice goes.
	imp.Endpoints = imp.Endpoints[:1]
	e.imp.Upsert(imp)
	e.reconcileAll(t)
	slices = e.slices(t, "frontend")
	require.Len(t, slices, 2)
	for _, s := range slices {
		assert.Equal(t, "a", s.Labels[api.LabelMCSSourceCluster])
	}
	assert.Equal(t, 1.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_gc_deletions_total", map[string]string{"kind": "endpoint_slice"}))
}

func TestDeleteCascades(t *testing.T) {
	e := newEnv(t, Options{}, testNamespace())
	e.sync(t, testImport())
	e.imp.Delete(testImport())
	e.reconcileAll(t)
	_, ok := e.serviceImport(t, "frontend")
	assert.False(t, ok)
	_, ok = e.derived(t, "frontend")
	assert.False(t, ok)
	assert.Empty(t, e.slices(t, "frontend"))
	assert.Equal(t, 1.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_gc_deletions_total", map[string]string{"kind": "service_import"}))
	assert.Equal(t, 1.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_gc_deletions_total", map[string]string{"kind": "service"}))
	assert.Equal(t, 2.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_gc_deletions_total", map[string]string{"kind": "endpoint_slice"}))
	calls := e.ownCalls()
	require.Len(t, calls, 2)
	assert.Nil(t, calls[1].st, "the export side learns the import is gone")
}

func TestSnapshotEndCollectsOrphans(t *testing.T) {
	e := newEnv(t, Options{}, testNamespace())
	orphan := testImport()
	orphan.Name = "orphan"
	e.sync(t, testImport(), orphan)
	_, ok := e.serviceImport(t, "orphan")
	require.True(t, ok)

	// A new snapshot without the orphan: before SNAPSHOT_END nothing is
	// deleted, after it the orphan is collected.
	e.imp.SnapshotStart()
	e.imp.Upsert(testImport())
	require.NoError(t, e.reconcile(t, testNS, "orphan"))
	_, ok = e.serviceImport(t, "orphan")
	assert.True(t, ok, "kept while the view is incomplete")
	e.imp.Synced("rev2")
	e.reconcileAll(t)
	_, ok = e.serviceImport(t, "orphan")
	assert.False(t, ok)
	_, ok = e.derived(t, "orphan")
	assert.False(t, ok)
	assert.Empty(t, e.slices(t, "orphan"))
	_, ok = e.serviceImport(t, "frontend")
	assert.True(t, ok)

	// Own statuses missing from a snapshot are cleared.
	e.imp.SnapshotStart()
	e.imp.Synced("rev3")
	calls := e.ownCalls()
	assert.Nil(t, calls[len(calls)-1].st)
	assert.Equal(t, 3.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_import_events_total", map[string]string{"type": "snapshot_end"}))
}

func TestOrphanSlicesAndServicesAreCollected(t *testing.T) {
	stray := mcstest.EndpointSlice(testNS, "stray", "derived-x")
	stray.Labels[api.LabelEndpointSliceManagedBy] = api.EndpointSliceManagedBy
	stray.Labels[api.LabelMCSServiceName] = "gone"
	svc := mcstest.Service(testNS, api.DerivedServiceName(testNS, "gone"), map[string]string{
		api.LabelManagedBy: api.ManagedByController, api.LabelMCSServiceName: "gone",
	})
	unmanaged := &mcsv1beta1.ServiceImport{Namespace: testNS, Name: "other-impl"}
	e := newEnv(t, Options{}, testNamespace(), stray, svc, unmanaged)
	e.sync(t)
	var list discoveryv1.EndpointSliceList
	require.NoError(t, e.k8s.List(mcstest.Ctx(t), &list))
	assert.Empty(t, list.Items)
	_, ok := e.derived(t, "gone")
	assert.False(t, ok)
	_, ok = e.serviceImport(t, "other-impl")
	assert.True(t, ok, "unmanaged imports are never collected")
}

func TestPendingNamespace(t *testing.T) {
	e := newEnv(t, Options{})
	e.sync(t, testImport())
	_, ok := e.serviceImport(t, "frontend")
	assert.False(t, ok)
	assert.Len(t, e.imp.Pending(), 1)
	assert.Equal(t, 1.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_imports_pending_namespace", nil))

	// The Namespace informer maps the new namespace to its imports.
	require.NoError(t, e.k8s.Create(mcstest.Ctx(t), testNamespace()))
	keys := e.imp.keysIn(testNS)
	require.Equal(t, []types.NamespacedName{{Namespace: testNS, Name: "frontend"}}, keys)
	require.NoError(t, e.reconcile(t, testNS, "frontend"))
	_, ok = e.serviceImport(t, "frontend")
	assert.True(t, ok)
	assert.Empty(t, e.imp.Pending())
	assert.Equal(t, 0.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_imports_pending_namespace", nil))
	assert.Empty(t, e.imp.keysIn("other"))
}

func TestTerminatingNamespaceIsPending(t *testing.T) {
	ns := testNamespace()
	ns.Finalizers = []string{"kubernetes"}
	e := newEnv(t, Options{}, ns)
	require.NoError(t, e.k8s.Delete(mcstest.Ctx(t), ns))
	e.sync(t, testImport())
	assert.Len(t, e.imp.Pending(), 1)

	// A pending import that the registry deletes is no longer pending.
	e.imp.Delete(testImport())
	e.reconcileAll(t)
	assert.Empty(t, e.imp.Pending())
}

func TestCreateNamespaces(t *testing.T) {
	e := newEnv(t, Options{CreateNamespaces: true})
	e.sync(t, testImport())
	var ns corev1.Namespace
	require.NoError(t, e.k8s.Get(mcstest.Ctx(t), types.NamespacedName{Name: testNS}, &ns))
	assert.Equal(t, api.ManagedByController, ns.Labels[api.LabelManagedBy])
	_, ok := e.serviceImport(t, "frontend")
	assert.True(t, ok)
}

func TestUnmanagedImportsAreSkippedUnlessAdopted(t *testing.T) {
	foreign := &mcsv1beta1.ServiceImport{
		Namespace: testNS, Name: "frontend", Labels: map[string]string{"owner": "someone-else"},
		Spec: mcsv1beta1.ServiceImportSpec{Type: mcsv1beta1.ClusterSetIP, Ports: []mcsv1beta1.ServicePort{{Port: 1}}},
	}
	e := newEnv(t, Options{}, testNamespace(), foreign)
	e.sync(t, testImport())
	si, _ := e.serviceImport(t, "frontend")
	assert.Empty(t, si.Labels[api.LabelManagedBy])
	_, ok := e.derived(t, "frontend")
	assert.False(t, ok)

	e.imp.opts.AdoptUnmanaged = true
	require.NoError(t, e.reconcile(t, testNS, "frontend"))
	si, _ = e.serviceImport(t, "frontend")
	assert.Equal(t, api.ManagedByController, si.Labels[api.LabelManagedBy])
	assert.Equal(t, int32(80), si.Spec.Ports[0].Port)
}

func TestDerivedServiceConflictsAndTypeChanges(t *testing.T) {
	squatter := mcstest.Service(testNS, api.DerivedServiceName(testNS, "frontend"), nil)
	var serviceDeletes atomic.Int32
	funcs := interceptor.Funcs{
		Delete: func(ctx context.Context, c ctrlclient.WithWatch, obj ctrlclient.Object, opts ...ctrlclient.DeleteOption) error {
			if _, ok := obj.(*corev1.Service); ok {
				serviceDeletes.Add(1)
			}
			return c.Delete(ctx, obj, opts...)
		},
	}
	e := newEnvWith(t, Options{}, funcs, testNamespace(), squatter)
	e.imp.SnapshotStart()
	e.imp.Upsert(testImport())
	e.imp.Synced("r")
	err := e.reconcile(t, testNS, "frontend")
	require.ErrorContains(t, err, "not managed")
	require.NoError(t, e.k8s.Delete(mcstest.Ctx(t), squatter))
	serviceDeletes.Store(0)

	require.NoError(t, e.reconcile(t, testNS, "frontend"))
	assert.Zero(t, serviceDeletes.Load())

	// ClusterSetIP to Headless re-creates the derived Service, since
	// clusterIP is immutable.
	imp := testImport()
	imp.Type = mcsv1.ServiceType_SERVICE_TYPE_HEADLESS
	e.imp.Upsert(imp)
	e.reconcileAll(t)
	assert.Equal(t, int32(1), serviceDeletes.Load())
	svc, _ := e.derived(t, "frontend")
	assert.Equal(t, corev1.ClusterIPNone, svc.Spec.ClusterIP)
	si, _ := e.serviceImport(t, "frontend")
	assert.Equal(t, mcsv1beta1.Headless, si.Spec.Type)
	assert.Empty(t, si.Spec.IPs)
	e.imp.Upsert(imp)
	e.reconcileAll(t)
	assert.Equal(t, int32(1), serviceDeletes.Load())
}

func TestClusterIPsBecomeImportIPs(t *testing.T) {
	// The fake client allocates no ClusterIP; an interceptor plays the API
	// server.
	funcs := interceptor.Funcs{
		Apply: func(ctx context.Context, c ctrlclient.WithWatch, obj runtime.ApplyConfiguration, opts ...ctrlclient.ApplyOption) error {
			if svc, ok := obj.(*corev1ac.ServiceApplyConfiguration); ok && svc.Spec.ClusterIP == nil {
				svc.Spec.WithClusterIP("10.96.0.10").WithClusterIPs("10.96.0.10", "fd00::10", "extra")
			}
			return c.Apply(ctx, obj, opts...)
		},
	}
	e := newEnvWith(t, Options{}, funcs, testNamespace())
	e.sync(t, testImport())
	si, _ := e.serviceImport(t, "frontend")
	assert.Equal(t, []string{"10.96.0.10", "fd00::10"}, si.Spec.IPs)
	assert.Equal(t, []string{"10.96.0.10"}, clusterIPs(&corev1.Service{Spec: corev1.ServiceSpec{ClusterIP: "10.96.0.10"}}))
	assert.Empty(t, clusterIPs(&corev1.Service{Spec: corev1.ServiceSpec{ClusterIP: corev1.ClusterIPNone}}))
}

func TestApplyErrors(t *testing.T) {
	boom := errors.New("boom")
	for _, tt := range []struct {
		name  string
		funcs interceptor.Funcs
	}{
		{"get", interceptor.Funcs{Get: func(context.Context, ctrlclient.WithWatch, ctrlclient.ObjectKey, ctrlclient.Object, ...ctrlclient.GetOption) error {
			return boom
		}}},
		{"apply", interceptor.Funcs{Apply: func(context.Context, ctrlclient.WithWatch, runtime.ApplyConfiguration, ...ctrlclient.ApplyOption) error {
			return boom
		}}},
		{"status", interceptor.Funcs{SubResourcePatch: func(context.Context, ctrlclient.Client, string, ctrlclient.Object, ctrlclient.Patch, ...ctrlclient.SubResourcePatchOption) error {
			return boom
		}}},
		{"list", interceptor.Funcs{List: func(context.Context, ctrlclient.WithWatch, ctrlclient.ObjectList, ...ctrlclient.ListOption) error {
			return boom
		}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			e := newEnvWith(t, Options{}, tt.funcs, testNamespace())
			e.imp.SnapshotStart()
			e.imp.Upsert(testImport())
			e.imp.Synced("r")
			assert.Error(t, e.reconcile(t, testNS, "frontend"))
			e.imp.Delete(testImport())
			if tt.name != "apply" && tt.name != "status" {
				assert.Error(t, e.reconcile(t, testNS, "frontend"))
			}
		})
	}
}

func TestWatchNamespacesFilter(t *testing.T) {
	e := newEnv(t, Options{WatchNamespaces: []string{"other"}}, testNamespace())
	e.sync(t, testImport())
	_, ok := e.serviceImport(t, "frontend")
	assert.False(t, ok)
	assert.Empty(t, e.drain())
	require.NoError(t, e.reconcile(t, testNS, "frontend"))
	assert.Len(t, e.ownCalls(), 1, "own statuses are reported for every namespace")
}

func TestStartWatchesRegistry(t *testing.T) {
	fakeReg := mcstest.StartRegistry(t)
	e := newEnv(t, Options{Registry: fakeReg.Client, WatchNamespaces: []string{testNS}, IdleTimeout: time.Second}, testNamespace())
	assert.True(t, e.imp.NeedLeaderElection())
	fakeReg.Server.PutExport(mcstest.ClusterSet, "b", testNS, "frontend", &mcsv1.ServiceExportSpec{
		Type:  mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP,
		Ports: []*mcsv1.ServicePort{{Name: "http", Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: 80}},
	}, []*mcsv1.Endpoint{endpoint("10.1.0.1", true, httpPort)})

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- e.imp.Start(ctx) }()
	defer func() {
		cancel()
		require.NoError(t, <-done)
	}()
	var got *mcsv1beta1.ServiceImport
	require.Eventually(t, func() bool {
		e.reconcileAll(t)
		si, ok := e.serviceImport(t, "frontend")
		got = si
		return ok
	}, 5*time.Second, 10*time.Millisecond)
	assert.Equal(t, []mcsv1beta1.ClusterStatus{{Cluster: "b"}}, got.Status.Clusters)
	require.Eventually(t, func() bool {
		e.reconcileAll(t)
		return len(e.slices(t, "frontend")) == 1
	}, 5*time.Second, 10*time.Millisecond)

	fakeReg.Server.DeleteExport(mcstest.ClusterSet, "b", testNS, "frontend")
	require.Eventually(t, func() bool {
		e.reconcileAll(t)
		_, ok := e.serviceImport(t, "frontend")
		return !ok
	}, 5*time.Second, 10*time.Millisecond)
}

func TestUnchangedImportsAreNotReapplied(t *testing.T) {
	var applies atomic.Int32
	funcs := interceptor.Funcs{
		Apply: func(ctx context.Context, c ctrlclient.WithWatch, obj runtime.ApplyConfiguration, opts ...ctrlclient.ApplyOption) error {
			applies.Add(1)
			return c.Apply(ctx, obj, opts...)
		},
	}
	e := newEnvWith(t, Options{}, funcs, testNamespace())
	e.sync(t, testImport())
	n := applies.Load()
	require.Positive(t, n)

	// A snapshot after a reconnect delivers the same import: no writes.
	e.sync(t, testImport())
	assert.Equal(t, n, applies.Load())

	// A deleted managed object makes the importer forget and re-apply.
	e.imp.forgetApplied(types.NamespacedName{Namespace: testNS, Name: "frontend"})
	e.imp.Upsert(testImport())
	e.reconcileAll(t)
	assert.Greater(t, applies.Load(), n)

	// A changed import is applied.
	n = applies.Load()
	imp := testImport()
	imp.Endpoints = imp.Endpoints[:1]
	e.imp.Upsert(imp)
	e.reconcileAll(t)
	assert.Greater(t, applies.Load(), n)
}
