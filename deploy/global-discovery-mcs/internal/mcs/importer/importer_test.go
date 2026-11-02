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
	"google.golang.org/protobuf/types/known/timestamppb"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	discoveryv1ac "k8s.io/client-go/applyconfigurations/discovery/v1"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/event"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/mcstest"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
)

type ownCall struct {
	key string
	st  *mcsv1.ClusterStatus
	gen uint64
}

type env struct {
	k8s ctrlclient.WithWatch
	imp *Importer
	reg *prometheus.Registry
	rec *events.FakeRecorder

	mu  sync.Mutex
	own []ownCall
}

func newEnv(t *testing.T, opts Options, objs ...ctrlclient.Object) *env {
	t.Helper()
	return newEnvWith(t, opts, interceptor.Funcs{}, objs...)
}

func newEnvWith(t *testing.T, opts Options, funcs interceptor.Funcs, objs ...ctrlclient.Object) *env {
	t.Helper()
	e := &env{k8s: mcstest.FakeClientWith(t, funcs, objs...), reg: prometheus.NewRegistry(), rec: events.NewFakeRecorder(100)}
	m := metrics.New()
	require.NoError(t, m.Register(e.reg))
	opts.Client, opts.Metrics, opts.Logger, opts.Recorder = e.k8s, m, testr.New(t), e.rec
	opts.Identity = mcstest.Identity()
	opts.OwnStatus = func(namespace, name string, st *mcsv1.ClusterStatus, gen uint64) {
		e.mu.Lock()
		defer e.mu.Unlock()
		e.own = append(e.own, ownCall{namespace + "/" + name, st, gen})
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
	assert.Equal(t, derived+"-a-v4-0", slices[0].Name)
	assert.Equal(t, derived, slices[0].Labels[discoveryv1.LabelServiceName])
	assert.Equal(t, "a", slices[0].Labels[api.LabelMCSSourceCluster])
	assert.Equal(t, api.EndpointSliceManagedBy, slices[0].Labels[api.LabelEndpointSliceManagedBy])
	assert.Equal(t, si.UID, slices[0].OwnerReferences[0].UID)
	assert.Equal(t, "b", slices[2].Labels[api.LabelMCSSourceCluster])
	assert.Equal(t, 1.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_imports", nil))
	assert.Equal(t, 1.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_import_events_total", map[string]string{"type": "upsert"}))

	// Our cluster's status and endpoint generation went to the export side.
	calls := e.ownCalls()
	require.Len(t, calls, 1)
	assert.Equal(t, testNS+"/frontend", calls[0].key)
	assert.Equal(t, "a", calls[0].st.GetCluster())
	assert.Equal(t, uint64(7), calls[0].gen)

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

	// Checked again after a delay: the Namespace informer reports the
	// deletion, but the requeue covers a missed event.
	res, err := e.imp.Reconcile(mcstest.Ctx(t), ctrl.Request{Namespace: testNS, Name: "frontend"})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{RequeueAfter: TerminatingNamespaceRequeue}, res)

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
	res, err := e.imp.Reconcile(mcstest.Ctx(t), ctrl.Request{Namespace: testNS, Name: "frontend"})
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{RequeueAfter: ForeignImportRequeue}, res, "checked again as a backstop to the watch")

	// The foreign ServiceImport goes away: the import is applied with no
	// registry change.
	require.NoError(t, e.k8s.Delete(mcstest.Ctx(t), foreign))
	require.NoError(t, e.reconcile(t, testNS, "frontend"))
	si, _ = e.serviceImport(t, "frontend")
	assert.Equal(t, api.ManagedByController, si.Labels[api.LabelManagedBy])
	_, ok = e.derived(t, "frontend")
	assert.True(t, ok)
	require.NoError(t, e.k8s.Delete(mcstest.Ctx(t), si))
	var svc corev1.Service
	require.NoError(t, e.k8s.Get(mcstest.Ctx(t), types.NamespacedName{Namespace: testNS, Name: api.DerivedServiceName(testNS, "frontend")}, &svc))
	require.NoError(t, e.k8s.Delete(mcstest.Ctx(t), &svc))
	require.NoError(t, e.k8s.Create(mcstest.Ctx(t), &mcsv1beta1.ServiceImport{
		Namespace: testNS, Name: "frontend", Labels: map[string]string{"owner": "someone-else"},
		Spec: mcsv1beta1.ServiceImportSpec{Type: mcsv1beta1.ClusterSetIP, Ports: []mcsv1beta1.ServicePort{{Port: 1}}},
	}))
	e.imp.forgetApplied(types.NamespacedName{Namespace: testNS, Name: "frontend"})
	require.NoError(t, e.reconcile(t, testNS, "frontend"))
	_, ok = e.derived(t, "frontend")
	assert.False(t, ok, "blocked again by a new foreign ServiceImport")

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
	assert.Equal(t, []string{"10.96.0.10", "fd00::10"}, si.Spec.IPs, "from the answer of Apply, without another read")
	assert.Equal(t, []string{"10.96.0.10"}, clusterIPs(nil, "10.96.0.10"))
	assert.Empty(t, clusterIPs([]string{corev1.ClusterIPNone}, corev1.ClusterIPNone))
}

// countingReader counts the reads that bypass the cache.
type countingReader struct {
	ctrlclient.Reader
	gets, lists atomic.Int32
}

func (r *countingReader) Get(ctx context.Context, key ctrlclient.ObjectKey, obj ctrlclient.Object, opts ...ctrlclient.GetOption) error {
	r.gets.Add(1)
	return r.Reader.Get(ctx, key, obj, opts...)
}

func (r *countingReader) List(ctx context.Context, list ctrlclient.ObjectList, opts ...ctrlclient.ListOption) error {
	r.lists.Add(1)
	return r.Reader.List(ctx, list, opts...)
}

// The ClusterIPs come from the answer of Apply: a first creation and a
// re-created Service need no uncached read.
func TestClusterIPsNeedNoUncachedRead(t *testing.T) {
	funcs := interceptor.Funcs{
		Apply: func(ctx context.Context, c ctrlclient.WithWatch, obj runtime.ApplyConfiguration, opts ...ctrlclient.ApplyOption) error {
			if svc, ok := obj.(*corev1ac.ServiceApplyConfiguration); ok && svc.Spec.ClusterIP == nil {
				ip := "10.96.0.10"
				if len(svc.Spec.IPFamilies) > 0 && svc.Spec.IPFamilies[0] == corev1.IPv6Protocol {
					ip = "fd00::10"
				}
				svc.Spec.WithClusterIP(ip).WithClusterIPs(ip)
			}
			return c.Apply(ctx, obj, opts...)
		},
	}
	e := newEnvWith(t, Options{ClusterIPFamilies: []corev1.IPFamily{corev1.IPv4Protocol, corev1.IPv6Protocol}}, funcs, testNamespace())
	reader := &countingReader{Reader: e.k8s}
	e.imp.opts.APIReader = reader
	e.sync(t, testImport())
	si, _ := e.serviceImport(t, "frontend")
	assert.Equal(t, []string{"10.96.0.10"}, si.Spec.IPs)

	// The primary family flips: the derived Service is re-created and the
	// new address lands in spec.ips.
	imp := testImport()
	imp.IpFamilies = []mcsv1.IPFamily{mcsv1.IPFamily_IP_FAMILY_IPV6}
	e.imp.Upsert(imp)
	e.reconcileAll(t)
	si, _ = e.serviceImport(t, "frontend")
	assert.Equal(t, []string{"fd00::10"}, si.Spec.IPs)
	svc, _ := e.derived(t, "frontend")
	assert.Equal(t, []corev1.IPFamily{corev1.IPv6Protocol}, svc.Spec.IPFamilies)
	assert.Zero(t, reader.gets.Load(), "no uncached Get")
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

// The hash of an import covers what the controller renders: a resync in an
// exporting cluster, which bumps timestamps and generations, does not
// re-apply an unchanged import.
func TestImportHashIgnoresTimestampsAndGenerations(t *testing.T) {
	a, b := testImport(), testImport()
	b.UpdatedAt = timestamppb.Now()
	b.Clusters[0].ExportGeneration = 99
	b.Clusters[0].EndpointCount, b.Clusters[0].LeaseState = 5, mcsv1.LeaseState_LEASE_STATE_ACTIVE
	b.Clusters[0].Conditions = []*mcsv1.Condition{{Type: "Ready", Reason: "Ready"}}
	b.Endpoints[0].Generation = 42
	b.Endpoints[0].EndpointCount = 3
	assert.Equal(t, importHash(a), importHash(b))

	c := testImport()
	c.Endpoints[0].Endpoints[0].Conditions.Ready = false
	assert.NotEqual(t, importHash(a), importHash(c), "an endpoint condition is rendered")
	d := testImport()
	d.Clusters[2].Participating = true
	assert.NotEqual(t, importHash(a), importHash(d), "participation is rendered in the status")
	f := testImport()
	f.Endpoints[0].Region = "eu-west-1"
	assert.NotEqual(t, importHash(a), importHash(f), "the region is a slice label")

	var applies atomic.Int32
	funcs := interceptor.Funcs{
		Apply: func(ctx context.Context, c ctrlclient.WithWatch, obj runtime.ApplyConfiguration, opts ...ctrlclient.ApplyOption) error {
			applies.Add(1)
			return c.Apply(ctx, obj, opts...)
		},
	}
	e := newEnvWith(t, Options{}, funcs, testNamespace())
	e.sync(t, a)
	n := applies.Load()
	e.imp.Upsert(b)
	e.reconcileAll(t)
	assert.Equal(t, n, applies.Load(), "a resync of an exporting cluster costs no apply")
}

// A pending import deleted from the registry while the watch was down is
// dropped by the next snapshot: it has no managed objects for the garbage
// collection to find.
func TestPendingImportGoneFromSnapshotIsDropped(t *testing.T) {
	e := newEnv(t, Options{})
	e.sync(t, testImport())
	require.Len(t, e.imp.Pending(), 1)
	e.sync(t)
	assert.Empty(t, e.imp.Pending())
	assert.Equal(t, 0.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_imports_pending_namespace", nil))
}

// An apply that started before a managed object was deleted (forgetApplied)
// does not record its hash, so the deleted object is re-created by the next
// reconcile instead of being skipped as applied.
func TestForgetDuringApplyIsNotRecorded(t *testing.T) {
	e := newEnv(t, Options{}, testNamespace())
	key := types.NamespacedName{Namespace: testNS, Name: "frontend"}
	hash, epoch := e.imp.appliedHash(key)
	require.Empty(t, hash)
	e.imp.forgetApplied(key)
	e.imp.setApplied(key, "h1", epoch)
	got, epoch2 := e.imp.appliedHash(key)
	assert.Empty(t, got, "the apply started before the forget")
	e.imp.setApplied(key, "h2", epoch2)
	got, _ = e.imp.appliedHash(key)
	assert.Equal(t, "h2", got)

	// End to end: the derived Service is deleted during the apply.
	var once sync.Once
	funcs := interceptor.Funcs{
		Apply: func(ctx context.Context, c ctrlclient.WithWatch, obj runtime.ApplyConfiguration, opts ...ctrlclient.ApplyOption) error {
			err := c.Apply(ctx, obj, opts...)
			if _, ok := obj.(*discoveryv1ac.EndpointSliceApplyConfiguration); ok && err == nil {
				once.Do(func() {
					// The watch reports the deletion while the slices are applied.
					svc := &corev1.Service{Namespace: testNS, Name: api.DerivedServiceName(testNS, "frontend")}
					require.NoError(t, c.Delete(ctx, svc))
					e.imp.forgetApplied(key)
				})
			}
			return err
		},
	}
	e = newEnvWith(t, Options{}, funcs, testNamespace())
	e.sync(t, testImport())
	_, ok := e.derived(t, "frontend")
	require.False(t, ok, "deleted during the apply")
	e.imp.Upsert(testImport())
	e.reconcileAll(t)
	_, ok = e.derived(t, "frontend")
	assert.True(t, ok, "re-created: the apply that raced the deletion did not record its hash")
}

// An import whose objects cannot be applied reports it: Ready=False with
// reason ApplyFailed and a Warning Event, cleared by the next success.
func TestApplyFailureIsReportedOnTheServiceImport(t *testing.T) {
	e := newEnv(t, Options{}, testNamespace())
	e.sync(t, testImport())

	bad := testImport()
	bad.Ports = append(bad.Ports, &mcsv1.ServicePort{Protocol: mcsv1.Protocol_PROTOCOL_TCP, Port: 81})
	e.imp.Upsert(bad)
	for _, key := range e.drain() {
		ns, name, _ := cut(key)
		require.ErrorContains(t, e.reconcile(t, ns, name), "not valid Service ports")
	}
	si, _ := e.serviceImport(t, "frontend")
	ready := meta.FindStatusCondition(si.Status.Conditions, string(mcsv1beta1.ServiceImportConditionReady))
	require.NotNil(t, ready)
	assert.Equal(t, metav1.ConditionFalse, ready.Status)
	assert.Equal(t, api.ConditionReasonApplyFailed, ready.Reason)
	assert.Contains(t, ready.Message, "derived Service")
	assert.Contains(t, ready.Message, "81/TCP has no name")
	assert.NotNil(t, meta.FindStatusCondition(si.Status.Conditions, api.ConditionConflict), "the other conditions stay")
	select {
	case ev := <-e.rec.Events:
		assert.Contains(t, ev, EventApplyFailed)
		assert.Contains(t, ev, "Warning")
	default:
		t.Fatal("no Event")
	}
	svc, _ := e.derived(t, "frontend")
	assert.Len(t, svc.Spec.Ports, 1, "the invalid ports were not applied")

	// The next successful apply clears the condition.
	e.imp.Upsert(testImport())
	e.reconcileAll(t)
	si, _ = e.serviceImport(t, "frontend")
	ready = meta.FindStatusCondition(si.Status.Conditions, string(mcsv1beta1.ServiceImportConditionReady))
	assert.Equal(t, metav1.ConditionTrue, ready.Status)
	assert.Equal(t, api.ConditionReasonReady, ready.Reason)
}

// Slices the previous apply wrote and the current one does not are deleted
// by name, even when the informer cache does not know them yet; a shrinking
// slice count also lists the API server instead of the cache.
func TestStaleSlicesDeletedWhenTheCacheLags(t *testing.T) {
	var lagging atomic.Bool
	funcs := interceptor.Funcs{
		List: func(ctx context.Context, c ctrlclient.WithWatch, list ctrlclient.ObjectList, opts ...ctrlclient.ListOption) error {
			if _, ok := list.(*discoveryv1.EndpointSliceList); ok && lagging.Load() {
				// The cache has not seen any slice yet.
				return nil
			}
			return c.List(ctx, list, opts...)
		},
	}
	e := newEnvWith(t, Options{EndpointsPerSlice: 1}, funcs, testNamespace())
	reader := &countingReader{Reader: e.k8s}
	e.imp.opts.APIReader = reader
	one := testImport()
	one.Endpoints = []*mcsv1.ClusterEndpoints{{Cluster: "a", Endpoints: []*mcsv1.Endpoint{endpoint("10.0.0.1", true, httpPort)}}}
	e.sync(t, one)
	require.Len(t, e.slices(t, "frontend"), 1)

	lagging.Store(true)
	three := testImport()
	three.Endpoints = []*mcsv1.ClusterEndpoints{{Cluster: "a", Endpoints: []*mcsv1.Endpoint{
		endpoint("10.0.0.1", true, httpPort), endpoint("10.0.0.2", true, httpPort), endpoint("10.0.0.3", true, httpPort),
	}}}
	e.imp.Upsert(three)
	e.reconcileAll(t)
	lagging.Store(false)
	require.Len(t, e.slices(t, "frontend"), 3)
	lagging.Store(true)
	e.imp.Upsert(one)
	e.reconcileAll(t)
	lagging.Store(false)
	slices := e.slices(t, "frontend")
	require.Len(t, slices, 1, "the two slices the cache never showed are gone")
	assert.Equal(t, api.DerivedServiceName(testNS, "frontend")+"-a-v4-0", slices[0].Name)
	assert.Equal(t, int32(1), reader.lists.Load(), "the shrink listed the API server")
	assert.Equal(t, 2.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_gc_deletions_total", map[string]string{"kind": "endpoint_slice"}))
}

// The periodic resync forgets every applied hash and reconciles every
// import, repairing drift the watches did not report.
func TestResyncReappliesEveryImport(t *testing.T) {
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
	svc, _ := e.derived(t, "frontend")
	svc.Spec.Ports[0].Port = 9999
	require.NoError(t, e.k8s.Update(mcstest.Ctx(t), svc))

	e.imp.Resync()
	assert.Equal(t, []string{testNS + "/frontend"}, e.drain())
	require.NoError(t, e.reconcile(t, testNS, "frontend"))
	assert.Greater(t, applies.Load(), n)
	svc, _ = e.derived(t, "frontend")
	assert.Equal(t, int32(80), svc.Spec.Ports[0].Port, "drift repaired")
	assert.Equal(t, 1.0, mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_resyncs_total", map[string]string{"reason": "import_periodic"}))

	// The periodic runner fires on its own.
	e.imp.opts.ResyncPeriod = 5 * time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go e.imp.periodicResync(ctx)
	require.Eventually(t, func() bool {
		return mcstest.MetricValue(t, e.reg, "dynamo_mcs_controller_resyncs_total", map[string]string{"reason": "import_periodic"}) >= 3
	}, 5*time.Second, time.Millisecond)
}

// The controller's own writes are not drift; foreign changes to what it
// renders are.
func TestDriftPredicates(t *testing.T) {
	e := newEnv(t, Options{}, testNamespace())
	e.sync(t, testImport())
	key := types.NamespacedName{Namespace: testNS, Name: "frontend"}
	svc, _ := e.derived(t, "frontend")
	si, _ := e.serviceImport(t, "frontend")
	slice := e.slices(t, "frontend")[0]
	assert.True(t, e.imp.wrote(key, writtenService, svc.ResourceVersion))
	assert.True(t, e.imp.wrote(key, writtenImport, si.ResourceVersion))
	assert.True(t, e.imp.wrote(key, writtenSlice+slice.Name, slice.ResourceVersion))

	own := event.UpdateEvent{ObjectOld: svc.DeepCopy(), ObjectNew: svc}
	assert.False(t, e.imp.driftByLabel(own), "the controller's own write")
	edited := svc.DeepCopy()
	edited.ResourceVersion = "foreign"
	edited.Spec.Ports[0].Port = 9999
	assert.True(t, e.imp.driftByLabel(event.UpdateEvent{ObjectOld: svc, ObjectNew: edited}))
	relabelled := svc.DeepCopy()
	relabelled.ResourceVersion = "foreign"
	relabelled.Labels["x"] = "y"
	assert.True(t, e.imp.driftByLabel(event.UpdateEvent{ObjectOld: svc, ObjectNew: relabelled}))
	noop := svc.DeepCopy()
	noop.ResourceVersion = "foreign"
	noop.Annotations = map[string]string{"note": "x"}
	assert.False(t, e.imp.driftByLabel(event.UpdateEvent{ObjectOld: svc, ObjectNew: noop}), "nothing rendered changed")
	unmanaged := mcstest.Service(testNS, "theirs", nil)
	assert.False(t, e.imp.driftByLabel(event.UpdateEvent{ObjectOld: unmanaged, ObjectNew: unmanaged}))

	moved := slice.DeepCopy()
	moved.ResourceVersion = "foreign"
	moved.Endpoints[0].Addresses = []string{"192.0.2.1"}
	assert.True(t, e.imp.driftByLabel(event.UpdateEvent{ObjectOld: &slice, ObjectNew: moved}))
	assert.False(t, e.imp.driftByLabel(event.UpdateEvent{ObjectOld: slice.DeepCopy(), ObjectNew: &slice}))

	siEdited := si.DeepCopy()
	siEdited.ResourceVersion = "foreign"
	siEdited.Spec.Ports = nil
	assert.True(t, e.imp.serviceImportChanged(event.UpdateEvent{ObjectOld: si, ObjectNew: siEdited}))
	assert.False(t, e.imp.serviceImportChanged(event.UpdateEvent{ObjectOld: si.DeepCopy(), ObjectNew: si}))
	foreign := &mcsv1beta1.ServiceImport{Namespace: testNS, Name: "other", Labels: map[string]string{"owner": "x"}}
	released := foreign.DeepCopy()
	released.Labels = nil
	assert.True(t, e.imp.serviceImportChanged(event.UpdateEvent{ObjectOld: foreign, ObjectNew: released}), "an ownership label change of a foreign import")
	assert.False(t, e.imp.serviceImportChanged(event.UpdateEvent{ObjectOld: foreign, ObjectNew: foreign.DeepCopy()}))
	assert.False(t, e.imp.serviceImportChanged(event.UpdateEvent{ObjectOld: svc, ObjectNew: svc}), "not a ServiceImport")
}

// Narrowing --watch-namespaces leaves managed objects in namespaces the
// caches no longer cover; the sweep deletes them.
func TestSweepUnwatchedDeletesManagedObjects(t *testing.T) {
	e := newEnv(t, Options{WatchNamespaces: []string{testNS}}, testNamespace(), &corev1.Namespace{Name: "other"})
	e.sync(t, testImport())
	e.imp.opts.WatchNamespaces = nil
	other := testImport()
	other.Namespace = "other"
	e.imp.Upsert(other)
	e.reconcileAll(t)
	e.imp.opts.WatchNamespaces = []string{testNS}
	var list discoveryv1.EndpointSliceList
	require.NoError(t, e.k8s.List(mcstest.Ctx(t), &list, ctrlclient.InNamespace("other")))
	require.NotEmpty(t, list.Items)

	require.NoError(t, e.imp.SweepUnwatched(mcstest.Ctx(t)))
	require.NoError(t, e.k8s.List(mcstest.Ctx(t), &list, ctrlclient.InNamespace("other")))
	assert.Empty(t, list.Items)
	err := e.k8s.Get(mcstest.Ctx(t), types.NamespacedName{Namespace: "other", Name: "frontend"}, &mcsv1beta1.ServiceImport{})
	assert.True(t, apierrors.IsNotFound(err))
	err = e.k8s.Get(mcstest.Ctx(t), types.NamespacedName{Namespace: "other", Name: api.DerivedServiceName("other", "frontend")}, &corev1.Service{})
	assert.True(t, apierrors.IsNotFound(err))
	_, ok := e.serviceImport(t, "frontend")
	assert.True(t, ok, "the watched namespace is untouched")
	assert.Len(t, e.slices(t, "frontend"), 2)

	e.imp.opts.WatchNamespaces = nil
	require.NoError(t, e.imp.SweepUnwatched(mcstest.Ctx(t)), "every namespace watched: nothing to do")
}

func TestDetectClusterIPFamilies(t *testing.T) {
	ctx := mcstest.Ctx(t)
	fams, err := DetectClusterIPFamilies(ctx, mcstest.FakeClient(t))
	require.NoError(t, err)
	assert.Nil(t, fams, "no kubernetes Service: unknown")

	k8s := &corev1.Service{Namespace: "default", Name: "kubernetes", Spec: corev1.ServiceSpec{
		ClusterIP: "10.96.0.1", ClusterIPs: []string{"10.96.0.1", "fd00::1"},
	}}
	fams, err = DetectClusterIPFamilies(ctx, mcstest.FakeClient(t, k8s))
	require.NoError(t, err)
	assert.Equal(t, []corev1.IPFamily{corev1.IPv4Protocol, corev1.IPv6Protocol}, fams, "inferred from the ClusterIPs")

	k8s.Spec.IPFamilies = []corev1.IPFamily{corev1.IPv6Protocol}
	fams, err = DetectClusterIPFamilies(ctx, mcstest.FakeClient(t, k8s))
	require.NoError(t, err)
	assert.Equal(t, []corev1.IPFamily{corev1.IPv6Protocol}, fams)

	boom := errors.New("boom")
	failing := mcstest.FakeClientWith(t, interceptor.Funcs{Get: func(context.Context, ctrlclient.WithWatch, ctrlclient.ObjectKey, ctrlclient.Object, ...ctrlclient.GetOption) error {
		return boom
	}})
	_, err = DetectClusterIPFamilies(ctx, failing)
	assert.ErrorIs(t, err, boom)
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
