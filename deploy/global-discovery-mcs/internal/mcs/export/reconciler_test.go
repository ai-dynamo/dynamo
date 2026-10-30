// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package export

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlclient "sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	mcsv1beta1 "sigs.k8s.io/mcs-api/pkg/apis/v1beta1"

	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api"
	mcsv1 "github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/api/gen/go/dynamo/mcs/v1"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/mcstest"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/internal/mcs/metrics"
	"github.com/ai-dynamo/dynamo/deploy/global-discovery-mcs/pkg/client"
)

const ns = "default"

type fakeLease struct {
	unregistered atomic.Bool
	kicks        atomic.Int32
}

func (f *fakeLease) Registered() bool { return !f.unregistered.Load() }
func (f *fakeLease) Kick()            { f.kicks.Add(1) }
func (f *fakeLease) WaitRegistered(ctx context.Context) error {
	for !f.Registered() {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(5 * time.Millisecond):
		}
	}
	return nil
}

type env struct {
	fakeReg *mcstest.Registry
	k8s     ctrlclient.WithWatch
	r       *Reconciler
	lease   *fakeLease
	reg     *prometheus.Registry
	rec     *events.FakeRecorder
}

func newEnv(t *testing.T, register bool, objs ...ctrlclient.Object) *env {
	t.Helper()
	return newEnvWith(t, register, interceptor.Funcs{}, objs...)
}

func newEnvWith(t *testing.T, register bool, funcs interceptor.Funcs, objs ...ctrlclient.Object) *env {
	t.Helper()
	e := &env{fakeReg: mcstest.StartRegistry(t), k8s: mcstest.FakeClientWith(t, funcs, objs...), lease: &fakeLease{},
		reg: prometheus.NewRegistry(), rec: events.NewFakeRecorder(100)}
	if register {
		e.fakeReg.Server.EnsureCluster(mcstest.ClusterSet, mcstest.Cluster, mcstest.Region)
	}
	m := metrics.New()
	require.NoError(t, m.Register(e.reg))
	e.r = NewReconciler(Options{
		Client: e.k8s, Registry: e.fakeReg.Client, Identity: mcstest.Identity(), Lease: e.lease, Metrics: m,
		Recorder: e.rec, NotRegisteredRequeue: time.Second,
	})
	return e
}

func (e *env) reconcile(t *testing.T, name string) (ctrl.Result, error) {
	t.Helper()
	return e.r.Reconcile(mcstest.Ctx(t), ctrl.Request{Namespace: ns, Name: name})
}

func (e *env) mustReconcile(t *testing.T, name string) ctrl.Result {
	t.Helper()
	res, err := e.reconcile(t, name)
	require.NoError(t, err)
	return res
}

func (e *env) registryExports(t *testing.T) map[string]*mcsv1.ServiceExport {
	t.Helper()
	resp, err := e.fakeReg.Client.ServiceExports().List(mcstest.Ctx(t), &mcsv1.ListServiceExportsRequest{Clusterset: mcstest.ClusterSet, Cluster: mcstest.Cluster})
	require.NoError(t, err)
	out := map[string]*mcsv1.ServiceExport{}
	for _, x := range resp.GetExports() {
		out[x.GetNamespace()+"/"+x.GetName()] = x
	}
	return out
}

// webExport returns the ServiceExport web.
func (e *env) webExport(t *testing.T) *mcsv1beta1.ServiceExport {
	t.Helper()
	var se mcsv1beta1.ServiceExport
	require.NoError(t, e.k8s.Get(mcstest.Ctx(t), types.NamespacedName{Namespace: ns, Name: "web"}, &se))
	return &se
}

func (e *env) metric(t *testing.T, name string, labels map[string]string) float64 {
	t.Helper()
	return mcstest.MetricValue(t, e.reg, name, labels)
}

// drain empties the reconciler's event channel and returns the keys.
func (e *env) drain() []string {
	var keys []string
	for {
		select {
		case ev := <-e.r.events:
			keys = append(keys, ev.Object.Namespace+"/"+ev.Object.Name)
		default:
			return keys
		}
	}
}

func condition(se *mcsv1beta1.ServiceExport, t string) *metav1.Condition {
	return meta.FindStatusCondition(se.Status.Conditions, t)
}

func webObjects(addresses ...string) []ctrlclient.Object {
	svc := mcstest.Service(ns, "web", map[string]string{api.LabelExport: api.ExportEnabled, tierLabel: "gold"})
	return []ctrlclient.Object{svc, mcstest.ServiceExport(ns, "web"), mcstest.EndpointSlice(ns, "web-abc", "web", addresses...)}
}

func TestExportWritesSpecEndpointsAndStatus(t *testing.T) {
	imported := mcstest.EndpointSlice(ns, "imported", "web", "10.9.9.9")
	imported.Labels[api.LabelEndpointSliceManagedBy] = api.EndpointSliceManagedBy
	e := newEnv(t, true, append(webObjects("10.0.0.1", "10.0.0.2"), imported)...)
	assert.Equal(t, ctrl.Result{}, e.mustReconcile(t, "web"))

	se := e.webExport(t)
	assert.True(t, controllerutil.ContainsFinalizer(se, api.FinalizerServiceExport))
	assert.Equal(t, metav1.ConditionTrue, condition(se, api.ConditionValid).Status)
	assert.Equal(t, api.ConditionReasonValid, condition(se, api.ConditionValid).Reason)
	assert.Equal(t, metav1.ConditionUnknown, condition(se, api.ConditionReady).Status)
	assert.Equal(t, api.ConditionReasonPending, condition(se, api.ConditionReady).Reason)
	assert.Nil(t, condition(se, api.ConditionConflict))

	x := e.registryExports(t)[ns+"/web"]
	require.NotNil(t, x)
	assert.Equal(t, mcsv1.ServiceType_SERVICE_TYPE_CLUSTER_SET_IP, x.GetSpec().GetType())
	assert.Equal(t, "gold", x.GetSpec().GetLabels()[tierLabel])
	assert.Equal(t, se.CreationTimestamp.Time.UTC(), x.GetSpec().GetExportCreationTimestamp().AsTime())
	assert.Equal(t, int32(2), x.GetEndpointCount(), "the imported slice is not exported")
	assert.Equal(t, 1, e.fakeReg.Server.CallCount("PutServiceExport"))
	assert.Equal(t, 1, e.fakeReg.Server.CallCount("ReplaceEndpoints"))
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_exports", map[string]string{"state": "exported"}))
	assert.Equal(t, 2.0, e.metric(t, "dynamo_mcs_controller_exported_endpoints", nil))

	// Nothing changed: no writes.
	e.mustReconcile(t, "web")
	assert.Equal(t, 1, e.fakeReg.Server.CallCount("PutServiceExport"))
	assert.Equal(t, 1, e.fakeReg.Server.CallCount("ReplaceEndpoints"))

	// The registry reports this cluster's conditions through the importer.
	imp, err := e.fakeReg.Client.ServiceImports().Get(mcstest.Ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: mcstest.ClusterSet, Namespace: ns, Name: "web"})
	require.NoError(t, err)
	e.r.SetOwnStatus(ns, "web", imp.GetServiceImport().GetClusters()[0], imp.GetServiceImport().GetEndpoints()[0].GetGeneration())
	assert.Equal(t, []string{ns + "/web"}, e.drain())
	e.mustReconcile(t, "web")
	se = e.webExport(t)
	assert.Equal(t, metav1.ConditionTrue, condition(se, api.ConditionReady).Status)
	assert.Equal(t, metav1.ConditionFalse, condition(se, api.ConditionConflict).Status)
	assert.Equal(t, api.ConditionReasonNoConflicts, condition(se, api.ConditionConflict).Reason)

	// The registry no longer reports it: the export is written again.
	e.r.SetOwnStatus(ns, "web", nil, 0)
	e.drain()
	e.mustReconcile(t, "web")
	se = e.webExport(t)
	assert.Equal(t, metav1.ConditionUnknown, condition(se, api.ConditionReady).Status)
	assert.Nil(t, condition(se, api.ConditionConflict))
	assert.Equal(t, 2, e.fakeReg.Server.CallCount("PutServiceExport"))
	assert.Equal(t, 2, e.fakeReg.Server.CallCount("ReplaceEndpoints"))
}

// The registry's report of this cluster tells the reconciler what the
// registry still holds; a missing record or endpoint set is written again
// although nothing changed locally and the heartbeats are healthy.
func TestRegistryLossClearsSentHashes(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	e.mustReconcile(t, "web")
	require.Equal(t, 1, e.fakeReg.Server.CallCount("PutServiceExport"))
	require.Equal(t, 1, e.fakeReg.Server.CallCount("ReplaceEndpoints"))
	imp, err := e.fakeReg.Client.ServiceImports().Get(mcstest.Ctx(t), &mcsv1.GetServiceImportRequest{Clusterset: mcstest.ClusterSet, Namespace: ns, Name: "web"})
	require.NoError(t, err)
	own, gen := imp.GetServiceImport().GetClusters()[0], imp.GetServiceImport().GetEndpoints()[0].GetGeneration()
	require.NotZero(t, gen)

	t.Log("A healthy report changes nothing")
	e.r.SetOwnStatus(ns, "web", own, gen)
	e.drain()
	e.mustReconcile(t, "web")
	assert.Equal(t, 1, e.fakeReg.Server.CallCount("PutServiceExport"))
	assert.Equal(t, 1, e.fakeReg.Server.CallCount("ReplaceEndpoints"))

	t.Log("The endpoint set is gone while the cluster participates: endpoints resent")
	e.r.SetOwnStatus(ns, "web", own, 0)
	e.drain()
	e.mustReconcile(t, "web")
	assert.Equal(t, 1, e.fakeReg.Server.CallCount("PutServiceExport"))
	assert.Equal(t, 2, e.fakeReg.Server.CallCount("ReplaceEndpoints"))

	t.Log("A cluster that does not participate has no endpoint set; nothing is resent")
	expired := &mcsv1.ClusterStatus{Cluster: mcstest.Cluster, Participating: false}
	e.r.SetOwnStatus(ns, "web", expired, 0)
	e.drain()
	e.mustReconcile(t, "web")
	assert.Equal(t, 2, e.fakeReg.Server.CallCount("ReplaceEndpoints"))

	t.Log("The record is gone: spec and endpoints resent")
	e.r.SetOwnStatus(ns, "web", nil, 0)
	e.drain()
	e.mustReconcile(t, "web")
	assert.Equal(t, 2, e.fakeReg.Server.CallCount("PutServiceExport"))
	assert.Equal(t, 3, e.fakeReg.Server.CallCount("ReplaceEndpoints"))

	t.Log("A write that started before the loss was reported does not record its hash")
	key := types.NamespacedName{Namespace: ns, Name: "web"}
	_, view := e.r.state(key)
	e.r.SetOwnStatus(ns, "web", nil, 0)
	e.r.recordSent(key, view, func(st *exportState) { st.specHash = "written" })
	_, after := e.r.state(key)
	assert.Empty(t, after.specHash)
}

func TestScalingReplacesEndpointsWithAHigherGeneration(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	e.mustReconcile(t, "web")
	first := e.registryExports(t)[ns+"/web"].GetEndpointsGeneration()

	var s discoveryv1.EndpointSlice
	require.NoError(t, e.k8s.Get(mcstest.Ctx(t), types.NamespacedName{Namespace: ns, Name: "web-abc"}, &s))
	s.Endpoints = append(s.Endpoints, discoveryv1.Endpoint{Addresses: []string{"10.0.0.2"}}, discoveryv1.Endpoint{Addresses: []string{"10.0.0.3"}})
	require.NoError(t, e.k8s.Update(mcstest.Ctx(t), &s))
	e.mustReconcile(t, "web")

	x := e.registryExports(t)[ns+"/web"]
	assert.Equal(t, int32(3), x.GetEndpointCount())
	assert.Greater(t, x.GetEndpointsGeneration(), first)
	assert.Equal(t, 1, e.fakeReg.Server.CallCount("PutServiceExport"), "the spec did not change")
	assert.Equal(t, 2, e.fakeReg.Server.CallCount("ReplaceEndpoints"))
}

func TestStaleGenerationRetriesAboveTheStoredGeneration(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	future := uint64(time.Now().Add(24 * time.Hour).UnixMilli())
	ctx := mcstest.Ctx(t)
	key := func() (string, string, string, string) { return mcstest.ClusterSet, mcstest.Cluster, ns, "web" }
	cs, cl, n, name := key()
	_, err := e.fakeReg.Client.ServiceExports().Put(ctx, &mcsv1.PutServiceExportRequest{Clusterset: cs, Cluster: cl, Namespace: n, Name: name,
		Spec: &mcsv1.ServiceExportSpec{}, Generation: future})
	require.NoError(t, err)
	_, err = e.fakeReg.Client.ServiceExports().ReplaceEndpoints(ctx, &mcsv1.ReplaceEndpointsRequest{Clusterset: cs, Cluster: cl, Namespace: n, Name: name,
		Generation: future})
	require.NoError(t, err)

	e.mustReconcile(t, "web")
	x := e.registryExports(t)[ns+"/web"]
	assert.Equal(t, future+1, x.GetGeneration())
	assert.Equal(t, future+1, x.GetEndpointsGeneration())
	assert.Equal(t, int32(1), x.GetEndpointCount())
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_export_writes_total", map[string]string{"op": "put_export", "result": "stale_generation"}))
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_endpoint_replacements_total", map[string]string{"result": "stale_generation"}))
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_endpoint_replacements_total", map[string]string{"result": "success"}))
}

func TestInvalidExportsAreWithdrawn(t *testing.T) {
	e := newEnv(t, true, mcstest.ServiceExport(ns, "web"))
	e.mustReconcile(t, "web")
	se := e.webExport(t)
	assert.Equal(t, metav1.ConditionFalse, condition(se, api.ConditionValid).Status)
	assert.Equal(t, api.ConditionReasonNoService, condition(se, api.ConditionValid).Reason)
	assert.Equal(t, metav1.ConditionFalse, condition(se, api.ConditionReady).Status)
	assert.Equal(t, 1, e.fakeReg.Server.CallCount("DeleteServiceExport"))
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_exports", map[string]string{"state": "invalid"}))
	assert.Contains(t, <-e.rec.Events, EventInvalid)

	e.mustReconcile(t, "web")
	assert.Equal(t, 1, e.fakeReg.Server.CallCount("DeleteServiceExport"), "withdrawn once")

	svc := &corev1.Service{
		Namespace: ns, Name: "web",
		Spec: corev1.ServiceSpec{Type: corev1.ServiceTypeExternalName, ExternalName: "example.com"},
	}
	require.NoError(t, e.k8s.Create(mcstest.Ctx(t), svc))
	e.mustReconcile(t, "web")
	se = e.webExport(t)
	assert.Equal(t, api.ConditionReasonInvalidServiceType, condition(se, api.ConditionValid).Reason)
	assert.Empty(t, e.registryExports(t))

	svc.Spec = mcstest.Service(ns, "web", nil).Spec
	require.NoError(t, e.k8s.Update(mcstest.Ctx(t), svc))
	e.mustReconcile(t, "web")
	se = e.webExport(t)
	assert.Equal(t, metav1.ConditionTrue, condition(se, api.ConditionValid).Status)
	assert.Contains(t, e.registryExports(t), ns+"/web")

	// Withdrawing fails: the status is written and the error retried.
	require.NoError(t, e.k8s.Delete(mcstest.Ctx(t), svc))
	e.fakeReg.Server.FailNext("DeleteServiceExport", codes.Internal)
	_, err := e.reconcile(t, "web")
	require.Error(t, err)
	assert.Equal(t, api.ConditionReasonNoService, condition(e.webExport(t), api.ConditionValid).Reason)
}

func TestWaitsForTheClusterRegistration(t *testing.T) {
	e := newEnv(t, false, webObjects("10.0.0.1")...)
	e.lease.unregistered.Store(true)
	assert.Equal(t, ctrl.Result{RequeueAfter: time.Second}, e.mustReconcile(t, "web"))
	se := e.webExport(t)
	assert.Equal(t, metav1.ConditionUnknown, condition(se, api.ConditionReady).Status)
	assert.Contains(t, condition(se, api.ConditionReady).Message, "register")
	assert.Zero(t, e.fakeReg.Server.CallCount("PutServiceExport"))
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_exports", map[string]string{"state": "pending"}))

	// The lease believes it is registered, the registry does not.
	e.lease.unregistered.Store(false)
	assert.Equal(t, ctrl.Result{RequeueAfter: time.Second}, e.mustReconcile(t, "web"))
	assert.Equal(t, int32(1), e.lease.kicks.Load())
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_export_writes_total", map[string]string{"op": "put_export", "result": "not_registered"}))

	e.fakeReg.Server.EnsureCluster(mcstest.ClusterSet, mcstest.Cluster, "")
	e.mustReconcile(t, "web")
	assert.Contains(t, e.registryExports(t), ns+"/web")
}

func TestWriteFailures(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	e.fakeReg.Server.FailNextWith("PutServiceExport", status.Error(codes.InvalidArgument, "spec.labels has 70 entries"))
	assert.Equal(t, ctrl.Result{}, e.mustReconcile(t, "web"), "a permanent failure is not retried")
	ready := condition(e.webExport(t), api.ConditionReady)
	assert.Equal(t, metav1.ConditionFalse, ready.Status)
	assert.Equal(t, "Failed", ready.Reason)
	assert.Contains(t, ready.Message, "70 entries")
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_exports", map[string]string{"state": "failed"}))

	e.fakeReg.Server.FailNext("PutServiceExport", codes.Internal)
	_, err := e.reconcile(t, "web")
	require.Error(t, err)
	assert.Equal(t, 2.0, e.metric(t, "dynamo_mcs_controller_export_writes_total", map[string]string{"op": "put_export", "result": "error"}))

	e.fakeReg.Server.FailNext("ReplaceEndpoints", codes.Internal)
	_, err = e.reconcile(t, "web")
	require.Error(t, err)
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_endpoint_replacements_total", map[string]string{"result": "error"}))

	e.mustReconcile(t, "web")
	assert.Equal(t, metav1.ConditionUnknown, condition(e.webExport(t), api.ConditionReady).Status)
	assert.Equal(t, int32(1), e.registryExports(t)[ns+"/web"].GetEndpointCount())
}

func TestDeletionWithdrawsAndRemovesTheFinalizer(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	e.mustReconcile(t, "web")
	require.Contains(t, e.registryExports(t), ns+"/web")
	require.NoError(t, e.k8s.Delete(mcstest.Ctx(t), e.webExport(t)))

	e.fakeReg.Server.FailNext("DeleteServiceExport", codes.Internal)
	_, err := e.reconcile(t, "web")
	require.Error(t, err)
	assert.True(t, controllerutil.ContainsFinalizer(e.webExport(t), api.FinalizerServiceExport), "kept until the registry confirms")

	e.mustReconcile(t, "web")
	assert.Empty(t, e.registryExports(t))
	err = e.k8s.Get(mcstest.Ctx(t), types.NamespacedName{Namespace: ns, Name: "web"}, &mcsv1beta1.ServiceExport{})
	assert.True(t, apierrors.IsNotFound(err), "%v", err)
	assert.Equal(t, 0.0, e.metric(t, "dynamo_mcs_controller_exports", map[string]string{"state": "exported"}))

	// A deleted export without our finalizer is a no-op.
	e.mustReconcile(t, "web")
}

func TestExportGoneWithoutFinalizer(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	e.mustReconcile(t, "web")
	se := e.webExport(t)
	controllerutil.RemoveFinalizer(se, api.FinalizerServiceExport)
	require.NoError(t, e.k8s.Update(mcstest.Ctx(t), se))
	require.NoError(t, e.k8s.Delete(mcstest.Ctx(t), se))
	e.mustReconcile(t, "web")
	assert.Empty(t, e.registryExports(t))
}

func TestHeadlessAndForwardedLabels(t *testing.T) {
	svc := mcstest.Service(ns, "relay", map[string]string{tierLabel: "gold", api.Prefix + "team": "inference", "app": "relay"})
	svc.Spec.ClusterIP = corev1.ClusterIPNone
	svc.Annotations = map[string]string{modelsAnnotation: "m1"}
	se := mcstest.ServiceExport(ns, "relay")
	se.Spec.ExportedAnnotations = map[string]string{"extra": "1"}
	e := newEnv(t, true, svc, se)
	e.mustReconcile(t, "relay")
	spec := e.registryExports(t)[ns+"/relay"].GetSpec()
	assert.Equal(t, mcsv1.ServiceType_SERVICE_TYPE_HEADLESS, spec.GetType())
	assert.Equal(t, map[string]string{tierLabel: "gold", api.Prefix + "team": "inference"}, spec.GetLabels())
	assert.Equal(t, map[string]string{modelsAnnotation: "m1", "extra": "1"}, spec.GetAnnotations())
}

func TestTruncatedEndpointSetEmitsAnEvent(t *testing.T) {
	var addrs []string
	for i := range api.MaxEndpointsPerExport + 5 {
		addrs = append(addrs, fmt.Sprintf("10.%d.%d.%d", i/65536, (i/256)%256, i%256))
	}
	e := newEnv(t, true, webObjects(addrs...)...)
	e.mustReconcile(t, "web")
	assert.Equal(t, int32(api.MaxEndpointsPerExport), e.registryExports(t)[ns+"/web"].GetEndpointCount())
	assert.Contains(t, <-e.rec.Events, EventEndpointsTruncated)
}

func TestNamespaceFilter(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	e.r.opts.WatchesNamespace = func(n string) bool { return n == "other" }
	e.mustReconcile(t, "web")
	assert.Zero(t, e.fakeReg.Server.CallCount("PutServiceExport"))
}

func TestMapFunctions(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	ctx := mcstest.Ctx(t)
	reqs := e.r.exportOfService(ctx, mcstest.Service(ns, "web", nil))
	assert.Equal(t, []ctrl.Request{{Namespace: ns, Name: "web"}}, reqs)
	assert.Empty(t, e.r.exportOfService(ctx, mcstest.Service(ns, "other", nil)))
	assert.Len(t, e.r.exportOfSlice(ctx, mcstest.EndpointSlice(ns, "web-xyz", "web")), 1)
	assert.Empty(t, e.r.exportOfSlice(ctx, &discoveryv1.EndpointSlice{Namespace: ns, Name: "x"}))
}

func TestResyncCollectsGarbageAndResends(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	e.mustReconcile(t, "web")
	e.fakeReg.Server.PutExport(mcstest.ClusterSet, mcstest.Cluster, ns, "orphan", &mcsv1.ServiceExportSpec{}, nil)
	e.fakeReg.Server.PutExport(mcstest.ClusterSet, "b", ns, "theirs", &mcsv1.ServiceExportSpec{}, nil)

	require.NoError(t, e.r.Resync(mcstest.Ctx(t), metrics.ResyncStartup))
	assert.Equal(t, []string{ns + "/web"}, e.drain())
	assert.NotContains(t, e.registryExports(t), ns+"/orphan")
	resp, err := e.fakeReg.Client.ServiceExports().List(mcstest.Ctx(t), &mcsv1.ListServiceExportsRequest{Clusterset: mcstest.ClusterSet, Cluster: "b"})
	require.NoError(t, err)
	assert.Len(t, resp.GetExports(), 1, "another cluster's export is not ours")
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_gc_deletions_total", map[string]string{"kind": "registry_export"}))
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_resyncs_total", map[string]string{"reason": "startup"}))
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_resync_attempts_total", map[string]string{"reason": "startup"}))

	// Everything is written again.
	e.mustReconcile(t, "web")
	assert.Equal(t, 2, e.fakeReg.Server.CallCount("PutServiceExport"))
	assert.Equal(t, 2, e.fakeReg.Server.CallCount("ReplaceEndpoints"))

	// Failed resyncs count as attempts only.
	e.fakeReg.Server.FailNext("ListServiceExports", codes.Internal)
	require.Error(t, e.r.Resync(mcstest.Ctx(t), metrics.ResyncReconnect))
	e.fakeReg.Server.PutExport(mcstest.ClusterSet, mcstest.Cluster, ns, "orphan", &mcsv1.ServiceExportSpec{}, nil)
	e.fakeReg.Server.FailNext("DeleteServiceExport", codes.Internal)
	require.Error(t, e.r.Resync(mcstest.Ctx(t), metrics.ResyncReconnect))
	assert.Equal(t, 2.0, e.metric(t, "dynamo_mcs_controller_resync_attempts_total", map[string]string{"reason": "reconnect"}))
	assert.Equal(t, 0.0, e.metric(t, "dynamo_mcs_controller_resyncs_total", map[string]string{"reason": "reconnect"}))
}

// A ServiceExport created, and exported by a concurrent reconcile, between
// the two lists of a resync is not an orphan: the registry is listed first,
// so the later cache list includes the ServiceExport of any record the
// resync saw.
func TestResyncKeepsAnExportWrittenDuringTheResync(t *testing.T) {
	var e *env
	var once sync.Once
	funcs := interceptor.Funcs{
		List: func(ctx context.Context, c ctrlclient.WithWatch, list ctrlclient.ObjectList, opts ...ctrlclient.ListOption) error {
			if _, ok := list.(*mcsv1beta1.ServiceExportList); ok {
				once.Do(func() {
					// A reconcile ran between the registry list and this one.
					require.NoError(t, c.Create(ctx, mcstest.ServiceExport(ns, "late")))
					e.fakeReg.Server.PutExport(mcstest.ClusterSet, mcstest.Cluster, ns, "late", &mcsv1.ServiceExportSpec{}, nil)
				})
			}
			return c.List(ctx, list, opts...)
		},
	}
	e = newEnvWith(t, true, funcs, webObjects("10.0.0.1")...)
	require.NoError(t, e.r.Resync(mcstest.Ctx(t), metrics.ResyncStartup))
	assert.Contains(t, e.registryExports(t), ns+"/late", "written during the resync, kept")
	assert.Equal(t, 0.0, e.metric(t, "dynamo_mcs_controller_gc_deletions_total", map[string]string{"kind": "registry_export"}))
	assert.ElementsMatch(t, []string{ns + "/web", ns + "/late"}, e.drain())
}

// Narrowing --watch-namespaces leaves ServiceExports with the finalizer in
// namespaces the caches no longer cover; the sweep releases them.
func TestSweepUnwatchedReleasesFinalizers(t *testing.T) {
	inside := mcstest.ServiceExport("a", "web")
	controllerutil.AddFinalizer(inside, api.FinalizerServiceExport)
	outside := mcstest.ServiceExport("b", "web")
	controllerutil.AddFinalizer(outside, api.FinalizerServiceExport)
	foreign := mcstest.ServiceExport("b", "theirs")
	foreign.Finalizers = []string{"other.example/finalizer"}
	e := newEnv(t, true, inside, outside, foreign)
	e.r.opts.WatchesNamespace = func(n string) bool { return n == "a" }
	ctx := mcstest.Ctx(t)
	finalizers := func(ns, name string) []string {
		var se mcsv1beta1.ServiceExport
		require.NoError(t, e.k8s.Get(ctx, types.NamespacedName{Namespace: ns, Name: name}, &se))
		return se.Finalizers
	}

	t.Log("Every namespace watched: nothing to sweep")
	require.NoError(t, e.r.SweepUnwatched(ctx))
	assert.Equal(t, []string{api.FinalizerServiceExport}, finalizers("b", "web"))

	t.Log("Restricted to namespace a: b's ServiceExport is released, a's and foreign finalizers stay")
	e.r.opts.Restricted = true
	require.NoError(t, e.r.SweepUnwatched(ctx))
	assert.Empty(t, finalizers("b", "web"))
	assert.Equal(t, []string{api.FinalizerServiceExport}, finalizers("a", "web"))
	assert.Equal(t, []string{"other.example/finalizer"}, finalizers("b", "theirs"))
	require.NoError(t, e.r.SweepUnwatched(ctx), "idempotent")
}

// holdingClient returns a registry client of e whose ListServiceExports call
// blocks, once armed with hold, until gate is closed; held receives a value
// when the call is inside the hold.
func holdingClient(t *testing.T, e *env, hold *atomic.Bool, held chan<- struct{}, gate <-chan struct{}) *client.Client {
	t.Helper()
	block := func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
		if strings.HasSuffix(method, "/ListServiceExports") && hold.CompareAndSwap(true, false) {
			held <- struct{}{}
			<-gate
		}
		return invoker(ctx, method, req, reply, cc, opts...)
	}
	c, err := client.New(context.Background(), client.Config{
		URLs: []string{e.fakeReg.Running.URL()}, AllowInsecureHTTP: true, Backoff: mcstest.FastBackoff, Logger: mcstest.QuietLogger,
		ConnectTimeout: time.Second, RPCTimeout: 5 * time.Second, DialOptions: []grpc.DialOption{grpc.WithChainUnaryInterceptor(block)},
	})
	require.NoError(t, err)
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func TestResyncer(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	var hold atomic.Bool
	held, gate := make(chan struct{}, 1), make(chan struct{})
	e.r.opts.Registry = holdingClient(t, e, &hold, held, gate)
	e.lease.unregistered.Store(true)
	s := NewResyncer(ResyncerOptions{Reconciler: e.r, Lease: e.lease, Logger: testr.New(t),
		Backoff: client.Backoff{Initial: 5 * time.Millisecond, Max: 20 * time.Millisecond}})
	assert.True(t, s.NeedLeaderElection())
	metric := func(reason string) (attempts, completed float64) {
		return e.metric(t, "dynamo_mcs_controller_resync_attempts_total", map[string]string{"reason": reason}),
			e.metric(t, "dynamo_mcs_controller_resyncs_total", map[string]string{"reason": reason})
	}
	// A reconnect on a replica that is not the leader yet is covered by the
	// startup resync.
	s.Trigger(metrics.ResyncReconnect)
	e.fakeReg.Server.FailNext("ListServiceExports", codes.Internal)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Start(ctx) }()
	defer func() {
		cancel()
		require.NoError(t, <-done)
	}()
	time.Sleep(20 * time.Millisecond)
	assert.Zero(t, e.fakeReg.Server.CallCount("ListServiceExports"), "waits for the registration")
	e.lease.unregistered.Store(false)
	require.Eventually(t, func() bool {
		attempts, completed := metric(metrics.ResyncStartup)
		return attempts == 2 && completed == 1
	}, 5*time.Second, 5*time.Millisecond, "retried after a failure; the failed attempt is not a resync")
	time.Sleep(20 * time.Millisecond)
	_, reconnects := metric(metrics.ResyncReconnect)
	assert.Zero(t, reconnects, "the trigger before the startup resync was drained")

	// Two triggers while a resync is held inside its registry list coalesce
	// into one, with the first reason.
	hold.Store(true)
	s.Trigger(metrics.ResyncReconnect)
	select {
	case <-held:
	case <-time.After(5 * time.Second):
		t.Fatal("the resync did not reach the registry list")
	}
	s.Trigger(metrics.ResyncReconnect)
	s.Trigger(metrics.ResyncReregister)
	close(gate)
	require.Eventually(t, func() bool {
		_, completed := metric(metrics.ResyncReconnect)
		return completed == 2
	}, 5*time.Second, 5*time.Millisecond, "the held resync and the one coalesced trigger")
	time.Sleep(20 * time.Millisecond)
	attempts, completed := metric(metrics.ResyncReregister)
	assert.Zero(t, attempts+completed, "triggers coalesce")
}

func TestResyncerPeriodic(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	s := NewResyncer(ResyncerOptions{Reconciler: e.r, Lease: e.lease, Logger: testr.New(t), Period: 10 * time.Millisecond,
		Backoff: client.Backoff{Initial: 5 * time.Millisecond, Max: 20 * time.Millisecond}})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Start(ctx) }()
	defer func() {
		cancel()
		require.NoError(t, <-done)
	}()
	require.Eventually(t, func() bool {
		return e.metric(t, "dynamo_mcs_controller_resyncs_total", map[string]string{"reason": "periodic"}) >= 2
	}, 5*time.Second, 5*time.Millisecond, "resyncs every period without a trigger")
	assert.Equal(t, 1.0, e.metric(t, "dynamo_mcs_controller_resyncs_total", map[string]string{"reason": "startup"}))
}

func TestResetSentDuringAWriteStillResends(t *testing.T) {
	e := newEnv(t, true, webObjects("10.0.0.1")...)
	key := types.NamespacedName{Namespace: ns, Name: "web"}
	_, view := e.r.state(key)
	e.r.ResetSent()
	e.r.recordSent(key, view, func(st *exportState) { st.specHash = "written" })
	_, after := e.r.state(key)
	assert.Empty(t, after.specHash)
}
