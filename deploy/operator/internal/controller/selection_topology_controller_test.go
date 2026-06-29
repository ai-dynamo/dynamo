/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"encoding/json"
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/enginemetadata/sglang"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/selectionservice"
)

type selectionRecorder struct {
	mu         sync.Mutex
	posts      []selectionservice.WorkerRequest
	getQueries []url.Values
	deletes    []uint64
	workers    map[uint64]selectionservice.WorkerRecord
}

const (
	selectionWorkersPath  = "/workers"
	testOwnerKindService  = "Service"
	testSelectorNamespace = "selection"
	testTenantA           = "tenant-a"
)

type fakeSGLangMetadataFetcher struct {
	snapshot sglang.MetadataSnapshot
}

func (f fakeSGLangMetadataFetcher) Fetch(context.Context) (sglang.MetadataSnapshot, error) {
	return f.snapshot, nil
}

func reconcileWorkerService(t *testing.T, r *SelectionTopologyReconciler, service *corev1.Service) reconcile.Result {
	t.Helper()
	result, err := r.Reconcile(context.Background(), reconcile.Request{NamespacedName: client.ObjectKeyFromObject(service)})
	require.NoError(t, err)
	return result
}

func recorderPosts(recorder *selectionRecorder) []selectionservice.WorkerRequest {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	return append([]selectionservice.WorkerRequest(nil), recorder.posts...)
}

func recorderDeletes(recorder *selectionRecorder) []uint64 {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	return append([]uint64(nil), recorder.deletes...)
}

func recorderGetQueries(recorder *selectionRecorder) []url.Values {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	return append([]url.Values(nil), recorder.getQueries...)
}

func recorderWorker(recorder *selectionRecorder, workerID uint64) (selectionservice.WorkerRecord, bool) {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	worker, ok := recorder.workers[workerID]
	return worker, ok
}

func containsWorkerListQuery(queries []url.Values, modelName string, tenantID string) bool {
	for _, query := range queries {
		if query.Get("model_name") == modelName && query.Get("tenant_id") == tenantID {
			return true
		}
	}
	return false
}

func TestSelectionTopologyReconcilerReconcilesWorkerIntoSelectionService(t *testing.T) {
	sglangServer := newFakeSGLangServer(t)
	defer sglangServer.Close()
	selectionServer, recorder := newFakeSelectionServer(t)
	defer selectionServer.Close()

	service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
	service.Annotations[consts.KubeAnnotationDynamoSelectionTenantID] = "tenant-from-service"
	pod.Annotations[consts.KubeAnnotationDynamoSelectionTenantID] = "tenant-from-pod"
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)

	result := reconcileWorkerService(t, reconciler, service)
	assert.Equal(t, selectionTopologyResyncAfter, result.RequeueAfter)

	posts := recorderPosts(recorder)
	require.Len(t, posts, 1)
	posted := posts[0]
	getQueries := recorderGetQueries(recorder)
	assert.True(t, containsWorkerListQuery(getQueries, "", "tenant-from-service"), "queries = %#v", getQueries)
	assert.Equal(t, workerIDForPod(pod), posted.WorkerID)
	assert.Equal(t, "qwen", posted.ModelName)
	assert.Equal(t, "tenant-from-service", posted.TenantID)
	assert.Equal(t, sglangServer.URL, posted.Endpoint)
	assert.Equal(t, "zone-a", posted.TopologyDomains["zone"])
	assert.Equal(t, selectionMetadataManagedByVal, posted.Metadata[selectionMetadataManagedBy])
	assert.Equal(t, testOwnerKindService, posted.Metadata[selectionMetadataOwnerKind])
	assert.Equal(t, selectionAdapterExternalSGLang, posted.Metadata[selectionMetadataAdapter])
	assert.Equal(t, selectionServer.URL, posted.Metadata[selectionMetadataSelectorURL])
	assert.Empty(t, recorderDeletes(recorder))
}

func TestSelectionTopologyReconcilerRegistersFixtureBackedSGLangWorker(t *testing.T) {
	sglangServer := newFixtureSGLangServer(t, nil)
	defer sglangServer.Close()
	selectionServer, recorder := newFakeSelectionServer(t)
	defer selectionServer.Close()

	service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)

	result := reconcileWorkerService(t, reconciler, service)
	assert.Equal(t, selectionTopologyResyncAfter, result.RequeueAfter)

	posts := recorderPosts(recorder)
	require.Len(t, posts, 1)
	assertFixtureBackedSGLangWorker(t, posts[0], service, pod, sglangServer.URL, selectionServer.URL)
}

func TestSelectionTopologyReconcilerDefaultsToRequiredKVEvents(t *testing.T) {
	sglangServer := newFakeSGLangServer(t)
	sglangServer.omitKVEvents = true
	defer sglangServer.Close()
	selectionServer, recorder := newFakeSelectionServer(t)
	defer selectionServer.Close()

	service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)

	result := reconcileWorkerService(t, reconciler, service)

	assert.Equal(t, selectionTopologyRequeueAfter, result.RequeueAfter)
	assert.Empty(t, recorderPosts(recorder))
	assert.Empty(t, recorderDeletes(recorder))
}

func TestSelectionTopologyReconcilerAllowsMissingKVEventsWhenExplicitlyOptional(t *testing.T) {
	sglangServer := newFakeSGLangServer(t)
	sglangServer.omitKVEvents = true
	defer sglangServer.Close()
	selectionServer, recorder := newFakeSelectionServer(t)
	defer selectionServer.Close()

	service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
	service.Annotations[consts.KubeAnnotationDynamoSelectionRequireKVEvents] = consts.KubeLabelValueFalse
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)

	result := reconcileWorkerService(t, reconciler, service)

	assert.Equal(t, selectionTopologyResyncAfter, result.RequeueAfter)
	posts := recorderPosts(recorder)
	require.Len(t, posts, 1)
	assert.Empty(t, posts[0].KVEventsEndpoints)
}

func TestParseOptionalBool(t *testing.T) {
	tests := []struct {
		name         string
		value        string
		defaultValue bool
		want         bool
		wantErr      bool
	}{
		{name: "empty uses default", value: "", defaultValue: true, want: true},
		{name: "whitespace uses default", value: " \n\t", defaultValue: false, want: false},
		{name: "trims true", value: " true\n", defaultValue: false, want: true},
		{name: "trims false", value: "\tfalse ", defaultValue: true, want: false},
		{name: "invalid", value: "maybe", defaultValue: true, wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseOptionalBool(tt.value, tt.defaultValue)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestSelectionTopologyReconcilerUsesInjectedSGLangMetadataClientFactory(t *testing.T) {
	selectionServer, recorder := newFakeSelectionServer(t)
	defer selectionServer.Close()

	service, pod, slice := rawSGLangWorkerObjects(t, "http://10.0.0.1:30000", selectionServer.URL)
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)

	factoryCalls := 0
	reconciler.SGLangMetadataClientFactory = func(endpoint string) (sglangMetadataFetcher, error) {
		factoryCalls++
		assert.Equal(t, "http://10.0.0.1:30000", endpoint)
		return fakeSGLangMetadataFetcher{snapshot: testSGLangMetadataSnapshot(t)}, nil
	}

	reconcileWorkerService(t, reconciler, service)

	assert.Equal(t, 1, factoryCalls)
	posts := recorderPosts(recorder)
	require.Len(t, posts, 1)
	assert.Equal(t, "http://10.0.0.1:30000", posts[0].Endpoint)
}

func TestSelectionTopologyReconcilerHandlesOverlappingWorkerServices(t *testing.T) {
	t.Run("rejects annotated overlap", func(t *testing.T) {
		ctx := context.Background()
		sglangServer := newFakeSGLangServer(t)
		defer sglangServer.Close()
		selectionServer, recorder := newFakeSelectionServer(t)
		defer selectionServer.Close()

		serviceA, pod, sliceA := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
		serviceA.CreationTimestamp = metav1.NewTime(time.Unix(1, 0))
		reconciler := newSelectionTopologyTestReconciler(t, serviceA, pod, sliceA)

		reconcileWorkerService(t, reconciler, serviceA)
		require.Len(t, recorderPosts(recorder), 1)

		serviceB := rawSGLangWorkerService(selectionServer.URL)
		serviceB.Name = "worker-service-b"
		serviceB.UID = "worker-service-b-uid"
		serviceB.CreationTimestamp = metav1.NewTime(time.Unix(2, 0))
		serviceB.Annotations[consts.KubeAnnotationDynamoSelectionTenantID] = "tenant-b"
		host, port := splitTestServerAddress(t, sglangServer.URL)
		sliceB := rawSGLangEndpointSlice(host, port, serviceB.Namespace)
		sliceB.Name = "slice-b"
		sliceB.Labels[discoveryv1.LabelServiceName] = serviceB.Name
		require.NoError(t, reconciler.Create(ctx, serviceB))
		require.NoError(t, reconciler.Create(ctx, sliceB))

		reconcileWorkerService(t, reconciler, serviceB)

		posts := recorderPosts(recorder)
		require.Len(t, posts, 1)
		assert.Equal(t, testTenantA, posts[0].TenantID)
		assert.Empty(t, recorderDeletes(recorder))
		workerID := workerIDForPod(pod)
		record, ok := recorderWorker(recorder, workerID)
		require.True(t, ok)
		assert.Equal(t, testTenantA, record.TenantID)
		assert.Equal(t, serviceA.Name, record.Metadata[selectionMetadataOwnerName])
		assert.Equal(t, selectionservice.WorkerLifecycleSchedulable, record.Lifecycle)

		reconcileWorkerService(t, reconciler, serviceA)
		assert.Len(t, recorderPosts(recorder), 1)
		assert.Empty(t, recorderDeletes(recorder))
		record, ok = recorderWorker(recorder, workerID)
		require.True(t, ok)
		assert.Equal(t, testTenantA, record.TenantID)
		assert.Equal(t, selectionservice.WorkerLifecycleSchedulable, record.Lifecycle)
	})

	t.Run("ignores plain service overlap", func(t *testing.T) {
		sglangServer := newFakeSGLangServer(t)
		defer sglangServer.Close()
		selectionServer, recorder := newFakeSelectionServer(t)
		defer selectionServer.Close()

		service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
		plainService := &corev1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "plain-service",
				Namespace: service.Namespace,
				UID:       "plain-service-uid",
			},
		}
		host, port := splitTestServerAddress(t, sglangServer.URL)
		plainSlice := rawSGLangEndpointSlice(host, port, plainService.Namespace)
		plainSlice.Name = "plain-slice"
		plainSlice.Labels[discoveryv1.LabelServiceName] = plainService.Name
		reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice, plainService, plainSlice)

		reconcileWorkerService(t, reconciler, service)

		posts := recorderPosts(recorder)
		require.Len(t, posts, 1)
		assert.Equal(t, workerIDForPod(pod), posts[0].WorkerID)
		assert.Empty(t, recorderDeletes(recorder))
	})
}

func TestSelectionTopologyReconcilerFailsClosedAndDeactivatesKnownWorker(t *testing.T) {
	t.Run("missing required fixture metadata", func(t *testing.T) {
		ctx := context.Background()
		sglangServer := newFixtureSGLangServer(t, func(serverInfo map[string]any) {
			delete(serverInfo, "page_size")
		})
		defer sglangServer.Close()
		selectionServer, recorder := newFakeSelectionServer(t)
		defer selectionServer.Close()

		service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
		reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)
		workerID := workerIDForPod(pod)
		recorder.workers[workerID] = selectionservice.WorkerRecord{
			WorkerID:  workerID,
			ModelName: "Qwen/Qwen3-0.6B",
			TenantID:  testTenantA,
			Metadata:  selectionTestWorkerMetadata(service, selectionServer.URL, selectionAdapterExternalSGLang),
		}

		result, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(service)})
		require.NoError(t, err)
		assert.NotZero(t, result.RequeueAfter)
		assert.Empty(t, recorderPosts(recorder))
		assert.Equal(t, []uint64{workerID}, recorderDeletes(recorder))
		deactivated, ok := recorderWorker(recorder, workerID)
		require.True(t, ok)
		assert.Equal(t, selectionservice.WorkerLifecycleUnschedulable, deactivated.Lifecycle)
	})

	t.Run("registered worker becomes invalid", func(t *testing.T) {
		ctx := context.Background()
		sglangServer := newFakeSGLangServer(t)
		defer sglangServer.Close()
		selectionServer, recorder := newFakeSelectionServer(t)
		defer selectionServer.Close()

		service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
		reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)

		reconcileWorkerService(t, reconciler, service)
		sglangServer.invalid = true
		result, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(service)})
		require.NoError(t, err)

		assert.NotZero(t, result.RequeueAfter)
		assert.Equal(t, []uint64{workerIDForPod(pod)}, recorderDeletes(recorder))
	})

	t.Run("registered worker with zero max prefill becomes invalid", func(t *testing.T) {
		ctx := context.Background()
		sglangServer := newFakeSGLangServer(t)
		defer sglangServer.Close()
		selectionServer, recorder := newFakeSelectionServer(t)
		defer selectionServer.Close()

		service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
		reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)

		reconcileWorkerService(t, reconciler, service)
		sglangServer.zeroMaxPrefillTokens = true
		result, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(service)})
		require.NoError(t, err)

		assert.NotZero(t, result.RequeueAfter)
		assert.Equal(t, []uint64{workerIDForPod(pod)}, recorderDeletes(recorder))
	})

	t.Run("tombstoned worker becomes valid again", func(t *testing.T) {
		ctx := context.Background()
		sglangServer := newFakeSGLangServer(t)
		defer sglangServer.Close()
		selectionServer, recorder := newFakeSelectionServer(t)
		defer selectionServer.Close()

		service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
		reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)
		workerID := workerIDForPod(pod)

		reconcileWorkerService(t, reconciler, service)
		sglangServer.invalid = true
		result, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(service)})
		require.NoError(t, err)
		require.NotZero(t, result.RequeueAfter)

		sglangServer.invalid = false
		reconcileWorkerService(t, reconciler, service)

		posts := recorderPosts(recorder)
		require.Len(t, posts, 2)
		assert.Equal(t, workerID, posts[1].WorkerID)
		restored, ok := recorderWorker(recorder, workerID)
		require.True(t, ok)
		assert.Equal(t, selectionservice.WorkerLifecycleSchedulable, restored.Lifecycle)
	})
}

func TestSelectionTopologyReconcilerDeactivatesKnownWorkerWhenAdapterIsUnsupported(t *testing.T) {
	ctx := context.Background()
	sglangServer := newFakeSGLangServer(t)
	defer sglangServer.Close()
	selectionServer, recorder := newFakeSelectionServer(t)
	defer selectionServer.Close()

	service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)

	reconcileWorkerService(t, reconciler, service)
	updated := service.DeepCopy()
	updated.Annotations[consts.KubeAnnotationDynamoSelectionAdapter] = "external-vllm"
	require.NoError(t, reconciler.Update(ctx, updated))

	result, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(service)})
	require.NoError(t, err)
	assert.Equal(t, selectionTopologyResyncAfter, result.RequeueAfter)

	assert.Equal(t, []uint64{workerIDForPod(pod)}, recorderDeletes(recorder))
}

func TestSelectionTopologyReconcilerReconcilesScopedCatalogDeactivation(t *testing.T) {
	tests := []struct {
		name          string
		readyEndpoint bool
		seedWorkers   func(*corev1.Service, string) []selectionservice.WorkerRecord
		wantDeleted   []uint64
		wantPreserved []uint64
		wantPost      bool
	}{
		{
			name:          "stale restart worker",
			readyEndpoint: true,
			seedWorkers: func(service *corev1.Service, selectorURL string) []selectionservice.WorkerRecord {
				return []selectionservice.WorkerRecord{{
					WorkerID:  99,
					ModelName: "qwen",
					TenantID:  testTenantA,
					Metadata:  selectionTestWorkerMetadata(service, selectorURL, selectionAdapterExternalSGLang),
				}}
			},
			wantDeleted: []uint64{99},
			wantPost:    true,
		},
		{
			name:          "no ready endpoint",
			readyEndpoint: false,
			seedWorkers: func(service *corev1.Service, selectorURL string) []selectionservice.WorkerRecord {
				return []selectionservice.WorkerRecord{{
					WorkerID:  99,
					ModelName: "qwen",
					TenantID:  testTenantA,
					Metadata:  selectionTestWorkerMetadata(service, selectorURL, selectionAdapterExternalSGLang),
				}}
			},
			wantDeleted: []uint64{99},
		},
		{
			name:          "stale scoped workers deleted",
			readyEndpoint: true,
			seedWorkers: func(service *corev1.Service, selectorURL string) []selectionservice.WorkerRecord {
				return []selectionservice.WorkerRecord{
					{
						WorkerID:  100,
						ModelName: "other-model",
						TenantID:  testTenantA,
						Metadata:  selectionTestWorkerMetadata(service, selectorURL, selectionAdapterExternalSGLang),
					},
					{
						WorkerID:  101,
						ModelName: "qwen",
						TenantID:  "tenant-b",
						Metadata:  selectionTestWorkerMetadata(service, selectorURL, selectionAdapterExternalSGLang),
					},
					{
						WorkerID:  102,
						ModelName: "qwen",
						TenantID:  testTenantA,
						Metadata:  selectionTestWorkerMetadata(service, selectorURL, "external-vllm"),
					},
				}
			},
			wantDeleted: []uint64{100, 101, 102},
			wantPost:    true,
		},
		{
			name:          "non owner preserve",
			readyEndpoint: true,
			seedWorkers: func(*corev1.Service, string) []selectionservice.WorkerRecord {
				return []selectionservice.WorkerRecord{{
					WorkerID:  100,
					ModelName: "qwen",
					TenantID:  testTenantA,
					Metadata: map[string]string{
						selectionMetadataManagedBy: "someone-else",
					},
				}}
			},
			wantPreserved: []uint64{100},
			wantPost:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sglangServer := newFakeSGLangServer(t)
			defer sglangServer.Close()
			selectionServer, recorder := newFakeSelectionServer(t)
			defer selectionServer.Close()

			service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
			slice.Endpoints[0].Conditions.Ready = ptr.To(tt.readyEndpoint)
			objects := []client.Object{service, slice}
			if tt.readyEndpoint {
				objects = append(objects, pod)
			}
			reconciler := newSelectionTopologyTestReconciler(t, objects...)
			for _, worker := range tt.seedWorkers(service, selectionServer.URL) {
				recorder.workers[worker.WorkerID] = worker
			}

			reconcileWorkerService(t, reconciler, service)

			deletes := recorderDeletes(recorder)
			for _, workerID := range tt.wantDeleted {
				assert.Contains(t, deletes, workerID)
			}
			for _, workerID := range tt.wantPreserved {
				assert.NotContains(t, deletes, workerID)
				_, ok := recorderWorker(recorder, workerID)
				assert.Truef(t, ok, "worker %d should be preserved", workerID)
			}
			posts := recorderPosts(recorder)
			if tt.wantPost {
				require.Len(t, posts, 1)
				assert.Equal(t, workerIDForPod(pod), posts[0].WorkerID)
			} else {
				assert.Empty(t, posts)
			}
			if len(tt.wantDeleted) > 0 {
				reconcileWorkerService(t, reconciler, service)
				assert.Equal(t, deletes, recorderDeletes(recorder))
			}
		})
	}
}

func TestSelectionTopologyReconcilerDeactivatesOrphanedOwnerServiceWorker(t *testing.T) {
	for _, tt := range []struct {
		name         string
		ownerPatch   func(*corev1.Service)
		includeOwner bool
	}{
		{name: "owner Service deleted"},
		{
			name: "owner selection annotations removed",
			ownerPatch: func(service *corev1.Service) {
				service.Annotations = nil
			},
			includeOwner: true,
		},
		{
			name: "owner selector URL changed",
			ownerPatch: func(service *corev1.Service) {
				service.Annotations[consts.KubeAnnotationDynamoSelectionServiceURL] = "http://other-selector.default.svc"
			},
			includeOwner: true,
		},
		{
			name: "owner tenant changed",
			ownerPatch: func(service *corev1.Service) {
				service.Annotations[consts.KubeAnnotationDynamoSelectionTenantID] = "tenant-b"
			},
			includeOwner: true,
		},
		{
			name: "owner adapter changed",
			ownerPatch: func(service *corev1.Service) {
				service.Annotations[consts.KubeAnnotationDynamoSelectionAdapter] = "external-vllm"
			},
			includeOwner: true,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			sglangServer := newFakeSGLangServer(t)
			defer sglangServer.Close()
			selectionServer, recorder := newFakeSelectionServer(t)
			defer selectionServer.Close()

			service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
			orphanService := rawSGLangWorkerService(selectionServer.URL)
			orphanService.Name = "deleted-worker-service"
			orphanService.UID = "deleted-worker-service-uid"
			orphanWorkerID := uint64(99)
			recorder.workers[orphanWorkerID] = selectionservice.WorkerRecord{
				WorkerID:  orphanWorkerID,
				ModelName: "orphan-model",
				TenantID:  testTenantA,
				Lifecycle: selectionservice.WorkerLifecycleSchedulable,
				Endpoint:  "http://orphan-worker.default.svc:30000",
				Metadata:  selectionTestWorkerMetadata(orphanService, selectionServer.URL, selectionAdapterExternalSGLang),
			}
			objects := []client.Object{service, pod, slice}
			if tt.includeOwner {
				owner := orphanService.DeepCopy()
				tt.ownerPatch(owner)
				objects = append(objects, owner)
			}
			reconciler := newSelectionTopologyTestReconciler(t, objects...)

			reconcileWorkerService(t, reconciler, service)

			currentWorkerID := workerIDForPod(pod)
			deletes := recorderDeletes(recorder)
			assert.Contains(t, deletes, orphanWorkerID)
			assert.NotContains(t, deletes, currentWorkerID)
			orphan, ok := recorderWorker(recorder, orphanWorkerID)
			require.True(t, ok)
			assert.Equal(t, selectionservice.WorkerLifecycleUnschedulable, orphan.Lifecycle)
			posts := recorderPosts(recorder)
			require.Len(t, posts, 1)
			assert.Equal(t, currentWorkerID, posts[0].WorkerID)
		})
	}
}

func TestSelectionTopologyReconcilerDeactivatesOldSelectorURLOnChange(t *testing.T) {
	for _, tt := range []struct {
		name           string
		invalidSGLang  bool
		wantRequeue    time.Duration
		wantNewUpsert  bool
		wantNewDeletes bool
	}{
		{name: "new URL registers", wantNewUpsert: true, wantRequeue: selectionTopologyResyncAfter},
		{name: "new URL metadata invalid", invalidSGLang: true, wantRequeue: selectionTopologyRequeueAfter},
	} {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			sglangServer := newFakeSGLangServer(t)
			defer sglangServer.Close()
			oldSelectionServer, oldRecorder := newFakeSelectionServer(t)
			defer oldSelectionServer.Close()
			newSelectionServer, newRecorder := newFakeSelectionServer(t)
			defer newSelectionServer.Close()

			service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, oldSelectionServer.URL)
			reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)

			reconcileWorkerService(t, reconciler, service)
			sglangServer.invalid = tt.invalidSGLang
			updated := service.DeepCopy()
			updated.Annotations[consts.KubeAnnotationDynamoSelectionServiceURL] = newSelectionServer.URL
			require.NoError(t, reconciler.Update(ctx, updated))
			result, err := reconciler.Reconcile(ctx, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(service)})
			require.NoError(t, err)

			workerID := workerIDForPod(pod)
			assert.Equal(t, tt.wantRequeue, result.RequeueAfter)
			assert.Equal(t, []uint64{workerID}, recorderDeletes(oldRecorder))
			newPosts := recorderPosts(newRecorder)
			if tt.wantNewUpsert {
				require.Len(t, newPosts, 1)
				assert.Equal(t, workerID, newPosts[0].WorkerID)
			} else {
				assert.Empty(t, newPosts)
			}
			if tt.wantNewDeletes {
				assert.NotEmpty(t, recorderDeletes(newRecorder))
			} else {
				assert.Empty(t, recorderDeletes(newRecorder))
			}
		})
	}
}

func TestSelectionTopologyWorkerMatchesDesired(t *testing.T) {
	expected := selectionservice.WorkerRequest{
		WorkerID:              1,
		ModelName:             "qwen",
		TenantID:              testTenantA,
		Endpoint:              "http://worker.default.svc:30000",
		KVEventsEndpoints:     map[uint32]string{0: "tcp://worker.default.svc:5557"},
		BlockSize:             16,
		DataParallelStartRank: 0,
		DataParallelSize:      2,
		MaxNumBatchedTokens:   1024,
		TotalKVBlocks:         4096,
		StableRoutingID:       "pod-uid",
		IsEagle:               ptr.To(true),
		TopologyDomains:       map[string]string{"zone": "zone-a"},
		Metadata:              map[string]string{selectionMetadataManagedBy: selectionMetadataManagedByVal},
	}
	for _, tt := range []struct {
		name   string
		mutate func(selectionservice.WorkerRecord) selectionservice.WorkerRecord
		want   bool
	}{
		{name: "matches", want: true},
		{name: "incomplete lifecycle", mutate: func(worker selectionservice.WorkerRecord) selectionservice.WorkerRecord {
			worker.Lifecycle = selectionservice.WorkerLifecycleIncomplete
			worker.NotSchedulableReasons = []string{"max_num_batched_tokens is required while queueing is enabled"}
			return worker
		}},
		{name: "endpoint drift", mutate: func(worker selectionservice.WorkerRecord) selectionservice.WorkerRecord {
			worker.Endpoint = "http://stale"
			return worker
		}},
		{name: "block size drift", mutate: func(worker selectionservice.WorkerRecord) selectionservice.WorkerRecord {
			worker.BlockSize = ptr.To(uint32(32))
			return worker
		}},
		{name: "KV events drift", mutate: func(worker selectionservice.WorkerRecord) selectionservice.WorkerRecord {
			worker.KVEventsEndpoints = map[uint32]string{0: "tcp://stale:5557"}
			return worker
		}},
		{name: "metadata drift", mutate: func(worker selectionservice.WorkerRecord) selectionservice.WorkerRecord {
			worker.Metadata = maps.Clone(worker.Metadata)
			worker.Metadata["extra"] = "stale"
			return worker
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			actual := selectionWorkerRecordFromRequest(expected)
			if tt.mutate != nil {
				actual = tt.mutate(actual)
			}
			assert.Equal(t, tt.want, catalogWorkerMatchesDesired(actual, expected))
		})
	}

	t.Run("defaulted worker key matches omitted desired key", func(t *testing.T) {
		desired := expected
		desired.ModelName = ""
		desired.TenantID = ""
		actual := selectionWorkerRecordFromRequest(desired)
		actual.ModelName = selectionservice.DefaultModelName
		actual.TenantID = selectionservice.DefaultTenantID
		assert.True(t, catalogWorkerMatchesDesired(actual, desired))
	})
}

func TestSelectionTopologyReconcilerRepostsDriftedCatalogWorker(t *testing.T) {
	sglangServer := newFakeSGLangServer(t)
	defer sglangServer.Close()
	selectionServer, recorder := newFakeSelectionServer(t)
	defer selectionServer.Close()

	service, pod, slice := rawSGLangWorkerObjects(t, sglangServer.URL, selectionServer.URL)
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, slice)

	reconcileWorkerService(t, reconciler, service)

	workerID := workerIDForPod(pod)
	recorder.mu.Lock()
	worker := recorder.workers[workerID]
	worker.Endpoint = "http://stale-worker.default.svc:30000"
	recorder.workers[workerID] = worker
	recorder.posts = nil
	recorder.mu.Unlock()

	reconcileWorkerService(t, reconciler, service)

	posts := recorderPosts(recorder)
	require.Len(t, posts, 1)
	assert.Equal(t, workerID, posts[0].WorkerID)
}

func TestSelectionTopologyReconcilerEndpointSliceDeletionKeepsRemainingServiceWorkers(t *testing.T) {
	ctx := context.Background()
	sglangA := newFakeSGLangServer(t)
	defer sglangA.Close()
	sglangB := newFakeSGLangServer(t)
	defer sglangB.Close()
	selectionServer, recorder := newFakeSelectionServer(t)
	defer selectionServer.Close()

	service, podA, sliceA := rawSGLangWorkerObjects(t, sglangA.URL, selectionServer.URL)
	_, podB, sliceB := rawSGLangWorkerObjects(t, sglangB.URL, selectionServer.URL)
	podB.Name = "worker-1"
	podB.UID = "pod-uid-1"
	sliceA.Name = "slice-a"
	sliceB.Name = "slice-b"
	sliceB.Endpoints[0].TargetRef.Name = podB.Name
	reconciler := newSelectionTopologyTestReconciler(t, service, podA, podB, sliceA, sliceB)

	reconcileWorkerService(t, reconciler, service)
	require.NoError(t, reconciler.Delete(ctx, sliceA))
	reconcileWorkerService(t, reconciler, service)

	workerA := workerIDForPod(podA)
	workerB := workerIDForPod(podB)
	deletes := recorderDeletes(recorder)
	assert.Contains(t, deletes, workerA)
	assert.NotContains(t, deletes, workerB)
	_, ok := recorderWorker(recorder, workerB)
	assert.True(t, ok, "remaining worker should stay registered")
}

func TestSelectionTopologyReconcilerRegistersFixtureWorkerOnEverySelectorReplica(t *testing.T) {
	sglangServer := newFixtureSGLangServer(t, nil)
	defer sglangServer.Close()
	selectorA, recorderA := newFakeSelectionServer(t)
	defer selectorA.Close()
	selectorB, recorderB := newFakeSelectionServer(t)
	defer selectorB.Close()

	service, pod, workerSlice := rawNamespacedSGLangWorkerObjects(t, sglangServer.URL, "http://selector.selection.svc", "workers")
	service.Annotations[consts.KubeAnnotationDynamoSelectionServiceURL] = " http://selector.selection.svc/ "
	service.Annotations[consts.KubeAnnotationDynamoSelectionRequireKVEvents] = consts.KubeLabelValueTrue
	selectorService := rawSelectorService()
	selectorService.Namespace = testSelectorNamespace
	selectorSliceA := selectorEndpointSliceForURL(t, "selector-a", selectorA.URL, selectorService.Namespace)
	selectorSliceB := selectorEndpointSliceForURL(t, "selector-b", selectorB.URL, selectorService.Namespace)
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, workerSlice, selectorService, selectorSliceA, selectorSliceB)

	reconcileWorkerService(t, reconciler, service)

	postsA := recorderPosts(recorderA)
	postsB := recorderPosts(recorderB)
	require.Len(t, postsA, 1)
	require.Len(t, postsB, 1)
	selectorScopeKey := selectorServiceScopeKey(testSelectorNamespace, "selector", "")
	assertFixtureBackedSGLangWorker(t, postsA[0], service, pod, sglangServer.URL, selectorScopeKey)
	assertFixtureBackedSGLangWorker(t, postsB[0], service, pod, sglangServer.URL, selectorScopeKey)
	assert.Equal(t, postsA[0].Metadata[selectionMetadataSelectorURL], postsB[0].Metadata[selectionMetadataSelectorURL])
}

func TestSelectionTopologyReconcilerUsesSelectorEndpointSliceTargetPort(t *testing.T) {
	sglangServer := newFixtureSGLangServer(t, nil)
	defer sglangServer.Close()
	selectorServer, recorder := newFakeSelectionServer(t)
	defer selectorServer.Close()

	service, pod, workerSlice := rawNamespacedSGLangWorkerObjects(t, sglangServer.URL, "http://selector.selection.svc:80", "workers")
	service.Annotations[consts.KubeAnnotationDynamoSelectionRequireKVEvents] = consts.KubeLabelValueTrue
	selectorService := rawSelectorService()
	selectorService.Namespace = testSelectorNamespace
	selectorHost, selectorTargetPort := splitTestServerAddress(t, selectorServer.URL)
	selectorSlice := selectorEndpointSlice("selector-a", selectorHost, selectorTargetPort, selectorService.Namespace)
	selectorService.Spec.Ports = []corev1.ServicePort{
		{
			Name:       consts.DynamoSystemPortName,
			Port:       80,
			TargetPort: intstr.FromInt(int(selectorTargetPort)),
		},
	}
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, workerSlice, selectorService, selectorSlice)

	reconcileWorkerService(t, reconciler, service)

	posts := recorderPosts(recorder)
	require.Len(t, posts, 1)
	assert.Equal(t, selectorServiceScopeKey(testSelectorNamespace, "selector", "80"), posts[0].Metadata[selectionMetadataSelectorURL])
}

func TestSelectionTopologyReconcilerFallsBackToEndpointSlicePortForNamedSelectorTargetPort(t *testing.T) {
	sglangServer := newFixtureSGLangServer(t, nil)
	defer sglangServer.Close()
	selectorServer, recorder := newFakeSelectionServer(t)
	defer selectorServer.Close()

	service, pod, workerSlice := rawNamespacedSGLangWorkerObjects(t, sglangServer.URL, "http://selector.selection.svc:80", "workers")
	service.Annotations[consts.KubeAnnotationDynamoSelectionRequireKVEvents] = consts.KubeLabelValueTrue
	selectorService := rawSelectorService()
	selectorService.Namespace = testSelectorNamespace
	selectorHost, selectorTargetPort := splitTestServerAddress(t, selectorServer.URL)
	selectorSlice := selectorEndpointSlice("selector-a", selectorHost, selectorTargetPort, selectorService.Namespace)
	selectorService.Spec.Ports = []corev1.ServicePort{
		{
			Port:       80,
			TargetPort: intstr.FromString(consts.DynamoSystemPortName),
		},
	}
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, workerSlice, selectorService, selectorSlice)

	reconcileWorkerService(t, reconciler, service)

	posts := recorderPosts(recorder)
	require.Len(t, posts, 1)
	assert.Equal(t, selectorServiceScopeKey(testSelectorNamespace, "selector", "80"), posts[0].Metadata[selectionMetadataSelectorURL])
}

func TestSelectionTopologyReconcilerUsesStableSelectorScopeForServiceTargets(t *testing.T) {
	ctx := context.Background()
	rawURL := "http://selector.selection.svc"
	scopeKey := selectorServiceScopeKey(testSelectorNamespace, "selector", "")

	t.Run("service DNS fallback", func(t *testing.T) {
		reconciler := newSelectionTopologyTestReconciler(t)
		selectorTargets, err := reconciler.selectorTargetsForURL(ctx, rawURL, "workers")
		require.NoError(t, err)
		require.Equal(t, []selectorTarget{{
			TargetURL: rawURL,
			ScopeKey:  scopeKey,
		}}, selectorTargets)
	})

	t.Run("resolved service endpoints", func(t *testing.T) {
		selectorService := rawSelectorService()
		selectorService.Namespace = testSelectorNamespace
		selectorSlice := selectorEndpointSlice("selector-a", "10.2.0.3", 8092, selectorService.Namespace)
		reconciler := newSelectionTopologyTestReconciler(t, selectorService, selectorSlice)
		selectorTargets, err := reconciler.selectorTargetsForURL(ctx, rawURL, "workers")
		require.NoError(t, err)
		require.Equal(t, []selectorTarget{{
			TargetURL: "http://10.2.0.3:8092",
			ScopeKey:  scopeKey,
		}}, selectorTargets)
	})

	t.Run("invalid URL", func(t *testing.T) {
		reconciler := newSelectionTopologyTestReconciler(t)
		_, err := reconciler.selectorTargetsForURL(ctx, "http://", "workers")
		require.ErrorContains(t, err, "must include a host")
	})
}

func TestSelectionTopologyReconcilerRequeuesWhenSelectorHasNoReadyTargets(t *testing.T) {
	sglangServer := newFakeSGLangServer(t)
	defer sglangServer.Close()

	service, pod, workerSlice := rawNamespacedSGLangWorkerObjects(t, sglangServer.URL, "http://selector.selection.svc", "workers")
	selectorService := rawSelectorService()
	selectorService.Namespace = testSelectorNamespace
	selectorSlice := selectorEndpointSlice("selector-a", "10.2.0.3", 8092, selectorService.Namespace)
	selectorSlice.Endpoints[0].Conditions.Ready = ptr.To(false)
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, workerSlice, selectorService, selectorSlice)

	result, err := reconciler.Reconcile(context.Background(), reconcile.Request{NamespacedName: client.ObjectKeyFromObject(service)})

	require.NoError(t, err)
	assert.Equal(t, selectionTopologyRequeueAfter, result.RequeueAfter)
}

func TestSelectionTopologyReconcilerSelectorEndpointSliceChangeEnqueuesWorkerService(t *testing.T) {
	ctx := context.Background()
	sglangServer := newFakeSGLangServer(t)
	defer sglangServer.Close()
	selectorA, recorderA := newFakeSelectionServer(t)
	defer selectorA.Close()
	selectorB, recorderB := newFakeSelectionServer(t)
	defer selectorB.Close()

	service, pod, workerSlice := rawNamespacedSGLangWorkerObjects(t, sglangServer.URL, "http://selector.selection.svc", "workers")
	selectorService := rawSelectorService()
	selectorService.Namespace = testSelectorNamespace
	selectorSliceA := selectorEndpointSliceForURL(t, "selector-a", selectorA.URL, selectorService.Namespace)
	selectorSliceB := selectorEndpointSliceForURL(t, "selector-b", selectorB.URL, selectorService.Namespace)
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, workerSlice, selectorService, selectorSliceA)

	reconcileWorkerService(t, reconciler, service)
	require.NoError(t, reconciler.Create(ctx, selectorSliceB))
	requests := reconciler.findWorkerServicesForEndpointSlice(ctx, selectorSliceB)
	assert.True(t, containsRequest(requests, client.ObjectKeyFromObject(service)), "requests = %#v", requests)
	reconcileWorkerService(t, reconciler, service)

	postsA := recorderPosts(recorderA)
	postsB := recorderPosts(recorderB)
	assert.NotEmpty(t, postsA)
	require.Len(t, postsB, 1)
	assert.Equal(t, workerIDForPod(pod), postsB[0].WorkerID)
}

func TestSelectionTopologyReconcilerIgnoresInactiveSelectorCleanupFailure(t *testing.T) {
	ctx := context.Background()
	sglangServer := newFakeSGLangServer(t)
	defer sglangServer.Close()
	selectorA, recorderA := newFakeSelectionServer(t)
	selectorB, recorderB := newFakeSelectionServer(t)
	defer selectorB.Close()

	service, pod, workerSlice := rawNamespacedSGLangWorkerObjects(t, sglangServer.URL, "http://selector.selection.svc", "workers")
	selectorService := rawSelectorService()
	selectorService.Namespace = testSelectorNamespace
	selectorSliceA := selectorEndpointSliceForURL(t, "selector-a", selectorA.URL, selectorService.Namespace)
	selectorSliceB := selectorEndpointSliceForURL(t, "selector-b", selectorB.URL, selectorService.Namespace)
	reconciler := newSelectionTopologyTestReconciler(t, service, pod, workerSlice, selectorService, selectorSliceA)

	reconcileWorkerService(t, reconciler, service)
	require.Len(t, recorderPosts(recorderA), 1)

	selectorA.Close()
	require.NoError(t, reconciler.Delete(ctx, selectorSliceA))
	require.NoError(t, reconciler.Create(ctx, selectorSliceB))
	reconcileWorkerService(t, reconciler, service)

	postsB := recorderPosts(recorderB)
	require.Len(t, postsB, 1)
	assert.Equal(t, workerIDForPod(pod), postsB[0].WorkerID)
}

func TestSelectionTopologyReconcilerWatchMappers(t *testing.T) {
	ctx := context.Background()
	service, workerPod, workerSlice := rawNamespacedSGLangWorkerObjects(t, "http://10.0.0.1:30000", "http://selector.selection.svc", "workers")
	service.Annotations[consts.KubeAnnotationDynamoSelectionServiceURL] = " http://selector.selection.svc/ "
	selectorService := rawSelectorService()
	selectorService.Namespace = testSelectorNamespace
	selectorPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "selector-0",
			Namespace: selectorService.Namespace,
			UID:       "selector-pod-uid",
		},
	}
	selectorSlice := selectorEndpointSlice("selector-a", "10.0.1.1", 8080, selectorService.Namespace)
	selectorSlice.Endpoints[0].TargetRef = &corev1.ObjectReference{
		Kind:      "Pod",
		Name:      selectorPod.Name,
		Namespace: selectorPod.Namespace,
	}
	plainService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "plain-service",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationDynamoSelectionTenantID: testTenantA,
			},
		},
	}
	plainPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "plain-worker",
			Namespace: "default",
			UID:       "plain-worker-uid",
		},
	}
	plainSlice := rawSGLangEndpointSlice("10.0.0.2", 30000, plainService.Namespace)
	plainSlice.Labels[discoveryv1.LabelServiceName] = plainService.Name
	plainSlice.Endpoints[0].TargetRef.Name = plainPod.Name
	urlOnlyService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "url-only-service",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationDynamoSelectionServiceURL: "http://selector.default.svc",
			},
		},
	}
	reconciler := newSelectionTopologyTestReconciler(t,
		service, workerPod, workerSlice,
		selectorService, selectorPod, selectorSlice,
		plainService, plainPod, plainSlice, urlOnlyService)

	want := client.ObjectKeyFromObject(service)
	for _, tt := range []struct {
		name string
		got  []reconcile.Request
		want bool
	}{
		{name: "selector Pod", got: reconciler.findWorkerServicesForPod(ctx, selectorPod), want: true},
		{name: "worker Pod", got: reconciler.findWorkerServicesForPod(ctx, workerPod), want: true},
		{name: "worker EndpointSlice", got: reconciler.findWorkerServicesForEndpointSlice(ctx, workerSlice), want: true},
		{name: "non-selection worker Pod", got: reconciler.findWorkerServicesForPod(ctx, plainPod)},
		{name: "non-selection worker EndpointSlice", got: reconciler.findWorkerServicesForEndpointSlice(ctx, plainSlice)},
		{name: "selector ref without adapter", got: reconciler.findWorkerServicesForSelectorKey(ctx, types.NamespacedName{Namespace: "default", Name: "selector"})},
	} {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, containsRequest(tt.got, want), "requests = %#v", tt.got)
		})
	}
}

func TestSelectionTopologyServicePredicate(t *testing.T) {
	predicate := selectionTopologyServicePredicate()
	workerService := rawSGLangWorkerService("http://selector.default.svc")
	plainService := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "plain-service",
			Namespace: "default",
		},
	}
	auxiliaryAnnotationOnly := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "tenant-only-service",
			Namespace: "default",
			Annotations: map[string]string{
				consts.KubeAnnotationDynamoSelectionTenantID: testTenantA,
			},
		},
	}
	removedAnnotations := workerService.DeepCopy()
	removedAnnotations.Annotations = nil

	createTests := []struct {
		name    string
		service *corev1.Service
		want    bool
	}{
		{name: "plain service", service: plainService, want: false},
		{name: "auxiliary annotation only", service: auxiliaryAnnotationOnly, want: false},
		{name: "annotated worker service", service: workerService, want: true},
	}
	for _, tt := range createTests {
		t.Run("create "+tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, predicate.Create(event.CreateEvent{Object: tt.service}))
		})
	}

	updateTests := []struct {
		name string
		old  *corev1.Service
		new  *corev1.Service
		want bool
	}{
		{name: "neither version has activation annotations", old: plainService, new: plainService.DeepCopy(), want: false},
		{name: "old version had activation annotations", old: workerService, new: removedAnnotations, want: true},
		{name: "new version has activation annotations", old: plainService, new: workerService, want: true},
	}
	for _, tt := range updateTests {
		t.Run("update "+tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, predicate.Update(event.UpdateEvent{ObjectOld: tt.old, ObjectNew: tt.new}))
		})
	}

	deleteTests := []struct {
		name    string
		service *corev1.Service
		want    bool
	}{
		{name: "annotated worker service", service: workerService, want: true},
		{name: "plain service", service: plainService, want: false},
	}
	for _, tt := range deleteTests {
		t.Run("delete "+tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, predicate.Delete(event.DeleteEvent{Object: tt.service}))
		})
	}
}

func containsRequest(requests []reconcile.Request, want types.NamespacedName) bool {
	for _, request := range requests {
		if request.NamespacedName == want {
			return true
		}
	}
	return false
}

type fakeSGLangServer struct {
	*httptest.Server
	invalid              bool
	omitKVEvents         bool
	zeroMaxPrefillTokens bool
}

func newFakeSGLangServer(t *testing.T) *fakeSGLangServer {
	t.Helper()
	fakeServer := &fakeSGLangServer{}
	fakeServer.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_, _ = w.Write([]byte(`{"data":[{"id":"qwen"}]}`))
		case "/model_info":
			_, _ = w.Write([]byte(`{"model_path":"/models/qwen","is_generation":true}`))
		case "/server_info":
			if fakeServer.invalid {
				_, _ = w.Write([]byte(`{"page_size":0,"disaggregation_mode":"null"}`))
				return
			}
			if fakeServer.omitKVEvents {
				_, _ = w.Write([]byte(`{"page_size":16,"dp_size":2,"max_prefill_tokens":1024,"max_total_num_tokens":4096,"disaggregation_mode":"null"}`))
				return
			}
			if fakeServer.zeroMaxPrefillTokens {
				_, _ = w.Write([]byte(`{"page_size":16,"dp_size":2,"max_prefill_tokens":0,"max_total_num_tokens":4096,"disaggregation_mode":"null","kv_events":{"publisher":"zmq","endpoint_host":"*","endpoint_port_base":5557,"topic":"","block_size":16,"dp_size":2}}`))
				return
			}
			_, _ = w.Write([]byte(`{"page_size":16,"dp_size":2,"max_prefill_tokens":1024,"max_total_num_tokens":4096,"disaggregation_mode":"null","kv_events":{"publisher":"zmq","endpoint_host":"*","endpoint_port_base":5557,"topic":"","block_size":16,"dp_size":2}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	return fakeServer
}

func newFixtureSGLangServer(t *testing.T, mutateServerInfo func(map[string]any)) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			_, _ = w.Write(sglangFixtureBytes(t, "models.json"))
		case "/model_info":
			_, _ = w.Write(sglangFixtureBytes(t, "model_info_generation.json"))
		case "/server_info":
			payload := sglangFixtureBytes(t, "server_info_qwen3_huggingface_legacy_kv_events_config.json")
			if mutateServerInfo != nil {
				var serverInfo map[string]any
				if err := json.Unmarshal(payload, &serverInfo); err != nil {
					t.Fatalf("decode SGLang server_info fixture: %v", err)
				}
				mutateServerInfo(serverInfo)
				var err error
				payload, err = json.Marshal(serverInfo)
				if err != nil {
					t.Fatalf("encode SGLang server_info fixture: %v", err)
				}
			}
			_, _ = w.Write(payload)
		default:
			http.NotFound(w, r)
		}
	}))
}

func sglangFixtureBytes(t *testing.T, name string) []byte {
	t.Helper()
	payload, err := os.ReadFile(filepath.Join("..", "enginemetadata", "sglang", "testdata", name))
	require.NoErrorf(t, err, "read SGLang fixture %s", name)
	return payload
}

func testSGLangMetadataSnapshot(t *testing.T) sglang.MetadataSnapshot {
	t.Helper()
	return sglang.MetadataSnapshot{
		Models: sglang.ModelsResponse{Data: []sglang.Model{{ID: "qwen"}}},
		ModelInfo: sglang.ModelInfo{
			ModelPath:    "/models/qwen",
			IsGeneration: ptr.To(true),
		},
		ServerInfo: jsonRoundTripSGLangServerInfo(t, map[string]any{
			"page_size":             16,
			"dp_size":               2,
			"max_total_num_tokens":  4096,
			"max_prefill_tokens":    1024,
			"disaggregation_mode":   "null",
			"speculative_algorithm": "EAGLE",
			"kv_events": map[string]any{
				"publisher":          "zmq",
				"endpoint_host":      "*",
				"endpoint_port_base": 5557,
				"topic":              "",
				"block_size":         16,
				"dp_size":            2,
			},
		}),
	}
}

func jsonRoundTripSGLangServerInfo(t *testing.T, values map[string]any) sglang.ServerInfo {
	t.Helper()
	payload, err := json.Marshal(values)
	require.NoError(t, err)
	var result sglang.ServerInfo
	require.NoError(t, json.Unmarshal(payload, &result))
	return result
}

func assertFixtureBackedSGLangWorker(
	t *testing.T,
	worker selectionservice.WorkerRequest,
	service *corev1.Service,
	pod *corev1.Pod,
	workerEndpoint string,
	selectorScopeKey string,
) {
	t.Helper()
	assert.Equal(t, workerIDForPod(pod), worker.WorkerID)
	assert.Equal(t, "Qwen/Qwen3-0.6B", worker.ModelName)
	assert.Equal(t, testTenantA, worker.TenantID)
	assert.Equal(t, workerEndpoint, worker.Endpoint)
	assert.Equal(t, uint32(16), worker.BlockSize)
	assert.Equal(t, uint32(1), worker.DataParallelSize)
	assert.Equal(t, uint32(0), worker.DataParallelStartRank)
	assert.Equal(t, uint64(16384), worker.MaxNumBatchedTokens)
	assert.Equal(t, uint64(10570), worker.TotalKVBlocks)
	parsedEndpoint, err := url.Parse(workerEndpoint)
	require.NoError(t, err)
	kvHost := parsedEndpoint.Hostname()
	assert.Equal(t, "tcp://"+net.JoinHostPort(kvHost, "5557"), worker.KVEventsEndpoints[0])
	assert.Empty(t, worker.StableRoutingID)
	assert.Equal(t, "zone-a", worker.TopologyDomains["zone"])
	assert.Equal(t, selectionMetadataManagedByVal, worker.Metadata[selectionMetadataManagedBy])
	assert.Equal(t, testOwnerKindService, worker.Metadata[selectionMetadataOwnerKind])
	assert.Equal(t, service.Namespace, worker.Metadata[selectionMetadataOwnerNS])
	assert.Equal(t, service.Name, worker.Metadata[selectionMetadataOwnerName])
	assert.Equal(t, string(service.UID), worker.Metadata[selectionMetadataOwnerUID])
	assert.Equal(t, selectionAdapterExternalSGLang, worker.Metadata[selectionMetadataAdapter])
	assert.Equal(t, selectorScopeKey, worker.Metadata[selectionMetadataSelectorURL])
}

func newFakeSelectionServer(t *testing.T) (*httptest.Server, *selectionRecorder) {
	t.Helper()
	recorder := &selectionRecorder{workers: map[uint64]selectionservice.WorkerRecord{}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == selectionWorkersPath:
			var worker selectionservice.WorkerRequest
			if err := json.NewDecoder(r.Body).Decode(&worker); err != nil {
				t.Fatalf("decode worker request: %v", err)
			}
			recorder.mu.Lock()
			recorder.posts = append(recorder.posts, worker)
			record := selectionWorkerRecordFromRequest(worker)
			recorder.workers[worker.WorkerID] = record
			recorder.mu.Unlock()
			w.WriteHeader(http.StatusCreated)
			if err := json.NewEncoder(w).Encode(record); err != nil {
				t.Fatalf("encode worker record: %v", err)
			}
		case r.Method == http.MethodGet && r.URL.Path == selectionWorkersPath:
			recorder.mu.Lock()
			recorder.getQueries = append(recorder.getQueries, r.URL.Query())
			workers := make([]selectionservice.WorkerRecord, 0, len(recorder.workers))
			for _, worker := range recorder.workers {
				if modelName := r.URL.Query().Get("model_name"); modelName != "" && worker.ModelName != modelName {
					continue
				}
				if tenantID := r.URL.Query().Get("tenant_id"); tenantID != "" && worker.TenantID != tenantID {
					continue
				}
				workers = append(workers, worker)
			}
			recorder.mu.Unlock()
			if err := json.NewEncoder(w).Encode(workers); err != nil {
				t.Fatalf("encode workers: %v", err)
			}
		case r.Method == http.MethodDelete:
			parts := strings.Split(strings.TrimPrefix(r.URL.Path, selectionWorkersPath+"/"), "/")
			workerID, err := strconv.ParseUint(parts[0], 10, 64)
			if err != nil {
				t.Fatalf("parse worker delete path %q: %v", r.URL.Path, err)
			}
			recorder.mu.Lock()
			recorder.deletes = append(recorder.deletes, workerID)
			record := recorder.workers[workerID]
			record.WorkerID = workerID
			record.Lifecycle = selectionservice.WorkerLifecycleUnschedulable
			recorder.workers[workerID] = record
			recorder.mu.Unlock()
			if err := json.NewEncoder(w).Encode(record); err != nil {
				t.Fatalf("encode deactivated worker record: %v", err)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	return server, recorder
}

func selectionWorkerRecordFromRequest(worker selectionservice.WorkerRequest) selectionservice.WorkerRecord {
	return selectionservice.WorkerRecord{
		WorkerID:              worker.WorkerID,
		ModelName:             worker.ModelName,
		TenantID:              worker.TenantID,
		Lifecycle:             selectionservice.WorkerLifecycleSchedulable,
		Endpoint:              worker.Endpoint,
		KVEventsEndpoints:     worker.KVEventsEndpoints,
		BlockSize:             ptr.To(worker.BlockSize),
		DataParallelStartRank: ptr.To(worker.DataParallelStartRank),
		DataParallelSize:      ptr.To(worker.DataParallelSize),
		MaxNumBatchedTokens:   ptr.To(worker.MaxNumBatchedTokens),
		TotalKVBlocks:         ptr.To(worker.TotalKVBlocks),
		StableRoutingID:       worker.StableRoutingID,
		IsEagle:               worker.IsEagle,
		TopologyDomains:       worker.TopologyDomains,
		Metadata:              worker.Metadata,
	}
}

func rawSGLangPod(selectorURL string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-0",
			Namespace: "default",
			UID:       "pod-uid",
			Annotations: map[string]string{
				consts.KubeAnnotationDynamoSelectionAdapter:    selectionAdapterExternalSGLang,
				consts.KubeAnnotationDynamoSelectionServiceURL: selectorURL,
				consts.KubeAnnotationDynamoSelectionTenantID:   testTenantA,
			},
			Labels: map[string]string{
				consts.DynamoTopologyLabelKey("zone"): "zone-a",
			},
		},
	}
}

func rawSGLangWorkerObjects(t *testing.T, workerURL, selectorURL string) (*corev1.Service, *corev1.Pod, *discoveryv1.EndpointSlice) {
	t.Helper()
	return rawNamespacedSGLangWorkerObjects(t, workerURL, selectorURL, "default")
}

func rawNamespacedSGLangWorkerObjects(t *testing.T, workerURL, selectorURL, namespace string) (*corev1.Service, *corev1.Pod, *discoveryv1.EndpointSlice) {
	t.Helper()
	host, port := splitTestServerAddress(t, workerURL)
	service := rawSGLangWorkerService(selectorURL)
	service.Namespace = namespace
	pod := rawSGLangPod("")
	pod.Namespace = namespace
	clearSelectionAnnotations(pod)
	return service, pod, rawSGLangEndpointSlice(host, port, namespace)
}

func clearSelectionAnnotations(pod *corev1.Pod) {
	for _, key := range []string{
		consts.KubeAnnotationDynamoSelectionAdapter,
		consts.KubeAnnotationDynamoSelectionServiceURL,
		consts.KubeAnnotationDynamoSelectionTenantID,
		consts.KubeAnnotationDynamoSelectionRequireKVEvents,
	} {
		delete(pod.Annotations, key)
	}
}

func rawSGLangWorkerService(selectorURL string) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-service",
			Namespace: "default",
			UID:       "worker-service-uid",
			Annotations: map[string]string{
				consts.KubeAnnotationDynamoSelectionAdapter:    selectionAdapterExternalSGLang,
				consts.KubeAnnotationDynamoSelectionServiceURL: selectorURL,
				consts.KubeAnnotationDynamoSelectionTenantID:   testTenantA,
			},
		},
	}
}

func selectionTestWorkerMetadata(service *corev1.Service, selectorScopeKey string, adapter string) map[string]string {
	return map[string]string{
		selectionMetadataManagedBy:   selectionMetadataManagedByVal,
		selectionMetadataOwnerKind:   testOwnerKindService,
		selectionMetadataOwnerNS:     service.Namespace,
		selectionMetadataOwnerName:   service.Name,
		selectionMetadataOwnerUID:    string(service.UID),
		selectionMetadataAdapter:     adapter,
		selectionMetadataSelectorURL: selectorScopeKey,
	}
}

func rawSGLangEndpointSlice(address string, port int32, namespace string) *discoveryv1.EndpointSlice {
	return &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "slice-0",
			Namespace: namespace,
			Labels: map[string]string{
				discoveryv1.LabelServiceName: "worker-service",
			},
		},
		Ports: []discoveryv1.EndpointPort{
			{
				Name: ptr.To(consts.DynamoSystemPortName),
				Port: ptr.To(port),
			},
		},
		Endpoints: []discoveryv1.Endpoint{
			{
				Addresses:  []string{address},
				Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Name:      "worker-0",
					Namespace: namespace,
				},
			},
		},
	}
}

func rawSelectorService() *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "selector",
			Namespace: "default",
			UID:       "selector-service-uid",
		},
	}
}

func selectorEndpointSlice(name string, address string, port int32, namespace string) *discoveryv1.EndpointSlice {
	return &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				discoveryv1.LabelServiceName: "selector",
			},
		},
		Ports: []discoveryv1.EndpointPort{
			{
				Name: ptr.To(consts.DynamoSystemPortName),
				Port: ptr.To(port),
			},
		},
		Endpoints: []discoveryv1.Endpoint{
			{
				Addresses:  []string{address},
				Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(true)},
			},
		},
	}
}

func selectorEndpointSliceForURL(t *testing.T, name, rawURL, namespace string) *discoveryv1.EndpointSlice {
	t.Helper()
	host, port := splitTestServerAddress(t, rawURL)
	return selectorEndpointSlice(name, host, port, namespace)
}

func newSelectionTopologyTestReconciler(t *testing.T, objects ...client.Object) *SelectionTopologyReconciler {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	require.NoError(t, discoveryv1.AddToScheme(scheme))
	return &SelectionTopologyReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(scheme).
			WithIndex(&discoveryv1.EndpointSlice{}, selectionEndpointSlicePodIndex, func(obj client.Object) []string {
				return endpointSliceTargetPodNames(obj.(*discoveryv1.EndpointSlice))
			}).
			WithIndex(&corev1.Service{}, selectorServiceIndex, func(obj client.Object) []string {
				return selectorServiceIndexKeys(obj.(*corev1.Service))
			}).
			WithObjects(objects...).
			Build(),
		SelectionClientFactory: defaultSelectionClientFactory,
	}
}

func splitTestServerAddress(t *testing.T, rawURL string) (string, int32) {
	t.Helper()
	parsed, err := url.Parse(rawURL)
	require.NoError(t, err)
	host, portString, err := net.SplitHostPort(parsed.Host)
	require.NoError(t, err)
	port, err := strconv.ParseInt(portString, 10, 32)
	require.NoError(t, err)
	return host, int32(port)
}
