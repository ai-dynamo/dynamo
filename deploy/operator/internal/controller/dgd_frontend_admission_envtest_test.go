//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/podcache"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/workermetadata"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func TestFrontendAdmissionReadsWorkerMetadataFromTheCache(t *testing.T) {
	ctx := context.Background()
	env := sharedEnv.ForTest(t)

	t.Log("Count the worker metadata lists the manager sends to the API server")
	var lists atomic.Int64
	config := env.RESTConfig()
	config.WrapTransport = func(next http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(request *http.Request) (*http.Response, error) {
			if request.Method == http.MethodGet &&
				strings.HasSuffix(request.URL.Path, "/dynamoworkermetadatas") &&
				!request.URL.Query().Has("watch") {
				lists.Add(1)
			}
			return next.RoundTrip(request)
		})
	}

	t.Log("Start a manager with the operator's cache configuration and a worker metadata informer")
	cacheOptions := cache.Options{DefaultNamespaces: map[string]cache.Config{env.Namespace(): {}}}
	require.NoError(t, podcache.Configure(&cacheOptions))
	workermetadata.Configure(&cacheOptions)
	mgr, err := ctrl.NewManager(config, ctrl.Options{
		Scheme:  env.Scheme(),
		Metrics: metricsserver.Options{BindAddress: "0"},
		Cache:   cacheOptions,
	})
	require.NoError(t, err)
	_, err = mgr.GetCache().GetInformer(ctx, workermetadata.New())
	require.NoError(t, err)
	managerCtx, stopManager := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- mgr.Start(managerCtx) }()
	t.Cleanup(func() {
		stopManager()
		if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
			t.Errorf("manager stopped with error: %v", err)
		}
	})
	syncCtx, cancelSync := context.WithTimeout(ctx, 30*time.Second)
	defer cancelSync()
	require.True(t, mgr.GetCache().WaitForCacheSync(syncCtx))

	t.Log("A ready frontend publishes the workers it serves")
	frontend := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "frontend-0",
			Namespace: env.Namespace(),
			Labels: map[string]string{
				consts.KubeLabelDynamoGraphDeploymentName: "test-dgd",
				consts.KubeLabelDynamoComponentType:       consts.ComponentTypeFrontend,
				appsv1.DefaultDeploymentUniqueLabelKey:    admissionTestTemplate,
			},
		},
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "frontend", Image: "frontend"}}},
	}
	require.NoError(t, env.Client().Create(ctx, frontend))
	frontend.Status = corev1.PodStatus{
		Phase:      corev1.PodRunning,
		Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
	}
	require.NoError(t, env.Client().Status().Update(ctx, frontend))
	require.NoError(t, env.Client().Create(ctx, admissionTestMetadata(frontend,
		admissionTestRecord(true, "test-dgd-1/backend/generate/1"))))

	t.Log("The rollout reads the record once the cache observes it")
	dgd := createTestDGD("test-dgd", map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
		"frontend": {ComponentType: consts.ComponentTypeFrontend, Replicas: ptr.To(int32(1))},
	})
	dgd.Namespace = env.Namespace()
	rollout := newDGDWorkerRolloutReconciler(mgr.GetClient(), mgr.GetCache(), nil)
	require.Eventually(t, func() bool {
		view, err := rollout.resolveFrontendAdmission(ctx, dgd)
		return err == nil && view.gated && !view.pending &&
			len(view.admissions) == 1 && len(view.admissions[0].Members) == 1
	}, 30*time.Second, 100*time.Millisecond)

	t.Log("Further rollout steps read the metadata without listing it from the API server")
	listed := lists.Load()
	for range 5 {
		_, err := rollout.resolveFrontendAdmission(ctx, dgd)
		require.NoError(t, err)
	}
	assert.Equal(t, listed, lists.Load())
}

// BenchmarkFrontendServingFromCache measures, against a real API server and
// informer cache, what one rollout step reads to decide which replacement
// workers the frontends serve: the frontend view and each role's serving
// counts.
func BenchmarkFrontendServingFromCache(b *testing.B) {
	for _, workersPerRole := range []int{100, 1000} {
		b.Run(fmt.Sprintf("worker_pods=%d", 4*workersPerRole), func(b *testing.B) {
			ctx := context.Background()
			env := sharedEnv.ForTest(b)

			b.Log("Start a manager with the operator's cache configuration and Pod index")
			cacheOptions := cache.Options{DefaultNamespaces: map[string]cache.Config{env.Namespace(): {}}}
			require.NoError(b, podcache.Configure(&cacheOptions))
			workermetadata.Configure(&cacheOptions)
			mgr, err := ctrl.NewManager(env.RESTConfig(), ctrl.Options{
				Scheme:  env.Scheme(),
				Metrics: metricsserver.Options{BindAddress: "0"},
				Cache:   cacheOptions,
			})
			require.NoError(b, err)
			require.NoError(b, mgr.GetFieldIndexer().IndexField(ctx, &corev1.Pod{}, dgdComponentPodIndex, dgdComponentPodIndexValues))
			_, err = mgr.GetCache().GetInformer(ctx, workermetadata.New())
			require.NoError(b, err)
			managerCtx, stopManager := context.WithCancel(ctx)
			done := make(chan error, 1)
			go func() { done <- mgr.Start(managerCtx) }()
			b.Cleanup(func() {
				stopManager()
				<-done
			})
			require.True(b, mgr.GetCache().WaitForCacheSync(ctx))

			b.Log("Create ready worker pods of two roles and two generations, each with its model card")
			var members []string
			for _, role := range []string{consts.ComponentTypePrefill, consts.ComponentTypeDecode} {
				for _, hash := range []string{"oldhash1", "newhash1"} {
					for i := range workersPerRole {
						pod := admissionTestPod(fmt.Sprintf("%s-%s-%d", role, hash, i), role, role, hash, "test-dgd-"+role+"-"+hash, true)
						card := fmt.Sprintf("test-dgd-%s/%s/generate/%x", hash, role, i)
						benchmarkCreateReadyPod(b, env.Client(), env.Namespace(), pod)
						require.NoError(b, env.Client().Create(ctx, admissionTestMetadata(pod, admissionTestCards(card))))
						members = append(members, card)
					}
				}
			}

			b.Log("Two ready frontends serve every worker")
			for i := range 2 {
				frontend := admissionTestFrontendPod(fmt.Sprintf("frontend-%d", i), admissionTestTemplate, true)
				benchmarkCreateReadyPod(b, env.Client(), env.Namespace(), frontend)
				require.NoError(b, env.Client().Create(ctx, admissionTestMetadata(frontend, admissionTestRecord(true, members...))))
			}

			b.Log("Wait until the cache serves every record")
			dgd := admissionTestRollingDGD(2, "0", map[string]int32{})
			dgd.Namespace = env.Namespace()
			rollout := newDGDWorkerRolloutReconciler(mgr.GetClient(), mgr.GetCache(), nil)
			require.Eventually(b, func() bool {
				view, err := rollout.resolveFrontendAdmission(ctx, dgd)
				if err != nil || view.pending || len(view.admissions) != 2 {
					return false
				}
				serving, err := rollout.frontendServing(ctx, dgd, view, consts.ComponentTypeDecode)
				return err == nil && serving.byEveryFrontend["test-dgd-decode-newhash1"] == int32(workersPerRole)
			}, 5*time.Minute, 500*time.Millisecond)

			b.Log("Read the view and both roles' serving counts")
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				view, err := rollout.resolveFrontendAdmission(ctx, dgd)
				if err != nil {
					b.Fatal(err)
				}
				for _, role := range []string{consts.ComponentTypePrefill, consts.ComponentTypeDecode} {
					if _, err := rollout.frontendServing(ctx, dgd, view, role); err != nil {
						b.Fatal(err)
					}
				}
			}
		})
	}
}

// benchmarkCreateReadyPod creates pod in namespace and marks it Ready.
func benchmarkCreateReadyPod(b *testing.B, kubeClient client.Client, namespace string, pod *corev1.Pod) {
	b.Helper()
	ctx := context.Background()
	pod.Namespace = namespace
	pod.UID = ""
	pod.Spec = corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "worker"}}}
	status := pod.Status
	require.NoError(b, kubeClient.Create(ctx, pod))
	pod.Status = status
	require.NoError(b, kubeClient.Status().Update(ctx, pod))
}
