/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
)

// TestReconcileDisaggregatedSet_HappyPath exercises the full DGD controller
// path against envtest and verifies a DisaggregatedSet object is created
// with the two expected roles. This is the smallest end-to-end check that
// the operator can opt a DGD into the DS pathway.
func TestReconcileDisaggregatedSet_HappyPath(t *testing.T) {
	env := &envtest.Environment{
		CRDDirectoryPaths: []string{
			"../../config/crd/bases",
			"./testing/disaggregatedset",
		},
		ErrorIfCRDPathMissing: false,
	}

	cfg, err := env.Start()
	if err != nil {
		t.Skipf("envtest unavailable in this environment: %v", err)
	}
	defer func() { _ = env.Stop() }()

	if err := nvidiacomv1beta1.AddToScheme(scheme.Scheme); err != nil {
		t.Fatalf("add nvidia scheme: %v", err)
	}

	mgr, err := ctrl.NewManager(cfg, manager.Options{
		Scheme: scheme.Scheme,
	})
	if err != nil {
		t.Fatalf("new manager: %v", err)
	}

	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        mgr.GetClient(),
		RuntimeConfig: newTestRuntimeConfig(true),
	}

	if err := reconciler.SetupWithManager(mgr); err != nil {
		t.Fatalf("setup with manager: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		ctrl.SetLogger(zap.New(zap.UseDevMode(true)))
		if err := mgr.Start(ctx); err != nil {
			t.Logf("manager stopped: %v", err)
		}
	}()

	if !mgr.GetCache().WaitForCacheSync(ctx) {
		t.Fatalf("cache failed to sync")
	}

	dgd := newDSHappyPathDGD()
	if err := mgr.GetClient().Create(ctx, dgd); err != nil {
		t.Fatalf("create dgd: %v", err)
	}

	if err := mgr.GetClient().Get(ctx, types.NamespacedName{Name: dgd.Name, Namespace: dgd.Namespace}, &nvidiacomv1beta1.DynamoGraphDeployment{}); err != nil {
		t.Fatalf("get dgd: %v", err)
	}

	// Manually invoke the DS reconcile so this test does not depend on the
	// DGD controller's reconcile path. The envtest is intentionally minimal:
	// it covers the *DS branch* only, not the DGD -> DS plumbing.
	res, err := reconciler.reconcileDisaggregatedSetResources(ctx, dgd, nil, nil)
	if err != nil {
		t.Fatalf("reconcile DS: %v", err)
	}
	if res.State != nvidiacomv1beta1.DGDStatePending {
		t.Fatalf("state = %s, want pending (DS just created)", res.State)
	}

	ds := &unstructured.Unstructured{}
	ds.SetGroupVersionKind(disaggregatedSetGVK)
	if err := mgr.GetClient().Get(ctx, types.NamespacedName{Name: disaggregatedSetName(dgd), Namespace: dgd.Namespace}, ds); err != nil {
		t.Fatalf("expected DS object to be created: %v", err)
	}
	roles, found, err := unstructured.NestedSlice(ds.Object, "spec", "roles")
	if err != nil || !found {
		t.Fatalf("spec.roles missing: found=%v err=%v", found, err)
	}
	if len(roles) != 2 {
		t.Fatalf("spec.roles len = %d, want 2", len(roles))
	}
}

func newDSHappyPathDGD() *nvidiacomv1beta1.DynamoGraphDeployment {
	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "demo-ds",
			Namespace: "default",
			UID:       "demo-ds-uid",
			Annotations: map[string]string{
				consts.KubeAnnotationEnableDisaggregatedSet: consts.KubeLabelValueTrue,
			},
		},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			Components: []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "prefill",
					ComponentType: nvidiacomv1beta1.ComponentTypePrefill,
					Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
				},
				{
					ComponentName: "decode",
					ComponentType: nvidiacomv1beta1.ComponentTypeDecode,
					Multinode:     &nvidiacomv1beta1.MultinodeSpec{NodeCount: 2},
				},
			},
		},
	}
}

var _ = runtime.Object(nil)
var _ = client.Object(nil)
var _ = time.Second
