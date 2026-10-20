// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestDGDResourceSyncerDeleteIfControlledBy(t *testing.T) {
	const name = "aux-resource"
	const seededUID = types.UID("seeded-uid")
	errGet := errors.New("apiserver unavailable")
	errDeleteNotFound := apierrors.NewNotFound(schema.GroupResource{Resource: "objects"}, name)
	errConflict := apierrors.NewConflict(schema.GroupResource{Resource: "objects"}, name, errors.New("UID precondition failed"))

	// The protocol does not depend on the object kind, so the full matrix runs
	// for Service. ConfigMap runs only the error cases, which check its label.
	kinds := []struct {
		label     string
		obj       client.Object
		errorOnly bool
	}{
		{label: "service", obj: &corev1.Service{}},
		{label: "ConfigMap", obj: &corev1.ConfigMap{}, errorOnly: true},
	}

	cases := []struct {
		name string
		// seedController is the controller of the seeded object: "dgd",
		// "other", or "" for no seeded object.
		seedController string
		getErr         error
		deleteErr      error
		// wantErr is the exact error text with %s for the kind label; "" means success.
		wantErr      string
		wantCause    error
		wantConflict bool
		wantDelete   bool
		wantPresent  bool
	}{
		{
			name:           "owned object is deleted pinned to its UID",
			seedController: "dgd",
			wantDelete:     true,
		},
		{
			name:           "object controlled by another owner is left alone",
			seedController: "other",
			wantPresent:    true,
		},
		{
			name: "missing object is success",
		},
		{
			name:           "get failure is wrapped with the kind label",
			seedController: "dgd",
			getErr:         errGet,
			wantErr:        "failed to get %s " + name + ": " + errGet.Error(),
			wantCause:      errGet,
			wantPresent:    true,
		},
		{
			name:           "delete-time NotFound is success",
			seedController: "dgd",
			deleteErr:      errDeleteNotFound,
			wantDelete:     true,
			wantPresent:    true,
		},
		{
			name:           "delete conflict propagates",
			seedController: "dgd",
			deleteErr:      errConflict,
			wantErr:        "failed to delete %s " + name + ": " + errConflict.Error(),
			wantCause:      errConflict,
			wantConflict:   true,
			wantDelete:     true,
			wantPresent:    true,
		},
	}

	for _, kind := range kinds {
		for _, tc := range cases {
			if kind.errorOnly && tc.wantErr == "" {
				continue
			}
			t.Run(kind.label+"/"+tc.name, func(t *testing.T) {
				t.Log("Build the DGD and, when the case has one, the seeded object with its controller reference")
				ctx := context.Background()
				scheme := newDynamoGraphDeploymentControllerTestScheme(t)
				dgd := &v1beta1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{Name: "my-dgd", Namespace: "default", UID: types.UID("dgd-uid")},
				}
				other := &v1beta1.DynamoGraphDeployment{
					ObjectMeta: metav1.ObjectMeta{Name: "other-dgd", Namespace: "default", UID: types.UID("other-uid")},
				}
				objects := []client.Object{dgd}
				if tc.seedController != "" {
					seeded := kind.obj.DeepCopyObject().(client.Object)
					seeded.SetName(name)
					seeded.SetNamespace(dgd.Namespace)
					seeded.SetUID(seededUID)
					controller := dgd
					if tc.seedController == "other" {
						controller = other
					}
					if err := ctrl.SetControllerReference(controller, seeded, scheme); err != nil {
						t.Fatalf("failed to set the controller reference: %v", err)
					}
					objects = append(objects, seeded)
				}

				t.Log("Build a fake client that records Delete calls and injects the case's Get or Delete error")
				var deleteCalls int
				var deleteOptions client.DeleteOptions
				kubeClient := fake.NewClientBuilder().
					WithScheme(scheme).
					WithObjects(objects...).
					WithInterceptorFuncs(interceptor.Funcs{
						Get: func(ctx context.Context, c client.WithWatch, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
							if tc.getErr != nil && key.Name == name {
								return tc.getErr
							}
							return c.Get(ctx, key, obj, opts...)
						},
						Delete: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.DeleteOption) error {
							deleteCalls++
							for _, opt := range opts {
								opt.ApplyToDelete(&deleteOptions)
							}
							if tc.deleteErr != nil {
								return tc.deleteErr
							}
							return c.Delete(ctx, obj, opts...)
						},
					}).
					Build()
				syncer := newDGDResourceSyncer(kubeClient, nil)

				t.Log("Run the shared ownership-safe delete with a fresh object of the kind")
				key := types.NamespacedName{Name: name, Namespace: dgd.Namespace}
				err := syncer.deleteIfControlledBy(ctx, dgd, key, kind.obj.DeepCopyObject().(client.Object), kind.label)

				t.Log("Verify the returned error text, cause, and conflict classification")
				if tc.wantErr == "" {
					if err != nil {
						t.Fatalf("expected success, got %v", err)
					}
				} else {
					if err == nil {
						t.Fatal("expected an error, got nil")
					}
					if want := fmt.Sprintf(tc.wantErr, kind.label); err.Error() != want {
						t.Errorf("error = %q, want %q", err.Error(), want)
					}
					if !errors.Is(err, tc.wantCause) {
						t.Errorf("errors.Is(err, %v) = false for %v", tc.wantCause, err)
					}
					if apierrors.IsConflict(err) != tc.wantConflict {
						t.Errorf("apierrors.IsConflict(err) = %v, want %v", apierrors.IsConflict(err), tc.wantConflict)
					}
				}

				t.Log("Verify whether Delete ran, and that it carried a UID-only precondition")
				if !tc.wantDelete {
					if deleteCalls != 0 {
						t.Errorf("expected no Delete call, got %d", deleteCalls)
					}
				} else {
					if deleteCalls != 1 {
						t.Fatalf("expected one Delete call, got %d", deleteCalls)
					}
					if deleteOptions.Preconditions == nil || deleteOptions.Preconditions.UID == nil {
						t.Fatal("delete carried no UID precondition")
					}
					if got := *deleteOptions.Preconditions.UID; got != seededUID {
						t.Errorf("precondition UID = %q, want %q", got, seededUID)
					}
					if deleteOptions.Preconditions.ResourceVersion != nil {
						t.Errorf("precondition ResourceVersion = %q, want none", *deleteOptions.Preconditions.ResourceVersion)
					}
				}

				t.Log("Verify whether the object is still stored, unless the injected Get error hides it")
				if tc.getErr != nil {
					return
				}
				getErr := kubeClient.Get(ctx, key, kind.obj.DeepCopyObject().(client.Object))
				if tc.wantPresent && getErr != nil {
					t.Errorf("expected the %s to remain, got %v", kind.label, getErr)
				}
				if !tc.wantPresent && !apierrors.IsNotFound(getErr) {
					t.Errorf("expected the %s to be absent, got %v", kind.label, getErr)
				}
			})
		}
	}
}
