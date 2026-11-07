// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"errors"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestLPXPriorityPartialPublicationAndRecreation(t *testing.T) {
	for _, lag := range []bool{false, true} {
		name := "partial create failure"
		if lag {
			name = "AlreadyExists with watch lag"
		}
		t.Run(name, func(t *testing.T) {
			t.Log("Publish two replicas with one interrupted or not-yet-observed CREATE")
			ctx := t.Context()
			_, dgd, registry := newLPXTestDGD(t, lpx.PipelineSingle)
			component := dgd.GetComponentByName("lpx")
			component.Replicas = new(int32(2))
			component.LPX.Scheduling = &v1beta1.SchedulingSpec{PriorityClassName: new("old")}
			child := newLPXTestDeployment(t, dgd)
			r, selected := newPreparedLPXTestReconciler(t, registry, ctx, child, dgd)
			objects := lpxMaterializedObjects(t, r, child, dgd, selected)
			createLPXTestObjects(t, ctx, r.Client, objects...)
			pcs := findLPXTestPodCliqueSet(t, objects)
			baseClient := r.Client
			secondName := selected.requests[1].Name
			failCreate, hideSecond := true, false
			r.Client = interceptor.NewClient(r.Client.(client.WithWatch), interceptor.Funcs{
				Create: func(ctx context.Context, delegated client.WithWatch, object client.Object, opts ...client.CreateOption) error {
					request, ok := object.(*lpxv1alpha1.LPUPipelineRequest)
					if ok && request.Name == secondName && failCreate {
						if !lag {
							return errors.New("interrupted publication")
						}
						if err := delegated.Create(ctx, object, opts...); err != nil {
							return err
						}
						return apierrors.NewAlreadyExists(schema.GroupResource{Group: lpxv1alpha1.APIGroup, Resource: "lpupipelinerequests"}, request.Name)
					}
					return delegated.Create(ctx, object, opts...)
				},
				List: func(ctx context.Context, delegated client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
					if err := delegated.List(ctx, list, opts...); err != nil {
						return err
					}
					if requests, ok := list.(*lpxv1alpha1.LPUPipelineRequestList); ok && hideSecond {
						for index := range requests.Items {
							if requests.Items[index].Name == secondName {
								requests.Items = append(requests.Items[:index], requests.Items[index+1:]...)
								break
							}
						}
					}
					return nil
				},
			})
			key := ctrl.Request{NamespacedName: client.ObjectKeyFromObject(child)}
			_, err := r.Reconcile(ctx, key)
			if lag {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, "interrupted publication")
			}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), pcs))
			first := getTestPipelineRequest(t, ctx, baseClient, child.Namespace, selected.requests[0].Name)
			require.Equal(t, new("old"), first.Spec.PriorityClassName)

			t.Log("Retry current intent: only a genuinely missing request takes the edited class")
			component.LPX.Scheduling.PriorityClassName = new("new")
			updateTestDGD(t, r, child, dgd)
			failCreate, hideSecond = false, lag
			_, err = r.Reconcile(ctx, key)
			require.NoError(t, err)
			hideSecond = false
			_, err = r.Reconcile(ctx, key)
			require.NoError(t, err)
			require.Equal(t, first, getTestPipelineRequest(t, ctx, baseClient, first.Namespace, first.Name))
			second := getTestPipelineRequest(t, ctx, baseClient, child.Namespace, secondName)
			expected := "new"
			if lag {
				expected = "old"
			}
			require.Equal(t, new(expected), second.Spec.PriorityClassName)

			t.Log("Ordinary scale-in starts retirement, and scale-out waits for the exact finalizer")
			second.Finalizers = []string{"test.example/scheduler-release"}
			require.NoError(t, r.Update(ctx, second))
			component.Replicas = new(int32(1))
			updateTestDGD(t, r, child, dgd)
			for range 3 {
				_, err = r.Reconcile(ctx, key)
				require.NoError(t, err)
			}
			second = getTestPipelineRequest(t, ctx, baseClient, child.Namespace, secondName)
			require.False(t, second.DeletionTimestamp.IsZero())
			component.Replicas = new(int32(2))
			component.LPX.Scheduling.PriorityClassName = new("replacement")
			updateTestDGD(t, r, child, dgd)
			for range 2 {
				_, err = r.Reconcile(ctx, key)
				require.NoError(t, err)
			}
			require.Equal(t, second, getTestPipelineRequest(t, ctx, baseClient, second.Namespace, secondName))

			t.Log("After disappearance, the new replica gets current policy and the survivor stays byte-identical")
			second.Finalizers = nil
			require.NoError(t, r.Update(ctx, second))
			for range 4 {
				_, err = r.Reconcile(ctx, key)
				require.NoError(t, err)
			}
			replacement := getTestPipelineRequest(t, ctx, baseClient, child.Namespace, secondName)
			require.NotEqual(t, second.UID, replacement.UID)
			require.Equal(t, new("replacement"), replacement.Spec.PriorityClassName)
			require.Equal(t, first, getTestPipelineRequest(t, ctx, baseClient, first.Namespace, first.Name))
			currentPCS := &grovev1alpha1.PodCliqueSet{}
			require.NoError(t, r.Get(ctx, client.ObjectKeyFromObject(pcs), currentPCS))
			require.Equal(t, pcs.UID, currentPCS.UID)
			require.Equal(t, pcs.Spec.Template, currentPCS.Spec.Template)
		})
	}
}
