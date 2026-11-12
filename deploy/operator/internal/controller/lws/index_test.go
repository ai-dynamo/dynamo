/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lws

import (
	"context"
	"errors"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

func TestSetNameIndexValues(t *testing.T) {
	tests := []struct {
		name string
		obj  client.Object
		want []string
	}{
		{
			name: "LWS Pod",
			obj: &corev1.Pod{Labels: map[string]string{
				leaderworkersetv1.SetNameLabelKey: "decode",
			}},
			want: []string{"decode"},
		},
		{name: "Pod without set name", obj: &corev1.Pod{}},
		{name: "non-Pod object", obj: &corev1.ConfigMap{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Verify the index value of the case's object")
			assert.Equal(t, tt.want, SetNameIndexValues(tt.obj))
		})
	}
}

func TestListPodsBySetName(t *testing.T) {
	t.Log("Index Pods for one LWS and leave unrelated Pods nearby")
	reader := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithIndex(&corev1.Pod{}, SetNameIndex, SetNameIndexValues).
		WithObjects(
			&corev1.Pod{
				Name: "decode-0", Namespace: "inference",
				Labels: map[string]string{leaderworkersetv1.SetNameLabelKey: "decode"},
			},
			&corev1.Pod{
				Name: "decode-0-1", Namespace: "inference",
				Labels: map[string]string{leaderworkersetv1.SetNameLabelKey: "decode"},
			},
			&corev1.Pod{
				Name: "other", Namespace: "inference",
				Labels: map[string]string{leaderworkersetv1.SetNameLabelKey: "other"},
			},
			&corev1.Pod{
				Name: "elsewhere", Namespace: "other-namespace",
				Labels: map[string]string{leaderworkersetv1.SetNameLabelKey: "decode"},
			},
		).
		Build()

	t.Log("List only the namespace-scoped indexed candidates")
	pods, err := ListPodsBySetName(t.Context(), reader, "inference", "decode")
	require.NoError(t, err)
	names := make([]string, 0, len(pods))
	for i := range pods {
		names = append(names, pods[i].Name)
	}
	sort.Strings(names)
	assert.Equal(t, []string{"decode-0", "decode-0-1"}, names)
}

func TestListPodsBySetNameReturnsListErrors(t *testing.T) {
	t.Log("Fail every indexed list through the cache")
	reader := fake.NewClientBuilder().
		WithScheme(scheme.Scheme).
		WithIndex(&corev1.Pod{}, SetNameIndex, SetNameIndexValues).
		WithInterceptorFuncs(interceptor.Funcs{
			List: func(context.Context, client.WithWatch, client.ObjectList, ...client.ListOption) error {
				return errors.New("cache unavailable")
			},
		}).
		Build()

	t.Log("Verify infrastructure errors are not turned into Unknown")
	_, err := ListPodsBySetName(t.Context(), reader, "inference", "decode")
	require.ErrorContains(t, err, "cache unavailable")
}
