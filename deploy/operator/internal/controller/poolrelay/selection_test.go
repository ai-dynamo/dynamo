/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDynamoPoolRelaySelection(t *testing.T) {
	for _, tc := range []struct {
		name          string
		selector      *metav1.LabelSelector
		refs          []string
		want, missing []string
		invalid       bool
	}{
		{name: "labels", selector: &metav1.LabelSelector{MatchLabels: map[string]string{"group": "selected"}}, want: []string{"a"}},
		{name: "refs", refs: []string{"b"}, want: []string{"b"}},
		{name: "union", selector: &metav1.LabelSelector{MatchLabels: map[string]string{"group": "selected"}}, refs: []string{"b", "a", "missing"}, want: []string{"a", "b"}, missing: []string{"missing"}},
		{name: "all", selector: &metav1.LabelSelector{}, want: []string{"a", "b"}},
		{name: "empty-result", selector: &metav1.LabelSelector{MatchLabels: map[string]string{"group": "absent"}}, want: []string{}},
		{name: "deleting", refs: []string{"deleting"}, want: []string{}, missing: []string{"deleting"}},
		{name: "unspecified", invalid: true},
		{name: "duplicate", refs: []string{"a", "a"}, invalid: true},
		{name: "empty-name", refs: []string{""}, invalid: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Create namespace-local and foreign DGDs, including one being deleted")
			scheme := runtime.NewScheme()
			require.NoError(t, v1beta1.AddToScheme(scheme))
			now := metav1.Now()
			objects := []client.Object{
				&v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "a", Namespace: "local", Labels: map[string]string{"group": "selected"}}},
				&v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "b", Namespace: "local"}},
				&v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "foreign", Namespace: "other", Labels: map[string]string{"group": "selected"}}},
				&v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "deleting", Namespace: "local", DeletionTimestamp: &now, Finalizers: []string{"test"}}},
			}
			c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithObjects(objects...).Build()
			t.Log("Resolve the union without expanding a missing selector to wildcard")
			selected, missing, err := selectRelayDGDs(context.Background(), c, "local", tc.selector, tc.refs)
			if tc.invalid {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			names := make([]string, 0, len(selected))
			for _, dgd := range selected {
				names = append(names, dgd.Name)
			}
			require.Equal(t, tc.want, names)
			require.ElementsMatch(t, tc.missing, missing)
		})
	}
}

func TestDynamoPoolRelaySelectionLabelAndIdentityChanges(t *testing.T) {
	t.Log("Select a DGD through both labels and an explicit reference")
	ctx := context.Background()
	scheme := runtime.NewScheme()
	require.NoError(t, v1beta1.AddToScheme(scheme))
	dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "local", UID: "old", Generation: 1, Labels: map[string]string{"group": "selected"}}}
	c := fake.NewClientBuilder().WithReturnManagedFields().WithScheme(scheme).WithObjects(dgd).Build()
	selector := &metav1.LabelSelector{MatchLabels: map[string]string{"group": "selected"}}
	selected, _, err := selectRelayDGDs(ctx, c, "local", selector, []string{"model"})
	require.NoError(t, err)
	require.Len(t, selected, 1)

	t.Log("Changing labels without generation retains refs but withdraws selector-only selection")
	dgd.Labels = nil
	require.NoError(t, c.Update(ctx, dgd))
	selected, _, err = selectRelayDGDs(ctx, c, "local", selector, []string{"model"})
	require.NoError(t, err)
	require.Len(t, selected, 1)
	selected, _, err = selectRelayDGDs(ctx, c, "local", selector, nil)
	require.NoError(t, err)
	require.Empty(t, selected)

	t.Log("Deletion reports a missing ref and recreation carries the replacement UID")
	require.NoError(t, c.Delete(ctx, dgd))
	_, missing, err := selectRelayDGDs(ctx, c, "local", nil, []string{"model"})
	require.NoError(t, err)
	require.Equal(t, []string{"model"}, missing)
	replacement := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "model", Namespace: "local", UID: "new"}}
	require.NoError(t, c.Create(ctx, replacement))
	selected, missing, err = selectRelayDGDs(ctx, c, "local", nil, []string{"model"})
	require.NoError(t, err)
	require.Empty(t, missing)
	require.Equal(t, replacement.UID, selected[0].UID)
}
