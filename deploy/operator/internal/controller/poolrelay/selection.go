/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"fmt"
	"slices"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// selectRelayDGDs resolves a namespace-local union. An omitted selector selects
// nothing; an explicit empty selector selects everything. Deleting references
// remain missing until a replacement appears. reader must be non-nil.
func selectRelayDGDs(ctx context.Context, reader client.Reader, namespace string, selector *metav1.LabelSelector, names []string) ([]v1beta1.DynamoGraphDeployment, []string, error) {
	// Validate the contract before listing, including callers that bypass admission.
	if selector == nil && len(names) == 0 {
		return nil, nil, fmt.Errorf("supply sources.dgds.labelSelector or nonempty sources.dgds.refs")
	}
	requested := make(map[string]bool, len(names))
	for _, name := range names {
		if name == "" {
			return nil, nil, fmt.Errorf("sources.dgds.refs must contain nonempty names")
		}
		if _, exists := requested[name]; exists {
			return nil, nil, fmt.Errorf("duplicate sources.dgds.refs name %q", name)
		}
		requested[name] = false
	}
	matcher := labels.Nothing()
	if selector != nil {
		var err error
		matcher, err = metav1.LabelSelectorAsSelector(selector)
		if err != nil {
			return nil, nil, fmt.Errorf("invalid sources.dgds.labelSelector: %w", err)
		}
	}

	// One cached snapshot avoids inconsistent duplicate reads for refs and labels.
	var list v1beta1.DynamoGraphDeploymentList
	if err := reader.List(ctx, &list, client.InNamespace(namespace)); err != nil {
		return nil, nil, fmt.Errorf("list selected DGDs: %w", err)
	}
	selected := make([]v1beta1.DynamoGraphDeployment, 0)
	for _, dgd := range list.Items {
		if !dgd.DeletionTimestamp.IsZero() {
			continue
		}
		_, referenced := requested[dgd.Name]
		if referenced {
			requested[dgd.Name] = true
		}
		if referenced || matcher.Matches(labels.Set(dgd.Labels)) {
			selected = append(selected, dgd)
		}
	}

	// Stable ordering prevents selection permutations from changing generated sources.
	slices.SortFunc(selected, func(a, b v1beta1.DynamoGraphDeployment) int {
		if a.Name < b.Name {
			return -1
		}
		if a.Name > b.Name {
			return 1
		}
		return 0
	})
	missing := make([]string, 0)
	for name, found := range requested {
		if !found {
			missing = append(missing, name)
		}
	}
	slices.Sort(missing)
	return selected, missing, nil
}
