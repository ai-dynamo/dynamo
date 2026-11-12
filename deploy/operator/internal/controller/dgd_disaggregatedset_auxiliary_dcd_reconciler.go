/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"fmt"
	"sort"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type disaggregatedSetAuxiliaryDCDReconciler struct {
	client   client.Client
	recorder events.EventRecorder
}

func newDisaggregatedSetAuxiliaryDCDReconciler(
	k8sClient client.Client,
	recorder events.EventRecorder,
) *disaggregatedSetAuxiliaryDCDReconciler {
	return &disaggregatedSetAuxiliaryDCDReconciler{
		client:   k8sClient,
		recorder: recorder,
	}
}

func (r *disaggregatedSetAuxiliaryDCDReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment,
	selection disaggregatedSetSelection,
) ([]Resource, bool, error) {
	resources := []Resource{}
	modified := false
	syncer := newDGDResourceSyncer(r.client, r.recorder)
	for _, componentName := range sortedDCDKeys(dcds) {
		dcd := dcds[componentName]
		if _, selected := selection.componentToRole[componentName]; selected {
			continue
		}
		if err := preserveExistingBackendFramework(ctx, r.client, dcd); err != nil {
			return nil, false, fmt.Errorf("failed to preserve existing DynamoComponentDeployment backendFramework: %w", err)
		}
		wasModified, syncedDCD, err := commoncontroller.SyncResource(ctx, &syncer, dgd, func(context.Context) (*nvidiacomv1beta1.DynamoComponentDeployment, bool, error) {
			return dcd, false, nil
		})
		if err != nil {
			return nil, false, fmt.Errorf("failed to sync non-DisaggregatedSet DynamoComponentDeployment %s: %w", dcd.Name, err)
		}
		modified = modified || wasModified
		dcds[componentName] = syncedDCD
		resources = append(resources, syncedDCD)
	}
	return resources, modified, nil
}

func (r *disaggregatedSetAuxiliaryDCDReconciler) DeleteSelected(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	selection disaggregatedSetSelection,
) error {
	dcds, err := r.listOwnedSelectedDCDs(ctx, dgd, selection)
	if err != nil {
		return err
	}
	for i := range dcds {
		dcd := &dcds[i]
		deleteOptions := []client.DeleteOption{}
		if dcd.UID != "" {
			deleteOptions = append(deleteOptions, client.Preconditions{UID: &dcd.UID})
		}
		if err := r.client.Delete(ctx, dcd, deleteOptions...); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("failed to delete selected DynamoComponentDeployment %s/%s: %w", dcd.Namespace, dcd.Name, err)
		}
	}
	return nil
}

func (r *disaggregatedSetAuxiliaryDCDReconciler) listOwnedSelectedDCDs(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	selection disaggregatedSetSelection,
) ([]nvidiacomv1beta1.DynamoComponentDeployment, error) {
	dcdList := &nvidiacomv1beta1.DynamoComponentDeploymentList{}
	if err := r.client.List(ctx, dcdList, client.InNamespace(dgd.Namespace)); err != nil {
		return nil, fmt.Errorf("failed to list DynamoComponentDeployments for DisaggregatedSet cleanup: %w", err)
	}
	selectedDCDs := []nvidiacomv1beta1.DynamoComponentDeployment{}
	for _, dcd := range dcdList.Items {
		if !isControlledByBetaDGD(&dcd, dgd) {
			continue
		}
		componentName := dynamo.GetDCDComponentName(&dcd)
		if _, selected := selection.componentToRole[componentName]; selected {
			selectedDCDs = append(selectedDCDs, dcd)
		}
	}
	return selectedDCDs, nil
}

func sortedDCDKeys(dcds map[string]*nvidiacomv1beta1.DynamoComponentDeployment) []string {
	keys := make([]string, 0, len(dcds))
	for key := range dcds {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
