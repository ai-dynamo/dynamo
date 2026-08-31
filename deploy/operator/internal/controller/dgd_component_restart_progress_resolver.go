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

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// componentRestartProgressResolver owns the read-only DCD observations used
// to determine which components have not completed a requested restart.
type componentRestartProgressResolver struct {
	reader client.Reader
}

const dcdNotFoundReason = "resource not found"

func newComponentRestartProgressResolver(reader client.Reader) *componentRestartProgressResolver {
	return &componentRestartProgressResolver{reader: reader}
}

func (r *componentRestartProgressResolver) Resolve(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	inProgress []string,
) []string {
	logger := log.FromContext(ctx)
	updatedInProgress := make([]string, 0, len(inProgress))
	for _, componentName := range inProgress {
		isFullyUpdated, reason := r.checkComponentFullyUpdated(ctx, dgd, componentName)
		if !isFullyUpdated {
			logger.V(1).Info("component not fully updated", "componentName", componentName, "reason", reason)
			updatedInProgress = append(updatedInProgress, componentName)
		}
	}
	return updatedInProgress
}

func (r *componentRestartProgressResolver) checkComponentFullyUpdated(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	componentName string,
) (bool, string) {
	if currentWorkerHashes(dgd).empty() {
		resourceName := dynamo.GetDCDResourceName(dgd, componentName, "")
		return checkDCDReady(ctx, r.reader, resourceName, dgd.Namespace)
	}

	hashes, err := desiredWorkerHashes(dgd)
	if err != nil {
		return false, err.Error()
	}

	var lastReason string
	for _, hash := range activeWorkerHashCandidates(dgd, hashes) {
		resourceName := dynamo.GetDCDResourceName(dgd, componentName, hash)
		ready, reason := checkDCDReady(ctx, r.reader, resourceName, dgd.Namespace)
		if ready || reason != dcdNotFoundReason {
			return ready, reason
		}
		lastReason = reason
	}
	return false, lastReason
}

func checkDCDReady(
	ctx context.Context,
	reader client.Reader,
	resourceName string,
	namespace string,
) (bool, string) {
	logger := log.FromContext(ctx)
	dcd := &nvidiacomv1beta1.DynamoComponentDeployment{}
	err := reader.Get(ctx, types.NamespacedName{Name: resourceName, Namespace: namespace}, dcd)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.V(2).Info("DynamoComponentDeployment not found", "resourceName", resourceName)
			return false, dcdNotFoundReason
		}
		logger.V(1).Info("Failed to get DynamoComponentDeployment", "error", err, "resourceName", resourceName)
		return false, fmt.Sprintf("get error: %v", err)
	}

	logger.V(1).Info("CheckDCDFullyUpdated",
		"resourceName", resourceName,
		"generation", dcd.Generation,
		"observedGeneration", dcd.Status.ObservedGeneration,
		"conditionCount", len(dcd.Status.Conditions))

	if dcd.Status.ObservedGeneration < dcd.Generation {
		logger.V(1).Info("DynamoComponentDeployment spec not yet processed",
			"resourceName", resourceName,
			"generation", dcd.Generation,
			"observedGeneration", dcd.Status.ObservedGeneration)
		return false, fmt.Sprintf("spec not yet processed: generation=%d, observedGeneration=%d", dcd.Generation, dcd.Status.ObservedGeneration)
	}

	for _, condition := range dcd.Status.Conditions {
		if condition.Type != nvidiacomv1beta1.DynamoComponentDeploymentConditionTypeAvailable {
			continue
		}
		if condition.Status == metav1.ConditionTrue {
			return true, ""
		}
		logger.V(1).Info("DynamoComponentDeployment not available",
			"resourceName", resourceName,
			"status", condition.Status,
			"reason", condition.Reason,
			"message", condition.Message)
		return false, fmt.Sprintf("not available: %s", condition.Message)
	}

	logger.V(1).Info("DynamoComponentDeployment missing Available condition", "resourceName", resourceName)
	return false, "Available condition not found"
}
