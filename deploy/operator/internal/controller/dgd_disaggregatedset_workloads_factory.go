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
	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func (r *DynamoGraphDeploymentReconciler) newDisaggregatedSetWorkloadsReconciler(
	rollout *dgdWorkerRolloutReconciler,
) *disaggregatedSetWorkloadsReconciler {
	return newDisaggregatedSetWorkloadsReconciler(
		r.Client,
		r.Recorder,
		r.Config,
		r.RuntimeConfig,
		r.DockerSecretRetriever,
		rollout,
	)
}

func newDisaggregatedSetWorkloadsReconciler(
	k8sClient client.Client,
	recorder events.EventRecorder,
	config *configv1alpha1.OperatorConfiguration,
	runtimeConfig *commoncontroller.RuntimeConfig,
	dockerSecretRetriever DockerSecretRetriever,
	rollout *dgdWorkerRolloutReconciler,
) *disaggregatedSetWorkloadsReconciler {
	componentRenderer := newDCDWorkloadRenderer(k8sClient, config, runtimeConfig, dockerSecretRetriever)
	componentRestartProgress := newComponentRestartProgressResolver(k8sClient)
	return &disaggregatedSetWorkloadsReconciler{
		rollout:                  rollout,
		renderer:                 newDisaggregatedSetWorkloadRenderer(componentRenderer),
		resources:                newDisaggregatedSetResourceReconciler(k8sClient),
		stableResources:          newDisaggregatedSetStableResourcesReconciler(k8sClient, componentRenderer),
		auxiliaryDCDs:            newDisaggregatedSetAuxiliaryDCDReconciler(k8sClient, recorder),
		readiness:                newDisaggregatedSetReadinessResolver(k8sClient),
		restartProgress:          newDisaggregatedSetRestartProgressResolver(k8sClient, componentRestartProgress),
		componentRestartProgress: componentRestartProgress,
	}
}
