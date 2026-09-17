// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	consts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	lpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	deploymentUIDAnnotation = dynamo.LPXDeploymentUIDAnnotation
	deploymentUIDLabel      = deploymentUIDAnnotation
)

// renderPodCliqueSet renders the admitted DGD using the resolved build and plan.
// The DGD, child, workload and plan must be non-nil; the plan must use the
// child's PodCliqueSet identity. The renderer
// does not mutate its inputs. secretsRetriever may be nil when no secrets are used.
func (r *graphReconciler) renderPodCliqueSet(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	workload *lpx.SelectedWorkload,
	plan *lpx.MaterializationPlan,
) (*grovev1alpha1.PodCliqueSet, []client.Object, error) {
	// Apply shared Dynamo defaults before materializing the independently owned LPX roles.
	pcs, input, err := dynamo.RenderLPXBasePodCliqueSet(
		ctx,
		dgd,
		r.config,
		r.runtimeConfig,
		r.dockerSecretRetriever,
		workload,
		plan,
	)
	if err != nil {
		return nil, nil, err
	}
	resources, err := lpx.RenderSelectedNodeLocal(pcs, workload, plan, *input)
	if err != nil {
		return nil, nil, err
	}

	// Stamp stable ownership and identify the engine's serving endpoint.
	stampDeploymentIdentity(deployment, pcs, resources)
	serving := plan.ConductorTemplate
	if serving == "" {
		serving = plan.CyborgTemplate
	}

	// The child retains the last delivered LPX restart while other components restart.
	restartToken := deployment.Annotations[dynamo.LPXRestartAnnotation]
	for _, clique := range pcs.Spec.Template.Cliques {
		if restartToken != "" {
			clique.Annotations[consts.RestartAnnotation] = restartToken
		}
		delete(clique.Labels, dynamo.LPXServingLabel)
		if clique.Name == serving {
			clique.Labels[dynamo.LPXServingLabel] = consts.KubeLabelValueTrue
		} else {
			// Auxiliary roles must be private in both Service and Pod discovery modes.
			delete(clique.Labels, consts.KubeLabelDynamoDiscoveryEnabled)
			delete(clique.Labels, consts.KubeLabelDynamoDiscoveryBackend)
			delete(clique.Labels, consts.KubeLabelDynamoBaseModelHash)
		}
	}

	// Enforce the size budget after ownership, scheduling and discovery metadata are final.
	serialized, err := json.Marshal(pcs)
	if err != nil {
		return nil, nil, fmt.Errorf("serializing selected LPX PodCliqueSet: %w", err)
	}
	if len(serialized) > lpx.MaxRenderedPodCliqueSetBytes {
		return nil, nil, fmt.Errorf("rendered LPX PodCliqueSet is %d bytes; maximum is %d",
			len(serialized), lpx.MaxRenderedPodCliqueSetBytes)
	}
	return pcs, resources, nil
}

// stampDeploymentIdentity propagates stable ownership labels and annotations, never DGD revision.
func stampDeploymentIdentity(deployment *v1alpha1.LPXGraphDeployment, pcs *grovev1alpha1.PodCliqueSet, resources []client.Object) {
	// The initial DGD lookup validated this controller owner; rendering trusts that observation.
	dgdOwner := metav1.GetControllerOf(deployment)
	identity := map[string]string{
		lpx.DeploymentNameAnnotation: deployment.Name,
		deploymentUIDAnnotation:      string(deployment.UID),
		lpx.DGDUIDAnnotation:         string(dgdOwner.UID),
	}
	stamp := func(annotations *map[string]string) {
		if *annotations == nil {
			*annotations = make(map[string]string)
		}
		delete(*annotations, lpx.DGDGenerationAnnotation)
		maps.Copy(*annotations, identity)
	}
	stampOwnerLabel := func(object client.Object) {
		labels := object.GetLabels()
		if labels == nil {
			labels = make(map[string]string)
		}
		labels[deploymentUIDLabel] = string(deployment.UID)
		object.SetLabels(labels)
	}
	stamp(&pcs.Annotations)
	stampOwnerLabel(pcs)
	for _, clique := range pcs.Spec.Template.Cliques {
		stamp(&clique.Annotations)
	}
	for i := range pcs.Spec.Template.PodCliqueScalingGroupConfigs {
		stamp(&pcs.Spec.Template.PodCliqueScalingGroupConfigs[i].Annotations)
	}
	for _, resource := range resources {
		annotations := resource.GetAnnotations()
		stamp(&annotations)
		resource.SetAnnotations(annotations)
		stampOwnerLabel(resource)
	}
}
