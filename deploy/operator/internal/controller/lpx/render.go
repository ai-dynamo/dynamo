// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// renderPodCliqueSet renders the exact workload selected by controller preflight.
// The source, child, workload and plan must be non-nil and validated by that
// preflight; the plan must use the child's PodCliqueSet identity. The renderer
// does not mutate its inputs. secretsRetriever may be nil when no secrets are used.
func renderPodCliqueSet(
	ctx context.Context,
	source *v1beta1.DynamoGraphDeployment,
	operatorConfig *configv1alpha1.OperatorConfiguration,
	runtimeConfig *controller_common.RuntimeConfig,
	kubeClient client.Reader,
	secretsRetriever dynamo.SecretsRetriever,
	workload *dynamolpx.SelectedWorkload,
	plan *dynamolpx.MaterializationPlan,
	deployment *v1alpha1.LPXGraphDeployment,
) (*grovev1alpha1.PodCliqueSet, []client.Object, error) {
	// Apply shared Dynamo defaults before materializing the independently owned LPX roles.
	pcs, input, err := dynamo.RenderLPXBasePodCliqueSet(ctx, source, operatorConfig, runtimeConfig,
		kubeClient, secretsRetriever, workload, plan)
	if err != nil {
		return nil, nil, err
	}
	resources, err := dynamolpx.RenderSelectedNodeLocal(pcs, workload, plan, *input)
	if err != nil {
		return nil, nil, err
	}

	// Stamp stable ownership and identify the engine's serving endpoint.
	stampLPXIdentity(deployment, pcs, resources)
	serving := plan.ConductorTemplate
	if serving == "" {
		serving = plan.CyborgTemplate
	}

	// The child retains the last delivered LPX restart while other components restart.
	restartToken := deployment.Annotations[dynamo.LPXRestartAnnotation]
	for _, clique := range pcs.Spec.Template.Cliques {
		if restartToken != "" {
			clique.Annotations[commonconsts.RestartAnnotation] = restartToken
		}
		delete(clique.Labels, dynamo.LPXServingLabel)
		if clique.Name == serving {
			clique.Labels[dynamo.LPXServingLabel] = commonconsts.KubeLabelValueTrue
		} else {
			// Auxiliary roles must be private in both Service and Pod discovery modes.
			delete(clique.Labels, commonconsts.KubeLabelDynamoDiscoveryEnabled)
			delete(clique.Labels, commonconsts.KubeLabelDynamoDiscoveryBackend)
			delete(clique.Labels, commonconsts.KubeLabelDynamoBaseModelHash)
		}
	}

	// Enforce the size budget after ownership, scheduling and discovery metadata are final.
	serialized, err := json.Marshal(pcs)
	if err != nil {
		return nil, nil, fmt.Errorf("serializing selected LPX PodCliqueSet: %w", err)
	}
	if len(serialized) > dynamolpx.MaxRenderedPodCliqueSetBytes {
		return nil, nil, fmt.Errorf("rendered LPX PodCliqueSet is %d bytes; maximum is %d",
			len(serialized), dynamolpx.MaxRenderedPodCliqueSetBytes)
	}
	return pcs, resources, nil
}

// stampLPXIdentity propagates stable ownership labels and annotations, never source revision.
func stampLPXIdentity(deployment *v1alpha1.LPXGraphDeployment, pcs *grovev1alpha1.PodCliqueSet, resources []client.Object) {
	// Preflight validates the exact source controller owner before rendering resources.
	sourceOwner := metav1.GetControllerOf(deployment)
	identity := map[string]string{
		dynamolpx.DeploymentNameAnnotation: deployment.Name,
		lpxDeploymentUIDAnnotation:         string(deployment.UID),
		dynamolpx.DGDUIDAnnotation:         string(sourceOwner.UID),
	}
	stamp := func(annotations *map[string]string) {
		if *annotations == nil {
			*annotations = make(map[string]string)
		}
		delete(*annotations, dynamolpx.DGDGenerationAnnotation)
		maps.Copy(*annotations, identity)
	}
	stampOwnerLabel := func(object client.Object) {
		labels := object.GetLabels()
		if labels == nil {
			labels = make(map[string]string)
		}
		labels[lpxOwnerUIDLabel] = string(deployment.UID)
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
