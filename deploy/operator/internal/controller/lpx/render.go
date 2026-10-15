// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	consts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	lpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const deploymentUIDLabel = "lpx.nvidia.com/deployment-uid"

// renderWorkloads composes resolved workloads into one Grove envelope with
// runtime ConfigMaps and, for Kubernetes discovery, serving Services.
// Inputs must be non-nil except existingPCS, which is nil during creation.
// Workloads must contain every component group.
// Workload names must already be finalized.
// Inputs remain read-only.
func (r *graphReconciler) renderWorkloads(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	workloads []*lpx.Workload,
	existingPCS *grovev1alpha1.PodCliqueSet,
) (*grovev1alpha1.PodCliqueSet, []client.Object, error) {
	// Shared defaults and queue resolution belong to the single PCS envelope.
	pcs, err := dynamo.RenderLPXPodCliqueSet(ctx, dgd, r.config, r.runtimeConfig, dynamo.PCSNameForLPX(deployment), existingPCS)
	if err != nil {
		return nil, nil, err
	}

	var resources []client.Object

	for _, workload := range workloads {
		cliqueStart := len(pcs.Spec.Template.Cliques)

		// Render this workload's roles using the full graph for shared defaults.
		rendered, err := dynamo.RenderLPXWorkload(pcs, dgd, r.config, r.runtimeConfig, r.dockerSecretRetriever, workload)
		if err != nil {
			return nil, nil, err
		}

		// Discovery selects only this workload's conductor; all auxiliary roles stay private.
		serving := workload.ConductorTemplate()
		for _, clique := range pcs.Spec.Template.Cliques[cliqueStart:] {
			if token := deployment.Annotations[dynamo.LPXRestartAnnotation]; token != "" {
				clique.Annotations[consts.RestartAnnotation] = token
			}
			delete(clique.Labels, dynamo.LPXServingLabel)
			if clique.Name == serving {
				clique.Labels[dynamo.LPXServingLabel] = consts.KubeLabelValueTrue
			} else {
				delete(clique.Labels, consts.KubeLabelDynamoDiscoveryEnabled)
				delete(clique.Labels, consts.KubeLabelDynamoDiscoveryBackend)
				delete(clique.Labels, consts.KubeLabelDynamoBaseModelHash)
			}
		}

		resources = append(resources, rendered...)

		// Kubernetes discovery exposes only this workload's serving role within its PCS.
		if commoncontroller.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, dgd.Annotations) {
			component := dgd.GetComponentByName(workload.ServingComponentName())
			service, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
				ServiceName: workload.ResourcePrefix() + "-serve", Namespace: deployment.Namespace,
				ComponentType: string(component.ComponentType), ComponentName: component.ComponentName,
				DynamoNamespace: dgd.GetDynamoNamespaceForComponent(component), IsK8sDiscovery: true,
				Labels:      dynamo.GetDGDComponentResourceLabels(dgd, component.ComponentName, component),
				Annotations: dynamo.GetDGDComponentResourceAnnotations(dgd, component.ComponentName, component),
			})
			if err != nil {
				return nil, nil, err
			}
			service.Spec.Selector[dynamo.LPXServingLabel] = consts.KubeLabelValueTrue
			service.Spec.Selector[grovecommon.LabelPartOfKey] = pcs.Name
			resources = append(resources, service)
		}

	}

	stampDeploymentIdentity(deployment, pcs, resources)

	// Enforce the aggregate size budget after identity and discovery metadata are final.
	serialized, err := json.Marshal(pcs)
	if err != nil {
		return nil, nil, fmt.Errorf("serializing selected LPX PodCliqueSet: %w", err)
	}
	if len(serialized) > lpx.MaxRenderedPodCliqueSetBytes {
		return nil, nil, fmt.Errorf("rendered LPX PodCliqueSet is %d bytes; maximum is %d", len(serialized), lpx.MaxRenderedPodCliqueSetBytes)
	}
	return pcs, resources, nil
}

// stampDeploymentIdentity propagates stable ownership labels and annotations, never DGD revision.
func stampDeploymentIdentity(deployment *v1alpha1.LPXGraphDeployment, pcs *grovev1alpha1.PodCliqueSet, resources []client.Object) {
	stamp := func(annotations *map[string]string) {
		if *annotations == nil {
			*annotations = make(map[string]string)
		}
		(*annotations)[lpx.DeploymentNameAnnotation] = deployment.Name
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
