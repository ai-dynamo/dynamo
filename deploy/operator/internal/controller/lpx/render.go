// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package lpx

import (
	"context"
	"crypto/sha256"
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

// renderPodCliqueSet composes resolved workloads into one Grove envelope with
// runtime ConfigMaps and, for Kubernetes discovery, serving Services.
// Inputs must be non-nil and workloads must contain every resolved component
// group in name order. Inputs remain read-only.
func (r *graphReconciler) renderPodCliqueSet(
	ctx context.Context,
	deployment *v1alpha1.LPXGraphDeployment,
	dgd *v1beta1.DynamoGraphDeployment,
	workloads []*lpx.Workload,
) (*grovev1alpha1.PodCliqueSet, []client.Object, error) {
	// Shared defaults and queue resolution belong to the single PCS envelope.
	pcs, err := dynamo.RenderLPXPodCliqueSet(ctx, dgd, r.config, r.runtimeConfig, dynamo.PCSNameForLPX(deployment))
	if err != nil {
		return nil, nil, err
	}

	var (
		resources []client.Object
		digest    = sha256.New()
	)

	for _, workload := range workloads {
		// Render this workload's roles using the full graph for shared defaults.
		rendered, err := dynamo.RenderLPXWorkload(pcs, dgd, r.config, r.runtimeConfig, r.dockerSecretRetriever, workload)
		if err != nil {
			return nil, nil, err
		}
		resources = append(resources, rendered...)

		// Kubernetes discovery exposes only this workload's serving role within its PCS.
		if commoncontroller.IsK8sDiscoveryEnabled(r.config.Discovery.Backend, dgd.Annotations) {
			component := dgd.GetComponentByName(workload.Name())
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
			service.Spec.Selector[lpx.ServingLabel] = consts.KubeLabelValueTrue
			service.Spec.Selector[grovecommon.LabelPartOfKey] = pcs.Name
			resources = append(resources, service)
		}

		writeIdentityHashField(digest, workload.Name(), workload.Digest().String())
	}

	pcs.Annotations[lpx.WorkloadDigestAnnotation] = fmt.Sprintf("sha256:%x", digest.Sum(nil))
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

// stampDeploymentIdentity propagates stable ownership labels and annotations and the
// delivered restart token, never DGD revision.
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
		if token := deployment.Annotations[dynamo.LPXRestartAnnotation]; token != "" {
			clique.Annotations[consts.RestartAnnotation] = token
		}
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
