/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package mutation

import (
	"context"
	"net/http"
	"slices"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"gomodules.xyz/jsonpatch/v2"
	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"

	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	dynamolpx "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	internalwebhook "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook"
)

const podCyborgPlacementWebhookPath = "/mutate-core-v1-pod-cyborg-placement"

// PodCyborgPlacementMutator holds each Cyborg Pod that the LPX controller places
// by fabric pod out of scheduling until the controller has pinned it.
//
// Grove sets only its own scheduling gate on a new Pod and can remove it before
// the LPX controller observes the Pod. Kubernetes accepts a new gate only at Pod
// creation, so admission adds a second gate that only the LPX controller removes,
// in the write that sets the Pod's node selector.
type PodCyborgPlacementMutator struct {
	config        *configv1alpha1.OperatorConfiguration
	runtimeConfig *commoncontroller.RuntimeConfig
	scheme        *runtime.Scheme
}

// NewPodCyborgPlacementMutator creates the mutator for the non-nil configuration
// that the LPX controller also uses.
func NewPodCyborgPlacementMutator(config *configv1alpha1.OperatorConfiguration, runtimeConfig *commoncontroller.RuntimeConfig) *PodCyborgPlacementMutator {
	return &PodCyborgPlacementMutator{config: config, runtimeConfig: runtimeConfig}
}

func (h *PodCyborgPlacementMutator) RegisterWithManager(mgr manager.Manager, gate features.Gate) error {
	h.scheme = mgr.GetScheme()
	webhook := internalwebhook.WithGate((&admission.Webhook{Handler: h}).WithRecoverPanic(true), gate)
	mgr.GetWebhookServer().Register(podCyborgPlacementWebhookPath, webhook)
	return nil
}

func (h *PodCyborgPlacementMutator) Handle(ctx context.Context, req admission.Request) admission.Response {
	// Kubernetes accepts a new scheduling gate only when the Pod is created.
	if req.Operation != admissionv1.Create {
		return admission.Allowed("not a pod create")
	}

	// A gate without a controller that removes it holds the Pod forever. The LPX
	// controller places Pods only with Grove and with cluster-wide Node access.
	enabled := features.MustGateFrom(ctx)
	if !enabled.Enabled(features.LPX) || !enabled.Enabled(features.Grove) || h.config.Namespace.Restricted != "" {
		return admission.Allowed("LPX Cyborg placement is not active")
	}

	// The LPX controller reconciles only the namespaces that its event filter passes.
	pod := &corev1.Pod{}
	if err := admission.NewDecoder(h.scheme).Decode(req, pod); err != nil {
		return admission.Errored(http.StatusBadRequest, err)
	}
	if !commoncontroller.NamespaceAllowed(h.config, h.runtimeConfig, pod, req.Namespace) {
		return admission.Allowed("the LPX controller does not reconcile this namespace")
	}

	// The LPX controller marks the Cyborg Pods that it places. It finds them through
	// their LPX deployment and their PodClique. A Pod without these, such as a copy
	// or a Pod of another component with the same label, gets no gate.
	if pod.Labels[dynamolpx.CyborgPlacementLabel] != dynamolpx.CyborgPlacementPerFabricPod ||
		pod.Annotations[lpxv1alpha1.PodRoleAnnotation] != lpxv1alpha1.PodRoleCyborgWorker ||
		pod.Annotations[dynamolpx.DeploymentNameAnnotation] == "" {
		return admission.Allowed("pod is not a Cyborg Pod with fabric-pod placement")
	}
	owner := metav1.GetControllerOf(pod)
	if owner == nil || owner.Kind != "PodClique" || schema.FromAPIVersionAndKind(owner.APIVersion, owner.Kind).Group != grovev1alpha1.SchemeGroupVersion.Group {
		return admission.Allowed("pod is not controlled by a PodClique")
	}
	if slices.ContainsFunc(pod.Spec.SchedulingGates, func(gate corev1.PodSchedulingGate) bool {
		return gate.Name == dynamolpx.CyborgPlacementGate
	}) {
		return admission.Allowed("pod already has the placement gate")
	}

	// Send one operation that adds the gate, so that the patch cannot change another Pod field.
	gate := corev1.PodSchedulingGate{Name: dynamolpx.CyborgPlacementGate}
	if len(pod.Spec.SchedulingGates) == 0 {
		return admission.Patched("held until the LPX controller selects a fabric pod",
			jsonpatch.NewOperation("add", "/spec/schedulingGates", []corev1.PodSchedulingGate{gate}))
	}
	return admission.Patched("held until the LPX controller selects a fabric pod",
		jsonpatch.NewOperation("add", "/spec/schedulingGates/-", gate))
}
