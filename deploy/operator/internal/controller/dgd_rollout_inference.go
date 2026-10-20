/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
)

var rolloutWorkerMetadataGVK = schema.GroupVersionKind{Group: "nvidia.com", Version: "v1alpha1", Kind: "DynamoWorkerMetadata"}

// rolloutInferenceTarget identifies one replacement model member, not merely a generation label.
type rolloutInferenceTarget struct {
	Model      string
	InstanceID uint64
	Prefill    bool
	Paths      []string
}

// replacementInferenceReady gates managed Kubernetes-discovery text generation only.
// dgd must be non-nil. Embeddings and non-Kubernetes discovery retain their existing rollout behavior.
func (r *dgdWorkerRolloutReconciler) replacementInferenceReady(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment, componentName, newDCDName string) (bool, error) {
	if !commoncontroller.IsK8sDiscoveryEnabled(r.discoveryBackend, dgd.Annotations) {
		return true, nil
	}

	// External frontends are outside the managed DGD rollout contract.
	frontendComponents := make(map[string]bool)
	for _, spec := range dgd.Spec.Components {
		if string(spec.ComponentType) == consts.ComponentTypeFrontend {
			frontendComponents[spec.ComponentName] = false
		}
	}
	if len(frontendComponents) == 0 {
		return true, nil
	}

	// Select by DCD identity and Pod UID so stale metadata cannot certify a replacement.
	pods := &corev1.PodList{}
	if err := r.List(ctx, pods, client.InNamespace(dgd.Namespace), client.MatchingLabels{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name,
	}); err != nil {
		return false, err
	}
	newPods := make(map[types.UID]bool)
	var frontends []corev1.Pod
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil || !rolloutPodReady(&pod) {
			continue
		}
		if pod.Labels[consts.KubeLabelDynamoSelector] == newDCDName && pod.Labels[consts.KubeLabelDynamoComponent] == componentName {
			newPods[pod.UID] = true
		}
		if _, ok := frontendComponents[pod.Labels[consts.KubeLabelDynamoComponent]]; ok {
			frontends = append(frontends, pod)
			frontendComponents[pod.Labels[consts.KubeLabelDynamoComponent]] = true
		}
	}
	for name, ready := range frontendComponents {
		if !ready {
			return false, fmt.Errorf("no ready frontend Pod for component %s", name)
		}
	}

	// The cached metadata is emitted by released workers as well as current workers.
	metadata := &unstructured.UnstructuredList{}
	metadata.SetGroupVersionKind(rolloutWorkerMetadataGVK.GroupVersion().WithKind("DynamoWorkerMetadataList"))
	if err := r.List(ctx, metadata, client.InNamespace(dgd.Namespace)); err != nil {
		return false, err
	}
	targets, found, err := replacementInferenceTargets(metadata.Items, newPods)
	if err != nil {
		return false, err
	}
	if !found {
		return false, fmt.Errorf("replacement model metadata is pending")
	}
	if len(targets) == 0 {
		return true, nil // Non-text models have no worker attribution on their inference response.
	}

	// Probe each frontend directly; Service load balancing cannot prove admission by every replica.
	probeCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	defer cancel()
	group, probeCtx := errgroup.WithContext(probeCtx)
	group.SetLimit(4)
	for _, frontend := range frontends {
		group.Go(func() error {
			address, err := rolloutFrontendAddress(&frontend)
			if err != nil {
				return err
			}
			for _, target := range targets {
				for _, path := range target.Paths {
					// WorkerSet selection precedes instance targeting in older frontends.
					// A wrong-set failure is safe; retry it without accepting an old-worker response.
					for attempt := 0; attempt < 4; attempt++ {
						err = r.probeReplacementInference(probeCtx, address, path, target)
						if err == nil || probeCtx.Err() != nil {
							break
						}
					}
					if err != nil {
						return fmt.Errorf("frontend %s replacement %d: %w", frontend.Name, target.InstanceID, err)
					}
				}
			}
			return nil
		})
	}
	if err := group.Wait(); err != nil {
		return false, err
	}
	return true, nil
}

// replacementInferenceTargets requires metadata owned by currently ready replacement Pods.
func replacementInferenceTargets(metadata []unstructured.Unstructured, newPods map[types.UID]bool) ([]rolloutInferenceTarget, bool, error) {
	var targets []rolloutInferenceTarget
	found := false
	selected := make(map[string]rolloutInferenceTarget)
	for _, object := range metadata {
		owner := metav1.GetControllerOf(&object)
		if object.GetDeletionTimestamp() != nil || owner == nil || owner.Kind != "Pod" || !newPods[owner.UID] {
			continue
		}

		// Decode typed IDs without converting them through float64.
		data, _, err := unstructured.NestedMap(object.Object, "spec", "data", "model_cards")
		if err != nil {
			return nil, false, err
		}
		raw, err := json.Marshal(data)
		if err != nil {
			return nil, false, err
		}
		var cards map[string]struct {
			InstanceID uint64 `json:"instance_id"`
			Card       struct {
				DisplayName string `json:"display_name"`
				ModelType   string `json:"model_type"`
				WorkerType  string `json:"worker_type"`
			} `json:"card_json"`
		}
		if err := json.Unmarshal(raw, &cards); err != nil {
			return nil, false, err
		}
		for _, card := range cards {
			if card.Card.ModelType == "" {
				return nil, false, fmt.Errorf("replacement model card lacks model type")
			}
			found = true
			target := rolloutInferenceTarget{Model: card.Card.DisplayName, InstanceID: card.InstanceID,
				Prefill: card.Card.WorkerType == "prefill" || strings.Contains(card.Card.ModelType, "Prefill")}
			if strings.Contains(card.Card.ModelType, "Chat") {
				target.Paths = append(target.Paths, "/v1/chat/completions")
			}
			if strings.Contains(card.Card.ModelType, "Completions") {
				target.Paths = append(target.Paths, "/v1/completions")
			}
			if target.Prefill && len(target.Paths) == 0 {
				target.Paths = []string{"/v1/chat/completions", "/v1/completions"}
			}
			if len(target.Paths) == 0 {
				continue
			}
			if target.Model == "" || target.InstanceID == 0 {
				return nil, false, fmt.Errorf("replacement model card lacks model name or instance ID")
			}

			// One replacement member per model/role is sufficient to retain usable capacity.
			key := target.Model + ":" + strconv.FormatBool(target.Prefill)
			if previous, ok := selected[key]; !ok || target.InstanceID < previous.InstanceID {
				selected[key] = target
			}
		}
	}
	for _, target := range selected {
		targets = append(targets, target)
	}
	sort.Slice(targets, func(i, j int) bool {
		if targets[i].Model != targets[j].Model {
			return targets[i].Model < targets[j].Model
		}
		return targets[i].InstanceID < targets[j].InstanceID
	})
	return targets, found, nil
}

func rolloutPodReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

func rolloutFrontendAddress(pod *corev1.Pod) (string, error) {
	if pod.Status.PodIP == "" {
		return "", fmt.Errorf("frontend %s has no Pod IP", pod.Name)
	}
	for _, container := range pod.Spec.Containers {
		for _, port := range container.Ports {
			if port.Name == consts.DynamoContainerPortName {
				return "http://" + net.JoinHostPort(pod.Status.PodIP, strconv.Itoa(int(port.ContainerPort))), nil
			}
		}
	}
	return "", fmt.Errorf("frontend %s has no named HTTP port", pod.Name)
}

func (r *dgdWorkerRolloutReconciler) probeReplacementInference(ctx context.Context, address, path string, target rolloutInferenceTarget) error {
	nvext := map[string]any{"extra_fields": []string{"worker_id"}}
	if target.Prefill {
		nvext["prefill_worker_id"] = target.InstanceID
	} else {
		nvext["backend_instance_id"] = target.InstanceID
	}
	// A fresh prefill prompt prevents a full-prefix cache hit from skipping the phase being verified.
	prompt := "Hi"
	if target.Prefill {
		prompt += " " + strconv.FormatInt(time.Now().UnixNano(), 10)
	}
	body := map[string]any{"model": target.Model, "max_tokens": 1, "stream": false, "temperature": 0, "nvext": nvext}
	if path == "/v1/chat/completions" {
		body["messages"] = []map[string]string{{"role": "user", "content": prompt}}
	} else {
		body["prompt"] = prompt
	}

	// Bound network time and response size; never follow a redirect away from the selected Pod.
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, address+path, bytes.NewReader(raw))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := r.inferenceClient.Do(request)
	if err != nil {
		return err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("%s returned HTTP %d", path, response.StatusCode)
	}

	// A 200 alone could have come from an old worker or an ignored routing hint.
	var result struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
		Nvext struct {
			WorkerID struct {
				Decode  uint64 `json:"decode_worker_id"`
				Prefill uint64 `json:"prefill_worker_id"`
			} `json:"worker_id"`
		} `json:"nvext"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&result); err != nil {
		return fmt.Errorf("invalid inference response: %w", err)
	}
	actual := result.Nvext.WorkerID.Decode
	if target.Prefill {
		actual = result.Nvext.WorkerID.Prefill
	}
	if actual != target.InstanceID || len(result.Choices) != 1 || result.Usage.CompletionTokens < 1 ||
		(result.Choices[0].FinishReason != "stop" && result.Choices[0].FinishReason != "length") {
		return fmt.Errorf("%s did not complete inference attributed to replacement worker %d", path, target.InstanceID)
	}
	return nil
}

// isDGDFrontendPod limits admission-related Pod events to managed serving frontends.
func isDGDFrontendPod(object client.Object) bool {
	return object.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName] != "" &&
		object.GetLabels()[consts.KubeLabelDynamoComponentType] == consts.ComponentTypeFrontend
}

func mapDGDServingPodToRequests(ctx context.Context, object client.Object) []ctrl.Request {
	if isDGDFrontendPod(object) {
		return []ctrl.Request{{NamespacedName: types.NamespacedName{Namespace: object.GetNamespace(),
			Name: object.GetLabels()[consts.KubeLabelDynamoGraphDeploymentName]}}}
	}
	return mapDGDWorkerPodToRequests(ctx, object)
}

func (r *DynamoGraphDeploymentReconciler) mapRolloutMetadataToDGDRequests(ctx context.Context, object client.Object) []ctrl.Request {
	owner := metav1.GetControllerOf(object)
	if owner == nil || owner.Kind != "Pod" {
		return nil
	}

	// Resolve the live owner through the cache instead of trusting a reusable Pod name.
	pod := &corev1.Pod{}
	if err := r.Get(ctx, types.NamespacedName{Namespace: object.GetNamespace(), Name: owner.Name}, pod); err != nil || pod.UID != owner.UID {
		return nil
	}
	return mapDGDServingPodToRequests(ctx, pod)
}
