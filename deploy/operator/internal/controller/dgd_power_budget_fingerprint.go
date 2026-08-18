/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

type powerInventoryFingerprint struct {
	Status          nvidiacomv1beta1.DynamoGraphPowerBudgetStatus `json:"status"`
	AllPodsReported bool                                          `json:"allPodsReported,omitempty"`
	ReportedPodUIDs []string                                      `json:"reportedPodUIDs,omitempty"`
	Workers         []powerInventoryWorker                        `json:"workers"`
	DCDs            []powerInventoryDCD                           `json:"dcds"`
	ScalingAdapters []powerInventoryScalingAdapter                `json:"scalingAdapters"`
	Pods            []powerInventoryPod                           `json:"pods"`
	Rollout         powerInventoryRollout                         `json:"rollout"`
}

type powerInventoryWorker struct {
	Name                 string `json:"name"`
	ComponentType        string `json:"componentType"`
	DGDReplicas          int32  `json:"dgdReplicas"`
	RequestedCap         string `json:"requestedCap"`
	GPUProduct           string `json:"gpuProduct"`
	PhysicalGPUCount     int    `json:"physicalGPUCount"`
	PhysicalGPUCountOK   bool   `json:"physicalGPUCountOK"`
	NodeCount            int32  `json:"nodeCount"`
	CheckpointConfigured bool   `json:"checkpointConfigured"`
}

type powerInventoryDCD struct {
	Name               string                                   `json:"name"`
	UID                string                                   `json:"uid"`
	Generation         int64                                    `json:"generation"`
	ComponentName      string                                   `json:"componentName"`
	ObservedGeneration int64                                    `json:"observedGeneration"`
	ReplicaStatus      *nvidiacomv1beta1.ComponentReplicaStatus `json:"replicaStatus,omitempty"`
}

type powerInventoryScalingAdapter struct {
	Name          string `json:"name"`
	UID           string `json:"uid"`
	Generation    int64  `json:"generation"`
	DGDName       string `json:"dgdName"`
	ComponentName string `json:"componentName"`
	Replicas      int32  `json:"replicas"`
}

type powerInventoryPod struct {
	Name        string                   `json:"name"`
	UID         string                   `json:"uid"`
	Component   string                   `json:"component"`
	Node        string                   `json:"node"`
	Deleting    bool                     `json:"deleting"`
	Terminal    bool                     `json:"terminal"`
	ReportState string                   `json:"reportState"`
	Report      *powerbudget.AgentReport `json:"report,omitempty"`
}

type powerInventoryRollout struct {
	Phase             nvidiacomv1beta1.RollingUpdatePhase `json:"phase,omitempty"`
	UpdatedComponents []string                            `json:"updatedComponents,omitempty"`
}

func calculatePowerInventoryFingerprint(
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	inventory dgdPowerBudgetInventory,
	desired nvidiacomv1beta1.DynamoGraphPowerBudgetStatus,
	history powerReportHistory,
) (string, error) {
	desired.InventoryEpoch = 0
	desired.Conditions = append([]metav1.Condition(nil), desired.Conditions...)
	for i := range desired.Conditions {
		desired.Conditions[i].LastTransitionTime = metav1.Time{}
	}

	semantic := powerInventoryFingerprint{
		Status:          desired,
		AllPodsReported: history.All,
		ReportedPodUIDs: append([]string(nil), history.PodUIDs...),
	}
	sort.Strings(semantic.ReportedPodUIDs)
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if !dynamo.IsWorkerComponent(string(component.ComponentType)) {
			continue
		}
		gpuCount, gpuErr := dra.ExtractGPUCountFromResourceRequirements(
			dynamo.GetMainContainerResources(component),
		)
		dgdReplicas := int32(0)
		if component.Replicas != nil {
			dgdReplicas = *component.Replicas
		}
		requestedCap := ""
		gpuProduct := ""
		if component.PodTemplate != nil {
			requestedCap = component.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit]
			gpuProduct = component.PodTemplate.Spec.NodeSelector[qualifiedGPUProductLabel]
		}
		semantic.Workers = append(semantic.Workers, powerInventoryWorker{
			Name:                 component.ComponentName,
			ComponentType:        string(component.ComponentType),
			DGDReplicas:          dgdReplicas,
			RequestedCap:         requestedCap,
			GPUProduct:           gpuProduct,
			PhysicalGPUCount:     gpuCount,
			PhysicalGPUCountOK:   gpuErr == nil,
			NodeCount:            component.GetNumberOfNodes(),
			CheckpointConfigured: componentCheckpointConfigured(component),
		})
	}
	sort.Slice(semantic.Workers, func(i, j int) bool { return semantic.Workers[i].Name < semantic.Workers[j].Name })

	for i := range inventory.DCDs {
		dcd := &inventory.DCDs[i]
		var replicaStatus *nvidiacomv1beta1.ComponentReplicaStatus
		if dcd.Status.Component != nil {
			copy := *dcd.Status.Component
			copy.ComponentNames = append([]string(nil), copy.ComponentNames...)
			sort.Strings(copy.ComponentNames)
			replicaStatus = &copy
		}
		semantic.DCDs = append(semantic.DCDs, powerInventoryDCD{
			Name:               dcd.Name,
			UID:                string(dcd.UID),
			Generation:         dcd.Generation,
			ComponentName:      dcd.Spec.ComponentName,
			ObservedGeneration: dcd.Status.ObservedGeneration,
			ReplicaStatus:      replicaStatus,
		})
	}

	for i := range inventory.ScalingAdapters {
		adapter := &inventory.ScalingAdapters[i]
		semantic.ScalingAdapters = append(semantic.ScalingAdapters, powerInventoryScalingAdapter{
			Name:          adapter.Name,
			UID:           string(adapter.UID),
			Generation:    adapter.Generation,
			DGDName:       adapter.Spec.DGDRef.Name,
			ComponentName: adapter.Spec.DGDRef.ServiceName,
			Replicas:      adapter.Spec.Replicas,
		})
	}

	for i := range inventory.Pods {
		pod := &inventory.Pods[i]
		item := powerInventoryPod{
			Name:      pod.Name,
			UID:       string(pod.UID),
			Component: pod.Labels[consts.KubeLabelDynamoComponent],
			Node:      pod.Spec.NodeName,
			Deleting:  !pod.DeletionTimestamp.IsZero(),
			Terminal:  isTerminalPhase(pod.Status.Phase),
		}
		encoded := pod.Annotations[powerbudget.AgentReportAnnotation]
		switch {
		case encoded == "":
			item.ReportState = "missing"
		default:
			report, err := powerbudget.DecodeAgentReport([]byte(encoded))
			if err != nil {
				item.ReportState = "invalid"
			} else {
				item.ReportState = "valid"
				for gpuIndex := range report.GPUs {
					report.GPUs[gpuIndex].ObservedAt = time.Time{}
				}
				item.Report = &report
			}
		}
		semantic.Pods = append(semantic.Pods, item)
	}

	if dgd.Status.RollingUpdate != nil {
		semantic.Rollout.Phase = dgd.Status.RollingUpdate.Phase
		semantic.Rollout.UpdatedComponents = append(
			[]string(nil),
			dgd.Status.RollingUpdate.UpdatedComponents...,
		)
		sort.Strings(semantic.Rollout.UpdatedComponents)
	}

	encoded, err := json.Marshal(semantic)
	if err != nil {
		return "", fmt.Errorf("encode semantic power inventory: %w", err)
	}
	digest := sha256.Sum256(encoded)
	return hex.EncodeToString(digest[:]), nil
}

func componentCheckpointConfigured(component *nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec) bool {
	return component.Experimental != nil && component.Experimental.Checkpoint != nil &&
		(component.Experimental.Checkpoint.Enabled ||
			ptr.Deref(component.Experimental.Checkpoint.CheckpointRef, "") != "")
}
