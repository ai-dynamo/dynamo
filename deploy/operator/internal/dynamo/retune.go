// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"path"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/utils/ptr"
)

const retuneTraceVolumeName = "dynamo-retune-traces"

// RetuneLaunchConfiguration is the versioned semantic contract shared by the
// frontend and Planner, not a projection of their workload launch syntax.
type RetuneLaunchConfiguration struct {
	Version               int                       `json:"version"`
	ConfigHash            string                    `json:"configHash,omitempty"`
	FrontendComponentName string                    `json:"frontendComponentName"`
	AdminPort             int32                     `json:"adminPort"`
	TraceStorage          RetuneTraceStorage        `json:"traceStorage"`
	Router                RetuneRouterConfiguration `json:"router"`
}

// RetuneTraceStorage is the owned shared trace layout in the launch contract.
type RetuneTraceStorage struct {
	ClaimName   string `json:"claimName"`
	MountPath   string `json:"mountPath"`
	SegmentsDir string `json:"segmentsDir"`
	WindowsDir  string `json:"windowsDir"`
}

// RetuneRouterConfiguration names the runtime-owned profile and its typed inputs.
type RetuneRouterConfiguration struct {
	Profile            string  `json:"profile"`
	OverlapScoreCredit float64 `json:"overlapScoreCredit"`
	PrefillLoadScale   float64 `json:"prefillLoadScale"`
	Temperature        float64 `json:"temperature"`
}

// RetuneConfiguration compiles semantic intent without reading workload recipes.
// Callers publish its hash only after both target lowerings are synchronized.
// dgd must not be nil. A nil configuration means no Planner requested Retune.
func RetuneConfiguration(dgd *v1beta1.DynamoGraphDeployment) (string, *RetuneLaunchConfiguration, error) {
	var planner *v1beta1.DynamoComponentDeploymentSharedSpec
	var frontend *v1beta1.DynamoComponentDeploymentSharedSpec
	frontendCount := 0
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.ComponentType == v1beta1.ComponentTypeFrontend {
			frontend = component
			frontendCount++
		}
		if component.Retune == nil || !component.Retune.Enabled {
			continue
		}
		if component.ComponentType != v1beta1.ComponentTypePlanner {
			return "", nil, fmt.Errorf("component %q: retune is supported only for type planner", component.ComponentName)
		}
		if planner != nil {
			return "", nil, fmt.Errorf("retune requires at most one enabled Planner component")
		}
		planner = component
	}
	if planner == nil {
		return "", nil, nil
	}

	// Stored or admission-bypassed graphs must satisfy the same wiring invariants.
	if frontendCount != 1 {
		return "", nil, fmt.Errorf("component %q: retune requires exactly one frontend component, found %d", planner.ComponentName, frontendCount)
	}
	if planner.Retune.TraceStorage == nil || planner.Retune.TraceStorage.ClaimName == "" {
		return "", nil, fmt.Errorf("component %q: enabled retune requires traceStorage.claimName", planner.ComponentName)
	}
	configuration := &RetuneLaunchConfiguration{
		Version:               1,
		FrontendComponentName: frontend.ComponentName,
		TraceStorage: RetuneTraceStorage{
			ClaimName: planner.Retune.TraceStorage.ClaimName,
			MountPath: "/trace", SegmentsDir: "segments", WindowsDir: "windows",
		},
		AdminPort: 8002,
		Router:    RetuneRouterConfiguration{Profile: "kv-v1", OverlapScoreCredit: 1, PrefillLoadScale: 1},
	}

	// Default the three public inputs identically for admission-bypassed objects.
	if router := planner.Retune.Router; router != nil {
		configuration.Router.OverlapScoreCredit = ptr.Deref(router.OverlapScoreCredit, 1)
		configuration.Router.PrefillLoadScale = ptr.Deref(router.PrefillLoadScale, 1)
		configuration.Router.Temperature = ptr.Deref(router.Temperature, 0)
	}
	for _, value := range []float64{configuration.Router.OverlapScoreCredit, configuration.Router.PrefillLoadScale, configuration.Router.Temperature} {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return "", nil, fmt.Errorf("component %q: retune router values must be finite and non-negative", planner.ComponentName)
		}
	}

	// Bind semantic intent and selected identities, never scaling or global generation.
	binding := struct {
		Namespace     string                     `json:"namespace"`
		Name          string                     `json:"name"`
		UID           string                     `json:"uid"`
		Planner       string                     `json:"planner"`
		Configuration *RetuneLaunchConfiguration `json:"configuration"`
	}{dgd.Namespace, dgd.Name, string(dgd.UID), planner.ComponentName, configuration}
	encoded, err := json.Marshal(binding)
	if err != nil {
		return "", nil, fmt.Errorf("encode Retune target: %w", err)
	}
	digest := sha256.Sum256(encoded)
	configuration.ConfigHash = hex.EncodeToString(digest[:])
	return planner.ComponentName, configuration, nil
}

// applyDGDComponentRetune compiles graph-owned bindings into a component copy so
// both direct rendering and the DGD-to-DCD boundary retain identical wiring.
// component and dgd must not be nil; dgd is not mutated.
func applyDGDComponentRetune(component *v1beta1.DynamoComponentDeploymentSharedSpec, dgd *v1beta1.DynamoGraphDeployment) error {
	plannerName, configuration, err := RetuneConfiguration(dgd)
	if err != nil || configuration == nil {
		return err
	}
	isFrontend := component.ComponentName == configuration.FrontendComponentName
	if !isFrontend && component.ComponentName != plannerName {
		return nil
	}
	capability, err := json.Marshal(configuration)
	if err != nil {
		return fmt.Errorf("encode Retune launch contract: %w", err)
	}

	// Compile only into private templates, leaving the authored DGD untouched.
	for _, template := range EnsureComponentPodTemplates(component) {
		main := ensureMainContainer(template)
		if err := addRetuneTraceMount(&template.Spec, main, configuration.TraceStorage); err != nil {
			return fmt.Errorf("component %q: %w", component.ComponentName, err)
		}
		var bindings []corev1.EnvVar
		if isFrontend {
			bindings = retuneFrontendEnvironment(dgd.Name, configuration)
			bindings = append(bindings, corev1.EnvVar{Name: "DYN_FRONTEND_RETUNE_CAPABILITY", Value: string(capability)})
		} else {
			// A launch-time binding prevents an old Planner from using a newer
			// target contract with the PVC still mounted in its existing Pod.
			bindings = []corev1.EnvVar{{Name: "DYN_PLANNER_RETUNE_CAPABILITY", Value: string(capability)}}
		}
		owned := make(map[string]bool, len(bindings))
		for _, binding := range bindings {
			owned[binding.Name] = true
		}
		for _, binding := range main.Env {
			if !owned[binding.Name] {
				bindings = append(bindings, binding)
			}
		}
		main.Env = bindings
	}
	return nil
}

func retuneFrontendEnvironment(dgdName string, configuration *RetuneLaunchConfiguration) []corev1.EnvVar {
	// DYN_POD_NAME precedes DYN_REQUEST_TRACE_FILE_PATH even in legacy sorted
	// environments; POD_NAME does not. Bind the DGD directory without an alias.
	return []corev1.EnvVar{
		{Name: "DYN_POD_NAME", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "metadata.name"}}},
		{Name: "DYN_REQUEST_TRACE", Value: "1"},
		{Name: "DYN_REQUEST_TRACE_RECORDS", Value: "request_end"},
		{Name: "DYN_REQUEST_TRACE_SINKS", Value: "file"},
		{Name: "DYN_REQUEST_TRACE_FILE_FORMAT", Value: "jsonl_gz"},
		{Name: "DYN_REQUEST_TRACE_FILE_PATH", Value: path.Join(configuration.TraceStorage.MountPath, configuration.TraceStorage.SegmentsDir, dgdName, "$(DYN_POD_NAME)", "req")},
		{Name: "DYN_ENABLE_FRONTEND_ADMIN_API", Value: "1"},
		{Name: "DYN_FRONTEND_ADMIN_HOST", ValueFrom: &corev1.EnvVarSource{FieldRef: &corev1.ObjectFieldSelector{FieldPath: "status.podIP"}}},
		{Name: "DYN_FRONTEND_ADMIN_PORT", Value: strconv.Itoa(int(configuration.AdminPort))},
	}
}

func addRetuneTraceMount(pod *corev1.PodSpec, main *corev1.Container, storage RetuneTraceStorage) error {
	// A direct writable mount already exposing the requested claim is reusable.
	reusable := false
	for _, mount := range main.VolumeMounts {
		mountPath := path.Clean(mount.MountPath)
		if strings.HasPrefix(mountPath, storage.MountPath+"/") {
			return fmt.Errorf("retune trace directory %q is obscured by volumeMount %q", storage.MountPath, mount.MountPath)
		}
		if mountPath != storage.MountPath {
			continue
		}
		if mount.ReadOnly || mount.SubPath != "" || mount.SubPathExpr != "" {
			return fmt.Errorf("retune requires a direct writable trace mount at %q, without subPath", storage.MountPath)
		}
		matchesClaim := false
		for _, volume := range pod.Volumes {
			if volume.Name == mount.Name && volume.PersistentVolumeClaim != nil && volume.PersistentVolumeClaim.ClaimName == storage.ClaimName && !volume.PersistentVolumeClaim.ReadOnly {
				matchesClaim = true
				break
			}
		}
		if !matchesClaim {
			return fmt.Errorf("retune trace mount at %q must reference PVC %q", storage.MountPath, storage.ClaimName)
		}
		reusable = true
	}
	if reusable {
		return nil
	}

	// Never replace unrelated authored volumes or relocate their mounts.
	for _, mount := range main.VolumeMounts {
		if mount.Name == retuneTraceVolumeName {
			return fmt.Errorf("retune volume %q is already mounted at %q", retuneTraceVolumeName, mount.MountPath)
		}
	}
	found := false
	for _, volume := range pod.Volumes {
		if volume.Name != retuneTraceVolumeName {
			continue
		}
		if volume.PersistentVolumeClaim == nil || volume.PersistentVolumeClaim.ClaimName != storage.ClaimName || volume.PersistentVolumeClaim.ReadOnly {
			return fmt.Errorf("retune volume %q must reference writable PVC %q", retuneTraceVolumeName, storage.ClaimName)
		}
		found = true
	}
	if !found {
		pod.Volumes = append(pod.Volumes, corev1.Volume{Name: retuneTraceVolumeName, PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{ClaimName: storage.ClaimName}})
	}
	main.VolumeMounts = append(main.VolumeMounts, corev1.VolumeMount{Name: retuneTraceVolumeName, MountPath: storage.MountPath})
	return nil
}
