/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"fmt"
	"maps"
	"net/url"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/common"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	controllercommon "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/pelletier/go-toml/v2"
	corev1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/utils/ptr"
)

const (
	lpuConfigVolumeName = "config"
	lpuConfigMountPath  = "/configs"
)

type lpuModelStorage struct {
	volume corev1.Volume
	mount  corev1.VolumeMount
}

func boundedAuxiliaryName(root, suffix string) string {
	candidate := root + suffix
	if len(candidate) <= validation.DNS1123SubdomainMaxLength {
		return candidate
	}

	// Distinguish overlong resource names using the full name, including its suffix.
	digest := sha256.Sum256([]byte(candidate))
	hashSuffix := fmt.Sprintf("-%x", digest[:4])
	prefix := strings.TrimRight(candidate[:validation.DNS1123SubdomainMaxLength-len(hashSuffix)], "-.")
	return prefix + hashSuffix
}

// LPUConfigMapName names the immutable runtime table using its Pod-template content hash.
func LPUConfigMapName(root, configHash string) string {
	return boundedAuxiliaryName(root, fmt.Sprintf("-lpu-%.16s", configHash))
}

func renderLPUConfigMap(
	namespace string,
	materializationName string,
	modelStoragePath string,
	projections []*ModelProjection,
	agents []ExpectedAgent,
) (*corev1.ConfigMap, error) {
	modelConfig, err := lpuModelConfig(projections, modelStoragePath)
	if err != nil {
		return nil, fmt.Errorf("render model_config.toml: %w", err)
	}
	var modelTOML strings.Builder
	if err := toml.NewEncoder(&modelTOML).Encode(modelConfig); err != nil {
		return nil, fmt.Errorf("render model_config.toml: %w", err)
	}

	datacenterRacks := make(map[string]any, len(agents))
	for _, agent := range agents {
		availableNodes := make([]int, agent.Replicas)
		for index := range availableNodes {
			availableNodes[index] = index
		}
		datacenterRacks[agent.TemplateName] = map[string]any{
			"available_nodes": availableNodes,
			"preferred_start": 0,
			"node_name_template": fmt.Sprintf(
				"%s-%s-{node}.${GROVE_HEADLESS_SERVICE}",
				"${GROVE_PCSG_NAME}-${GROVE_PCSG_INDEX}",
				agent.TemplateName,
			),
		}
	}
	var datacenterTOML strings.Builder
	_ = toml.NewEncoder(&datacenterTOML).Encode(map[string]any{
		"model_base_paths": []string{modelStoragePath},
		"datacenters":      map[string]any{"racks": datacenterRacks},
	})

	data := resolvedPartitionData(projections)
	if projections[0].pipeline == PipelineLPX {
		data["gas_dir"] = modelConfig["iop"].(map[string]any)["model_path"].(string)
	}
	data["model_config.toml"] = modelTOML.String()
	data["datacenter.toml"] = datacenterTOML.String()
	return renderRuntimeConfigMap(namespace, materializationName+"-lpu", data)
}

func renderRuntimeConfigMap(namespace, namePrefix string, data map[string]string) (*corev1.ConfigMap, error) {
	// Name immutable configuration from the content hash used by Pod templates.
	configMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
		},
		Immutable: ptr.To(true),
		Data:      data,
	}
	configMap.Name = boundedAuxiliaryName(namePrefix, fmt.Sprintf("-%.16s", LPUConfigMapHash(configMap)))

	// Reject oversized configuration before any caller can publish it.
	totalSize := 0
	for _, value := range data {
		totalSize += len(value)
	}
	if totalSize > corev1.MaxSecretSize {
		return nil, fmt.Errorf(
			"rendered LPX ConfigMap %q data is %d bytes; maximum is %d",
			configMap.Name,
			totalSize,
			corev1.MaxSecretSize,
		)
	}
	return configMap, nil
}

// LPUConfigMapHash returns the hash stamped on LPU runtime Pod templates.
// configMap must be non-nil, with valid native metadata from rendering or Kubernetes.
// Its concrete content is always serializable; the input is not mutated.
func LPUConfigMapHash(configMap *corev1.ConfigMap) string {
	contentHash, _ := controllercommon.GetSpecHash(configMap)

	// The graph's extra-resource annotation hashes each resource's spec hash.
	// Apply the same second hash here so LPX can render that annotation directly.
	return fmt.Sprintf("%x", sha256.Sum256([]byte(contentHash)))
}

func lpuModelStorageBinding(spec corev1.PodSpec) (lpuModelStorage, error) {
	container := common.FindContainerByName(spec.Containers, commonconsts.MainContainerName)
	mountIndex := slices.IndexFunc(container.VolumeMounts, func(mount corev1.VolumeMount) bool {
		return mount.Name == commonconsts.ModelStorageVolumeName
	})
	if mountIndex < 0 {
		return lpuModelStorage{}, fmt.Errorf(
			"selected LPX main container requires model storage volume mount %q",
			commonconsts.ModelStorageVolumeName,
		)
	}
	mount := container.VolumeMounts[mountIndex]
	if strings.TrimSpace(mount.MountPath) == "" {
		return lpuModelStorage{}, fmt.Errorf("model storage volume %q has no mount path", mount.Name)
	}
	volumeIndex := slices.IndexFunc(spec.Volumes, func(volume corev1.Volume) bool { return volume.Name == mount.Name })
	if volumeIndex < 0 {
		return lpuModelStorage{}, fmt.Errorf("selected LPX podTemplate has no model storage volume %q", mount.Name)
	}
	return lpuModelStorage{volume: spec.Volumes[volumeIndex], mount: mount}, nil
}

// withLPUModelStorage requires a nonnil spec and a unique container retained from its unchanged Containers slice.
func withLPUModelStorage(spec *corev1.PodSpec, container *corev1.Container, storage lpuModelStorage) error {
	volumeIndex := slices.IndexFunc(spec.Volumes, func(existing corev1.Volume) bool { return existing.Name == storage.volume.Name })
	if volumeIndex < 0 {
		spec.Volumes = append(spec.Volumes, storage.volume)
	} else if !apiequality.Semantic.DeepEqual(spec.Volumes[volumeIndex], storage.volume) {
		return fmt.Errorf("selected Cyborg podTemplate volume %q conflicts with model storage", storage.volume.Name)
	}

	mountIndex := slices.IndexFunc(container.VolumeMounts, func(existing corev1.VolumeMount) bool {
		return existing.Name == storage.mount.Name || existing.MountPath == storage.mount.MountPath
	})
	if mountIndex < 0 {
		container.VolumeMounts = append(container.VolumeMounts, storage.mount)
	} else if !apiequality.Semantic.DeepEqual(container.VolumeMounts[mountIndex], storage.mount) {
		return fmt.Errorf("selected Cyborg main container conflicts with model storage mount %q", storage.mount.MountPath)
	}
	return nil
}

func lpuModelConfig(projections []*ModelProjection, modelStoragePath string) (map[string]any, error) {
	if projections[0].pipeline != PipelineSpecDecode {
		config, err := nestedLPUModelConfig(projections[0], modelStoragePath)
		if err != nil {
			return nil, err
		}
		config["type"] = "Single"
		return config, nil
	}

	// Derive the pipeline-wide speculative-decoding controls from the draft model.
	config := make(map[string]any)

	// Promote the draft's setup timeouts so Nova applies them to every model.
	setup, err := modelObjectSetting(projections[0], "setup")
	if err != nil {
		return nil, err
	}
	delete(setup, "setup_ops_format")
	delete(setup, "resolved_partitions_dir")
	if len(setup) > 0 {
		config["setup"] = setup
	}

	// Promote the only model-provided speculative-decoding scalar currently consumed by the runtime.
	if value, present := projections[0].configuredBuild.runtimeSettings[v3MaxSWADKVCBlocksDraft]; present {
		config[v3MaxSWADKVCBlocksDraft] = value
	}

	draftCount := len(projections) - 1
	draft, err := nestedLPUModelConfig(projections[0], modelStoragePath)
	if err != nil {
		return nil, err
	}
	// Draft setup values configure the SpecDecode parent. Keep the child setup
	// limited to the operator-owned LPU runtime bindings so Nova can inherit the
	// parent timeouts exactly as it did with the legacy pipeline field.
	draft["setup"] = map[string]any{
		"setup_ops_format":        agentSetupOpsFormat,
		"resolved_partitions_dir": lpuConfigMountPath,
	}
	target, err := nestedLPUModelConfig(projections[draftCount], modelStoragePath)
	if err != nil {
		return nil, err
	}
	config["type"] = "SpecDecode"
	config["draft"] = draft
	config["target"] = target
	config["num_drafts"] = draftCount
	setConfigDefault(config, "draft_to_target_port", 23456)
	setConfigDefault(config, "target_to_draft_port", 23457)
	setConfigDefault(config, "head_to_head_port", 23458)
	return config, nil
}

func nestedLPUModelConfig(projection *ModelProjection, modelStoragePath string) (map[string]any, error) {
	settings := maps.Clone(projection.configuredBuild.runtimeSettings)

	// Lift Nova's model-level sections out of the IOP settings object.
	scheduler, err := modelObjectSetting(projection, "scheduler")
	if err != nil {
		return nil, err
	}
	if scheduler == nil {
		scheduler = map[string]any{}
	}
	delete(settings, "scheduler")
	setup, err := modelObjectSetting(projection, "setup")
	if err != nil {
		return nil, err
	}
	if setup == nil {
		setup = map[string]any{}
	}
	delete(settings, "setup")

	// Keep operator-owned LPU runtime setup bindings authoritative.
	setup["setup_ops_format"] = agentSetupOpsFormat
	setup["resolved_partitions_dir"] = lpuConfigMountPath
	delete(settings, v3MaxSWADKVCBlocksDraft)

	// Lift additional programs into the Nova model object when configured.
	var extraPrograms []any
	if configured, present := settings["extra_programs"]; present {
		var ok bool
		extraPrograms, ok = configured.([]any)
		if !ok {
			return nil, fmt.Errorf("model %q settings.extra_programs must be an array", projection.model)
		}
		delete(settings, "extra_programs")
		if projection.configuredBuild.Family == BuildFamilyXT &&
			projection.pipeline == PipelineSingle && len(extraPrograms) > 0 {
			return nil, fmt.Errorf("model %q settings.extra_programs is not supported with setup_ops_format %q", projection.model, agentSetupOpsFormat)
		}
	}

	build := &projection.configuredBuild
	buildRef := build.Path
	if projection.pipeline == PipelineLPX ||
		projection.configuredBuild.Family == BuildFamilyHX {
		buildRef = lpuRuntimeBuildRef(projection, modelStoragePath)
	}
	modelPath, err := buildRuntimePath(buildRef, modelStoragePath)
	if err != nil {
		return nil, fmt.Errorf("model %q model_path: %w", projection.model, err)
	}
	if _, overridden := settings["tokenizer_path"]; !overridden &&
		strings.TrimSpace(build.RuntimeTokenizerPath) != "" {
		settings["tokenizer_path"] = filepath.Join(modelPath, build.RuntimeTokenizerPath)
	}
	if projection.configuredBuild.Family == BuildFamilyHX ||
		projection.pipeline == PipelineSingle {
		if err := validateRuntimeTokenizerSettings(settings); err != nil {
			return nil, fmt.Errorf("model %q settings: %w", projection.model, err)
		}
	}
	if build.SupportsCPUEmbeddings {
		if enabled, _ := settings["cpu_embeddings"].(bool); enabled {
			if _, overridden := settings["embedding_path"]; !overridden &&
				strings.TrimSpace(build.RuntimeTokenEmbeddingsPath) != "" {
				settings["embedding_path"] = filepath.Join(modelPath, build.RuntimeTokenEmbeddingsPath)
			}
		}
	}
	if projection.configuredBuild.Family == BuildFamilyXT &&
		projection.pipeline == PipelineSpecDecode {
		setConfigDefault(settings, "model_path", modelPath)
	} else {
		settings["model_path"] = modelPath
	}
	config := map[string]any{
		"scheduler": scheduler,
		"setup":     setup,
		"iop":       settings,
	}
	if extraPrograms != nil {
		config["extra_programs"] = extraPrograms
	}
	return config, nil
}

func modelObjectSetting(projection *ModelProjection, name string) (map[string]any, error) {
	configured, present := projection.configuredBuild.runtimeSettings[name]
	if !present {
		return nil, nil
	}
	object, ok := configured.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("model %q settings.%s must be an object", projection.model, name)
	}
	return maps.Clone(object), nil
}

func lpuRuntimeBuildRef(projection *ModelProjection, modelStoragePath string) string {
	buildRef := projection.configuredBuild.Path
	snapshotRef, snapshotErr := url.Parse(buildRef)
	runtimeRef := strings.TrimSpace(projection.runtimeBuildRef)
	runtimeURL, runtimeErr := url.Parse(runtimeRef)
	if snapshotErr != nil || runtimeErr != nil || snapshotRef.Scheme != BuildSchemeFile ||
		runtimeRef == "" || runtimeURL.Scheme != "" || filepath.IsAbs(runtimeRef) {
		return buildRef
	}
	cleaned := filepath.Clean(runtimeRef)
	if cleaned == "." || cleaned == ".." || strings.HasPrefix(cleaned, ".."+string(filepath.Separator)) {
		return buildRef
	}
	return (&url.URL{Scheme: BuildSchemeFile, Path: filepath.Join(modelStoragePath, cleaned)}).String()
}

func resolvedPartitionData(projections []*ModelProjection) map[string]string {
	keys := [...]string{"nodes_per_partition", "partition_indices", "partition_ids", "partition_models",
		"partition_node_offsets", "partition_paths", "topologies"}

	// Omit model-identity columns that the XT Single runtime never consumes.
	includeModelColumns := projections[0].configuredBuild.Family != BuildFamilyXT ||
		projections[0].pipeline != PipelineSingle
	var columns [len(keys)]strings.Builder

	// Accumulate each projection's runtime partitions into the surviving columns.
	for _, projection := range projections {
		// Render configured V2 runtime metadata without reapplying scheduler-shape validation to collapsed partitions.
		partitions := projection.partitions
		v2Runtime := projection.configuredBuild.Family == BuildFamilyXT &&
			len(projection.configuredBuild.Partitions) != 0
		if v2Runtime {
			partitions = projection.configuredBuild.Partitions
		}

		offset := int64(0)
		for index, partition := range partitions {
			var nodes string
			var endpointCount int64
			if v2Runtime {
				nodeCount := partition.effectiveNodeCount()
				nodes, endpointCount = strconv.Itoa(nodeCount), int64(nodeCount)
			} else {
				endpointCount = partition.HXExtent[1] * partition.HXExtent[2] * partition.HXExtent[3]
				nodes = strconv.FormatInt(endpointCount, 10)
			}
			// Project one row and populate optional model identity only when consumed.
			row := [len(keys)]string{nodes, "",
				strconv.FormatUint(uint64(uint32(partition.SourcePartitionID)), 10), "",
				strconv.FormatInt(offset, 10), partition.PartPath, partition.Topology.Raw}
			if includeModelColumns {
				row[1] = strconv.Itoa(index)
				row[3] = projection.model
			}

			// Write only columns that survive into the ConfigMap.
			for column := range row {
				if !includeModelColumns && (column == 1 || column == 3) {
					continue
				}
				columns[column].WriteString(row[column])
				columns[column].WriteByte('\n')
			}
			offset += endpointCount
		}
	}
	data := make(map[string]string, len(keys))

	// Materialize only the runtime-visible columns.
	for column := range keys {
		if !includeModelColumns && (column == 1 || column == 3) {
			continue
		}
		data[keys[column]] = strings.TrimSuffix(columns[column].String(), "\n")
	}
	return data
}

func setConfigDefault(values map[string]any, key string, value any) {
	if _, present := values[key]; !present {
		values[key] = value
	}
}

func withLPUConfigVolume(spec *corev1.PodSpec, configMapName string, allowOverrides bool) error {
	found := false
	for _, volume := range spec.Volumes {
		if volume.Name != lpuConfigVolumeName {
			continue
		}
		if !allowOverrides && (found ||
			volume.ConfigMap == nil ||
			volume.ConfigMap.Name != configMapName ||
			len(volume.ConfigMap.Items) != 0 ||
			volume.ConfigMap.DefaultMode != nil ||
			volume.ConfigMap.Optional != nil) {
			return fmt.Errorf("selected LPX podTemplate volume %q is reserved for ConfigMap %q", lpuConfigVolumeName, configMapName)
		}
		found = true
	}
	if !found {
		spec.Volumes = append(spec.Volumes, corev1.Volume{
			Name: lpuConfigVolumeName,
			VolumeSource: corev1.VolumeSource{ConfigMap: &corev1.ConfigMapVolumeSource{
				LocalObjectReference: corev1.LocalObjectReference{Name: configMapName},
			}},
		})
	}
	container := common.FindContainerByName(spec.Containers, commonconsts.MainContainerName)
	found = false
	for _, mount := range container.VolumeMounts {
		if mount.Name == lpuConfigVolumeName || mount.MountPath == lpuConfigMountPath {
			if !allowOverrides && (found ||
				mount.Name != lpuConfigVolumeName ||
				mount.MountPath != lpuConfigMountPath ||
				mount.ReadOnly ||
				mount.RecursiveReadOnly != nil ||
				mount.SubPath != "" ||
				mount.MountPropagation != nil ||
				mount.SubPathExpr != "") {
				return fmt.Errorf("selected LPX main container reserves volume %q at %q", lpuConfigVolumeName, lpuConfigMountPath)
			}
			found = found || mount.MountPath == lpuConfigMountPath
		}
	}
	if !found {
		container.VolumeMounts = append(container.VolumeMounts, corev1.VolumeMount{
			Name:      lpuConfigVolumeName,
			MountPath: lpuConfigMountPath,
		})
	}
	return nil
}
