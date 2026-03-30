package workload

import (
	"fmt"
	"strings"
)

const DefaultCheckpointArtifactVersion = "1"

type StorageConfig struct {
	Type     string
	PVCName  string
	BasePath string
}

type ResolvedStorage struct {
	Type     string
	Location string
	PVCName  string
	BasePath string
}

func ResolveCheckpointStorage(checkpointID string, version string, config StorageConfig) (ResolvedStorage, error) {
	resolved, err := resolveStorageConfig(config)
	if err != nil {
		return ResolvedStorage{}, err
	}
	version = strings.TrimSpace(version)
	if version == "" {
		version = DefaultCheckpointArtifactVersion
	}
	resolved.Location = strings.TrimRight(resolved.BasePath, "/") + "/" + checkpointID + "/versions/" + version
	return resolved, nil
}

func ResolveRestoreStorage(checkpointID string, location string, storageType string, config StorageConfig) (ResolvedStorage, error) {
	resolved, err := resolveStorageConfig(config)
	if err != nil {
		return ResolvedStorage{}, err
	}
	if strings.TrimSpace(storageType) != "" && strings.TrimSpace(storageType) != resolved.Type {
		return ResolvedStorage{}, fmt.Errorf("checkpoint storage type %q does not match configured storage type %q", storageType, resolved.Type)
	}
	location = strings.TrimSpace(location)
	if location == "" {
		resolved, err = ResolveCheckpointStorage(checkpointID, "", config)
		if err != nil {
			return ResolvedStorage{}, err
		}
		return resolved, nil
	}
	resolved.Location = location
	return resolved, nil
}

func resolveStorageConfig(config StorageConfig) (ResolvedStorage, error) {
	storageType := strings.TrimSpace(config.Type)
	if storageType == "" {
		storageType = StorageTypePVC
	}
	if storageType != StorageTypePVC {
		return ResolvedStorage{}, fmt.Errorf("checkpoint storage type %q is not supported", storageType)
	}
	pvcName := strings.TrimSpace(config.PVCName)
	if pvcName == "" {
		return ResolvedStorage{}, fmt.Errorf("checkpoint pvc name is required")
	}
	basePath := strings.TrimSpace(config.BasePath)
	if basePath == "" {
		return ResolvedStorage{}, fmt.Errorf("checkpoint base path is required")
	}
	return ResolvedStorage{
		Type:     storageType,
		PVCName:  pvcName,
		BasePath: strings.TrimRight(basePath, "/"),
	}, nil
}
