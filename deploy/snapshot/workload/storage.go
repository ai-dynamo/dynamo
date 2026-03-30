package workload

import (
	"fmt"
	"strings"
)

const DefaultCheckpointArtifactVersion = "1"

type Storage struct {
	Type     string
	Location string
	PVCName  string
	BasePath string
}

func ResolveCheckpointStorage(checkpointID string, version string, storage Storage) (Storage, error) {
	resolved, err := resolveStorageConfig(storage)
	if err != nil {
		return Storage{}, err
	}
	version = strings.TrimSpace(version)
	if version == "" {
		version = DefaultCheckpointArtifactVersion
	}
	resolved.Location = strings.TrimRight(resolved.BasePath, "/") + "/" + checkpointID + "/versions/" + version
	return resolved, nil
}

func ResolveRestoreStorage(checkpointID string, location string, storageType string, storage Storage) (Storage, error) {
	resolved, err := resolveStorageConfig(storage)
	if err != nil {
		return Storage{}, err
	}
	if strings.TrimSpace(storageType) != "" && strings.TrimSpace(storageType) != resolved.Type {
		return Storage{}, fmt.Errorf("checkpoint storage type %q does not match configured storage type %q", storageType, resolved.Type)
	}
	location = strings.TrimSpace(location)
	if location == "" {
		resolved, err = ResolveCheckpointStorage(checkpointID, "", storage)
		if err != nil {
			return Storage{}, err
		}
		return resolved, nil
	}
	resolved.Location = location
	return resolved, nil
}

func resolveStorageConfig(storage Storage) (Storage, error) {
	storageType := strings.TrimSpace(storage.Type)
	if storageType == "" {
		storageType = StorageTypePVC
	}
	if storageType != StorageTypePVC {
		return Storage{}, fmt.Errorf("checkpoint storage type %q is not supported", storageType)
	}
	pvcName := strings.TrimSpace(storage.PVCName)
	if pvcName == "" {
		return Storage{}, fmt.Errorf("checkpoint pvc name is required")
	}
	basePath := strings.TrimSpace(storage.BasePath)
	if basePath == "" {
		return Storage{}, fmt.Errorf("checkpoint base path is required")
	}
	return Storage{
		Type:     storageType,
		PVCName:  pvcName,
		BasePath: strings.TrimRight(basePath, "/"),
	}, nil
}
