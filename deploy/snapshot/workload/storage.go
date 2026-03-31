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

func ResolveRestoreStorage(checkpointID string, version string, location string, storage Storage) (Storage, error) {
	resolved, err := resolveStorageConfig(storage)
	if err != nil {
		return Storage{}, err
	}
	location = strings.TrimSpace(location)
	if location == "" {
		resolved, err = ResolveCheckpointStorage(checkpointID, version, storage)
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
	basePath := strings.TrimSpace(storage.BasePath)
	if basePath == "" {
		return Storage{}, fmt.Errorf("checkpoint base path is required")
	}
	return Storage{
		Type:     storageType,
		PVCName:  strings.TrimSpace(storage.PVCName),
		BasePath: strings.TrimRight(basePath, "/"),
	}, nil
}
