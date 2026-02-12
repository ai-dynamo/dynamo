package restore

import (
	"fmt"

	criurpc "github.com/checkpoint-restore/go-criu/v7/rpc"
	"google.golang.org/protobuf/proto"

	"github.com/ai-dynamo/dynamo/deploy/chrek/pkg/checkpoint"
)

// GenerateExtMountMaps generates external mount mappings for CRIU restore.
// It reuses the exact dump-time ext-mount plan persisted in checkpoint manifest.
func GenerateExtMountMaps(data *checkpoint.CheckpointManifest) ([]*criurpc.ExtMountMap, error) {
	if data == nil {
		return nil, fmt.Errorf("checkpoint manifest is required")
	}
	if len(data.CRIUDump.ExtMnt) == 0 {
		return nil, fmt.Errorf("checkpoint manifest is missing criuDump.extMnt")
	}

	maps := []*criurpc.ExtMountMap{{
		Key: proto.String("/"),
		Val: proto.String("."),
	}}
	addedMounts := map[string]struct{}{"/": {}}

	// Replay dump-time ext-mount plan exactly, with restore-specific root remap.
	for _, mount := range data.CRIUDump.ExtMnt {
		key := mount.Key
		if key == "" || key == "/" {
			continue
		}
		if _, exists := addedMounts[key]; exists {
			continue
		}
		val := mount.Val
		if val == "" {
			val = key
		}
		maps = append(maps, &criurpc.ExtMountMap{
			Key: proto.String(key),
			Val: proto.String(val),
		})
		addedMounts[key] = struct{}{}
	}

	return maps, nil
}
