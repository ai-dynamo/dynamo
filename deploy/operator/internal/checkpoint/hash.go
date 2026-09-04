/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package checkpoint

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
)

// DGDCheckpointID returns the snapshot artifact ID for an automatic DGD-owned
// checkpoint. The DGD UID prevents cross-DGD reuse; the component name and
// worker hash/generation prevent reuse across incompatible worker generations
// inside the same DGD.
func DGDCheckpointID(namespace, dgdName, dgdUID, componentName, workerHash string) string {
	data, _ := json.Marshal(struct {
		Namespace     string `json:"namespace,omitempty"`
		DGDName       string `json:"dgdName"`
		DGDUID        string `json:"dgdUID,omitempty"`
		ComponentName string `json:"componentName"`
		WorkerHash    string `json:"workerHash,omitempty"`
	}{
		Namespace:     namespace,
		DGDName:       dgdName,
		DGDUID:        dgdUID,
		ComponentName: componentName,
		WorkerHash:    workerHash,
	})
	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])[:32]
}
