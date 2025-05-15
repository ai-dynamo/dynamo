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

package archive

import (
	"archive/tar"
	"bytes"
	"fmt"
	"io"
	"path/filepath"
)

func ExtractFileFromTar(tarData []byte, fileName string) (*bytes.Buffer, error) {
	// Create a tar reader
	tarReader := tar.NewReader(bytes.NewReader(tarData))

	// Iterate through tar archive
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break // End of archive
		}
		if err != nil {
			return nil, fmt.Errorf("error reading tar file: %w", err)
		}

		// Check if the current file is the desired YAML file
		if header.Typeflag == tar.TypeReg && (header.Name == fileName || filepath.Base(header.Name) == fileName) {
			var content bytes.Buffer
			_, err = content.ReadFrom(tarReader)
			if err != nil {
				return nil, fmt.Errorf("error extracting file: %w", err)
			}
			return &content, nil
		}
	}
	return nil, fmt.Errorf("file %s not found in tar archive", fileName)
}
