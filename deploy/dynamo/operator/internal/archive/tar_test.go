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
	"bytes"
	"os"
	"reflect"
	"testing"
)

func TestExtractFileFromTar(t *testing.T) {
	// read test.tar file
	// it contains test.yaml at the root
	tarData, err := os.ReadFile("test.tar")
	if err != nil {
		t.Fatalf("Failed to read test.tar: %v", err)
	}
	// read test2.tar file
	// it contains test2.yaml inside a folder
	tarData2, err := os.ReadFile("test2.tar")
	if err != nil {
		t.Fatalf("Failed to read test2.tar: %v", err)
	}
	type args struct {
		tarData      []byte
		yamlFileName string
	}
	tests := []struct {
		name    string
		args    args
		want    *bytes.Buffer
		wantErr bool
	}{
		{
			name: "Test ExtractFileFromTar",
			args: args{
				tarData:      tarData,
				yamlFileName: "test.yaml",
			},
			want:    bytes.NewBufferString("property1: true\n"),
			wantErr: false,
		},
		{
			name: "Test ExtractFileFromTar",
			args: args{
				tarData:      tarData2,
				yamlFileName: "test.yaml",
			},
			want:    bytes.NewBufferString("property1: true\n"),
			wantErr: false,
		},
		{
			name: "Test ExtractFileFromTar, file not found",
			args: args{
				tarData:      tarData,
				yamlFileName: "test2.yaml",
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "Test ExtractFileFromTar, invalid content",
			args: args{
				tarData:      []byte("invalid content"),
				yamlFileName: "test.yaml",
			},
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ExtractFileFromTar(tt.args.tarData, tt.args.yamlFileName)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExtractFileFromTar() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ExtractFileFromTar() = %v, want %v", got, tt.want)
			}
		})
	}
}
