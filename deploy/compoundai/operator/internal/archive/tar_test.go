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
