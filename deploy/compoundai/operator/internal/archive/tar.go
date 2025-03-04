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
