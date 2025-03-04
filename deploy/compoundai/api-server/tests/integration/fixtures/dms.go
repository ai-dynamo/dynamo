package fixtures

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/services"
)

type MockDMSServer struct {
	server *httptest.Server
	throws *bool
}

func (s *MockDMSServer) Close() {
	s.server.Close()
}

func (s *MockDMSServer) Throws(throws bool) {
	s.throws = &throws
}

func CreateMockDMSServer(t *testing.T) *MockDMSServer {
	throws := false
	mockServer := MockDMSServer{}
	mockServer.throws = &throws
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if *mockServer.throws {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		response := services.DMSCreateResponse{
			Id: "abc123",
			Status: services.DMSResponseStatus{
				Status:  "success",
				Message: "DMS resource created successfully",
			},
			Configuration: map[string]string{
				"setting1": "value1",
				"setting2": "value2",
				"setting3": "value3",
			},
		}

		jsonResponse, err := json.Marshal(response)
		if err != nil {
			t.Fatalf("Failed to marshal JSON %s", err.Error())
		}

		w.WriteHeader(http.StatusOK)
		w.Write(jsonResponse)
	}))

	idx := strings.LastIndex(server.URL, ":")
	os.Setenv("DMS_HOST", "localhost")
	os.Setenv("DMS_PORT", server.URL[idx+1:])

	mockServer.server = server
	return &mockServer
}
