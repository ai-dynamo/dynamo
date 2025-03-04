package main

import "github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/runtime"

const (
	port = 8181
)

func main() {
	runtime.Runtime.StartServer(port)
}
