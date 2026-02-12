// middleware.go provides HTTP middleware for the server.
package httpApiServer

import (
	"log"
	"net/http"
	"time"
)

// LoggingMiddleware wraps an HTTP handler and logs request details.
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		log.Printf("Started %s %s", r.Method, r.URL.Path)
		next.ServeHTTP(w, r)
		log.Printf("Completed %s %s in %v", r.Method, r.URL.Path, time.Since(start))
	})
}
