# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum* ./
RUN go mod download

# Copy source
COPY cmd/ cmd/

# Build
RUN CGO_ENABLED=0 GOOS=linux go build -o body-transformer ./cmd

# Runtime stage
FROM alpine:3.19

RUN apk --no-cache add ca-certificates

WORKDIR /app
COPY --from=builder /app/body-transformer .

EXPOSE 9003

ENTRYPOINT ["./body-transformer"]
CMD ["-port", "9003"]

