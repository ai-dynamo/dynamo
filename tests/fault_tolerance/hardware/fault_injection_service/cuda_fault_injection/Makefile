# Makefile for CUDA Intercept Library
# Simulates various XID errors via LD_PRELOAD
#
# GCC Version: Requires gcc 4.8+ (any modern gcc works)
#              Tested with gcc 7.5, 9.4, 11.x, 13.x
#              Uses standard C99 features only

.PHONY: all clean test help

# Compiler settings
CC = gcc
CFLAGS = -fPIC -Wall -Wextra -std=c99
LDFLAGS = -shared
LDLIBS = -ldl
TARGET = cuda_intercept.so
SOURCE = cuda_intercept.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	@echo "Building CUDA fault injection library..."
	$(CC) $(CFLAGS) $(SOURCE) $(LDFLAGS) -o $(TARGET) $(LDLIBS)
	@echo "✓ Built: $(TARGET)"
	@echo ""
	@echo "Usage:"
	@echo "  LD_PRELOAD=./$(TARGET) python -c 'import torch; print(torch.cuda.is_available())'"
	@echo ""

clean:
	@echo "Cleaning..."
	rm -f $(TARGET)
	@echo "✓ Cleaned"

test: $(TARGET)
	@echo "Testing CUDA fault injection..."
	@echo ""
	@echo "Test 1: With fault injection enabled (default)"
	@CUDA_FAULT_INJECTION_ENABLED=1 LD_PRELOAD=./$(TARGET) python3 -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("Device count:", torch.cuda.device_count())' || echo "✓ Expected failure"
	@echo ""
	@echo "Test 2: With fault injection disabled"
	@CUDA_FAULT_INJECTION_ENABLED=0 LD_PRELOAD=./$(TARGET) python3 -c 'import torch; print("CUDA available:", torch.cuda.is_available()); print("Device count:", torch.cuda.device_count())'
	@echo ""

help:
	@echo "CUDA Fault Injection Library - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build the library"
	@echo "  make test     - Run basic tests"
	@echo "  make clean    - Remove built files"
	@echo "  make help     - Show this help"
	@echo ""
	@echo "Usage in tests:"
	@echo "  export LD_PRELOAD=/path/to/fake_cuda_xid79.so"
	@echo "  export CUDA_FAULT_INJECTION_ENABLED=1"
	@echo "  python -m vllm.entrypoints.api_server"

