# Makefile for Triton Server installation
# Builds Triton Server and copies artifacts only when necessary

CURRENT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
TMP_SERVER := /tmp/server
BUILD_INSTALL := $(TMP_SERVER)/build/install

# Target directories
WHEELHOUSE_DIR := $(CURRENT_DIR)/wheelhouse
LIB_DIR := $(CURRENT_DIR)/lib
BIN_DIR := $(CURRENT_DIR)/bin
BACKEND_DIR := $(CURRENT_DIR)/backends
# Target files
TRITON_LIB := $(LIB_DIR)/libtritonserver.so
TRITON_BIN := $(BIN_DIR)/tritonserver
TRITON_WHEEL := $(WHEELHOUSE_DIR)/.wheel_marker

# Main target - build everything
.PHONY: all
all: $(TRITON_LIB) $(TRITON_BIN) $(TRITON_WHEEL) $(BACKEND_DIR)
	@echo "Triton Server installation complete."
	@echo "Add to your environment:"
	@echo "  export LD_LIBRARY_PATH=$(LIB_DIR):$(BACKEND_DIR):\$$LD_LIBRARY_PATH"
	@echo "  export PATH=$(BIN_DIR):\$$PATH"

# Clone the repository
$(TMP_SERVER):
	@echo "Cloning Triton Server repository..."
	cd /tmp && git clone https://github.com/triton-inference-server/server.git

# Build Triton Server
$(BUILD_INSTALL)/lib/libtritonserver.so $(BUILD_INSTALL)/bin/tritonserver: $(TMP_SERVER)
	@echo "Building Triton Server (this may take a while)..."
	cd $(TMP_SERVER) && \
	uv venv .venv && \
	. .venv/bin/activate && \
	uv pip install distro requests && \
	python3 build.py \
	  --enable-logging \
	  --enable-stats \
	  --enable-metrics \
	  --endpoint=http \
	  --backend=identity

# Copy library
$(TRITON_LIB): $(BUILD_INSTALL)/lib/libtritonserver.so
	@echo "Copying libtriton.so..."
	@mkdir -p $(LIB_DIR)
	cp $(BUILD_INSTALL)/lib/libtritonserver.so $(LIB_DIR)/

# Copy binary
$(TRITON_BIN): $(BUILD_INSTALL)/bin/tritonserver
	@echo "Copying tritonserver binary..."
	@mkdir -p $(BIN_DIR)
	cp $(BUILD_INSTALL)/bin/tritonserver $(BIN_DIR)/

# Copy backends
$(BACKEND_DIR): $(BUILD_INSTALL)/backends
	@echo "Copying backends..."
	@mkdir -p $(BACKEND_DIR)
	cp -r $(BUILD_INSTALL)/backends/* $(BACKEND_DIR)/

# Copy wheels
$(TRITON_WHEEL): $(BUILD_INSTALL)/lib/libtritonserver.so
	@echo "Copying Python wheels..."
	@mkdir -p $(WHEELHOUSE_DIR)
	cp $(BUILD_INSTALL)/python/*.whl $(WHEELHOUSE_DIR)/
	@touch $(TRITON_WHEEL)
	@echo "Triton Server wheel built successfully."
	@ls -al $(WHEELHOUSE_DIR)

# Clean installed artifacts (keeps the build)
.PHONY: clean
clean:
	@echo "Cleaning installed artifacts..."
	rm -rf $(WHEELHOUSE_DIR) $(LIB_DIR) $(BIN_DIR)

# Full clean (removes everything including the cloned repo and build)
.PHONY: distclean
distclean: clean
	@echo "Cleaning build directory..."
	rm -rf $(TMP_SERVER)

# Show what would be built
.PHONY: status
status:
	@echo "Installation status:"
	@echo "  Repository: $(if $(wildcard $(TMP_SERVER)),✓ cloned,✗ not cloned)"
	@echo "  Built: $(if $(wildcard $(BUILD_INSTALL)/lib/libtritonserver.so),✓ yes,✗ no)"
	@echo "  Library: $(if $(wildcard $(TRITON_LIB)),✓ installed,✗ not installed)"
	@echo "  Binary: $(if $(wildcard $(TRITON_BIN)),✓ installed,✗ not installed)"
	@echo "  Wheels: $(if $(wildcard $(TRITON_WHEEL)),✓ installed,✗ not installed)"

.PHONY: help
help:
	@echo "Triton Server Installation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build and install Triton Server (default)"
	@echo "  clean      - Remove installed artifacts (keeps build cache)"
	@echo "  distclean  - Remove everything including build cache"
	@echo "  status     - Show installation status"
	@echo "  help       - Show this help message"

