# Makefile for rebuilding NIXL and kvbm
# Usage: make [target]
# Location: lib/bindings/kvbm/Makefile

.PHONY: all clean nixl kvbm rebuild info shutdown help

# Paths (absolute for flexibility)
WORKSPACE_ROOT := /workspace
NIXL_DIR := $(WORKSPACE_ROOT)/nixl
BUILD_DIR := $(NIXL_DIR)/build
KVBM_DIR := $(WORKSPACE_ROOT)/lib/bindings/kvbm
DIST_DIR := $(WORKSPACE_ROOT)/dist

# Default: rebuild everything
all: rebuild

# ============================================================================
# NIXL Targets
# ============================================================================

nixl-setup:
	@echo "========================================"
	@echo "Setting up NIXL build with meson..."
	@echo "========================================"
	cd $(NIXL_DIR) && meson setup $(BUILD_DIR) -Dinstall_headers=true -Ddisable_gds_backend=true

nixl-build:
	@echo "========================================"
	@echo "Building NIXL with ninja..."
	@echo "========================================"
	cd $(BUILD_DIR) && ninja

nixl-install:
	@echo "========================================"
	@echo "Installing NIXL..."
	@echo "========================================"
	cd $(BUILD_DIR) && ninja install
	ldconfig

nixl: nixl-build nixl-install
	@echo "✅ NIXL rebuild complete!"

# ============================================================================
# kvbm Targets
# ============================================================================

kvbm-build:
	@echo "========================================"
	@echo "Building kvbm Python bindings..."
	@echo "========================================"
	cd $(KVBM_DIR) && maturin build --profile dev --out $(DIST_DIR)

kvbm-install:
	@echo "========================================"
	@echo "Installing kvbm wheel (no deps)..."
	@echo "========================================"
	uv pip install --upgrade --force-reinstall --no-deps $(DIST_DIR)/kvbm*.whl

kvbm: kvbm-build kvbm-install
	@echo "✅ kvbm rebuild complete!"

# ============================================================================
# Combined Targets
# ============================================================================

# Rebuild both NIXL and kvbm (in order)
rebuild: nixl kvbm
	@echo "========================================"
	@echo "✅ Full rebuild complete!"
	@echo "   - NIXL rebuilt and installed"
	@echo "   - kvbm rebuilt and installed"
	@echo "========================================"

# Rebuild just kvbm (most common for code changes)
quick: kvbm
	@echo "✅ Quick kvbm rebuild complete!"

# Clean everything
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf $(DIST_DIR)/*.whl

# Show info about installed packages
info:
	@echo "========================================"
	@echo "NIXL Installation:"
	@echo "========================================"
	@ls -lh /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/libnixl.so 2>/dev/null || echo "Not found"
	@echo ""
	@echo "NIXL Plugins:"
	@ls -lh /opt/nvidia/nvda_nixl/lib/x86_64-linux-gnu/plugins/ | head -10
	@echo ""
	@echo "========================================"
	@echo "kvbm Python Package:"
	@echo "========================================"
	@pip show kvbm | grep -E "Name:|Version:|Location:"
	@echo ""
	@echo "Bundled NIXL in kvbm:"
	@ls -lh /opt/dynamo/venv/lib/python3.12/site-packages/kvbm.libs/libnixl*.so 2>/dev/null | head -3 || echo "Not found"

# ============================================================================
# Process Management
# ============================================================================

# Shutdown vLLM processes
shutdown:
	@echo "========================================"
	@echo "Shutting down vLLM processes..."
	@echo "========================================"
	@PIDS=$$(pgrep -f "python.*vllm.*serve" | grep -v $$$$); \
	if [ -n "$$PIDS" ]; then \
		echo "Found vLLM processes: $$PIDS"; \
		echo "Sending SIGTERM..."; \
		echo "$$PIDS" | xargs -r kill -TERM 2>/dev/null || true; \
		sleep 2; \
		REMAINING=$$(echo "$$PIDS" | xargs -r ps -p 2>/dev/null | grep -v PID | awk '{print $$1}' || true); \
		if [ -n "$$REMAINING" ]; then \
			echo "Processes still running: $$REMAINING"; \
			echo "Sending SIGKILL..."; \
			echo "$$REMAINING" | xargs -r kill -KILL 2>/dev/null || true; \
			sleep 1; \
		fi; \
		echo "✅ vLLM processes terminated"; \
	else \
		echo "No vLLM processes found"; \
	fi
	@echo "========================================"

# Help message
help:
	@echo "NIXL + kvbm Build Targets:"
	@echo ""
	@echo "  make             - Rebuild both NIXL and kvbm"
	@echo "  make quick       - Rebuild kvbm only (for Rust code changes)"
	@echo "  make nixl        - Rebuild and install NIXL only"
	@echo "  make kvbm        - Rebuild and install kvbm only"
	@echo "  make rebuild     - Same as 'make'"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make info        - Show installed package info"
	@echo "  make shutdown    - Terminate all vLLM processes"
	@echo "  make help        - Show this help"
	@echo ""
	@echo "Individual steps:"
	@echo "  make nixl-build            - Build NIXL"
	@echo "  make nixl-install          - Install NIXL"
	@echo "  make kvbm-build            - Build kvbm wheel"
	@echo "  make kvbm-install          - Install kvbm (no deps)"

