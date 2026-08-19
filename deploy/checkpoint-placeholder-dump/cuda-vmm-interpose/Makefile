# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

CC ?= cc
CUDA_HOME ?= /usr/local/cuda
BUILD_DIR ?= build

INTERPOSER := $(BUILD_DIR)/libsnapshot_cuda_vmm.so
COORDINATOR := $(BUILD_DIR)/snapshot-cuda-vmm
INTERPOSER_SOURCES := interpose.c multicast.c posix.c util.c
INTERPOSER_HEADERS := multicast.h posix.h protocol.h util.h
COORDINATOR_SOURCES := coordinator.c util.c
COORDINATOR_HEADERS := protocol.h util.h
COMMON_FLAGS := -std=gnu11 -O2 -Wall -Wextra -Werror

.PHONY: all clean

all: $(INTERPOSER) $(COORDINATOR)

$(BUILD_DIR):
	mkdir -p $@

$(INTERPOSER): $(INTERPOSER_SOURCES) $(INTERPOSER_HEADERS) | $(BUILD_DIR)
	$(CC) $(COMMON_FLAGS) -fPIC -shared -Wno-deprecated-declarations \
		-I$(CUDA_HOME)/include -o $@.tmp $(INTERPOSER_SOURCES) \
		-Wl,-Bsymbolic-functions -ldl -pthread
	readelf -d $@.tmp >/dev/null
	! readelf -d $@.tmp | grep -Eq 'NEEDED.*(libcuda|libcudart)'
	mv $@.tmp $@

$(COORDINATOR): $(COORDINATOR_SOURCES) $(COORDINATOR_HEADERS) | $(BUILD_DIR)
	$(CC) $(COMMON_FLAGS) -o $@ $(COORDINATOR_SOURCES)

clean:
	rm -rf $(BUILD_DIR)
