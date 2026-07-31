# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

CUDA_HOME ?= /usr/local/cuda
CXX ?= g++
CXXFLAGS ?= -O2 -g

TARGET := cuda-custom-storage-roundtrip
SHARED_DIR := ../../deploy/snapshot/cmd/cuda-checkpoint-helper
SOURCES := roundtrip.cpp $(SHARED_DIR)/custom_storage_operation.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(SHARED_DIR)/custom_storage_operation.h $(SHARED_DIR)/cuda_checkpoint_compat.h
	$(CXX) -std=c++20 $(CXXFLAGS) -Wall -Wextra -Werror \
		-I$(CUDA_HOME)/include \
		-I$(SHARED_DIR) \
		-L$(CUDA_HOME)/lib64/stubs \
		-Wl,-rpath,$(CUDA_HOME)/lib64 \
		-o $@ $(SOURCES) -lcuda

clean:
	$(RM) $(TARGET)
