# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

CUDA_HOME ?= /usr/local/cuda
CXX ?= g++
CXXFLAGS ?= -O2 -g

BENCHMARK_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
TARGET := $(BENCHMARK_DIR)/cuda-custom-storage-roundtrip
SOURCES := $(addprefix $(BENCHMARK_DIR)/, \
	custom_storage_operation.cpp \
	roundtrip.cpp \
	roundtrip_common.cpp \
	storage.cpp \
	workload.cpp)
HEADERS := $(wildcard $(BENCHMARK_DIR)/*.hpp)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) -std=c++20 $(CXXFLAGS) -Wall -Wextra -Werror \
		-I$(CUDA_HOME)/include \
		-I$(BENCHMARK_DIR) \
		-L$(CUDA_HOME)/lib64/stubs \
		-Wl,-rpath,$(CUDA_HOME)/lib64 \
		-o $@ $(SOURCES) -lcuda

clean:
	$(RM) $(TARGET)
