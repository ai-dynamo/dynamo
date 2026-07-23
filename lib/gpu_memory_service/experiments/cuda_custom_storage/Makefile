# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

CUDA_HOME ?= /usr/local/cuda
CXX ?= g++
CXXFLAGS ?= -O2 -g

TARGET := gms-custom-storage-roundtrip

.PHONY: all clean

all: $(TARGET)

$(TARGET): gms_roundtrip.cpp cuda_checkpoint_compat.h
	$(CXX) -std=c++20 $(CXXFLAGS) -Wall -Wextra -Werror \
		-I$(CUDA_HOME)/include \
		-L$(CUDA_HOME)/lib64/stubs \
		-Wl,-rpath,$(CUDA_HOME)/lib64 \
		-o $@ gms_roundtrip.cpp -lcuda

clean:
	$(RM) $(TARGET)
