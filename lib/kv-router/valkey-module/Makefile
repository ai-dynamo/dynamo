# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

VALKEY_SRC ?=
VALKEY_SERVER ?= $(VALKEY_SRC)/valkey-server
PYTHON ?= python3
SANITIZE ?=
TARGET := dynkv.so
SOURCES := $(sort $(wildcard *.c))
OBJECTS := $(SOURCES:.c=.o)
HEADERS := $(sort $(wildcard dynkv_*.h))

ifeq ($(strip $(VALKEY_SRC)),)
ifneq ($(MAKECMDGOALS),clean)
$(error VALKEY_SRC must point to a Valkey source tree containing valkeymodule.h)
endif
endif

CPPFLAGS += -I$(VALKEY_SRC)
CFLAGS += -std=gnu11 -fPIC -fno-common -Wall -Wextra -Werror
LDFLAGS += -shared -Wl,--version-script=dynkv.map

ifeq ($(SANITIZE),)
CFLAGS += -O3 -flto
LDFLAGS += -flto
else ifeq ($(SANITIZE),undefined)
CFLAGS += -O1 -g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined
LDFLAGS += -fsanitize=undefined -fno-sanitize-recover=undefined
else
$(error unsupported SANITIZE=$(SANITIZE); expected empty or undefined)
endif

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJECTS) dynkv.map
	$(CC) $(LDFLAGS) -o $@ $(OBJECTS)

%.o: %.c $(HEADERS) $(VALKEY_SRC)/valkeymodule.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# GNU make decides whether shared prerequisites are current before executing
# an earlier `clean` goal. Force both build stages for the documented
# `make clean all test` validation form.
ifneq ($(filter clean,$(MAKECMDGOALS)),)
.PHONY: force-rebuild-after-clean
$(OBJECTS) $(TARGET): force-rebuild-after-clean
force-rebuild-after-clean:
endif

clean:
	rm -f $(OBJECTS) $(TARGET)

test: $(TARGET)
	$(PYTHON) tests/test_dynkv_module.py --server $(VALKEY_SERVER) --module $(abspath $(TARGET))
