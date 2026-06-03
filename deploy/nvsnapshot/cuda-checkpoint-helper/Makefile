# TODO: port the full build (CUDA toolkit paths, cross-compile, container image)
# from dynamo's deploy/snapshot/Dockerfile in a follow-up PR. This stub exists
# so consumers see a documented build path next to the source.

CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
CUDA_HOME ?= /usr/local/cuda
LDFLAGS ?= -L$(CUDA_HOME)/lib64/stubs -lcuda

BIN := cuda-checkpoint-helper
SRC := src/main.c

build: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(BIN)

.PHONY: build clean
