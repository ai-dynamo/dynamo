# Build the SYCL vectorized_copy FFI shared library.
#
# Prerequisites: Intel oneAPI (icpx -fsycl)
#
# Usage:
#   make                     # Build libvectorized_copy_sycl.so
#   make clean               # Remove build artifacts
#   make install             # Copy .so to $DESTDIR (default: ..)
#   make test                # Build & run the self-test binary

CXX      := icpx
CXXFLAGS := -fsycl -O2 -Wall -Wextra -fPIC
LDFLAGS  := -fsycl -shared

LIB      := libvectorized_copy_sycl.so
FFI_SRC  := vectorized_copy_ffi.cpp
TEST_SRC := vectorized_copy.cpp
TEST_BIN := vectorized_copy_test
DESTDIR  ?= ..

.PHONY: all clean install test

all: $(LIB)

$(LIB): $(FFI_SRC) vectorized_copy_ffi.h
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $(FFI_SRC)

test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC)

install: $(LIB)
	install -m 755 $(LIB) $(DESTDIR)/$(LIB)

clean:
	rm -f $(LIB) $(TEST_BIN) *.o
