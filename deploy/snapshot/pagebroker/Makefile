PROTO := v1/pagebroker.proto
GTEST_FLAGS := $(shell pkg-config --cflags --libs gtest_main)

.PHONY: generate test

generate:
	protoc --proto_path=. --cpp_out=. $(PROTO)

test: generate broker.cpp daemon_test.cpp
	mkdir -p build
	c++ -I. -std=c++20 -Wall -Werror broker.cpp daemon_test.cpp v1/pagebroker.pb.cc -lprotobuf $(GTEST_FLAGS) -o build/pagebroker-test
	./build/pagebroker-test
