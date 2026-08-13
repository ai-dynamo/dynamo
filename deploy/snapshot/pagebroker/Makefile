PROTO := v1/pagebroker.proto
GTEST_FLAGS = $(shell pkg-config --cflags --libs gtest_main)
BROKER_SOURCES := broker.cpp checkpoint_transaction_descriptor.cpp posix_copy_engine.cpp restore_transaction_descriptor.cpp transaction.cpp transfer_engine.cpp
DAEMON_SOURCES := $(BROKER_SOURCES) connection_tracker.cpp daemon.cpp file_descriptor.cpp

.PHONY: daemon generate test

generate:
	protoc --proto_path=. --cpp_out=. $(PROTO)

daemon: generate
	c++ -I. -std=c++20 -Wall -Werror $(DAEMON_SOURCES) v1/pagebroker.pb.cc -lprotobuf -o pagebroker

test: generate $(BROKER_SOURCES) daemon_test.cpp
	mkdir -p build
	c++ -I. -std=c++20 -Wall -Werror $(BROKER_SOURCES) daemon_test.cpp v1/pagebroker.pb.cc -lprotobuf $(GTEST_FLAGS) -o build/pagebroker-test
	./build/pagebroker-test
