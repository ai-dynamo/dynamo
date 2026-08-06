CXX ?= g++
CXXFLAGS ?= -O2 -Wall -Wextra -Werror -std=c++23

.PHONY: all test clean
all: pagebroker

pagebroker: pagebroker.cpp pagebroker.hpp pagebroker.proto
	$(CXX) $(CXXFLAGS) -pthread -o $@ pagebroker.cpp

pagebroker-test: pagebroker.cpp pagebroker.hpp pagebroker_test.cpp
	$(CXX) $(CXXFLAGS) -pthread -DPAGEBROKER_TEST -o $@ pagebroker.cpp pagebroker_test.cpp

test: pagebroker-test
	./pagebroker-test

clean:
	rm -f pagebroker pagebroker-test
