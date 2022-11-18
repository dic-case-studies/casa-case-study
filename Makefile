UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
		OMPFLAGS += -fopenmp
endif
ifeq ($(UNAME_S),Darwin)
		OMPFLAGS += -Xpreprocessor -fopenmp -lomp
endif

CXXFLAGS=-std=c++14 -Wall -Wextra -pedantic -I include -march=native
DEBUGFLAGS=-fsanitize=address -g

LIBS= -lcasa_casa -lcasa_meas -lcasa_measures

CXX=g++

all: build/casa-bench build/min-max-bench

build/main: src/main.cpp include/ArrayMathOpt.hpp include/helpers.hpp

build/min-max-bench: src/min-max-bench.cpp

build/%: src/%.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) $(OMPFLAGS) $(OPT) $(LIBS)

clean:
	rm -rf build/* *app

.PHONY: all clean

