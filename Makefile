UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	OMPFLAGS += -fopenmp
else ifeq ($(UNAME_S),Darwin)
	OMPFLAGS += -Xpreprocessor -fopenmp -lomp
endif

processor := $(shell uname -m)
ifeq ($(processor),$(filter $(processor),aarch64 arm64))
    ARCH_CFLAGS += -march=armv8-a+fp+simd+crc
	ifeq ($(UNAME_S),Darwin)
		EXTRA_FLAGS += -L /opt/homebrew/Cellar/libomp/15.0.4/lib 
	endif
else ifeq ($(processor),$(filter $(processor),i386 x86_64))
    ARCH_CFLAGS=-march=native
	EXTRA_FLAGS=''
endif

CXXFLAGS=-std=c++14 -Wall -Wextra -pedantic -I include -O3
DEBUGFLAGS=-fsanitize=address -g

LIBS= -lcasa_casa -lcasa_meas -lcasa_measures

CXX=g++

all: build build/casa-bench build/min-max-bench build/min-max-pos-bench build/min-max-masked-bench 

build/main: src/main.cpp include/ArrayMathOpt.hpp include/helpers.hpp

build/%: src/%.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) $(OMPFLAGS) $(LIBS) $(EXTRA_FLAGS) $(ARCH_CFLAGS)

clean:
	rm -rf build/* *app

dir:
	mkdir -p build

.PHONY: all clean

