UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	OMPFLAGS += -fopenmp
else ifeq ($(UNAME_S),Darwin)
	OMPFLAGS += -Xpreprocessor -fopenmp -lomp
endif

CXXFLAGS=-std=c++14 -Wall -Wextra -pedantic -I include -O3

processor := $(shell uname -m)
ifeq ($(processor),$(filter $(processor),aarch64 arm64))
    ARCH_C_FLAGS += -march=armv8-a+fp+simd+crc 
	CUSTOM_CODE_FLAGS += -Darm64 -DGOLDEN -DSSE -DNEON -DASSERT 
else ifeq ($(processor),$(filter $(processor),i386 x86_64))
    ARCH_C_FLAGS += -march=native 
	CUSTOM_CODE_FLAGS += -Damd64 -DGOLDEN -DSSE -DAVX -DASSERT 
endif


DEBUGFLAGS=-fsanitize=address -g

LIBS= -lcasa_casa -lcasa_meas -lcasa_measures

CXX=g++

all: dir build/casa-bench build/min-max-bench build/min-max-pos-bench build/min-max-masked-bench 

build/%: src/%.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) $(ARCH_C_FLAGS) $(CUSTOM_CODE_FLAGS) $(OMPFLAGS) $(LIBS) $(OPT)

clean:
	rm -rf build/* *app

dir:
	mkdir -p build

.PHONY: all clean

