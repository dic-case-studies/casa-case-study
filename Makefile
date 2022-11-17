CXXFLAGS=-std=c++14 -Wall -Wextra -pedantic -I include -I /opt/homebrew/include -L /opt/homebrew/lib -lcasa_casa -lcasa_meas -lcasa_measures -L /opt/homebrew/Cellar/libomp/15.0.4/lib -Xpreprocessor -fopenmp -lomp -fsanitize=address -g

CXX=g++

all: build/main.o build/masks.o
	${CXX} ${CXXFLAGS} -o main-app build/main.o build/masks.o

build/main.o: main.cpp include/masks.hpp include/helpers.hpp
	${CXX} ${CXXFLAGS} -c main.cpp -o $@

build/masks.o: src/masks.cpp
	${CXX} ${CXXFLAGS} -c src/masks.cpp -o $@

clean:
	rm -rf build/* *app

