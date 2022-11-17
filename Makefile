CXXFLAGS=-std=c++14 -Wall -Wextra -pedantic -I /opt/homebrew/include -L /opt/homebrew/lib -lcasa_casa -lcasa_meas -lcasa_measures
CXX=g++

all: build/main.o
	${CXX} ${CXXFLAGS} -o main-app build/main.o 

build/main.o: main.cpp
	${CXX} ${CXXFLAGS} -c main.cpp -o $@

