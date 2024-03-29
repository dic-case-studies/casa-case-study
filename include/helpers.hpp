#ifndef include_helpers_hpp
#define include_helpers_hpp

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>
#include <cassert>

class WallClock {
  std::chrono::high_resolution_clock::time_point begin;

public:
  WallClock() { begin = std::chrono::high_resolution_clock::now(); }
  void tick() { begin = std::chrono::high_resolution_clock::now(); }
  void reset() { begin = std::chrono::high_resolution_clock::now(); }
  double elapsedTime() {
    std::chrono::high_resolution_clock::time_point end =
        std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
        .count();
  }
};

void assert_int(size_t expected, size_t actual, std::string str) {
  if (expected != actual) {
    std::cerr << str << " expected: " << expected << " actual: " << actual
              << std::endl;
    assert(expected == actual);
  }
}

void assert_float(float expected, float actual, std::string str) {
  float diff = fabs(expected - actual);
  float loss = (diff / expected) * 100;
  if (loss > 0.1) {
    std::cerr << str << " expected: " << expected << " actual: " << actual
              << " loss: " << loss << std::endl;
    assert(loss < 0.1);
  }
}

#endif /* include_helpers_hpp */
