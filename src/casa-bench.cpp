#include <casacore/casa/Arrays.h>
#include <iostream>
#include "helpers.hpp"
#include "ArrayMathOpt.hpp"

Timer stop_watch;

int main() {
  int SIZE = 4096;

  casacore::Matrix<casacore::Float> matrix(SIZE,SIZE);
  casacore::Matrix<casacore::Float> weight(SIZE,SIZE);

  float range = 10.0;
  float offset = -4.0;
  for (auto i = 0; i < SIZE; i++) {
    for (auto j = 0; j < SIZE; j++) {
      matrix(i, j) = offset + range * (rand() / (float)RAND_MAX);
      weight(i, j) = 1;
    }
  }

  casacore::IPosition minPos, maxPos;
  float min, max;

  stop_watch.start_timer();
  casacore::minMaxMasked(min, max, minPos, maxPos, matrix, weight);
  stop_watch.stop_timer();

  auto duration = stop_watch.time_elapsed();

  std::cout << "Time taken for casacore minMaxMasked: " << duration << " ms" << std::endl << std::endl;

  stop_watch.start_timer();
  minMaxMaskedParallel(min, max, minPos, maxPos, matrix, weight);
  stop_watch.stop_timer();
  
  duration = stop_watch.time_elapsed();
  std::cout << "Time taken for openmp minMaxMasked: " << duration << " ms" << std::endl;

  stop_watch.start_timer();
  minMaxMaskedSIMD(min, max, minPos, maxPos, matrix, weight);
  stop_watch.stop_timer();

  duration = stop_watch.time_elapsed();
  std::cout << "Time taken for SIMD minMaxMasked: " << duration << " ms" << std::endl;

  return 0;

}