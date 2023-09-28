#include "ArrayMathOpt.hpp"
#include "helpers.hpp"
#include <assert.h>
#include <casacore/casa/Arrays.h>
#include <cfloat>
#include <ctime>
#include <iostream>
#include <ostream>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << " usage: ./casa-bench <N>" << std::endl;
    return 1;
  }
  size_t SIZE = atoi(argv[1]);
  std::cout << "Matrix dim: " << SIZE << std::endl;

  casacore::Matrix<casacore::Float> matrix(SIZE, SIZE);
  casacore::Matrix<casacore::Float> weight(SIZE, SIZE);
  auto seed = time(0);
  std::cout << "seed = " << seed << std::endl;
  srand(seed);

  float range = 100.0;
  float offset = -50.0;
  for (auto i = 0; i < SIZE; i++) {
    for (auto j = 0; j < SIZE; j++) {
      matrix(i, j) = offset + range * (rand() / (float)RAND_MAX);
      weight(i, j) = (rand() / (float)RAND_MAX);
    }
  }

  casacore::IPosition expectedMinPos, expectedMaxPos;
  float expectedMin, expectedMax;
  {

    WallClock t;
    casacore::minMaxMasked(expectedMin, expectedMax, expectedMinPos,
                           expectedMaxPos, matrix, weight);

    auto duration = t.elapsedTime();

    std::cout << "Time taken for casacore minMaxMasked: " << duration << " us"
              << std::endl
              << std::endl;
  }
  /// Running openmp version of minMaxMasked ///

  {
    casacore::IPosition actualMinPos, actualMaxPos;
    float actualMin, actualMax;

    WallClock t;
    minMaxMaskedParallel(actualMin, actualMax, actualMinPos, actualMaxPos,
                         matrix, weight);

    auto duration = t.elapsedTime();
    std::cout << "Time taken for openmp minMaxMasked: " << duration << " us"
              << std::endl;

    ///Asserting on openmp results ///

    assert_float(expectedMin, actualMin, "openmp min");
    assert_float(expectedMax, actualMax, "openmp max");
    assert(expectedMinPos.isEqual(actualMinPos));
    assert(expectedMaxPos.isEqual(actualMaxPos));

    std::cout << "Assertion passed for openmp" << std::endl;
  }

  {
    casacore::IPosition actualMinPos, actualMaxPos;
    float actualMin, actualMax;
    /// Running SIMD version of minMaxMasked ///
    actualMin = FLT_MAX;
    actualMax = FLT_MIN;

    WallClock t;

    minMaxMaskedSIMD(actualMin, actualMax, actualMinPos, actualMaxPos, matrix,
                     weight);

    auto duration = t.elapsedTime();
    std::cout << "Time taken for SIMD minMaxMasked: " << duration << " us"
              << std::endl;

    ///Asserting on SIMD results ///

    assert_float(expectedMin, actualMin, "SIMD min");
    assert_float(expectedMax, actualMax, "SIMD max");

    assert(expectedMinPos.isEqual(actualMinPos));
    assert(expectedMaxPos.isEqual(actualMaxPos));

    std::cout << "Assertion passed for SIMD" << std::endl;
  }
  std::cout << std::endl;
  std::cout << "-----------------------------" << std::endl;
  std::cout << std::endl;

  return 0;
}