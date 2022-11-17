#include <casacore/casa/Arrays.h>
// #incu
#include <iostream>

int main() {
  int SIZE = 10;

  std::cout << std::endl << "Vector" << std::endl;
  casacore::Vector<casacore::Float> vector(SIZE);
  casacore::Vector<casacore::Float> weight(SIZE);
  float range = 10.0;
  float offset = -4.0;
  for (int i = 0; i < SIZE; ++i) {
    vector(i) = offset + range * (rand() / (float)RAND_MAX);
    weight(i) = 1;
  }

  casacore::IPosition minPos, maxPos;
  float min, max;

  casacore::minMax(min, max, vector);

  std::cout << "Min: " << min << "Max: " << max << std::endl;

  casacore::minMaxMasked(min, max, minPos, maxPos, vector, weight);

  std::cout << "Min: " << min << "Max: " << max << std::endl;

  for (int i = 0; i < SIZE; ++i) {
    std::cout << vector(i) << " ";
  }
  std::cout << std::endl;

  return 0;

}