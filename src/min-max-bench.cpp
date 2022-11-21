/*
  g++ -o min-max min-max.cc -O3 -march=native
  ./min-max <N>
*/
#include "helpers.hpp"
#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <ctime>
#include <emmintrin.h>
#include <float.h>
#include <immintrin.h>
#include <iostream>
#include <limits.h>
#include <math.h>
#include <smmintrin.h>
#include <xmmintrin.h>

#include <chrono>

void golden(float *arr, size_t N, float &min, float &max) {
  min = FLT_MAX;
  max = FLT_MIN;
  for (size_t i = 0; i < N; i++) {
    if (min > arr[i]) {
      min = arr[i];
    }
    if (max < arr[i]) {
      max = arr[i];
    }
  }
}

void print_vector(const __m128i v) {
  uint32_t temp[4];
  _mm_storeu_si128((__m128i_u *)temp, v);
  for (int i = 0; i < 4; i++) {
    std::cout << temp[i] << " ";
  }
  std::cout << std::endl;
}

void print_vector(const __m128 v) {
  float temp[4];
  _mm_storeu_ps(temp, v);
  for (int i = 0; i < 4; i++) {
    std::cout << temp[i] << " ";
  }
  std::cout << std::endl;
}

void simd_sse(float *arr, size_t N, float &min, float &max) {

  assert(N < (size_t)INT_MAX);

  const int simd_width = 4;
  __m128 arr_r = _mm_loadu_ps(arr);
  __m128 max_r = arr_r;
  __m128 min_r = arr_r;

  size_t quot = N / simd_width;
  size_t limit = quot * simd_width;

  for (size_t i = simd_width; i < limit; i += simd_width) {
    arr_r = _mm_loadu_ps(arr + i);

    min_r = _mm_min_ps(min_r, arr_r);
    max_r = _mm_max_ps(max_r, arr_r);
  }

  float max_tmp[simd_width];
  float min_tmp[simd_width];

  _mm_storeu_ps(min_tmp, min_r);
  _mm_storeu_ps(max_tmp, max_r);

  max = max_tmp[0];
  min = min_tmp[0];

  for (int i = 1; i < simd_width; i++) {
    if (max_tmp[i] > max) {
      max = max_tmp[i];
    }
    if (min_tmp[i] < min) {
      min = min_tmp[i];
    }
  }

  // Min max for reminder
  for (size_t i = limit; i < N; i++) {
    if (max < arr[i]) {
      max = arr[i];
    }
    if (min > arr[i]) {
      min = arr[i];
    }
  }
}

void simd_avx(float *arr, size_t N, float &min, float &max) {

  assert(N < (size_t)INT_MAX);

  const int simd_width = 8;
  __m256 arr_r = _mm256_loadu_ps(arr);
  __m256 max_r = arr_r;
  __m256 min_r = arr_r;

  size_t quot = N / simd_width;
  size_t limit = quot * simd_width;

  for (size_t i = simd_width; i < limit; i += simd_width) {
    arr_r = _mm256_loadu_ps(arr + i);

    min_r = _mm256_min_ps(min_r, arr_r);

    max_r = _mm256_max_ps(max_r, arr_r);
  }

  // print_vector(max_r);
  // print_vector(max_pos_r);

  // Unload from vector register and reduce
  float max_tmp[simd_width];
  float min_tmp[simd_width];

  _mm256_storeu_ps(min_tmp, min_r);

  _mm256_storeu_ps(max_tmp, max_r);

  max = max_tmp[0];
  min = min_tmp[0];

  for (int i = 1; i < simd_width; i++) {
    if (max_tmp[i] > max) {
      max = max_tmp[i];
    }
    if (min_tmp[i] < min) {
      min = min_tmp[i];
    }
  }

  // Min max for reminder
  for (size_t i = limit; i < N; i++) {
    if (max < arr[i]) {
      max = arr[i];
    }
    if (min > arr[i]) {
      min = arr[i];
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << " usage: ./min-max <N>" << std::endl;
    return 1;
  }
  size_t x = atoi(argv[1]);
  size_t N = x * x;
  float *arr = new float[N];
  time_t seed = time(0);
  // time_t seed = 1668773523;
  std::cout << "Seed: " << seed << std::endl;
  srand(seed);

  float offset = 5.0f;
  float range = 1000.0f;
  for (size_t i = 0; i < N; i++) {
    arr[i] = offset + range * (rand() / (float)RAND_MAX);
  }

  std::cout << "Matrix dim: " << x << std::endl;

  float minExpected = FLT_MAX, maxExpected = FLT_MIN;
  {
    WallClock t;

    golden(arr, N, minExpected, maxExpected);

    std::cout << "Elapsed time golden: " << t.elapsedTime() << " us"
              << std::endl;
  }

  std::cout << "min: " << minExpected << " max " << maxExpected << std::endl;

  float minActual = FLT_MAX, maxActual = FLT_MIN;
  {
    WallClock t;

    simd_sse(arr, N, minActual, maxActual);

    std::cout << "Elapsed time SIMD SSE: " << t.elapsedTime() << " us"
              << std::endl;
  }

  assert_float(maxExpected, maxActual, "max");
  assert_float(minExpected, minActual, "min");

  minActual = 0.0f, maxActual = 0.0f;
  {
    WallClock t;

    simd_avx(arr, N, minActual, maxActual);

    std::cout << "Elapsed time SIMD AVX: " << t.elapsedTime() << " us"
              << std::endl;
  }

  assert_float(maxExpected, maxActual, "max");
  assert_float(minExpected, minActual, "min");

  delete[] arr;
}
