#include "helpers.hpp"
#include <algorithm>
#include <assert.h>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cstddef>
#include <ctime>
#include <float.h>
#include <iostream>
#include <limits.h>
#include <math.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#include "sse2neon.h"
#endif

#ifdef __APPLE__
#include <simd/simd.h>
#endif

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

#ifdef SSE
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
#endif

#ifdef AVX
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
#endif

#ifdef __ARM_NEON__
void simd_neon(float *arr, size_t N, float &min, float &max) {
  assert(N < (size_t)INT_MAX);

  const int simd_width = 4;
  float32x4_t arr_r = vld1q_f32(arr);
  float32x4_t max_r = arr_r;
  float32x4_t min_r = arr_r;

  size_t quot = N / simd_width;
  size_t limit = quot * simd_width;

  for (size_t i = simd_width; i < limit; i += simd_width) {
    arr_r = vld1q_f32(arr + i);

    min_r = vminq_f32(min_r, arr_r);

    max_r = vmaxq_f32(max_r, arr_r);
  }

  max = vmaxvq_f32(max_r);
  min = vminvq_f32(min_r);

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
#endif

#ifdef __APPLE__
void simd_apple(float *arr, size_t N, float &min, float &max) {
  simd::float4 min_v = simd::make_float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
  simd::float4 max_v = simd::make_float4(FLT_MIN, FLT_MIN, FLT_MIN, FLT_MIN);

  for (size_t i = 0; i < N; i += 4) {
    simd::float4 v =
        simd::make_float4(arr[i], arr[i + 1], arr[i + 2], arr[i + 3]);
    min_v = simd_min(v, min_v);
    max_v = simd_max(v, max_v);
  }
  min = simd_reduce_min(min_v);
  max = simd_reduce_max(max_v);
}
#endif

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

  float range = 100.0;
  float offset = -50.0;
  for (size_t i = 0; i < N; i++) {
    arr[i] = offset + range * (rand() / (float)RAND_MAX);
  }

  std::cout << "Matrix dim: " << x << std::endl;

#ifdef GOLDEN
  float minExpected = FLT_MAX, maxExpected = FLT_MIN;
  {
    WallClock t;

    golden(arr, N, minExpected, maxExpected);

    std::cout << "Elapsed time golden: " << t.elapsedTime() << " us"
              << std::endl;
  }

  std::cout << "min: " << minExpected << " max " << maxExpected << std::endl;
#endif

#ifdef SSE
  {
    float minActual = FLT_MAX, maxActual = FLT_MIN;
    WallClock t;

    simd_sse(arr, N, minActual, maxActual);

    std::cout << "Elapsed time SIMD SSE: " << t.elapsedTime() << " us"
              << std::endl;
#ifdef ASSERT
    assert_float(maxExpected, maxActual, "maxSSE");
    assert_float(minExpected, minActual, "minSSE");
#endif
  }
#endif

#ifdef AVX
  {
    float minActual = FLT_MAX, maxActual = FLT_MIN;
    WallClock t;

    simd_avx(arr, N, minActual, maxActual);

    std::cout << "Elapsed time SIMD AVX: " << t.elapsedTime() << " us"
              << std::endl;
#ifdef ASSERT
    assert_float(maxExpected, maxActual, "maxAVX");
    assert_float(minExpected, minActual, "minAVX");
#endif
  }
#endif

#ifdef __ARM_NEON__
  {
    float minActual = FLT_MAX, maxActual = FLT_MIN;
    WallClock t;

    simd_neon(arr, N, minActual, maxActual);

    std::cout << "Elapsed time SIMD NEON: " << t.elapsedTime() << " us"
              << std::endl;

#ifdef ASSERT
    assert_float(maxExpected, maxActual, "maxNEON");
    assert_float(minExpected, minActual, "minNEON");
#endif
  }
#endif

#ifdef __APPLE__
  {
    float minActual = FLT_MAX, maxActual = FLT_MIN;
    WallClock t;

    simd_apple(arr, N, minActual, maxActual);

    std::cout << "Elapsed time SIMD Apple: " << t.elapsedTime() << " us"
              << std::endl;

#ifdef ASSERT
    assert_float(maxExpected, maxActual, "maxApple");
    assert_float(minExpected, minActual, "minApple");
#endif
  }
#endif

  delete[] arr;
}
