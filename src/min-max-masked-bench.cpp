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

// #include <emmintrin.h>
// #include <immintrin.h>
// #include <smmintrin.h>
// #include <xmmintrin.h>

#include "sse2neon.h"

void golden(float *arr, float *weight, size_t N, float &min, size_t &minPos,
            float &max, size_t &maxPos) {
  min = FLT_MAX;
  max = FLT_MIN;
  for (size_t i = 0; i < N; i++) {
    float val = arr[i] * weight[i];
    if (min > val) {
      min = val;
      minPos = i;
    }
    if (max < val) {
      max = val;
      maxPos = i;
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

void simd_sse(float *arr, float *weight, size_t N, float &min, size_t &minPos,
              float &max, size_t &maxPos) {

  assert(N < (size_t)INT_MAX);

  const int simd_width = 4;
  __m128 arr_r = _mm_loadu_ps(arr);
  __m128 weight_r = _mm_loadu_ps(weight);
  arr_r = _mm_mul_ps(arr_r, weight_r);
  __m128 max_r = arr_r;
  __m128 min_r = arr_r;

  __m128i idx_r = _mm_setr_epi32(0, 1, 2, 3);

  __m128i min_pos_r = idx_r;
  __m128i max_pos_r = idx_r;

  __m128i inc = _mm_set1_epi32(simd_width);

  size_t quot = N / simd_width;
  size_t limit = quot * simd_width;

  for (size_t i = simd_width; i < limit; i += simd_width) {
    idx_r = _mm_add_epi32(idx_r, inc);
    arr_r = _mm_loadu_ps(arr + i);
    weight_r = _mm_loadu_ps(weight + i);
    arr_r = _mm_mul_ps(arr_r, weight_r);

    __m128i min_mask = _mm_castps_si128(_mm_cmplt_ps(arr_r, min_r));
    __m128i max_mask = _mm_castps_si128(_mm_cmpgt_ps(arr_r, max_r));

    min_r = _mm_min_ps(min_r, arr_r);
    min_pos_r = _mm_blendv_epi8(min_pos_r, idx_r, min_mask);

    max_r = _mm_max_ps(max_r, arr_r);
    max_pos_r = _mm_blendv_epi8(max_pos_r, idx_r, max_mask);
  }

  // Unload from vector register and reduce
  float max_tmp[simd_width];
  float min_tmp[simd_width];

  uint32_t max_pos_temp[simd_width];
  uint32_t min_pos_temp[simd_width];

  _mm_storeu_ps(min_tmp, min_r);
  _mm_storeu_si128((__m128i_u *)min_pos_temp, min_pos_r);

  _mm_storeu_ps(max_tmp, max_r);
  _mm_storeu_si128((__m128i_u *)max_pos_temp, max_pos_r);

  max = max_tmp[0];
  maxPos = max_pos_temp[0];
  min = min_tmp[0];
  minPos = min_pos_temp[0];

  for (int i = 1; i < simd_width; i++) {
    if (max_tmp[i] > max) {
      max = max_tmp[i];
      maxPos = max_pos_temp[i];
    } else if (max_tmp[i] == max) {
      maxPos = std::min(maxPos, size_t(max_pos_temp[i]));
    }
    if (min_tmp[i] < min) {
      min = min_tmp[i];
      minPos = min_pos_temp[i];
    } else if (min_tmp[i] == min) {
      minPos = std::min(minPos, size_t(min_pos_temp[i]));
    }
  }

  // Min max for reminder
  for (size_t i = limit; i < N; i++) {
    float val = arr[i] * weight[i];
    if (max < val) {
      max = val;
      maxPos = i;
    }
    if (min > val) {
      min = val;
      minPos = i;
    }
  }
}

void simd_avx(float *arr, float *weight, size_t N, float &min, size_t &minPos,
              float &max, size_t &maxPos) {

  assert(N < (size_t)INT_MAX);

  const int simd_width = 8;
  __m256 arr_r = _mm256_loadu_ps(arr);
  __m256 weight_r = _mm256_loadu_ps(weight);
  arr_r = _mm256_mul_ps(arr_r, weight_r);
  __m256 max_r = arr_r;
  __m256 min_r = arr_r;

  __m256i idx_r = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

  __m256i min_pos_r = idx_r;
  __m256i max_pos_r = idx_r;

  __m256i inc = _mm256_set1_epi32(simd_width);

  size_t quot = N / simd_width;
  size_t limit = quot * simd_width;

  for (size_t i = simd_width; i < limit; i += simd_width) {
    idx_r = _mm256_add_epi32(idx_r, inc);
    arr_r = _mm256_loadu_ps(arr + i);
    weight_r = _mm256_loadu_ps(weight + i);
    arr_r = _mm256_mul_ps(arr_r, weight_r);

    // _CMP_LT_OS
    __m256i min_mask =
        _mm256_castps_si256(_mm256_cmp_ps(arr_r, min_r, _CMP_LT_OQ));
    __m256i max_mask =
        _mm256_castps_si256(_mm256_cmp_ps(arr_r, max_r, _CMP_GT_OQ));

    min_r = _mm256_min_ps(min_r, arr_r);
    min_pos_r = _mm256_blendv_epi8(min_pos_r, idx_r, min_mask);

    max_r = _mm256_max_ps(max_r, arr_r);
    max_pos_r = _mm256_blendv_epi8(max_pos_r, idx_r, max_mask);
  }

  // print_vector(max_r);
  // print_vector(max_pos_r);

  // Unload from vector register and reduce
  float max_tmp[simd_width];
  float min_tmp[simd_width];

  uint32_t max_pos_temp[simd_width];
  uint32_t min_pos_temp[simd_width];

  _mm256_storeu_ps(min_tmp, min_r);
  _mm256_storeu_si256((__m256i_u *)min_pos_temp, min_pos_r);

  _mm256_storeu_ps(max_tmp, max_r);
  _mm256_storeu_si256((__m256i_u *)max_pos_temp, max_pos_r);

  max = max_tmp[0];
  maxPos = max_pos_temp[0];
  min = min_tmp[0];
  minPos = min_pos_temp[0];

  for (int i = 1; i < simd_width; i++) {
    if (max_tmp[i] > max) {
      max = max_tmp[i];
      maxPos = max_pos_temp[i];
    } else if (max_tmp[i] == max) {
      maxPos = std::min(maxPos, size_t(max_pos_temp[i]));
    }
    if (min_tmp[i] < min) {
      min = min_tmp[i];
      minPos = min_pos_temp[i];
    } else if (min_tmp[i] == min) {
      minPos = std::min(minPos, size_t(min_pos_temp[i]));
    }
  }

  // Min max for reminder
  for (size_t i = limit; i < N; i++) {
    float val = arr[i] * weight[i];
    if (max < val) {
      max = val;
      maxPos = i;
    }
    if (min > val) {
      min = val;
      minPos = i;
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
  float *weight = new float[N];
  time_t seed = time(0);
  // time_t seed = 1668773523;
  std::cout << "Seed: " << seed << std::endl;
  srand(seed);

  float offset = 5.0f;
  float range = 1000.0f;
  for (size_t i = 0; i < N; i++) {
    arr[i] = offset + range * (rand() / (float)RAND_MAX);
    weight[i] = (rand() / (float)RAND_MAX);
  }

  std::cout << "Matrix dim: " << x << std::endl;

  float minExpected = FLT_MAX, maxExpected = FLT_MIN;
  size_t minPosExpected = 0, maxPosExpected = 0;
  {
    WallClock t;

    golden(arr, weight, N, minExpected, minPosExpected, maxExpected,
           maxPosExpected);

    std::cout << "Elapsed time golden: " << t.elapsedTime() << " us"
              << std::endl;
  }

  std::cout << "min: " << minExpected << " max " << maxExpected << " minPos "
            << minPosExpected << " maxPos: " << maxPosExpected << std::endl;

  float minActual = FLT_MIN, maxActual = FLT_MIN;
  size_t minPosActual = 0, maxPosActual = 0;
  {
    WallClock t;

    simd_sse(arr, weight, N, minActual, minPosActual, maxActual, maxPosActual);

    std::cout << "Elapsed time SIMD SSE: " << t.elapsedTime() << " us"
              << std::endl;
  }

  assert_float(maxExpected, maxActual, "max");
  assert_float(minExpected, minActual, "min");
  assert_int(minPosExpected, minPosActual, "min_pos");
  assert_int(maxPosExpected, maxPosActual, "max_pos");

  minActual = FLT_MIN, maxActual = FLT_MIN;
  minPosActual = 0, maxPosActual = 0;
  {
    WallClock t;

    simd_avx(arr, weight, N, minActual, minPosActual, maxActual, maxPosActual);

    std::cout << "Elapsed time SIMD AVX: " << t.elapsedTime() << " us"
              << std::endl;
  }

  assert_float(maxExpected, maxActual, "max");
  assert_float(minExpected, minActual, "min");
  assert_int(minPosExpected, minPosActual, "min_pos");
  assert_int(maxPosExpected, maxPosActual, "max_pos");

  delete[] arr;
}
