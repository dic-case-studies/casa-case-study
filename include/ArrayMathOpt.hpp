#pragma once
#include <casacore/casa/Arrays.h>
#include <climits>

using casacore::Array;
using casacore::ArrayConformanceError;
using casacore::ArrayError;
using casacore::IPosition;

#ifdef amd64
#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <xmmintrin.h>
#endif

#ifdef arm64
#include "sse2neon.h"
#endif

template <typename T, typename Alloc>
void minMaxMaskedParallel(T &minVal, T &maxVal, IPosition &minPos,
                          IPosition &maxPos, const Array<T, Alloc> &array,
                          const Array<T, Alloc> &weight) {
  size_t n = array.nelements();
  if (n == 0) {
    throw(ArrayError("void minMax(T &min, T &max, IPosition &minPos,"
                     "IPosition &maxPos, const Array<T, Alloc> &array) - "
                     "const Array<T, Alloc> &weight) - "
                     "Array has no elements"));
  }
  if (!array.shape().isEqual(weight.shape())) {
    throw(ArrayConformanceError("void minMaxMasked(T &min, T &max,"
                                "IPosition &minPos, IPosition &maxPos, "
                                "const Array<T, Alloc> &array, "
                                "const Array<T, Alloc> &weight) - array "
                                "and weight do not have the same shape()"));
  }
  size_t minp = 0;
  size_t maxp = 0;
  T minv = array.data()[0] * weight.data()[0];
  T maxv = minv;
  if (array.contiguousStorage() && weight.contiguousStorage()) {
    const float *arrRaw = array.data();
    const float *weightRaw = weight.data();

#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      T tmp = arrRaw[i] * weightRaw[i];
#pragma omp critical
      {
        if (tmp < minv) {
          minv = tmp;
          minp = i;
        }
      }
#pragma omp critical
      {
        if (tmp > maxv) {
          maxv = tmp;
          maxp = i;
        }
      }
    }
  } else {
    typename Array<T, Alloc>::const_iterator iter = array.begin();
    typename Array<T, Alloc>::const_iterator witer = weight.begin();
    for (size_t i = 0; i < n; ++i, ++iter, ++witer) {
      T tmp = *iter * *witer;
      if (tmp < minv) {
        minv = tmp;
        minp = i;
      } else if (tmp > maxv) {
        maxv = tmp;
        maxp = i;
      }
    }
  }
  minPos.resize(array.ndim());
  maxPos.resize(array.ndim());
  minPos = toIPositionInArray(minp, array.shape());
  maxPos = toIPositionInArray(maxp, array.shape());
  minVal = minv;
  maxVal = maxv;
}

#ifdef __AVX__
inline void minMaxAVX(const float *arr, const float *weight, size_t N,
                      float &min, size_t &minPos, float &max, size_t &maxPos) {

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
#endif

template <typename T, typename Alloc>
void minMaxMaskedSIMD(T &minVal, T &maxVal, IPosition &minPos,
                      IPosition &maxPos, const Array<T, Alloc> &array,
                      const Array<T, Alloc> &weight) {
  size_t n = array.nelements();
  if (n == 0) {
    throw(ArrayError("void minMax(T &min, T &max, IPosition &minPos,"
                     "IPosition &maxPos, const Array<T, Alloc> &array) - "
                     "const Array<T, Alloc> &weight) - "
                     "Array has no elements"));
  }
  if (!array.shape().isEqual(weight.shape())) {
    throw(ArrayConformanceError("void minMaxMasked(T &min, T &max,"
                                "IPosition &minPos, IPosition &maxPos, "
                                "const Array<T, Alloc> &array, "
                                "const Array<T, Alloc> &weight) - array "
                                "and weight do not have the same shape()"));
  }
  size_t minp = 0;
  size_t maxp = 0;
  T minv = array.data()[0] * weight.data()[0];
  T maxv = minv;
  if (array.contiguousStorage() && weight.contiguousStorage()) {
    typename Array<T, Alloc>::const_contiter iter = array.cbegin();
    typename Array<T, Alloc>::const_contiter witer = weight.cbegin();
    for (size_t i = 0; i < n; ++i, ++iter, ++witer) {
      T tmp = *iter * *witer;
      if (tmp < minv) {
        minv = tmp;
        minp = i;
      } else if (tmp > maxv) {
        maxv = tmp;
        maxp = i;
      }
    }
  } else {
    typename Array<T, Alloc>::const_iterator iter = array.begin();
    typename Array<T, Alloc>::const_iterator witer = weight.begin();
    for (size_t i = 0; i < n; ++i, ++iter, ++witer) {
      T tmp = *iter * *witer;
      if (tmp < minv) {
        minv = tmp;
        minp = i;
      } else if (tmp > maxv) {
        maxv = tmp;
        maxp = i;
      }
    }
  }
  minPos.resize(array.ndim());
  maxPos.resize(array.ndim());
  minPos = toIPositionInArray(minp, array.shape());
  maxPos = toIPositionInArray(maxp, array.shape());
  minVal = minv;
  maxVal = maxv;
}

template <typename Alloc>
void minMaxMaskedSIMD(float &minVal, float &maxVal, IPosition &minPos,
                      IPosition &maxPos, const Array<float, Alloc> &array,
                      const Array<float, Alloc> &weight) {
  size_t n = array.nelements();
  if (n == 0) {
    throw(ArrayError("void minMax(float &min, float &max, IPosition &minPos,"
                     "IPosition &maxPos, const Array<float, Alloc> &array) - "
                     "const Array<float, Alloc> &weight) - "
                     "Array has no elements"));
  }
  if (!array.shape().isEqual(weight.shape())) {
    throw(ArrayConformanceError("void minMaxMasked(float &min, float &max,"
                                "IPosition &minPos, IPosition &maxPos, "
                                "const Array<float, Alloc> &array, "
                                "const Array<float, Alloc> &weight) - array "
                                "and weight do not have the same shape()"));
  }
  size_t minp = 0;
  size_t maxp = 0;
  float minv = array.data()[0] * weight.data()[0];
  float maxv = minv;
  if (array.contiguousStorage() && weight.contiguousStorage() && n < INT_MAX) {

#ifdef __AVX__
    const float *arrRaw = array.data();
    const float *weightRaw = weight.data();
    minMaxAVX(arrRaw, weightRaw, n, minv, minp, maxv, maxp);
#else
    typename Array<float, Alloc>::const_contiter iter = array.cbegin();
    typename Array<float, Alloc>::const_contiter witer = weight.cbegin();
    for (size_t i = 0; i < n; ++i, ++iter, ++witer) {
      float tmp = *iter * *witer;
      if (tmp < minv) {
        minv = tmp;
        minp = i;
      } else if (tmp > maxv) {
        maxv = tmp;
        maxp = i;
      }
    }
#endif
  } else {
    typename Array<float, Alloc>::const_iterator iter = array.begin();
    typename Array<float, Alloc>::const_iterator witer = weight.begin();
    for (size_t i = 0; i < n; ++i, ++iter, ++witer) {
      float tmp = *iter * *witer;
      if (tmp < minv) {
        minv = tmp;
        minp = i;
      } else if (tmp > maxv) {
        maxv = tmp;
        maxp = i;
      }
    }
  }
  minPos.resize(array.ndim());
  maxPos.resize(array.ndim());
  minPos = toIPositionInArray(minp, array.shape());
  maxPos = toIPositionInArray(maxp, array.shape());
  minVal = minv;
  maxVal = maxv;
}
