#pragma once
#include <casacore/casa/Arrays.h>

using casacore::Array;
using casacore::ArrayConformanceError;
using casacore::ArrayError;
using casacore::IPosition;

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
      if (tmp < minv) {
#pragma omp critical
        {
          minv = tmp;
          minp = i;
        }
      } else if (tmp > maxv) {
#pragma omp critical
        {
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
