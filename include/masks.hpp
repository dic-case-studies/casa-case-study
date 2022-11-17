#ifndef include_masks_hpp
#define include_masks_hpp

#include <casacore/casa/Arrays.h>

using casacore::IPosition;
using casacore::Array;

template<typename T, typename Alloc> 
void minMaxMaskedWithOpenMp(T &minVal, T &maxVal, 
                  IPosition &minPos, IPosition &maxPos,
                  const Array<T, Alloc> &array, const Array<T, Alloc> &weight) ;


#endif /* include_masks_hpp */