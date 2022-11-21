#!/usr/bin/env bash

set -e
set -x

declare -a SIZE=(1024 2048 4096 8192 16384 32768)
# declare -a SIZE=(1024 2048 4096 8192)
# declare -a SIZE=(1024)

mkdir -p stat
bench=casa-bench

outputFile=stat/$bench-result.txt

if [ -f $outputFile ]
then
    rm $outputFile
fi

echo "Running "

for sz in "${SIZE[@]}"
do
    ./build/$bench $sz >> $outputFile
done

echo "Done"


cat stat/$bench-result.txt | awk '                          \
  /Matrix dim/ {                              \
    size = $NF;                               \
  }                                           \
  /Time taken for casacore minMaxMasked/ {                     \
    golden = $(NF-1);                         \
  }                                           \
  /Time taken for openmp minMaxMasked/ {                   \
    openmp = $(NF-1);                            \
  }                                           \
  /Time taken for SIMD minMaxMasked/ {                   \
    SIMD = $(NF-1);                            \
    printf("%s, %s, %s, %s\n", size, golden, openmp,SIMD); \
  }                                           \
' > stat/$bench-stats.csv

echo "                                            \
  reset;                                          \
  set terminal png enhanced large font \"Helvetica,10\"; \
                                                         \
  set title \"$bench Benchmark\";                        \
  set xlabel \"Matrix Dim\";                             \
  set ylabel \"Execution time(us)\";                     \
  set key left top;                                      \
  set logscale x;                                        \
  set logscale y;                                        \
                                                         \
  plot \"stat/$bench-stats.csv\" using 1:2 with linespoint title \"Golden\", \
       \"stat/$bench-stats.csv\" using 1:3 with linespoint title \"Openmp\",    \
       \"stat/$bench-stats.csv\" using 1:4 with linespoint title \"SIMD\";    \
" | gnuplot > stat/$bench-performance.png
