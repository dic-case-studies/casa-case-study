#!/usr/bin/env bash

set -e
set -x

declare -a SIZE=(1024 2048 4096 8192 16384 32768)
# declare -a SIZE=(1024 2048)

rm -f stat/min-max-bench-result.txt stat/min-max-bench-stats.csv stat/min-max-bench-performance.png

for sz in "${SIZE[@]}"
do
  echo "min-max $sz"
  ./build/min-max-bench $sz >> stat/min-max-bench-result.txt
done

cat stat/min-max-bench-result.txt | awk '                          \
  /Matrix dim/ {                              \
    size = $NF;                               \
  }                                           \
  /Elapsed time golden/ {                     \
    golden = $(NF-1);                         \
  }                                           \
  /Elapsed time SIMD SSE/ {                   \
    sse = $(NF-1);                            \
  }                                           \
  /Elapsed time SIMD AVX/ {                   \
    avx = $(NF-1);                            \
    printf("%s, %s, %s, %s\n", size, golden, sse, avx); \
  }                                           \
' > stat/min-max-bench-stats.csv

echo "                                            \
  reset;                                          \
  set terminal png enhanced large font \"Helvetica,10\"; \
                                                         \
  set title \"minMax Benchmark\";                        \
  set xlabel \"Matrix Dim\";                             \
  set ylabel \"Execution time(us)\";                     \
  set key left top;                                      \
  set logscale x;                                        \
  set logscale y;                                        \
                                                         \
  plot \"stat/min-max-bench-stats.csv\" using 1:2 with linespoint title \"Golden\", \
       \"stat/min-max-bench-stats.csv\" using 1:3 with linespoint title \"SSE\",    \
       \"stat/min-max-bench-stats.csv\" using 1:4 with linespoint title \"AVX\";    \
" | gnuplot > stat/min-max-bench-performance.png
