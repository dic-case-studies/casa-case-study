#!/usr/bin/env bash

set -e
set -x

declare -a SIZE=(1024 2048 4096 8192 16384 32768)
# declare -a SIZE=(1024 2048 4096 8192)
# declare -a SIZE=(4096)

mkdir -p stat
bench=$1

rm -f stat/$bench-result.txt stat/$bench-stats.csv stat/$bench-performance.png

for sz in "${SIZE[@]}"
do
  echo "min-max $sz"
  ./build/$bench $sz >> stat/$bench-result.txt
done

cat stat/$bench-result.txt | awk '                          \
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
       \"stat/$bench-stats.csv\" using 1:3 with linespoint title \"SSE\",    \
       \"stat/$bench-stats.csv\" using 1:4 with linespoint title \"AVX\";    \
" | gnuplot > stat/$bench-performance.png
