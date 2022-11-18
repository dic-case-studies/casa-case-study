#!/usr/bin/env bash

set -e
set -x

declare -a SIZE=(1024 2048 4096 8192 16384 32768)
# declare -a SIZE=(1024 2048)

rm -f runs.txt

for sz in "${SIZE[@]}"
do
  echo "min-max $sz"
  ./min-max-bench $sz >> runs.txt
done

cat runs.txt | awk '                          \
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
' > stats.csv

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
  plot \"stats.csv\" using 1:2 with linespoint title \"Golden\", \
       \"stats.csv\" using 1:3 with linespoint title \"SSE\",    \
       \"stats.csv\" using 1:4 with linespoint title \"AVX\";    \
" | gnuplot > performance.png
