#!/usr/bin/env bash

set -e

declare -a SIZE=(1024 2048 4096 8192 16384 32768)

if [[ $# -ne 2 ]] 
then
  echo "Usage: $0 host bench"
  exit 1
fi

host=$1
bench=$2
mkdir -p stat/$host

rm -f stat/$host/$bench-result.txt stat/$host/$bench-stats.csv stat/$host/$bench-performance.png

for sz in "${SIZE[@]}"
do
  echo "min-max $sz"
  ./build/$bench $sz >> stat/$host/$bench-result.txt
done

cat stat/$host/$bench-result.txt | awk '                          \
  /Matrix dim/ {                              \
    size = $NF;                               \
  }                                           \
  /Elapsed time golden/ {                     \
    golden = $(NF-1);                         \
  }                                           \
  /Elapsed time SIMD SSE/ {                   \
    sse = $(NF-1);                            \
  }                                           \
  /Elapsed time SIMD NEON/ {                   \
    neon = $(NF-1);                            \
  }                                           \
  /Elapsed time SIMD Apple/ {                   \
    apple_simd = $(NF-1);                            \
    printf("%s, %s, %s, %s, %s\n", size, golden, sse, neon, apple_simd); \
  }                                           \
' > stat/$host/$bench-stats.csv

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
  plot \"stat/$host/$bench-stats.csv\" using 1:2 with linespoint title \"Golden\", \
       \"stat/$host/$bench-stats.csv\" using 1:3 with linespoint title \"SSE2NEON\",  \
       \"stat/$host/$bench-stats.csv\" using 1:4 with linespoint title \"NEON\",   \
       \"stat/$host/$bench-stats.csv\" using 1:5 with linespoint title \"APPLE SIMD\";   \
" | gnuplot > stat/$host/$bench-performance.png
