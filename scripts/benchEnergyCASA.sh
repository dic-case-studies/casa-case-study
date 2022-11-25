#!/usr/bin/env bash

set -e
set -x

declare -a CASES=(casa-bench)
declare -a METHODS=(GOLDEN SSE AVX)

declare -a SIZE=(1024 2048 4096 8192 16384 32768)
# declare -a SIZE=(1024 2048 4096 8192)
# declare -a SIZE=(4096)

host=$1
mkdir -p stat/$host/energy

for case in ${CASES[@]}
do  
  for method in ${METHODS[@]}
  do
      rm -f build/${case} stat/$host/energy/${case}-${method}-result.txt

      make build/${case} OPT="-UGOLDEN -USSE -UAVX -UASSERT -D$method"
      for sz in "${SIZE[@]}"
      do
      echo "Running ${case} ${method} ${sz}" >> stat/$host/energy/${case}-${method}-result.txt 
      perf stat -e power/energy-pkg/ ./build/${case} ${sz} &>> stat/$host/energy/${case}-${method}-result.txt 
      echo "----------------------------" >> stat/$host/energy/${case}-${method}-result.txt 
      done

      cat stat/$host/energy/${case}-${method}-result.txt   | awk '    \
        /Matrix dim/ {                              \
          size = $NF;                               \
        }                                           \
        /Time taken/ {                   \
          time = $(NF-1);                            \
        }                                           \
        /energy-pkg/ {                              \
          energy = $1;                            \
          printf("%s, %s, %s\n", size, energy, time); \
        }                                           \
      ' > stat/$host/energy/${case}-${method}-stats.csv
  done

  echo "                                            \
    reset;                                          \
    set terminal png enhanced large; \
                                                          \
    set title \"$bench Benchmark\";                        \
    set xlabel \"Matrix Dim\";                             \
    set ylabel \"Joules\";                     \
    set key left top;                                      \
    set logscale x;                                        \
                                                          \
    plot \"stat/$host/energy/${case}-GOLDEN-stats.csv\" using 1:2 with linespoint title \"Golden\", \
        \"stat/$host/energy/${case}-SSE-stats.csv\" using 1:2 with linespoint title \"SSE\",    \
        \"stat/$host/energy/${case}-AVX-stats.csv\" using 1:2 with linespoint title \"AVX\";    \
  " | gnuplot > stat/$host/energy/${case}-performance.png


done
