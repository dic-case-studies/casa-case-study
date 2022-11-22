declare -a CASES=(min-max-bench min-max-masked-bench min-max-pos-bench)
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
    make build/${case} OPT="-U GOLDEN -U SSE -U AVX -U ASSERT -D $method"
    for sz in "${SIZE[@]}"
    do
    echo "Running ${case} ${method} ${sz}" >> stat/$host/energy/${case}-result.txt  
    perf stat -e power/energy-pkg/ ./build/${case} ${sz} &>> stat/$host/energy/${case}-result.txt
    done
done
echo "----------------------------" >> stat/$host/energy/${case}-result.txt 
done

# for case in ${CASES[@]}
# do
# cat stat/$host/energy/$case-result.txt | awk '                          \
#   /Matrix dim/ {                              \
#     size = $NF;                               \
#   }                                           \
#   /Elapsed time golden/ {                     \
#     golden = $(NF-1);                         \
#   }                                           \
#   /Elapsed time SIMD SSE/ {                   \
#     sse = $(NF-1);                            \
#   }                                           \
#   /Elapsed time SIMD AVX/ {                   \
#     avx = $(NF-1);                            \
#     printf("%s, %s, %s, %s\n", size, golden, sse, avx); \
#   }                                           \
# ' > stat/$host/$bench-stats.csv
# done

    

