# Casa Case Study

## Build

Pre-requisites: casa
Install using following commands:

```sh
brew tap ska-sa/tap
brew install casacore
```

## Benchmarking

To run benchmarks:

```sh
# 1. Run make to build project
make
# 2. Run scripts to run benchmarks
bench.sh <executable-to-run>
# or for casa benchmaring
benchCASA.sh
```

-------------

## Performance

1. MinMax with OpenMp and SIMD

![Stats](./stat/min-max-bench-performance.png)

2. MinMaxPos with OpenMp and SIMD

![Stats](./stat/min-max-pos-bench-performance.png)

3. MinMaxMasked with OpenMp and SIMD

![Stats](./stat/min-max-masked-bench-performance.png)

4. Casa benchchmark with Openmp and SIMD

![Stats](./stat/casa-bench-performance.png)
