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

### i7-9750H

1. MinMax with OpenMp and SIMD

![Stats](./stat/i7-9750H/min-max-bench-performance.png)

2. MinMaxPos with OpenMp and SIMD

![Stats](./stat/i7-9750H/min-max-pos-bench-performance.png)

3. MinMaxMasked with OpenMp and SIMD

![Stats](./stat/i7-9750H/min-max-masked-bench-performance.png)

4. Casa benchchmark with Openmp and SIMD

![Stats](./stat/i7-9750H/casa-bench-performance.png)

### Ryzen5800X

1. MinMax with OpenMp and SIMD

![Stats](./stat/Ryzen5800X/min-max-bench-performance.png)

2. MinMaxPos with OpenMp and SIMD

![Stats](./stat/Ryzen5800X/min-max-pos-bench-performance.png)

3. MinMaxMasked with OpenMp and SIMD

![Stats](./stat/Ryzen5800X/min-max-masked-bench-performance.png)

4. Casa benchchmark with Openmp and SIMD

![Stats](./stat/Ryzen5800X/casa-bench-performance.png)

## Energy

### Ryzen 3700X

1. MinMax with OpenMp and SIMD

![Stats](./stat/Ryzen3700X/energy/min-max-bench-performance.png)

2. MinMaxPos with OpenMp and SIMD

![Stats](./stat/Ryzen3700X/energy/min-max-pos-bench-performance.png)

3. MinMaxMasked with OpenMp and SIMD

![Stats](./stat/Ryzen3700X/energy/min-max-masked-bench-performance.png)

### Ryzen 5800X

1. MinMax with OpenMp and SIMD

![Stats](./stat/Ryzen5800X/energy/min-max-bench-performance.png)

2. MinMaxPos with OpenMp and SIMD

![Stats](./stat/Ryzen5800X/energy/min-max-pos-bench-performance.png)

3. MinMaxMasked with OpenMp and SIMD

![Stats](./stat/Ryzen5800X/energy/min-max-masked-bench-performance.png)

