// helpers.hxx

#pragma once

__device__ static float atomicMin(float* address, float value) {
  int* addr_as_int = reinterpret_cast<int*>(address);
  int old = *addr_as_int;
  int expected;
  do {
    expected = old;
    old = atomicCAS(addr_as_int, expected, __float_as_int(::fminf(value, __int_as_float(expected))));
  } while (expected != old);
  return __int_as_float(old);
}

struct cuda_timer_t {
  float time;

  cuda_timer_t() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    cudaEventRecord(start_);
  }

  ~cuda_timer_t() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start() { cudaEventRecord(start_); }
  
  float stop() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&time, start_, stop_);

    return microseconds();
  }
  
  float microseconds() { return (long long)(1000 * time); }

 private:
  cudaEvent_t start_, stop_;
};