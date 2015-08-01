#include <thrust/execution_policy.h>
#include <thrust/version.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <thrust/system_error.h>
#include "cuda_runtime.h"

struct LessThanOrEqual {
  LessThanOrEqual(double min): x_min(min) {}

  double x_min;

  __host__ __device__
  bool operator()(const double x) {
    return (x <= x_min);
  }
};


// C wrapper function for thrust copy_if with LessThanOrEqual predicate
extern "C" void CopyIfLessThanOrEqual(double min,
                                      int *input,
                                      int input_count,
                                      double * stencil,
                                      int * output,
                                      int * output_count,
                                      void* cuda_stream) {

  int *end_pointer = thrust::copy_if(thrust::cuda::par.on((cudaStream_t)cuda_stream),
                                     input,
                                     input + input_count,
                                     stencil,
                                     output,
                                     LessThanOrEqual(min));
  *output_count = end_pointer - output;
}
