#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include "cuda_runtime.h"

extern "C" void CopyIfLessThanOrEqual( double min,
                                   int *const input,
                                   int input_count,
                                   double * stencil,
                                   int * output,
                                   int * output_count,
                                   void* cuda_stream);

int main(int argc, char **argv)
{
  int count = 1000000;

  thrust::host_vector<int> h_input(count);
  thrust::sequence(h_input.begin(), h_input.end());

  thrust::host_vector<double> h_stencil(count);
  thrust::fill(h_stencil.begin(), h_stencil.end() - count/2, 1.5);
  thrust::fill(h_stencil.begin()+count/2, h_stencil.end(), 2.0);

  thrust::host_vector<int> h_output(count);
  thrust::host_vector<int> h_output2(count);

  thrust::device_vector<int> d_input = h_input;
  thrust::device_vector<double> d_stencil =h_stencil;
  thrust::device_vector<int> d_output = h_output;
  thrust::device_vector<int> d_output2 = h_output2;
  int output_count, output_count2;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  CopyIfLessThanOrEqual(
                        1.75,
                        thrust::raw_pointer_cast(d_input.data()),
                        count,
                        thrust::raw_pointer_cast(d_stencil.data()),
                        thrust::raw_pointer_cast(d_output.data()),
                        &output_count,
                        stream);

  CopyIfLessThanOrEqual(
                        3.0,
                        thrust::raw_pointer_cast(d_input.data()),
                        count,
                        thrust::raw_pointer_cast(d_stencil.data()),
                        thrust::raw_pointer_cast(d_output2.data()),
                        &output_count2,
                        stream);

  std::cout<<"copied: "<<output_count<<", "<<output_count2<<std::endl;
}
