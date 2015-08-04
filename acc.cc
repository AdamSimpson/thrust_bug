#include "stdio.h"
#include "stdlib.h"
#include "openacc.h"

extern "C" void CopyIfLessThanOrEqual( double min,
                                   int *const input,
                                   int input_count,
                                   double * stencil,
                                   int * output,
                                   int * output_count,
                                   void* cuda_stream);

extern "C" void printStreamFlags(void *stream);

int main(int argc, char** argv)
{
  int count = 1000000;
  int *input = (int*)malloc(count*sizeof(int));
  double *stencil = (double*)malloc(count*sizeof(double));
  int *output = (int*)malloc(count*sizeof(int));
  int *output2 = (int*)malloc(count*sizeof(int));

  for(int i=0; i<count; i++)
    input[i] = i;

  for(int i=0; i<count/2; i++)
    stencil[i] = 1.5;
  for(int i=count/2; i<count; i++)
    stencil[i] = 2.0;

  #pragma acc enter data copyin(input[:count], stencil[:count], output[:count], output2[:count])

  int out_count, out_count2;
  void* cuda_stream = acc_get_cuda_stream(acc_async_sync);

  printStreamFlags(cuda_stream);

  #pragma acc host_data use_device(input, stencil, output, output2)
  {
  CopyIfLessThanOrEqual(1.75,
               input,
               count,
               stencil,
               output,
               &out_count,
               cuda_stream);

  CopyIfLessThanOrEqual(3.0,
               input,
               count,
               stencil,
               output2,
               &out_count2,
               cuda_stream);
  }

  printf("num copied: %d %d\n", out_count, out_count2);

  return 0;
}
