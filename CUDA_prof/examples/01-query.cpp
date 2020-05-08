/**
 * Detect the number of CUDA_prof capable devices.
 */

#include <iostream>

#include <cuda_runtime.h>


int main()
{
  int count = 0;

  cudaGetDeviceCount( &count );

  std::cout << count << " device(s) found.\n";
  
  return 0;
}
