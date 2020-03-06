#include <iostream>
#include <vector>


__global__ void fill( int * v, std::size_t size )
{
  auto tid = threadIdx.x;
  v[ tid ] = tid;
}


int main()
{
  std::vector< int > v( 100 );

  int * v_d = nullptr;

  cudaMalloc( &v_d, v.size() * sizeof( int ) );

  fill<<< 1, 100 >>>( v_d, v.size() );

  cudaMemcpy( v.data(), v_d, v.size() * sizeof( int ), cudaMemcpyDeviceToHost );

  for( auto x: v )
  {
    std::cout << x << std::endl;
  }

  return 0;
}