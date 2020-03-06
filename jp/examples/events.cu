#include <iostream>
#include <vector>

__global__ void matadd( int * m0, int * m1, std::size_t w, std::size_t h )
{
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;
  if( i < w && j < h )
    m0[ i * w + j ] +=  m1[ i * w + j ];// i * w + j;
}

int main() {
  std::vector< int > v0_h( 10000 );
  std::vector< int > v1_h( 10000 );  
  for( std::size_t i = 0 ; i < v0_h.size(); ++i ) {
    v0_h[ i ] = v1_h[ i ] = i;
  }
  int * v0_d = nullptr;
  int * v1_d = nullptr;
  cudaMalloc( &v0_d, v0_h.size() * sizeof( int ) );
  cudaMalloc( &v1_d, v0_h.size() * sizeof( int ) );
  cudaMemcpy( v0_d, v0_h.data(), v0_h.size() * sizeof( int ), cudaMemcpyHostToDevice );
  cudaMemcpy( v1_d, v1_h.data(), v0_h.size() * sizeof( int ), cudaMemcpyHostToDevice );
  dim3 t( 32, 32 );
  dim3 b( 4, 4 );

  cudaEvent_t start, stop;
  cudaEventCreate( &start );
  cudaEventCreate( &stop );
  
  cudaEventRecord( start );
  
  matadd<<< b, t >>>( v0_d, v1_d, 100, 100 );

  cudaEventRecord( stop );

  cudaEventSynchronize( stop );

  float elapsedTime;
  cudaEventElapsedTime( & elapsedTime, start, stop );
  std::cout << elapsedTime << std::endl;
  cudaEventDestroy( start );
  cudaEventDestroy( stop );
  auto err = cudaGetLastError();

  cudaMemcpy( v0_h.data(), v0_d, v0_h.size() * sizeof( int ), cudaMemcpyDeviceToHost );
  //for( auto const i: v0_h ) { std::cout << i << std::endl; }
  cudaFree( v0_d );
  cudaFree( v1_d );
  return 0;
}
