#include <iostream>
#include <vector>


__global__ void vecadd( int * v0, int * v1, std::size_t size )
{
  auto tid = threadIdx.x;
  v0[ tid ] += v1[ tid ];
}


int main()
{
  cudaError_t err;
  
  std::size_t const size = 100;
  std::size_t const sizeb = size * sizeof( int );
  
  int * v0_h = nullptr;
  int * v1_h = nullptr;
  
  int * v0_d = nullptr;
  int * v1_d = nullptr;
  
  err = cudaMallocHost( &v0_h, sizeb );
  if( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
  err = cudaMallocHost( &v1_h, sizeb );
  if( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }

  for( std::size_t i = 0 ; i < size ; ++i )
  {
    v0_h[ i ] = v1_h[ i ] = i;
  }

  err = cudaMalloc( &v0_d, sizeb );
  if( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
  err = cudaMalloc( &v1_d, sizeb );
  if( err != cudaSuccess ) { std::cerr << "Error" << std::endl; }
  
  cudaStream_t streams[ 2 ];

  for( std::size_t i = 0 ; i < 2 ; ++i )
  {
    cudaStreamCreate( &streams[ i ] );
  }

  for( std::size_t i = 0 ; i < 2 ; ++i )
  {
    cudaMemcpyAsync( v0_d + i*size/2, v0_h + i*size/2, sizeb/2, cudaMemcpyHostToDevice, streams[ i ] );
    cudaMemcpyAsync( v1_d + i*size/2, v1_h + i*size/2, sizeb/2, cudaMemcpyHostToDevice, streams[ i ] );
  }

  for( std::size_t i = 0 ; i < 2 ; ++i )
  {
    vecadd<<< 1, size/2, 0, streams[ i ] >>>( v0_d + i*size/2, v1_d + i*size/2, size/2 );
  }
  /*
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if( err != cudaSuccess )
  {
     std::cout << cudaGetErrorString( err ) << std::endl;
  }
*/

  for( std::size_t i = 0 ; i < 2 ; ++i )
  {
    cudaMemcpyAsync( v0_h + i*size/2, v0_d + i*size/2, sizeb/2, cudaMemcpyDeviceToHost, streams[ i ] );
  }

  cudaDeviceSynchronize();
  /*
  err = cudaGetLastError();
  if( err != cudaSuccess )
  {
     std::cout << cudaGetErrorString( err ) << std::endl;
  }
*/
  for( std::size_t i = 0 ; i < 2 ; ++i )
  {  
    cudaStreamDestroy( streams[ i ] );
  }

  for( std::size_t i = 0 ; i < size ; ++i )
  {
    std::cout << v0_h[ i ] << std::endl;
  }

  cudaFree( v0_d );
  cudaFree( v1_d );

  cudaFreeHost( v0_h );
  cudaFreeHost( v1_h );

  return 0;
}