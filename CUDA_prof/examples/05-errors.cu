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

  // Récupération du code erreur du kernel en cas de plantage.
  cudaDeviceSynchronize(); // Attente de la fin d'exécution du kernel.
  cudaError err = cudaGetLastError();
  if( err != cudaSuccess )
  {
    std::cerr << cudaGetErrorString( err ); // récupération du message associé au code erreur.
  }

  // Récupération du code erreur pour les fonctions CUDA_prof synchrones.
  err = cudaMemcpy( v.data(), v_d, v.size() * sizeof( int ), cudaMemcpyDeviceToHost );
  if( err != cudaSuccess )
  {
    std::cerr << cudaGetErrorString( err ); // récupération du message associé au code erreur.
  }

  for( auto x: v )
  {
    std::cout << x << std::endl;
  }

  return 0;
}